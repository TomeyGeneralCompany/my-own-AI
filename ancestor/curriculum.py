"""Controller-directed, owner-source-specific paragraph learning pipeline.

A learning objective cannot graduate through accumulated steps or repeated answers
alone; a separately supplied examination bank must pass the stage gate.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import random
import torch

from .controller import Controller,Objective,State
from .exams import ExamItem,run_exam
from .ledger import StageRequirement
from .training import Trainer,structured_prefix_batch
from .checkpoints import save_checkpoint

STAGE_ORDER = (
    "READINESS", "ELEMENTARY_ENGLISH", "WEBSTER_DICTIONARY",
    "GRAMMAR_READING_COMPOSITION", "ORDERED_MANUSCRIPTS",
    "FORMAL_MATHEMATICS", "CROSS_BOOK_SYNTHESIS",
    "FOUNDATION_CERTIFICATION",
)

@dataclass(frozen=True)
class Lesson:
    id: str
    source_id: str
    task: str
    prerequisites: tuple[str, ...] = ()
    exam_id: str = ""
    updates: int = 1

@dataclass(frozen=True)
class CourseStage:
    id: str
    lessons: tuple[Lesson, ...]
    retention_exam_ids: tuple[str, ...] = ()

def load_course(path: str | Path) -> list[CourseStage]:
    obj=json.loads(Path(path).read_text(encoding="utf-8"))
    stages=[]
    seen=set()
    order_index=-1
    for stage in obj["stages"]:
        if stage["id"] not in STAGE_ORDER:
            raise ValueError(f"Unknown training stage: {stage['id']}")
        new_index=STAGE_ORDER.index(stage["id"])
        if new_index <= order_index:
            raise ValueError("Curriculum stages out of prescribed order")
        order_index=new_index
        lessons=[]
        for item in stage["lessons"]:
            if item["id"] in seen:
                raise ValueError("Duplicate lesson ID")
            seen.add(item["id"])
            if item.get("updates",1)<1 or not item.get("exam_id"):
                raise ValueError("Every lesson needs a positive update budget and independent exam")
            lessons.append(Lesson(item["id"],item["source_id"],item["task"],
                                  tuple(item.get("prerequisites",[])),
                                  item["exam_id"],item.get("updates",1)))
        stages.append(CourseStage(stage["id"],tuple(lessons),
                                  tuple(stage.get("retention_exam_ids",[]))))
    return stages

def load_exam_bank(path: str | Path) -> dict[str,list[ExamItem]]:
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    bank={}
    for exam_id, questions in data.items():
        if not questions:
            raise ValueError(f"Empty exam: {exam_id}")
        bank[exam_id]=[ExamItem(item["id"],item["prompt"],item["expected"],item["category"],
                                tuple(item.get("source_ids",[]))) for item in questions]
    return bank

class CurriculumRunner:
    def __init__(self, controller: Controller, trainer: Trainer,
                 tokenizer_hash: str, corpus_hash: str, checkpoint_dir: str | Path):
        self.controller=controller
        self.trainer=trainer
        self.tokenizer_hash=tokenizer_hash
        self.corpus_hash=corpus_hash
        self.checkpoint_dir=Path(checkpoint_dir)

    def lesson_step(self, lesson: Lesson, *, reserve: int=128) -> float:
        """Generate the target from the approved source paragraph with earlier premises."""
        objective=Objective(lesson.id,lesson.source_id,lesson.task,lesson.prerequisites)
        window=self.controller.formulate(objective,reserve)
        tokenizer=self.controller.context.tokenizer
        target=self.controller.corpus.get(lesson.source_id).exact_text
        dependencies=[self.controller.corpus.get(src) for src in lesson.prerequisites]
        context_text="\n\n".join(f"[{item.id}]\n{item.exact_text}" for item in dependencies)
        sections=[
            ("<|AXIOM|>",self.controller.constitution.axiom_prefix()),
            ("<|CONTEXT|>",context_text),
            ("<|DERIVE|>",lesson.task+"\n"+target),
        ]
        batch=structured_prefix_batch(tokenizer,sections,self.trainer.config.block_size)
        if self.trainer.config.microbatch!=1 or self.trainer.config.accumulation!=1:
            raise ValueError("Curriculum lesson runner currently supports microbatch=1, accumulation=1")
        self.controller.state=State.TRAINING
        try:
            loss=self.trainer.step([batch])
            self.controller.ledger.append({"type":"LESSON_OPTIMIZATION",
                "objective_id":lesson.id,"source_id":lesson.source_id,
                "updates":self.trainer.completed_updates,"loss":loss,
                "constitution_hash":self.controller.constitution.digest()})
        finally:
            self.controller.state=State.READY
        return loss

    def teach(self, lesson: Lesson):
        objective=Objective(lesson.id,lesson.source_id,lesson.task,lesson.prerequisites)
        for _ in range(lesson.updates):
            self.lesson_step(lesson)
        self.controller.record_coverage(objective)
        return save_checkpoint(self.checkpoint_dir,self.trainer,
            tokenizer_hash=self.tokenizer_hash,constitution=self.controller.constitution,
            corpus_manifest_hash=self.corpus_hash,stage="LESSON_COMPLETED",
            objective_id=lesson.id)

    @torch.no_grad()
    def answer(self, prompt: str, *, context_sources: tuple[str,...]=(),
               max_new_tokens: int=48) -> str:
        tokenizer=self.controller.context.tokenizer
        sources=[(src,self.controller.corpus.get(src).exact_text) for src in context_sources]
        window=self.controller.context.build(sources,prompt,reserve=max_new_tokens)
        inp=torch.tensor([window.token_ids],dtype=torch.long)
        all_ids=self.trainer.model.generate(inp,max_new_tokens)
        new_ids=all_ids[0,len(window.token_ids):].tolist()
        end=tokenizer.special_id("<|END|>")
        if end in new_ids:
            new_ids=new_ids[:new_ids.index(end)]
        # Ordinary output must not synthesize privileged control delimiters.
        new_ids=[x for x in new_ids if x < tokenizer.base_vocab_size]
        try:
            return tokenizer.decode(new_ids)
        except UnicodeDecodeError:
            return ""

    def examine(self, exam_id: str, items: list[ExamItem], *,
                threshold: float=0.9, sources: tuple[str,...]=(),
                independence: bool=True):
        if not independence:
            raise ValueError("Cannot assert independent exam with answer-bearing retrieval")
        report=run_exam(exam_id,items,lambda prompt:self.answer(prompt,context_sources=sources),
                        threshold=threshold)
        self.controller.evaluate(exam_id,passed=report.passed,
            evidence={"questions":report.questions,"correct":report.correct,
                      "category":report.category,"answer_digest":report.answer_digest},
            independent=True)
        return report

    def run_stage(self, stage: CourseStage, exams: dict[str,list[ExamItem]],
                  *, threshold: float=0.9):
        if self.controller.state not in (State.READY,):
            raise RuntimeError("Controller not ready")
        required=[]
        for lesson in stage.lessons:
            self.teach(lesson)
            if lesson.exam_id not in exams:
                raise ValueError("Missing independent exam bank")
            result=self.examine(lesson.exam_id,exams[lesson.exam_id],threshold=threshold)
            required.append(lesson.exam_id)
            if not result.passed:
                self.controller.diagnose("INDEPENDENT_EXAM_FAILURE",lesson.prerequisites)
                # No autonomous model changes / no dishonest stage promotion.
                return False
        for retention_id in stage.retention_exam_ids:
            if retention_id not in exams:
                raise ValueError("Missing retention exam")
            if not self.examine(retention_id,exams[retention_id],threshold=threshold).passed:
                return False
        self.controller.stage_gate(StageRequirement(stage.id,
            tuple(x.source_id for x in stage.lessons),tuple(required),
            stage.retention_exam_ids))
        return True
