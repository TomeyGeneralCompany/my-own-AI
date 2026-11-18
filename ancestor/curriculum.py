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
from .training import Trainer,controller_completion_batch
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
    remediation_lessons: tuple[tuple[str,str], ...] = ()

@dataclass(frozen=True)
class CourseStage:
    id: str
    lessons: tuple[Lesson, ...]
    retention_exam_ids: tuple[str, ...] = ()
    approved_entry_count: int | None = None
    approved_document_ids: tuple[str, ...] = ()

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
            remedies=tuple((r["source_id"],r["task"]) for r in item.get("remediation_lessons",[]))
            lessons.append(Lesson(item["id"],item["source_id"],item["task"],
                                  tuple(item.get("prerequisites",[])),
                                  item["exam_id"],item.get("updates",1),remedies))
        stages.append(CourseStage(stage["id"],tuple(lessons),
                                  tuple(stage.get("retention_exam_ids",[])),
                                  stage.get("approved_entry_count"),
                                  tuple(stage.get("approved_document_ids",[]))))
    return stages

def load_exam_bank(path: str | Path) -> dict[str,list[ExamItem]]:
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    bank={}
    for exam_id, questions in data.items():
        if not questions:
            raise ValueError(f"Empty exam: {exam_id}")
        bank[exam_id]=[ExamItem(item["id"],item["prompt"],item["expected"],item["category"],
                                tuple(item.get("source_ids",[])),
                                item.get("grading","exact"),
                                tuple(item.get("required_reasoning_steps",[])),
                                tuple(item.get("required_axioms",[]))) for item in questions]
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
        """Teach every byte of a paragraph, segmenting only when necessary."""
        tokenizer=self.controller.context.tokenizer
        current=self.controller.corpus.get(lesson.source_id)
        dependencies=[self.controller.corpus.get(src) for src in lesson.prerequisites]
        context_text="\n\n".join(f"[{item.id}]\n{item.exact_text}" for item in dependencies)
        # Controller still validates mandatory authority/context. Current source
        # text is the target and must not be dropped or silently shortened.
        self.controller.context.build(
            [(item.id,item.exact_text) for item in dependencies],lesson.task,
            reserve=max(1,min(reserve,self.trainer.config.block_size//4)))
        header=[
            ("<|AXIOM|>",self.controller.constitution.axiom_prefix()),
            ("<|CONTEXT|>",context_text),
            ("<|DERIVE|>",lesson.task+"\n"),
        ]
        # The required training target and <|END|> must fit the neural block
        # even if the controller itself has a larger working-context tier.
        header_tokens=tokenizer.encode_structured(header)
        remaining=self.trainer.config.block_size-len(header_tokens)-1
        if remaining < 1:
            raise ValueError("Required premises plus task exceed neural working tier")
        segments=self.controller.context.paragraph_segments(current.exact_text,remaining)
        scores=[]
        self.controller.state=State.TRAINING
        try:
            for i,segment in enumerate(segments):
                batch=controller_completion_batch(
                    tokenizer,self.controller.constitution.axiom_prefix(),
                    context_text,lesson.task,segment,self.trainer.config.block_size)
                if self.trainer.config.microbatch!=1 or self.trainer.config.accumulation!=1:
                    raise ValueError("Lesson runner currently requires microbatch=1, accumulation=1")
                loss=self.trainer.step([batch])
                scores.append(loss)
                self.controller.ledger.append({"type":"LESSON_OPTIMIZATION",
                    "objective_id":lesson.id,"source_id":lesson.source_id,
                    "segment":i+1,"total_segments":len(segments),
                    "updates":self.trainer.completed_updates,"loss":loss,
                    "constitution_hash":self.controller.constitution.digest()})
        finally:
            self.controller.state=State.READY
        return sum(scores)/len(scores)

    def _checkpoint(self, stage: str, objective_id: str):
        return save_checkpoint(self.checkpoint_dir,self.trainer,
            tokenizer_hash=self.tokenizer_hash,constitution=self.controller.constitution,
            corpus_manifest_hash=self.corpus_hash,stage=stage,objective_id=objective_id,
            ledger_offset=len(self.controller.ledger.events()))

    def teach(self, lesson: Lesson):
        objective=Objective(lesson.id,lesson.source_id,lesson.task,lesson.prerequisites)
        for _ in range(lesson.updates):
            self.lesson_step(lesson)
        self.controller.record_coverage(objective)
        return self._checkpoint("LESSON_COMPLETED",lesson.id)

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

    def _require_complete_source_coverage(self, stage: CourseStage):
        claimed={lesson.source_id for lesson in stage.lessons}
        corpus=self.controller.corpus
        if stage.id=="WEBSTER_DICTIONARY":
            indexed={item.id for item in corpus.dictionary_records()}
            if (stage.approved_entry_count is None or not indexed
                    or len(indexed)!=stage.approved_entry_count or claimed!=indexed):
                raise ValueError("Webster stage must cover every owner-verified lexical entry")
        if stage.id=="ORDERED_MANUSCRIPTS":
            if not stage.approved_document_ids:
                raise ValueError("Approved full manuscript document inventory is required")
            indexed=set()
            for doc_id in stage.approved_document_ids:
                records=corpus.list_paragraphs(doc_id)
                if not records:
                    raise ValueError(f"Missing approved manuscript paragraphs: {doc_id}")
                indexed.update(item.id for item in records)
            if claimed!=indexed:
                raise ValueError("Manuscript stage must cover every numbered paragraph in approved books")

    def run_stage(self, stage: CourseStage, exams: dict[str,list[ExamItem]],
                  *, threshold: float=0.9):
        self._require_complete_source_coverage(stage)
        if self.controller.state != State.READY:
            raise RuntimeError("Controller not ready")
        for lesson in stage.lessons:
            if lesson.exam_id not in exams:
                raise ValueError("Missing independent exam bank")
        for retention_id in stage.retention_exam_ids:
            if retention_id not in exams:
                raise ValueError("Missing retention exam bank")
        completed=self.controller.ledger.exam_results()
        covered=self.controller.ledger.covered()
        required=[]
        for lesson in stage.lessons:
            # A previously tested lesson is not trained a second time simply
            # because the process was restarted; its source and exam identities
            # have to belong to the exact checkpoint-bound curriculum.
            if not (lesson.source_id in covered and completed.get(lesson.exam_id,False)):
                self.teach(lesson)
                result=self.examine(lesson.exam_id,exams[lesson.exam_id],threshold=threshold)
                self._checkpoint("EXAM_COMPLETED",lesson.id)
                if not result.passed:
                    self.controller.diagnose("INDEPENDENT_EXAM_FAILURE",lesson.prerequisites)
                    for n,(remedy_source,remedy_task) in enumerate(lesson.remediation_lessons,1):
                        # Remediation material is explicitly source-bound and
                        # owner-approved in the course. Never invent it.
                        self.controller.ledger.append({"type":"REMEDIATION_SELECTED",
                            "objective_id":lesson.id,"remediation_number":n,
                            "source_id":remedy_source})
                        self.controller.state=State.READY
                        substitute=Lesson(lesson.id+f":REMED-{n}",remedy_source,
                            remedy_task,lesson.prerequisites,lesson.exam_id,
                            lesson.updates)
                        self.teach(substitute)
                        result=self.examine(lesson.exam_id,exams[lesson.exam_id],
                                           threshold=threshold)
                        self._checkpoint("REMEDIATION_EXAMINED",lesson.id)
                        if result.passed:
                            break
                    if not result.passed:
                        self._checkpoint("REMEDIATION_REQUIRED",lesson.id)
                        return False
            required.append(lesson.exam_id)
        for retention_id in stage.retention_exam_ids:
            if not self.examine(retention_id,exams[retention_id],threshold=threshold).passed:
                self._checkpoint("RETENTION_FAILED",retention_id)
                return False
            self._checkpoint("RETENTION_PASSED",retention_id)
        self.controller.stage_gate(StageRequirement(stage.id,
            tuple(x.source_id for x in stage.lessons),tuple(required),
            stage.retention_exam_ids))
        self._checkpoint("STAGE_COMPLETED",stage.id)
        return True
