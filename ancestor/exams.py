"""Independent examination contracts. Self-reported answers are never accepted as grading."""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Callable

@dataclass(frozen=True)
class ExamItem:
    id: str
    prompt: str
    expected: str
    category: str
    source_ids: tuple[str,...] = ()
    grading: str = "exact"
    required_reasoning_steps: tuple[str,...] = ()
    required_axioms: tuple[str,...] = ()

@dataclass(frozen=True)
class ExamReport:
    exam_id: str
    questions: int
    correct: int
    passed: bool
    category: str
    answer_digest: str

def exact_match(answer: str, expected: str) -> bool:
    return answer == expected

def rubric_grade(answer: str, item: ExamItem) -> bool:
    if item.grading in ("exact","normalized","contains") and not item.expected:
        raise ValueError("Examination reference answer must not be empty")
    if item.grading=="ordered_derivation" and (not item.required_reasoning_steps or not item.required_axioms):
        raise ValueError("Axiomatic derivation requires both governing premises and ordered reasoning steps")
    if item.grading=="exact":
        core=(answer==item.expected)
    elif item.grading=="normalized":
        core=(answer.strip().casefold()==item.expected.strip().casefold())
    elif item.grading=="contains":
        core=(item.expected in answer)
    elif item.grading=="ordered_derivation":
        core=True
    else:
        raise ValueError("Unknown owner rubric mode")
    if not core:
        return False
    for principle in item.required_axioms:
        if principle not in answer:
            return False
    position=0
    for required_step in item.required_reasoning_steps:
        found=answer.find(required_step,position)
        if found<0:
            return False
        position=found+len(required_step)
    return True

def run_exam(exam_id: str, items: list[ExamItem],
             answerer: Callable[[str],str],
             grader: Callable[[str,str],bool] | None = None,
             *, threshold: float=0.9) -> ExamReport:
    if not items or not 0 < threshold <= 1:
        raise ValueError("Invalid examination")
    if len({item.id for item in items})!=len(items):
        raise ValueError("Duplicate examination item")
    categories={x.category for x in items}
    if len(categories)!=1:
        raise ValueError("Use independent per-category examinations")
    answers=[answerer(item.prompt) for item in items]
    correct=sum(bool(grader(answer,item.expected) if grader is not None else rubric_grade(answer,item))
                for answer,item in zip(answers,items))
    digest=sha256(json.dumps(answers,ensure_ascii=False).encode()).hexdigest()
    return ExamReport(exam_id,len(items),correct,correct/len(items)>=threshold,
                      next(iter(categories)),digest)

def distinct_premise_exam(base: ExamItem, variant: ExamItem) -> None:
    if base.prompt==variant.prompt or base.expected==variant.expected:
        raise ValueError("Changed-premise exam must change both premise and expected answer")
