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

def run_exam(exam_id: str, items: list[ExamItem],
             answerer: Callable[[str],str],
             grader: Callable[[str,str],bool] = exact_match,
             *, threshold: float=0.9) -> ExamReport:
    if not items or not 0 < threshold <= 1:
        raise ValueError("Invalid examination")
    if len({item.id for item in items})!=len(items):
        raise ValueError("Duplicate examination item")
    categories={x.category for x in items}
    if len(categories)!=1:
        raise ValueError("Use independent per-category examinations")
    answers=[answerer(item.prompt) for item in items]
    correct=sum(bool(grader(answer,item.expected)) for answer,item in zip(answers,items))
    digest=sha256(json.dumps(answers,ensure_ascii=False).encode()).hexdigest()
    return ExamReport(exam_id,len(items),correct,correct/len(items)>=threshold,
                      next(iter(categories)),digest)

def distinct_premise_exam(base: ExamItem, variant: ExamItem) -> None:
    if base.prompt==variant.prompt or base.expected==variant.expected:
        raise ValueError("Changed-premise exam must change both premise and expected answer")
