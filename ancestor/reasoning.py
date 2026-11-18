"""Owner-authored axiom-first derivation examples for neural instruction.

This module does not invent an external philosophical worldview or redefine axioms.
The exact derivation and its approved references come from the owner's corpus.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
from .corpus import Corpus
from .governance import Constitution
from .tokenizer import ByteBPE
from .training import controller_completion_batch

@dataclass(frozen=True)
class ReasoningExample:
    id: str
    axiom_ids: tuple[int,...]
    premise_source_ids: tuple[str,...]
    question: str
    derivation: str
    conclusion: str
    approved: bool
    discipline: str = ""

def load_reasoning_examples(path: str | Path) -> list[ReasoningExample]:
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    result=[]
    seen=set()
    for item in data:
        example=ReasoningExample(
            id=item["id"],
            axiom_ids=tuple(item["axiom_ids"]),
            premise_source_ids=tuple(item["premise_source_ids"]),
            question=item["question"],
            derivation=item["derivation"],
            conclusion=item["conclusion"],
            approved=bool(item.get("owner_approved",False)),
            discipline=item.get("discipline",""),
        )
        if not example.approved:
            raise PermissionError("Unapproved reasoning example cannot become training data")
        if example.id in seen:
            raise ValueError("Duplicate reasoning example ID")
        seen.add(example.id)
        if not example.axiom_ids or not example.premise_source_ids:
            raise ValueError("Reasoning requires axioms and referenced premises")
        if not example.question.strip() or not example.derivation.strip() or not example.conclusion.strip():
            raise ValueError("Derivation, conclusion and question are required")
        result.append(example)
    return result

def reasoning_sample(example: ReasoningExample, corpus: Corpus,
                     constitution: Constitution, tokenizer: ByteBPE,
                     *, context_length: int):
    axioms=constitution.axioms
    if any(i<0 or i>=len(axioms) for i in example.axiom_ids):
        raise ValueError("Unknown canonical axiom index")
    selected="\n".join(axioms[i] for i in example.axiom_ids)
    context="\n\n".join(f"[{sid}]\n{corpus.get(sid).exact_text}"
                        for sid in example.premise_source_ids)
    target="Derivation:\n"+example.derivation+"\nConsequence:\n"+example.conclusion
    return controller_completion_batch(tokenizer,selected,context,
                                       example.question,target,context_length)

def assert_constitution_is_authoritative(example: ReasoningExample,
                                          constitution: Constitution):
    """A source/example may cite axioms but cannot supply replacement axioms."""
    for index in example.axiom_ids:
        if index>=len(constitution.axioms) or index<0:
            raise ValueError("Missing approved axiomatic reference")
    if not example.approved:
        raise PermissionError("Owner approval missing")
