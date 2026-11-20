"""Stage-gated educational controller centered on the owner's source corpus."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from .corpus import Corpus
from .context import ContextBuilder, ContextWindow
from .governance import Constitution
from .ledger import LearningLedger, StageRequirement

class State(str, Enum):
    READY="READY"
    TRAINING="TRAINING"
    EXAMINING="EXAMINING"
    REMEDIATING="REMEDIATING"
    CHECKPOINTING="CHECKPOINTING"
    PAUSED="PAUSED"
    FAILED="FAILED"
    CERTIFIED="CERTIFIED"

@dataclass(frozen=True)
class Objective:
    objective_id: str
    record_id: str
    task: str
    required_dependencies: tuple[str,...] = ()

class Controller:
    def __init__(self, corpus: Corpus, context: ContextBuilder,
                 ledger: LearningLedger, constitution: Constitution):
        if context.constitution.digest() != constitution.digest():
            raise ValueError("Controller and context governance disagree")
        self.corpus=corpus
        self.context=context
        self.ledger=ledger
        self.constitution=constitution
        self.state=State.READY

    def formulate(self, objective: Objective, generation_reserve: int=128) -> ContextWindow:
        """Construct context with authoritative axioms and exact source prerequisites."""
        if self.state not in (State.READY,State.REMEDIATING):
            raise RuntimeError("Controller state does not permit objective formulation")
        required=[self.corpus.get(sid) for sid in objective.required_dependencies]
        current=self.corpus.get(objective.record_id)
        sources=[(record.id,record.exact_text) for record in [*required,current]]
        window=self.context.build(sources,objective.task,reserve=generation_reserve)
        if window.axiom_digest != self.constitution.digest():
            raise ValueError("Governance changed during context assembly")
        return window

    def record_coverage(self, objective: Objective):
        self.ledger.append({"type":"SOURCE_COVERED","source_id":objective.record_id,
                            "objective_id":objective.objective_id})

    def evaluate(self, exam_id: str, *, passed: bool, evidence: dict,
                 independent: bool):
        if not independent or not evidence:
            raise ValueError("Examinations require independent evidence")
        if "questions" not in evidence or "correct" not in evidence:
            raise ValueError("Missing examination counts")
        if evidence["questions"]<=0 or not 0<=evidence["correct"]<=evidence["questions"]:
            raise ValueError("Invalid examination evidence")
        self.ledger.append({"type":"EXAM_RESULT","exam_id":exam_id,
                            "passed":passed,"evidence":evidence,"independent":True})
        self.state=State.READY if passed else State.REMEDIATING

    def stage_gate(self, requirement: StageRequirement):
        self.ledger.advance(requirement)

    def diagnose(self, failure: str, prerequisite_ids: tuple[str,...]) -> dict:
        if self.state != State.REMEDIATING:
            raise RuntimeError("No failed objective to remediate")
        # No training method or weight resets are invented here.
        response={"failure":failure,"prerequisites":list(prerequisite_ids),
                  "action":"RETEACH_APPROVED_PREREQUISITES"}
        self.ledger.append({"type":"REMEDIATION_REQUESTED",**response})
        return response


    def cloud_exposure_selection(self, records, completed_updates: int, seed: int):
        """Authorize one source lesson under deterministic, replayed, shuffled epochs.

        This is exposure, not a stage advancement. Only run_stage with its exam
        banks may advance formal curriculum stages.
        """
        import random
        if self.state != State.READY:
            raise RuntimeError("Controller must be ready to authorize exposure")
        if completed_updates < 0 or not records:
            raise ValueError("Invalid training cursor or missing approved sources")
        if len({r.id for r in records}) != len(records):
            raise ValueError("Duplicate source record identifiers")
        epoch, slot = divmod(completed_updates, len(records))
        # Shuffle each full epoch; always revisit previously seen records.
        key = (seed, epoch, len(records), records[0].id, records[-1].id)
        if getattr(self, "_cloud_epoch_key", None) != key:
            permitted = {r[0] for r in self.corpus.db.execute("SELECT id FROM documents")}
            for r in records:
                if r.document_id not in permitted:
                    raise ValueError("Controller refused unapproved source")
            indices = list(range(len(records)))
            random.Random(f"{seed}:{epoch}:{self.constitution.digest()}").shuffle(indices)
            self._cloud_epoch_indices = indices
            self._cloud_epoch_key = key
        selected = self._cloud_epoch_indices[slot]
        current = records[selected]
        # The prior paragraph is the prior paragraph in its original document,
        # not the preceding position in the randomized training schedule.
        prev = records[selected - 1] if selected else None
        if prev is not None and prev.document_id != current.document_id:
            prev = None
        return current, prev, epoch

    def protected_axiom_revision(self, *args, **kwargs):
        raise PermissionError("The educational controller has no authority to change axioms")

    def authorize_architecture_replacement(self, *args, **kwargs):
        raise PermissionError("Training cannot replace neural architecture")
