"""Durable append-only event ledger and explicit stage gates."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import os

@dataclass(frozen=True)
class StageRequirement:
    stage_id: str
    required_source_ids: tuple[str, ...]
    required_exam_ids: tuple[str, ...]
    required_retention_ids: tuple[str, ...] = ()

class LearningLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)

    def append(self, event: dict) -> None:
        if "type" not in event:
            raise ValueError("Every event must have a type")
        with self.path.open("a",encoding="utf-8") as fh:
            fh.write(json.dumps(event,sort_keys=True,ensure_ascii=False)+"\n")
            fh.flush()
            os.fsync(fh.fileno())

    def events(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line]

    def covered(self) -> set[str]:
        return {x["source_id"] for x in self.events() if x["type"]=="SOURCE_COVERED"}

    def exam_results(self) -> dict[str,bool]:
        result = {}
        for event in self.events():
            if event["type"]=="EXAM_RESULT":
                result[event["exam_id"]]=bool(event["passed"])
        return result

    def stage_ready(self, requirement: StageRequirement) -> bool:
        covered = self.covered()
        exams = self.exam_results()
        return (set(requirement.required_source_ids) <= covered
                and all(exams.get(x,False) for x in requirement.required_exam_ids)
                and all(exams.get(x,False) for x in requirement.required_retention_ids))

    def advance(self, requirement: StageRequirement):
        if not self.stage_ready(requirement):
            raise PermissionError("Stage gate failed: coverage, competence or retention missing")
        self.append({"type":"STAGE_COMPLETED","stage_id":requirement.stage_id})
