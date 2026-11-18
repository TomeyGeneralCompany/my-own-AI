"""Exclusive active operating mode. No activity mode edits canonical governance."""
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class ModeManager:
    available: set[str] = field(default_factory=lambda: {
        "foundation_training","examination","conversation","coding","writing","trading"})
    active: str | None = None
    def activate(self,name: str):
        if name not in self.available:
            raise ValueError("Unapproved mode")
        if self.active is not None and self.active != name:
            raise RuntimeError("Suspend active mode before switching")
        self.active=name
    def suspend(self):
        prior=self.active
        self.active=None
        return prior
