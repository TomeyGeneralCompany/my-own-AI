"""Deterministic exclusive module socket with durable checkpoints."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile

@dataclass
class BasicModule:
    name: str
    state: dict | None = None
    loaded: bool = False

    def activate(self, previous_state):
        self.state = dict(previous_state)
        self.loaded = True

    def snapshot(self):
        if not self.loaded:
            raise RuntimeError("Inactive module")
        return dict(self.state or {})

    def deactivate(self):
        self.state = None
        self.loaded = False

class CheckpointStore:
    def __init__(self, directory):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def path(self, name):
        if not name.isidentifier():
            raise ValueError("Invalid module identifier")
        return self.directory / (name + ".json")

    def read(self, name):
        path = self.path(name)
        if not path.exists():
            return {}
        state = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            raise ValueError("Invalid checkpoint")
        return state

    def write(self, name, state):
        path = self.path(name)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.directory, delete=False) as handle:
            json.dump(state, handle, sort_keys=True, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
            tmp = Path(handle.name)
        try:
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

class ModuleController:
    def __init__(self, store, modules):
        self.store = store
        self.modules = {module.name: module for module in modules}
        if len(self.modules) != len(modules):
            raise ValueError("Duplicate modules")
        self.active = None

    def switch(self, target):
        if target not in self.modules:
            raise KeyError(target)
        if self.active is not None and self.active.name == target:
            return
        old = self.active
        new = self.modules[target]
        saved = self.store.read(target)
        if old is not None:
            self.store.write(old.name, old.snapshot())
            old.deactivate()
            self.active = None
        try:
            new.activate(saved)
        except Exception:
            if old is not None:
                old.activate(self.store.read(old.name))
                self.active = old
            raise
        self.active = new

    def checkpoint(self):
        if self.active is not None:
            self.store.write(self.active.name, self.active.snapshot())

    def close(self):
        if self.active is not None:
            self.checkpoint()
            self.active.deactivate()
            self.active = None

def default_modules():
    return [BasicModule(name) for name in ("conversation", "writing", "coding", "research")]
