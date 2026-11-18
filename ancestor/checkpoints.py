"""Atomic complete-state recovery checkpoints and verified promotion."""
from __future__ import annotations
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import tempfile
import torch
from .model import AncestorModel
from .training import Trainer
from .governance import Constitution

def _hash(path: Path) -> str:
    h=sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def save_checkpoint(directory: str | Path, trainer: Trainer, *, tokenizer_hash: str,
                    constitution: Constitution, corpus_manifest_hash: str,
                    stage: str, objective_id: str, ledger_offset: int = 0) -> Path:
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    payload={
        "model_state":trainer.model.state_dict(),
        "optimizer_state":trainer.optimizer.state_dict(),
        "model_config":trainer.model.manifest(),
        "train_config":asdict(trainer.config),
        "completed_updates":trainer.completed_updates,
        "tokens_seen":trainer.tokens_seen,
        "torch_rng":torch.get_rng_state(),
        "python_rng":random.getstate(),
        "tokenizer_hash":tokenizer_hash,
        "constitution_hash":constitution.digest(),
        "corpus_manifest_hash":corpus_manifest_hash,
        "stage":stage,"objective_id":objective_id,"ledger_offset":ledger_offset,
    }
    fd,name=tempfile.mkstemp(prefix=".ancestor-",suffix=".pt",dir=directory)
    os.close(fd)
    tmp=Path(name)
    try:
        torch.save(payload,tmp)
        # Read-back validation: never announce a half-written artifact.
        inspected=torch.load(tmp,map_location="cpu",weights_only=True)
        if inspected["completed_updates"] != trainer.completed_updates:
            raise ValueError("Checkpoint readback mismatch")
        digest=_hash(tmp)
        dest=directory / f"recovery-{trainer.completed_updates:012d}-{digest[:12]}.pt"
        os.replace(tmp,dest)
        manifest=dest.with_suffix(".json")
        fd,mname=tempfile.mkstemp(prefix=".manifest-",dir=directory)
        with os.fdopen(fd,"w") as f:
            json.dump({"file":dest.name,"sha256":digest,"completed_updates":trainer.completed_updates,
                       "stage":stage,"objective_id":objective_id,
                       "constitution_hash":constitution.digest(),"tokenizer_hash":tokenizer_hash},f,indent=2)
            f.flush();os.fsync(f.fileno())
        os.replace(mname,manifest)
        return dest
    finally:
        if tmp.exists():
            tmp.unlink()

def restore_checkpoint(path: str | Path, trainer: Trainer, *,
                       tokenizer_hash: str, constitution: Constitution,
                       corpus_manifest_hash: str) -> dict:
    path=Path(path)
    metadata=json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata["sha256"] != _hash(path):
        raise ValueError("Checkpoint file hash mismatch")
    state=torch.load(path,map_location="cpu",weights_only=True)
    expected={
        "tokenizer_hash":tokenizer_hash,
        "constitution_hash":constitution.digest(),
        "corpus_manifest_hash":corpus_manifest_hash,
    }
    for key,value in expected.items():
        if state[key] != value:
            raise ValueError(f"Checkpoint {key} mismatch; refusing reinitialization")
    if state["model_config"] != trainer.model.manifest():
        raise ValueError("Architecture mismatch")
    if state["train_config"] != asdict(trainer.config):
        raise ValueError("Training configuration mismatch")
    # State is validated prior to mutating the live model and optimizer.
    if set(state["model_state"]) != set(trainer.model.state_dict()):
        raise ValueError("Model tensor keys mismatch")
    for key,value in trainer.model.state_dict().items():
        if value.shape != state["model_state"][key].shape:
            raise ValueError(f"Tensor mismatch: {key}")
    trainer.model.load_state_dict(state["model_state"],strict=True)
    trainer.optimizer.load_state_dict(state["optimizer_state"])
    trainer.completed_updates=int(state["completed_updates"])
    trainer.tokens_seen=int(state["tokens_seen"])
    torch.set_rng_state(state["torch_rng"])
    random.setstate(state["python_rng"])
    return {"stage":state["stage"],"objective_id":state["objective_id"],
            "ledger_offset":state["ledger_offset"],"completed_updates":trainer.completed_updates}
