"""Train a new neural network on local text with no hosted inference."""
from dataclasses import asdict
import os
from pathlib import Path
import tempfile
import torch
from .model import ModelConfig, LocalTransformer

def collect_corpus(folder):
    if not folder.is_dir():
        raise ValueError("Missing corpus directory")
    files = sorted(folder.rglob("*.txt"))
    if not files:
        raise ValueError("Corpus has no .txt files")
    return b"\n".join(f.read_bytes() for f in files)

def train(corpus_dir, output, cfg, steps=20, batch_size=4, learning_rate=0.001, seed=42):
    cfg.validate()
    if steps < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("Invalid training settings")
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    raw = collect_corpus(corpus_dir)
    if len(raw) <= cfg.context_size:
        raise ValueError("Corpus too short for context")
    data = torch.tensor(list(raw), dtype=torch.long)
    model = LocalTransformer(cfg)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses = []
    for _ in range(steps):
        offsets = torch.randint(0, len(data) - cfg.context_size, (batch_size,))
        x = torch.stack([data[i:i+cfg.context_size] for i in offsets])
        y = torch.stack([data[i+1:i+cfg.context_size+1] for i in offsets])
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.item()))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".tmp", delete=False) as handle:
        tmp = Path(handle.name)
    try:
        torch.save({"model_config": asdict(cfg), "state_dict": model.state_dict(),
                    "steps": steps, "seed": seed, "final_loss": losses[-1]}, tmp)
        os.replace(tmp, output)
    finally:
        tmp.unlink(missing_ok=True)
    return {"steps": steps, "initial_loss": losses[0], "final_loss": losses[-1],
            "bytes": len(raw), "parameters": sum(p.numel() for p in model.parameters())}

def load_model(checkpoint):
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = LocalTransformer(ModelConfig(**data["model_config"]))
    model.load_state_dict(data["state_dict"])
    model.eval()
    return model
