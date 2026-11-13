"""Exact, sequential, resume-safe neural reading of *every* approved book unit.

No held-out source units are skipped: all text is read. Comprehension questions,
rather than source text, are held out. Source order is never shuffled.
"""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import torch
from torch.nn import functional as F

from .model import LocalTransformer, ModelConfig
from .tokenizer import ByteBPETokenizer
from .progressive_reading import overlapping_windows
from .train import _atomic_save


def source_units(folder):
    """Return every text unit, including repeats, in numbered/source order."""
    paths = sorted(Path(folder).glob("*.txt"))
    if not paths:
        raise ValueError("Missing complete approved work")
    result = []
    for position, path in enumerate(paths):
        text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
        if not text.strip():
            raise ValueError("Empty approved source unit: " + path.name)
        result.append({"id": f"unit-{position:08d}-{sha256(text.encode()).hexdigest()}",
                       "source": path.name, "text": text})
    return result


def read_all_units(*, tokenizer_path, source, output, parent,
                   max_updates=300, context=512, overlap=64, embed=128,
                   layers=4, heads=4, batch_size=2, threads=2, seed=42,
                   learning_rate=.0005, checkpoint_every=25):
    """One *complete* pass, with durable window cursor and exact supervised coverage."""
    if any(not isinstance(n, int) or n <= 0 for n in
           (max_updates, context, embed, layers, heads, batch_size, threads, checkpoint_every)):
        raise ValueError("Invalid neural reading settings")
    units = source_units(source)
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    windows = overlapping_windows(units, tokenizer, context, overlap)
    unit_manifest = [{"name": u["source"],
                      "sha256": sha256(u["text"].encode()).hexdigest()}
                     for u in units]
    ordered_digest = sha256(json.dumps(unit_manifest,sort_keys=True).encode()).hexdigest()
    cfg = ModelConfig(vocab_size=tokenizer.vocab_size, context_size=context,
                      embed_dim=embed, heads=heads, layers=layers)
    cfg.validate()
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    model = LocalTransformer(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    output = Path(output)
    identity = {"language_format": 1, "ordered_unit_sha256": ordered_digest,
                "tokenizer_sha256": tokenizer.fingerprint(),
                "model_config": asdict(cfg), "reading_format": 2}
    if output.exists():
        record = torch.load(output, map_location="cpu", weights_only=True)
        if any(record.get(key) != value for key, value in identity.items()):
            raise ValueError("Reading resume changed text, order, tokenizer, or model")
        model.load_state_dict(record["state_dict"])
        optimizer.load_state_dict(record["optimizer"])
        torch.set_rng_state(record["torch_rng"])
        cursor = int(record["next_window"])
        supervised = int(record["supervised_tokens"])
        updates = int(record["completed_steps"])
    else:
        inherited = torch.load(parent, map_location="cpu", weights_only=True)
        if inherited.get("language_format") != 1 or inherited.get("model_config") != asdict(cfg) or inherited.get("tokenizer_sha256") != tokenizer.fingerprint():
            raise ValueError("Incompatible learned ancestral weights")
        model.load_state_dict(inherited["state_dict"])
        cursor = 0
        supervised = 0
        updates = 0
    if not 0 <= cursor <= len(windows):
        raise ValueError("Corrupt ordered reading cursor")
    expected = sum(len(w[3]) for w in windows)
    if supervised > expected:
        raise ValueError("Supervised coverage exceeds approved text")
    def persist():
        _atomic_save({**identity, "tokenizer_state": tokenizer.state(),
                      "state_dict": model.state_dict(), "optimizer":optimizer.state_dict(),
                      "torch_rng":torch.get_rng_state(), "next_window":cursor,
                      "completed_steps":updates, "supervised_tokens":supervised,
                      "total_target_tokens":expected, "source_units":len(units),
                      "last_read_unit":windows[cursor-1][0] if cursor else None},
                     output)
    if not output.exists():
        persist()
    model.train()
    from .language_training import _batch
    used = 0
    while cursor < len(windows) and used < max_updates:
        selected = windows[cursor: min(cursor+batch_size, len(windows))]
        x, y = _batch([(w[1],w[2]) for w in selected],
                      list(range(len(selected))))
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1,logits.shape[-1]), y.reshape(-1),
                               ignore_index=-100)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite source-reading loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        supervised += sum(len(w[3]) for w in selected)
        cursor += len(selected)
        updates += 1
        used += 1
        if used % checkpoint_every == 0:
            persist()
    persist()
    completed = cursor == len(windows) and supervised == expected
    return {"read_complete": completed, "next_window": 0 if completed else cursor,
            "covered_targets": supervised, "expected_targets":expected,
            "source_units":len(units), "ordered_unit_sha256":ordered_digest,
            "last_read_unit":windows[cursor-1][0] if cursor else None,
            "updates_this_run":used, "completed_updates":updates}
