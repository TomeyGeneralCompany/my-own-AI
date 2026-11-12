"""Progressive, exhaustive language reading with overlapping token windows.

A pass reads every unique next-token target from every eligible training
document exactly once. Overlap tokens supply context but are masked from
loss to avoid repeatedly optimizing the same targets. Document boundaries
are never crossed. Progress and optimizer are saved atomically.
"""
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from .corpus import collect_documents
from .model import LocalTransformer, ModelConfig
from .tokenizer import BOS, EOS, ByteBPETokenizer
from .train import _atomic_save


def _reading_documents(folder):
    """Restore sequential sections to one continuous stream per source book.

    A PR #15 prepared corpus contains separate numbered section files plus
    a manifest. Rejoin only sections in the *same partition*, in original
    sequence. No training window may cross a validation/test boundary.
    Plain text corpora without a manifest keep per-file isolation.
    """
    folder = Path(folder)
    rows = collect_documents(folder)
    manifest_path = folder.parent / "manifest.json"
    if not manifest_path.is_file():
        return rows
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "approved_books_v1":
        return rows
    import re
    known = {re.sub(r"[^A-Za-z0-9_-]", "_", row["id"]): row["id"]
             for row in manifest.get("sources", [])}
    groups = {}
    for row in rows:
        stem = Path(row["source"]).stem
        prefix, sep, index = stem.rpartition("_")
        if not sep or not index.isdecimal() or prefix not in known:
            raise ValueError("Prepared book section name does not match approved manifest")
        groups.setdefault(prefix, []).append((int(index), row["text"]))
    combined = []
    for name, sections in sorted(groups.items()):
        ordered = sorted(sections)
        indices = [i for i, _ in ordered]
        if len(indices) != len(set(indices)):
            raise ValueError("Duplicate book section index")
        content = "\n\n".join(text for _, text in ordered)
        combined.append({"id": sha256(content.encode()).hexdigest(),
                         "source": known[name], "text": content})
    if not combined:
        raise ValueError("Prepared corpus has no readable book sections")
    return combined


def _identity(docs):
    rows = [(row["source"], row["id"]) for row in docs]
    return sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def overlapping_windows(documents, tokenizer, context, overlap):
    """Yield (source ID, input IDs, masked targets, covered target positions).

    The union of covered positions for a document equals [1..last token].
    Every target has exactly one supervised appearance in each reading pass.
    """
    if type(context) is not int or context < 16:
        raise ValueError("Reading context must be at least 16")
    if type(overlap) is not int or not 0 <= overlap < context:
        raise ValueError("Overlap must be smaller than model context")
    result = []
    for document in documents:
        stream = [BOS] + tokenizer.encode(document["text"]) + [EOS]
        if len(stream) < 3:
            continue
        start, first_unseen = 0, 1
        while first_unseen < len(stream):
            end = min(len(stream), start + context + 1)
            inputs = stream[start:end-1]
            positions = list(range(start + 1, end))
            labels = [token if position >= first_unseen else -100
                      for token, position in zip(stream[start+1:end], positions)]
            newly_read = [position for position in positions if position >= first_unseen]
            if not newly_read:
                raise RuntimeError("Progressive reading stalled without new tokens")
            result.append((document["id"], inputs, labels, newly_read))
            first_unseen = end
            if end == len(stream):
                break
            # Start sufficiently far back to provide overlap, but guarantee
            # the next block predicts at least one previously unseen token.
            start = end - 1 - overlap
    if not result:
        raise ValueError("No readable token windows")
    return result


def _batch(windows):
    from .language_training import _batch as padded_batch
    return padded_batch([(window[1], window[2]) for window in windows],
                        list(range(len(windows))))


def _validation_windows(model, windows):
    """Token-weighted validation, no validation target used in optimization."""
    was_training = model.training
    model.eval()
    total, tokens = 0., 0
    with torch.inference_mode():
        for window in windows:
            x, y = _batch([window])
            logits, _ = model(x)
            total += float(F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                ignore_index=-100, reduction="sum"))
            tokens += int((y != -100).sum())
    model.train(was_training)
    if not tokens:
        raise ValueError("No supervised validation tokens")
    return total / tokens


def _pass_order(windows, seed, pass_number):
    """Shuffle section order across passes while preserving within-section reading."""
    groups = {}
    for idx, window in enumerate(windows):
        groups.setdefault(window[0], []).append(idx)
    keys = sorted(groups)
    generator = torch.Generator().manual_seed(seed + pass_number)
    shuffled = torch.randperm(len(keys), generator=generator).tolist()
    return [index for position in shuffled for index in groups[keys[position]]]


def train_progressive(*, tokenizer_path, training, validation, output,
                      passes=5, overlap=64, embed=128, heads=4,
                      layers=4, context=512, lr=.0005, seed=42, threads=4,
                      checkpoint_every=25, resume=False, init_checkpoint=None,
                      patience=0, batch_size=1):
    """Exhaustively read complete book chunks, with resumable ordered cursor.

    Pass count is an upper bound, not a fabricated English proficiency gate.
    If patience > 0, validation stagnation stops further complete passes.
    """
    if any(type(value) is not int or value < 1 for value in (
        passes, embed, heads, layers, context, threads, checkpoint_every, batch_size)):
        raise ValueError("Invalid progressive reading dimensions")
    if (type(seed) is not int or type(patience) is not int or patience < 0
            or not math.isfinite(lr) or lr <= 0):
        raise ValueError("Invalid reading settings")
    if resume and init_checkpoint:
        raise ValueError("Cannot both resume and initialize a new generation")
    output = Path(output)
    if output.exists() and not resume:
        raise ValueError("Existing reading checkpoint requires --resume or a new output path")
    source_paths = [Path(p).resolve() for p in (tokenizer_path, training, validation)]
    if output.resolve() in source_paths:
        raise ValueError("Checkpoint must not replace a source")
    if init_checkpoint and Path(init_checkpoint).resolve() == output.resolve():
        raise ValueError("Parent checkpoint must not be overwritten")
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    cfg = ModelConfig(vocab_size=tokenizer.vocab_size, context_size=context,
                      embed_dim=embed, heads=heads, layers=layers)
    cfg.validate()
    train_docs, valid_docs = _reading_documents(training), _reading_documents(validation)
    train_ids = {row["id"] for row in train_docs}
    if train_ids & {row["id"] for row in valid_docs}:
        raise ValueError("Training and validation documents overlap")
    training_hash, validation_hash = _identity(train_docs), _identity(valid_docs)
    windows = overlapping_windows(train_docs, tokenizer, context, overlap)
    valid_windows = overlapping_windows(valid_docs, tokenizer, context, overlap)
    total_targets = sum(len(w[3]) for w in windows)
    valid_targets = sum(len(w[3]) for w in valid_windows)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    model = LocalTransformer(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    current_pass, next_index, update_count = 0, 0, 0
    best_loss, best_pass, no_improvement = math.inf, -1, 0
    parent_digest = None
    best_path = output.with_name(output.stem + ".best.pt")
    identity = {
        "reading_format": 1, "language_format": 1,
        "model_config": asdict(cfg), "tokenizer_sha256": tokenizer.fingerprint(),
        "training_sha256": training_hash, "validation_sha256": validation_hash,
        "overlap": overlap, "lr": lr, "seed": seed, "batch_size": batch_size,
    }
    if resume:
        payload = torch.load(output, map_location="cpu", weights_only=True)
        if any(payload.get(key) != value for key, value in identity.items()):
            raise ValueError("Reading resume requires matching corpus, tokenizer and configuration")
        model.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["torch_rng"])
        current_pass = int(payload["reading_pass"])
        next_index = int(payload["next_window"])
        update_count = int(payload["completed_steps"])
        best_loss = float(payload["best_validation_loss"])
        best_pass = int(payload["best_pass"])
        no_improvement = int(payload["passes_without_improvement"])
        parent_digest = payload.get("parent_sha256")
        if not 0 <= next_index < len(windows) and next_index != 0:
            raise ValueError("Invalid saved reading cursor")
        if passes < current_pass:
            raise ValueError("Target passes precede saved reading progress")
    elif init_checkpoint is not None:
        parent = Path(init_checkpoint)
        payload = torch.load(parent, map_location="cpu", weights_only=True)
        if (payload.get("language_format") != 1 or
                payload.get("model_config") != asdict(cfg) or
                payload.get("tokenizer_sha256") != tokenizer.fingerprint()):
            raise ValueError("Incompatible parent model/tokenizer for progressive reading")
        model.load_state_dict(payload["state_dict"])
        parent_digest = sha256(parent.read_bytes()).hexdigest()

    log = []
    started = perf_counter()
    best_marker = False
    last_validation = None

    def persist(*, measured=None, mark_best=False):
        state = {**identity, "tokenizer_state": tokenizer.state(),
                 "phase": "pretrain", "state_dict": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                 "reading_pass": current_pass, "next_window": next_index,
                 "completed_steps": update_count, "best_validation_loss": best_loss,
                 "best_pass": best_pass, "passes_without_improvement": no_improvement,
                 "validation_loss": measured,
                 "parent_sha256": parent_digest, "training_target_tokens": total_targets,
                 "validation_target_tokens": valid_targets}
        _atomic_save(state, output)
        if mark_best:
            _atomic_save(state, best_path)

    if current_pass == 0 and next_index == 0 and not resume:
        last_validation = _validation_windows(model, valid_windows)
        best_loss, best_pass = last_validation, 0
        persist(measured=last_validation, mark_best=True)
        log.append({"pass": 0, "validation_loss": last_validation,
                    "completed_windows": 0, "read_target_tokens": 0})
    elif resume:
        last_validation = payload["validation_loss"]

    stopped_reason = "target_passes"
    for pass_idx in range(current_pass, passes):
        order = _pass_order(windows, seed, pass_idx)
        start_at = next_index if pass_idx == current_pass else 0
        model.train()
        for cursor in range(start_at, len(order), batch_size):
            selected = [windows[k] for k in order[cursor:cursor + batch_size]]
            x, y = _batch(selected)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   y.reshape(-1), ignore_index=-100)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite reading loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            update_count += 1
            next_index = min(cursor + batch_size, len(order))
            current_pass = pass_idx
            if next_index < len(order) and update_count % checkpoint_every == 0:
                persist(measured=last_validation)
        current_pass = pass_idx + 1
        next_index = 0
        measured = _validation_windows(model, valid_windows)
        improved = math.isfinite(measured) and measured < best_loss
        if not math.isfinite(measured):
            raise ValueError("Non-finite validation loss")
        if improved:
            best_loss, best_pass, no_improvement = measured, current_pass, 0
        else:
            no_improvement += 1
        last_validation = measured
        persist(measured=measured, mark_best=improved)
        log.append({"pass": current_pass, "validation_loss": measured,
                    "best_validation_loss": best_loss,
                    "completed_windows": len(windows),
                    "read_target_tokens": total_targets,
                    "coverage": 1.0})
        if patience and no_improvement >= patience:
            stopped_reason = "validation_plateau"
            break
    return {
        "reading_format": 1, "completed_passes": current_pass,
        "completed_steps": update_count, "read_windows_per_pass": len(windows),
        "unique_supervised_tokens_per_pass": total_targets,
        "heldout_validation_tokens": valid_targets,
        "overlap": overlap, "context": context,
        "best_validation_loss": best_loss, "best_pass": best_pass,
        "last_validation_loss": last_validation,
        "elapsed_seconds": round(perf_counter() - started, 3),
        "stop_reason": stopped_reason,
        "checkpoint": str(output), "best_checkpoint": str(best_path),
        "tokenizer_sha256": tokenizer.fingerprint(),
        "passes": log,
        "warning": "Complete reading coverage is not evidence of English fluency.",
    }
