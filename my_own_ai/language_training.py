"""Original-weight CPU language pretraining and instruction learning.

This path is separate from legacy byte checkpoints: checkpoint and tokenizer
identities are explicit and old controller/proposal interfaces remain unchanged.
"""
from dataclasses import asdict
from hashlib import sha256
import json
import math
from time import perf_counter, monotonic
from pathlib import Path
import torch
from torch.nn import functional as F
from .model import ModelConfig, LocalTransformer
from .tokenizer import ByteBPETokenizer, BOS, EOS, USER, ASSISTANT, PAD
from .train import _atomic_save
from .corpus import collect_documents


def load_dialogues(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected nonempty dialogue examples")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "question", "answer"}:
            raise ValueError("Invalid dialogue schema")
        if (not all(isinstance(row[key], str) and row[key].strip() and len(row[key]) <= 2000
                    for key in ("id", "question", "answer"))
                or row["id"] in seen):
            raise ValueError("Invalid or duplicate dialogue")
        seen.add(row["id"])
    return rows


def _hash(obj):
    return sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                            separators=(",", ":")).encode()).hexdigest()


def _example(row, tokenizer, context):
    question = [BOS, USER] + tokenizer.encode(row["question"]) + [EOS, ASSISTANT]
    answer = tokenizer.encode(row["answer"]) + [EOS]
    tokens = question + answer
    if len(tokens) > context:
        raise ValueError(f"Dialogue exceeds context: {row['id']}")
    # Input at t predicts token t+1, so answer targets begin at answer's first token.
    inputs = tokens[:-1]
    labels = [-100] * (len(question) - 1) + answer
    assert len(inputs) == len(labels)
    return inputs, labels


def _text_sequences(folder, tokenizer, context):
    docs = collect_documents(folder)
    sequences = []
    for doc in docs:
        tokens = [BOS] + tokenizer.encode(doc["text"]) + [EOS]
        if len(tokens) < 4:
            continue
        # Each document remains independent, and blocks are not interleaved.
        for start in range(0, len(tokens) - 1, context):
            sample = tokens[start:start + context + 1]
            if len(sample) >= 3:
                sequences.append((sample[:-1], sample[1:]))
    if not sequences:
        raise ValueError("Corpus is too short for language pretraining")
    return sequences


def _batch(samples, indices):
    selected = [samples[i] for i in indices]
    maximum = max(len(x) for x, _ in selected)
    x = torch.full((len(selected), maximum), PAD, dtype=torch.long)
    y = torch.full_like(x, -100)
    for idx, (tokens, labels) in enumerate(selected):
        x[idx, :len(tokens)] = torch.tensor(tokens)
        y[idx, :len(labels)] = torch.tensor(labels)
    return x, y


def _loss(model, batch):
    x, y = batch
    logits, _ = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                           ignore_index=-100)


@torch.no_grad()
def _validation(model, samples):
    """Compute loss weighted by actual supervised tokens, not row count."""
    was_training = model.training
    model.eval()
    total_loss = 0.
    total_tokens = 0
    for index in range(len(samples)):
        x, y = _batch(samples, [index])
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                               ignore_index=-100, reduction="sum")
        total_loss += float(loss)
        total_tokens += int((y != -100).sum())
    model.train(was_training)
    if total_tokens == 0:
        raise ValueError("No supervised validation targets")
    return total_loss / total_tokens


def train_language(*, tokenizer_path, training, validation, output,
                   phase, steps, init_checkpoint=None, resume=False,
                   embed=64, layers=2, heads=4, context=256,
                   batch_size=2, lr=0.001, seed=42, checkpoint_every=20, threads=2,
                   time_budget_seconds=None):
    if phase not in ("pretrain", "instruction"):
        raise ValueError("Invalid language training phase")
    if any(type(x) is not int or x < 1 for x in
           (steps, embed, layers, heads, context, batch_size, checkpoint_every, threads)):
        raise ValueError("Invalid training dimensions")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("Invalid learning rate")
    if time_budget_seconds is not None and (not isinstance(time_budget_seconds, (int, float))
            or not math.isfinite(time_budget_seconds) or time_budget_seconds <= 0):
        raise ValueError("Invalid instruction time budget")
    if resume and init_checkpoint:
        raise ValueError("Resume and new-generation initialization are separate operations")
    targets = [Path(p).resolve() for p in (tokenizer_path, training, validation, output)]
    if len(set(targets)) < 4:
        raise ValueError("Output must not overwrite training, validation, or tokenizer")
    if init_checkpoint is not None and Path(init_checkpoint).resolve() == Path(output).resolve():
        raise ValueError("New generation cannot overwrite its parent")
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    cfg = ModelConfig(vocab_size=tokenizer.vocab_size, context_size=context,
                      embed_dim=embed, heads=heads, layers=layers)
    cfg.validate()
    if phase == "instruction":
        train_rows, validation_rows = load_dialogues(training), load_dialogues(validation)
        questions_train = {row["question"].casefold().strip() for row in train_rows}
        if questions_train & {row["question"].casefold().strip() for row in validation_rows}:
            raise ValueError("Training and validation questions overlap")
        train_samples = [_example(row, tokenizer, context) for row in train_rows]
        val_samples = [_example(row, tokenizer, context) for row in validation_rows]
        source_hash = _hash(train_rows)
        validation_hash = _hash(validation_rows)
    else:
        train_docs, validation_docs = collect_documents(training), collect_documents(validation)
        if {doc["id"] for doc in train_docs} & {doc["id"] for doc in validation_docs}:
            raise ValueError("Pretraining corpora overlap")
        train_samples = _text_sequences(training, tokenizer, context)
        val_samples = _text_sequences(validation, tokenizer, context)
        source_hash = _hash([(d["id"], d["source"]) for d in train_docs])
        validation_hash = _hash([(d["id"], d["source"]) for d in validation_docs])
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    model = LocalTransformer(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    sampler = torch.Generator().manual_seed(seed)
    # Uniform-by-block oversamples short tails of long books. In new
    # English prose runs, sample in proportion to actual supervised
    # tokens so each source token has comparable exposure.
    training_token_count = sum(len(labels) for _, labels in train_samples)
    validation_token_count = sum(len(labels) for _, labels in val_samples)
    sampling_policy = ("pretrain_target_token_weighted" if phase == "pretrain"
                       else "instruction_example_uniform")
    start = 0
    original = None
    best_loss = float('inf')
    best_step = 0
    best_path = Path(output).with_name(Path(output).stem + '.best.pt')
    if resume:
        payload = torch.load(output, map_location="cpu", weights_only=True)
        if any((payload.get(k) != v) for k, v in {
            "language_format": 1, "tokenizer_sha256": tokenizer.fingerprint(),
            "model_config": asdict(cfg), "phase": phase,
            "training_sha256": source_hash, "validation_sha256": validation_hash,
            "batch_size": batch_size, "lr": lr, "seed": seed}.items()):
            raise ValueError("Resume corpus, tokenizer, settings or phase changed")
        model.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["optimizer"])
        sampler.set_state(payload["sampler_rng"])
        torch.set_rng_state(payload["torch_rng"])
        start = int(payload["completed_steps"])
        # Preserve old PR #14 uniform sampling when continuing a checkpoint
        # that predates this policy; do not silently change its RNG stream.
        sampling_policy = payload.get("sampling_policy", "legacy_uniform")
        original = payload.get("parent_sha256")
        best_loss = float(payload.get("best_validation_loss", payload["validation_loss"]))
        best_step = int(payload.get("best_step", start))
    elif init_checkpoint:
        parent = Path(init_checkpoint)
        payload = torch.load(parent, map_location="cpu", weights_only=True)
        if (payload.get("language_format") != 1 or
                payload.get("tokenizer_sha256") != tokenizer.fingerprint() or
                payload.get("model_config") != asdict(cfg)):
            raise ValueError("Parent model and tokenizer are incompatible")
        model.load_state_dict(payload["state_dict"])
        original = sha256(parent.read_bytes()).hexdigest()
    if steps < start:
        raise ValueError("Requested steps precede saved progress")
    def save(step):
        nonlocal best_loss, best_step
        measured = _validation(model, val_samples)
        if measured < best_loss:
            best_loss, best_step = measured, step
        if not math.isfinite(measured):
            raise ValueError("Non-finite validation loss")
        state = {
            "language_format": 1, "model_config": asdict(cfg),
            "tokenizer_state": tokenizer.state(), "tokenizer_sha256": tokenizer.fingerprint(),
            "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
            "sampler_rng": sampler.get_state(), "torch_rng": torch.get_rng_state(),
            "completed_steps": step, "phase": phase, "seed": seed,
            "lr": lr, "batch_size": batch_size, "training_sha256": source_hash,
            "validation_sha256": validation_hash, "parent_sha256": original,
            "sampling_policy": sampling_policy,
            "training_target_tokens": training_token_count,
            "validation_target_tokens": validation_token_count,
            "validation_loss": measured,
            "best_validation_loss": best_loss, "best_step": best_step,
        }
        _atomic_save(state, output)
        if step == best_step:
            _atomic_save(state, best_path)
        model.train()
        return measured
    model.train()
    measured = None
    started = perf_counter()
    clock_start = monotonic()
    supervised_tokens = 0
    completed = start
    recent_update = 0.
    stop_reason = "target_steps"
    weights = torch.tensor([len(labels) for _, labels in train_samples],
                           dtype=torch.float64)
    for step in range(start + 1, steps + 1):
        if time_budget_seconds is not None and completed > start and (
                monotonic() - clock_start + max(5., recent_update * 2) >= time_budget_seconds):
            stop_reason = "time_budget"
            break
        update_start = monotonic()
        if sampling_policy == "pretrain_target_token_weighted":
            chosen = torch.multinomial(weights, batch_size, replacement=True,
                                       generator=sampler).tolist()
        else:
            chosen = torch.randint(len(train_samples), (batch_size,),
                                   generator=sampler).tolist()
        current_batch = _batch(train_samples, chosen)
        supervised_tokens += int((current_batch[1] != -100).sum())
        loss = _loss(model, current_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        completed = step
        recent_update = monotonic() - update_start
        if step % checkpoint_every == 0 or step == steps:
            measured = save(step)
    if completed > start and (not output.exists() or
                              int(torch.load(output, map_location="cpu", weights_only=True)["completed_steps"]) < completed):
        measured = save(completed)
    if measured is None:
        measured = float(torch.load(output, map_location="cpu", weights_only=True)["validation_loss"])
    elapsed = perf_counter() - started
    return {"phase": phase, "steps": steps, "completed_steps": completed,
            "stop_reason": stop_reason,
            "completed_this_run": completed - start,
            "trained_target_tokens": supervised_tokens,
            "training_target_tokens": training_token_count,
            "validation_target_tokens": validation_token_count,
            "effective_training_passes": round(
                supervised_tokens / max(training_token_count, 1), 3),
            "sampling_policy": sampling_policy,
            "elapsed_seconds": round(elapsed, 3),
            "target_tokens_per_second": round(supervised_tokens / max(elapsed, 1e-6), 2),
            "validation_loss": measured, "validation_perplexity": math.exp(min(measured, 50)),
            "best_validation_loss": best_loss, "best_step": best_step,
            "best_checkpoint": str(best_path),
            "tokenizer_sha256": tokenizer.fingerprint(), "parent_sha256": original,
            "parameters": sum(p.numel() for p in model.parameters()), "checkpoint": str(output)}


def load_language(checkpoint, tokenizer_path):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("language_format") != 1:
        raise ValueError("Not a language-model checkpoint; legacy weights require their original runtime")
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    if (tokenizer.fingerprint() != payload.get("tokenizer_sha256") or
            tokenizer.state() != payload.get("tokenizer_state")):
        raise ValueError("Tokenizer mismatch; generation blocked")
    model = LocalTransformer(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, tokenizer
