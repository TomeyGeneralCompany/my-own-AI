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
from .copy_curriculum import synthetic_copy_rows, VERSION as COPY_VERSION
from .comprehension_curriculum import (VERSION as COMPREHENSION_VERSION, cycle_size,
                                       lesson_groups, validation_lessons)
from .intellectual_memory import (new_memory, validate_memory, consolidate,
    replay_group, source_examples, contextual_question)
from .comprehension_curriculum import GROUPS
from .comprehension_transfer import exercise, grounded_row, author_examples, probe_rows


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
                   time_budget_seconds=None, english_drill_fraction=0.0,
                   curriculum_cycles=False, allow_curriculum_migration=False,
                   dynamic_copy=False, comprehension_cycles=False, tiered_memory=False,
                   transfer_learning=False, author_language_folder=None):
    if transfer_learning and not tiered_memory:
        raise ValueError("Transfer learning requires tiered comprehension memory")
    if tiered_memory and not comprehension_cycles:
        raise ValueError("Tiered memory requires comprehension cycles")
    if not 0 <= english_drill_fraction <= 1 or not math.isfinite(english_drill_fraction):
        raise ValueError("Invalid English drill sampling fraction")
    if comprehension_cycles and (not curriculum_cycles or dynamic_copy or phase != "instruction"):
        raise ValueError("Comprehension cycles require independent instruction mode")
    if dynamic_copy and not curriculum_cycles:
        raise ValueError("Dynamic copy requires complete instructional cycles")
    if curriculum_cycles and (phase != "instruction" or english_drill_fraction):
        raise ValueError("Ordered curriculum requires independent instruction examples")
    if allow_curriculum_migration and not curriculum_cycles:
        raise ValueError("Curriculum migration requires ordered instruction")
    if phase != "instruction" and english_drill_fraction:
        raise ValueError("English drill sampling requires instruction phase")
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
        if comprehension_cycles:
            validation_rows = validation_rows + validation_lessons(seed)
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
                       else "english_grounded_balanced" if english_drill_fraction
                       else "instruction_example_uniform")
    drill_indices = [i for i, row in enumerate(train_rows) if row["id"].startswith("english-")] if phase == "instruction" else []
    authored_indices = [i for i, row in enumerate(train_rows) if not row["id"].startswith("english-")] if phase == "instruction" else []
    if english_drill_fraction and (not drill_indices or not authored_indices):
        raise ValueError("Balanced curriculum needs English and original-author lessons")
    start = 0
    original = None
    curriculum_order, curriculum_cursor, curriculum_completed = [], 0, 0
    curriculum_lineage = []
    curriculum_length = (cycle_size(train_rows) if comprehension_cycles else
                         len(train_samples) * (2 if dynamic_copy else 1))
    comprehension_schedule = {"best": None, "bad_cycles": 0, "last_cycle": 0,
                              "lr": min(lr, 0.0005)}
    memory = None
    active_copy_cycle, active_copy_samples = None, []
    best_loss = float('inf')
    best_step = 0
    best_path = Path(output).with_name(Path(output).stem + '.best.pt')
    last_training_loss = None
    if resume:
        payload = torch.load(output, map_location="cpu", weights_only=True)
        stable = {"language_format": 1, "tokenizer_sha256": tokenizer.fingerprint(),
                  "model_config": asdict(cfg), "phase": phase,
                  "batch_size": batch_size, "lr": lr, "seed": seed}
        if any(payload.get(k) != v for k, v in stable.items()):
            raise ValueError("Resume architecture, tokenizer or optimizer settings changed")
        changed_corpus = (payload.get("training_sha256") != source_hash or
                          payload.get("validation_sha256") != validation_hash)
        old_policy = payload.get("sampling_policy", "legacy_uniform")
        new_policy = ("instruction_comprehension_cycles_v1" if comprehension_cycles else
                      "instruction_coverage_cycles" if curriculum_cycles else
                      "english_grounded_balanced" if english_drill_fraction else
                      "pretrain_target_token_weighted" if phase == "pretrain" else
                      "instruction_example_uniform")
        changing_policy = old_policy != new_policy
        if not changing_policy and payload.get("comprehension_generator_version") != (
                COMPREHENSION_VERSION if comprehension_cycles else None):
            raise ValueError("Comprehension generator changed during resume")
        if not changing_policy and payload.get("copy_generator_version") != (
                COPY_VERSION if dynamic_copy else None):
            raise ValueError("Copy curriculum generator changed during resume")
        if changed_corpus or changing_policy:
            if not (allow_curriculum_migration and curriculum_cycles and
                    old_policy in ("instruction_example_uniform", "legacy_uniform") and
                    not payload.get("curriculum_lineage")):
                raise ValueError("Curriculum changed without explicit permitted migration")
            curriculum_lineage = [{"prior_training_sha256": payload.get("training_sha256"),
                                   "prior_validation_sha256":payload.get("validation_sha256"),
                                   "prior_policy":old_policy,
                                   "migrated_at_update":int(payload["completed_steps"])}]
            best_loss, best_step = float("inf"), 0
        else:
            curriculum_lineage = payload.get("curriculum_lineage", [])
        if curriculum_cycles and not changing_policy:
            curriculum_order = list(payload.get("curriculum_order", []))
            curriculum_cursor = int(payload.get("curriculum_cursor", 0))
            curriculum_completed = int(payload.get("curriculum_completed", 0))
            if (len(curriculum_order) != curriculum_length or
                    sorted(curriculum_order) != list(range(curriculum_length)) or
                    not 0 <= curriculum_cursor < len(curriculum_order)):
                raise ValueError("Invalid saved curriculum coverage ledger")
        model.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["optimizer"])
        if comprehension_cycles and not changing_policy:
            comprehension_schedule = dict(payload["comprehension_schedule"])
            if not (0 < comprehension_schedule["lr"] <= lr and
                    comprehension_schedule["last_cycle"] <= curriculum_completed):
                raise ValueError("Invalid persisted comprehension schedule")
        if tiered_memory:
            memory = payload.get("intellectual_memory")
            if memory is None and curriculum_cursor:
                raise ValueError("Tiered memory migration requires a complete cycle boundary")
        sampler.set_state(payload["sampler_rng"])
        torch.set_rng_state(payload["torch_rng"])
        start = int(payload["completed_steps"])
        last_training_loss = payload.get("last_training_loss")
        # Preserve old PR #14 uniform sampling when continuing a checkpoint
        # that predates this policy; do not silently change its RNG stream.
        sampling_policy = new_policy
        if not curriculum_cycles and changing_policy and not (
                not english_drill_fraction and old_policy == "legacy_uniform"):
            raise ValueError("Instruction sampling policy changed during resume")
        original = payload.get("parent_sha256")
        if not (changed_corpus or changing_policy):
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
    def next_order(cycle):
        if comprehension_cycles:
            return lesson_groups(train_rows, cycle, seed)[1]
        return torch.randperm(curriculum_length, generator=sampler).tolist()
    if curriculum_cycles:
        sampling_policy = ("instruction_comprehension_cycles_v1" if comprehension_cycles
                           else "instruction_coverage_cycles")
        if not curriculum_order:
            curriculum_order = next_order(curriculum_completed)
        elif comprehension_cycles and curriculum_order != next_order(curriculum_completed):
            raise ValueError("Persisted comprehension lesson order mismatch")
    if comprehension_cycles:
        for group in optimizer.param_groups:
            group["lr"] = comprehension_schedule["lr"]
    transfer_state = None
    if transfer_learning:
        excerpts = author_examples(author_language_folder)
        author_samples = [_example(row, tokenizer, context) for row in excerpts
                          if len(tokenizer.encode(row['question'])) + len(tokenizer.encode(row['answer'])) + 5 <= context]
        prior_transfer = payload.get('transfer_state') if resume else None
        if prior_transfer is None:
            if curriculum_cursor:
                raise ValueError('Transfer migration requires a complete cycle boundary')
            transfer_state = {'version': 2, 'introduced_at_cycle': curriculum_completed,
                              'author_hash': _hash(excerpts), 'probe_history': {}, 'last_probe_cycle': curriculum_completed-1, 'transfer_batches': 0}
            comprehension_schedule.update(best=None, bad_cycles=0, lr=min(lr, 0.00025))
            for group in optimizer.param_groups:
                group['lr'] = comprehension_schedule['lr']
        else:
            transfer_state = prior_transfer
            if transfer_state['version'] != 2 or transfer_state['author_hash'] != _hash(excerpts):
                raise ValueError('Transfer source or version changed')
        train_samples = [_example(grounded_row(row), tokenizer, context) for row in train_rows]
    if tiered_memory:
        if memory is None:
            memory = new_memory(train_rows, seed, curriculum_completed)
        validate_memory(memory, train_rows, seed, curriculum_completed)
        # Stable training-only anchors measure retention without using held-out exams.
        anchors = {g: [_example(row, tokenizer, context) for row in source_examples(g, seed)]
                   for g in GROUPS}
    if steps < start:
        raise ValueError("Requested steps precede saved progress")
    def save(step):
        nonlocal best_loss, best_step
        measured = _validation(model, val_samples)
        if measured < best_loss:
            best_loss, best_step = measured, step
        if not math.isfinite(measured):
            raise ValueError("Non-finite validation loss")
        # Adjust only at completed cycles, using independent development loss.
        # Never use examination answers or scores to choose neural updates.
        if (comprehension_cycles and curriculum_cursor == 0 and
                curriculum_completed > comprehension_schedule["last_cycle"]):
            prior = comprehension_schedule["best"]
            if prior is None or measured < prior - 0.001:
                comprehension_schedule["best"] = measured
                comprehension_schedule["bad_cycles"] = 0
            else:
                comprehension_schedule["bad_cycles"] += 1
                if comprehension_schedule["bad_cycles"] >= 3:
                    comprehension_schedule["lr"] = max(0.000025, comprehension_schedule["lr"] * 0.5)
                    comprehension_schedule["bad_cycles"] = 0
            comprehension_schedule["last_cycle"] = curriculum_completed
            for group in optimizer.param_groups:
                group["lr"] = comprehension_schedule["lr"]
        if tiered_memory and curriculum_cursor == 0 and memory["covered_through"] < curriculum_completed - 1:
            if transfer_learning:
                probes = {g: [_example(row, tokenizer, context) for row in probe_rows(g, seed, curriculum_completed-1)] for g in GROUPS}
                losses = {g: _validation(model, probes[g]) for g in GROUPS}
                if not transfer_state['probe_history']:
                    for record in memory['records'].values():
                        record.update(tier='acquisition', streak=0, loss=None, best=None)
                transfer_state['probe_history'] = losses
                transfer_state['last_probe_cycle'] = curriculum_completed-1
                consolidate(memory, losses, curriculum_completed-1)
            else:
                consolidate(memory, {g: _validation(model, anchors[g]) for g in GROUPS}, curriculum_completed - 1)
        state = {
            **({"intellectual_memory": memory} if tiered_memory else {}),
            **({"transfer_state": transfer_state} if transfer_learning else {}),
            "language_format": 1, "model_config": asdict(cfg),
            "tokenizer_state": tokenizer.state(), "tokenizer_sha256": tokenizer.fingerprint(),
            "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
            "sampler_rng": sampler.get_state(), "torch_rng": torch.get_rng_state(),
            "completed_steps": step, "phase": phase, "seed": seed,
            "lr": lr, "batch_size": batch_size, "training_sha256": source_hash,
            "validation_sha256": validation_hash, "parent_sha256": original,
            "sampling_policy": sampling_policy,
            "curriculum_order": curriculum_order, "curriculum_cursor": curriculum_cursor,
            "curriculum_completed": curriculum_completed,
            "curriculum_lineage": curriculum_lineage,
            "copy_generator_version": COPY_VERSION if dynamic_copy else None,
            **({"comprehension_generator_version": COMPREHENSION_VERSION,
                "comprehension_schedule": comprehension_schedule} if comprehension_cycles else {}),
            "english_drill_fraction": english_drill_fraction,
            "training_target_tokens": training_token_count,
            "validation_target_tokens": validation_token_count,
            "validation_loss": measured,
            "last_training_loss": last_training_loss,
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
    step = start
    while step < steps or (curriculum_cycles and curriculum_cursor):
        step += 1
        if time_budget_seconds is not None and completed > start and (
                monotonic() - clock_start + max(5., recent_update * 2) >= time_budget_seconds):
            stop_reason = "time_budget"
            break
        update_start = monotonic()
        if sampling_policy in ("instruction_coverage_cycles", "instruction_comprehension_cycles_v1"):
            if comprehension_cycles and active_copy_cycle != curriculum_completed:
                rows, expected_order = lesson_groups(train_rows, curriculum_completed, seed)
                if expected_order != curriculum_order:
                    raise ValueError("Comprehension curriculum order mismatch")
                active_copy_samples = [_example(row, tokenizer, context) for row in rows]
                active_copy_cycle = curriculum_completed
            if dynamic_copy and active_copy_cycle != curriculum_completed:
                words = synthetic_copy_rows(curriculum_completed, len(train_samples), seed)
                active_copy_samples = [_example(row, tokenizer, context) for row in words]
                active_copy_cycle = curriculum_completed
            chosen = curriculum_order[curriculum_cursor:curriculum_cursor + batch_size]
            curriculum_cursor += len(chosen)
            if curriculum_cursor == len(curriculum_order):
                curriculum_completed += 1
                curriculum_cursor = 0
                curriculum_order = next_order(curriculum_completed)
        elif sampling_policy == "english_grounded_balanced":
            # Select independently from English mechanics and authored reasoning,
            # making neither curriculum silently drown out the other.
            chosen = []
            for _ in range(batch_size):
                is_drill = bool(torch.rand((), generator=sampler).item() < english_drill_fraction)
                pool = drill_indices if is_drill else authored_indices
                chosen.append(pool[int(torch.randint(len(pool), (1,), generator=sampler).item())])
        elif sampling_policy == "pretrain_target_token_weighted":
            chosen = torch.multinomial(weights, batch_size, replacement=True,
                                       generator=sampler).tolist()
        else:
            chosen = torch.randint(len(train_samples), (batch_size,),
                                   generator=sampler).tolist()
        current_batch = _batch(train_samples + active_copy_samples if (dynamic_copy or comprehension_cycles) else train_samples, chosen)
        supervised_tokens += int((current_batch[1] != -100).sum())
        loss = _loss(model, current_batch)
        if transfer_learning and step % 4 == 0:
            # Full ordinary coverage remains intact; transfer examples are supplementary.
            group = replay_group(memory, step)
            idx = step // 4
            if group == 'conditional':
                idx *= 2
            rows = [exercise(group, idx % 96, seed, curriculum_completed, variant=v) for v in (0, 1)]
            if group == 'conditional':
                rows[1] = exercise(group, (idx % 96)+1, seed, curriculum_completed, variant=0)
            samples = [_example(row, tokenizer, context) for row in rows]
            if idx % 4 == 0 and author_samples:
                samples.append(author_samples[(idx // 4) % len(author_samples)])
            extra = _batch(samples, list(range(len(samples))))
            supervised_tokens += int((extra[1] != -100).sum())
            loss = loss + 0.75 * _loss(model, extra)
            transfer_state['transfer_batches'] = transfer_state.get('transfer_batches', 0) + 1
        if tiered_memory and step % 16 == 0:
            # Preserve the full original curriculum, adding one bounded replay batch.
            group = replay_group(memory, step)
            row = source_examples(group, seed, (step // 16) % 97)[(step // 256) % 4]
            # Reserve the complete target and structural delimiters before adding context.
            limit = context - len(tokenizer.encode(row["answer"])) - 5
            enriched = dict(row, question=contextual_question(memory, row["question"], tokenizer, limit, group))
            replay = _batch([_example(row, tokenizer, context), _example(enriched, tokenizer, context)], [0, 1])
            supervised_tokens += int((replay[1] != -100).sum())
            loss = loss + 0.5 * _loss(model, replay)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        last_training_loss = float(loss.detach())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        completed = step
        recent_update = monotonic() - update_start
        if (step % checkpoint_every == 0 or step == steps or
                (comprehension_cycles and curriculum_cursor == 0)):
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
            "curriculum_cycles_completed": curriculum_completed,
            "curriculum_cursor": curriculum_cursor,
            "curriculum_examples_total": curriculum_length,
            "copy_generator_version": COPY_VERSION if dynamic_copy else None,
            "curriculum_migrated": bool(curriculum_lineage),
            **({"memory_tiers": {g: memory["records"][g]["tier"] for g in GROUPS}} if tiered_memory else {}),
            **({"transfer_probe_loss": transfer_state["probe_history"], "author_excerpt_count": len(author_samples), "transfer_batches": transfer_state["transfer_batches"]} if transfer_learning else {}),
            **({"comprehension_schedule": comprehension_schedule} if comprehension_cycles else {}),
            "effective_training_passes": round(
                supervised_tokens / max(training_token_count, 1), 3),
            "sampling_policy": sampling_policy,
            "elapsed_seconds": round(elapsed, 3),
            "target_tokens_per_second": round(supervised_tokens / max(elapsed, 1e-6), 2),
            "last_training_loss": last_training_loss,
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
