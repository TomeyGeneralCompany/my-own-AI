"""Resumable original-weight CPU training with held-out evaluation and checkpoints."""
from dataclasses import asdict
import math
import hashlib
import os
from pathlib import Path
import tempfile
import time
import torch
from .model import ModelConfig, LocalTransformer
from .corpus import collect_documents, corpus_fingerprint, split_documents

def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_corpus(folder):
    return b"\n".join(d["text"].encode("utf-8") for d in collect_documents(folder))

def _atomic_save(payload, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".tmp", delete=False) as f:
        path = Path(f.name)
    try:
        torch.save(payload, path)
        os.replace(path, output)
    finally:
        path.unlink(missing_ok=True)

def _choose_batch(documents, context_size, batch_size, generator):
    eligible = [torch.tensor(list(raw), dtype=torch.long) for raw in documents if len(raw) > context_size + 1]
    if not eligible:
        raise ValueError("Training documents shorter than context window")
    xs, ys = [], []
    for _ in range(batch_size):
        d = eligible[int(torch.randint(len(eligible), (1,), generator=generator))]
        offset = int(torch.randint(len(d) - context_size, (1,), generator=generator))
        xs.append(d[offset:offset+context_size])
        ys.append(d[offset+1:offset+context_size+1])
    return torch.stack(xs), torch.stack(ys)

@torch.no_grad()
def evaluate(model, documents, context_size, batch_size=4, batches=3, seed=8128):
    if not documents:
        return None
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    loss_total = 0.
    for _ in range(batches):
        x, y = _choose_batch(documents, context_size, batch_size, generator)
        _, loss = model(x, y)
        loss_total += float(loss)
    model.train()
    return loss_total / batches

def train(corpus_dir, output, cfg, steps=20, batch_size=4, learning_rate=0.001,
          seed=42, resume=False, checkpoint_every=10, validation_batches=2,
          schedule_steps=1000, warmup_steps=10, init_checkpoint=None):
    cfg.validate()
    if init_checkpoint is not None:
        if resume:
            raise ValueError("Cannot resume the same run while initializing a new corpus generation")
        if Path(init_checkpoint).resolve() == Path(output).resolve():
            raise ValueError("New generation must preserve its original source checkpoint")
    if steps < 1 or batch_size < 1 or learning_rate <= 0 or checkpoint_every < 1 or validation_batches < 1 or schedule_steps < 1 or warmup_steps < 0 or warmup_steps >= schedule_steps:
        raise ValueError("Invalid training settings")
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    docs = collect_documents(corpus_dir)
    fingerprint = corpus_fingerprint(docs)
    training_docs, validation_docs, split_strategy = split_documents(docs, cfg.context_size, seed)
    generator = torch.Generator().manual_seed(seed)
    if resume:
        if not Path(output).exists():
            raise ValueError("Resume checkpoint not found")
        state = torch.load(output, map_location="cpu", weights_only=True)
        if state.get("corpus_fingerprint") != fingerprint or state.get("split_seed") != seed:
            raise ValueError("Corpus or split seed changed since checkpoint")
        if state.get("model_config") != asdict(cfg):
            raise ValueError("Model configuration does not match checkpoint")
        if state.get("batch_size") != batch_size or state.get("learning_rate") != learning_rate or state.get("schedule_steps") != schedule_steps or state.get("warmup_steps") != warmup_steps:
            raise ValueError("Training settings do not match checkpoint")
        model = LocalTransformer(cfg)
        model.load_state_dict(state["state_dict"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        optimizer.load_state_dict(state["optimizer"])
        generator.set_state(state["sampler_rng"])
        start_step = int(state["steps"])
        first_loss = float(state["initial_loss"])
        previous_loss = float(state["final_loss"])
        if steps < start_step:
            raise ValueError("Target steps is below checkpoint progress")
    else:
        torch.manual_seed(seed)
        model = LocalTransformer(cfg)
        if init_checkpoint is not None:
            previous = torch.load(init_checkpoint, map_location="cpu", weights_only=True)
            if previous.get("model_config") != asdict(cfg):
                raise ValueError("Previous generation model configuration differs")
            model.load_state_dict(previous["state_dict"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        start_step = 0
        first_loss = None
        previous_loss = None
    source_generation = (state.get("initialized_from") if resume
                         else str(Path(init_checkpoint)) if init_checkpoint else None)
    source_digest = (state.get("initialized_from_sha256") if resume
                     else _sha256_file(init_checkpoint) if init_checkpoint else None)
    inherited_format = (state.get("proposal_format") if resume
                        else previous.get("proposal_format") if init_checkpoint else None)
    model.train()
    begin = time.perf_counter()
    # This horizon is immutable across resumed runs.
    def scheduled_rate(step):
        if warmup_steps and step <= warmup_steps:
            return learning_rate * step / warmup_steps
        progress = min(1.0, (step - warmup_steps) / (schedule_steps - warmup_steps))
        return learning_rate * (0.1 + 0.45 * (1.0 + math.cos(math.pi * progress)))
    def save(step, loss):
        validation = evaluate(model, validation_docs, cfg.context_size,
                              min(batch_size, 4), validation_batches)
        _atomic_save({"format_version": 2, "model_config": asdict(cfg),
                      "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "sampler_rng": generator.get_state(), "steps": step, "split_seed": seed,
                      "batch_size": batch_size, "learning_rate": learning_rate,
                      "schedule_steps": schedule_steps, "warmup_steps": warmup_steps,
                      "corpus_fingerprint": fingerprint, "split_strategy": split_strategy,
                      "initialized_from": source_generation,
                      "initialized_from_sha256": source_digest,
                      "proposal_format": inherited_format,
                      "initial_loss": first_loss, "final_loss": loss,
                      "validation_loss": validation}, output)
        return validation
    validation = None
    for step in range(start_step + 1, steps + 1):
        x, y = _choose_batch(training_docs, cfg.context_size, batch_size, generator)
        for group in optimizer.param_groups:
            group["lr"] = scheduled_rate(step)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        previous_loss = float(loss.detach())
        if first_loss is None:
            first_loss = previous_loss
        if step % checkpoint_every == 0 or step == steps:
            validation = save(step, previous_loss)
    if steps == start_step:
        validation = state.get("validation_loss")
    elapsed = time.perf_counter() - begin
    return {"steps": steps, "steps_completed_this_run": steps - start_step,
            "initialized_from": source_generation,
            "initialized_from_sha256": source_digest,
            "learning_rate": scheduled_rate(steps), "schedule_steps": schedule_steps,
            "initial_loss": first_loss, "final_loss": previous_loss,
            "validation_loss": validation,
            "validation_perplexity": math.exp(min(validation, 80)) if validation is not None else None,
            "split_strategy": split_strategy, "documents": len(docs),
            "parameters": sum(p.numel() for p in model.parameters()),
            "seconds": round(elapsed, 3), "training_tokens_per_second":
            round(((steps - start_step)*batch_size*cfg.context_size) / max(elapsed, .000001), 1)}

def load_model(checkpoint):
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = LocalTransformer(ModelConfig(**data["model_config"]))
    model.load_state_dict(data["state_dict"])
    model.proposal_format = data.get("proposal_format")
    model.eval()
    return model

def benchmark(checkpoint, iterations=10, prompt=b"Hello", generation_tokens=16):
    if iterations < 1 or generation_tokens < 1:
        raise ValueError("Positive benchmark inputs required")
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    model = load_model(checkpoint)
    start = time.perf_counter()
    for _ in range(iterations):
        model.generate(prompt, max_new_tokens=generation_tokens)
    seconds = time.perf_counter() - start
    return {"parameters": sum(p.numel() for p in model.parameters()),
            "generation_tokens_per_second": round(iterations*generation_tokens / max(seconds, .000001), 2),
            "seconds": round(seconds, 3), "iterations": iterations}
