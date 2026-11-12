"""CPU supervised training on worked, source-grounded reasoning proposals."""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re
from time import perf_counter
import unicodedata

import torch
from torch.nn import functional as F

from .corpus_eval import load_cases
from .functional_modules import Research, grounded_evidence
from .grounding import SPAN_FORMAT, span_prompt
from .module_memory import IDENTIFIER, encode_json
from .modules import hardware_metrics
from .reasoning import END_PROPOSAL, SUPERVISED_FORMAT, ProposalRequest, Stage, supervised_prefix
from .retrieval import PASSAGE_LIMIT
from .train import _atomic_save, load_model

FIELDS = {"id", "group", "stage", "question", "passages", "plan", "candidate", "feedback", "answer"}


def load_examples(path):
    with Path(path).open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty JSON list of examples")
    ids = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != FIELDS:
            raise ValueError("Worked example fields do not match the schema")
        for field in ("id", "group"):
            if not isinstance(row[field], str) or not IDENTIFIER.fullmatch(row[field]):
                raise ValueError(f"Invalid {field} in worked example")
        if row["id"] in ids:
            raise ValueError(f"Duplicate example ID: {row['id']}")
        ids.add(row["id"])
        for field in ("question", "plan", "candidate", "feedback", "answer"):
            value = row[field]
            if not isinstance(value, str) or "\0" in value or len(value) > 4000:
                raise ValueError(f"Invalid {field} in {row['id']}")
        if not row["question"].strip() or not row["answer"].strip():
            raise ValueError(f"Missing question or target in {row['id']}")
        passages = row["passages"]
        if (not isinstance(passages, list) or not 1 <= len(passages) <= 12
                or any(not isinstance(text, str) or not text.strip() or "\0" in text or len(text) > PASSAGE_LIMIT
                       for text in passages)):
            raise ValueError(f"Invalid passages in {row['id']}")
        stage = row["stage"]
        if stage not in (Stage.FORMULATE, Stage.CONSTRUCT, Stage.CORRECT):
            raise ValueError("Evaluate is controller-owned and cannot be a training target")
        if stage != Stage.FORMULATE:
            if not row["plan"].strip():
                raise ValueError(f"Missing plan in {row['id']}")
            if grounded_evidence(row["answer"], [{"text": text} for text in passages])[2]:
                raise ValueError(f"Unverified training target in {row['id']}")
        if stage == Stage.CORRECT and (not row["candidate"].strip() or not row["feedback"].strip()):
            raise ValueError(f"Correction requires a prior candidate and feedback in {row['id']}")
        if stage != Stage.CORRECT and (row["candidate"] or row["feedback"]):
            raise ValueError(f"Prior candidate and feedback belong to Correct in {row['id']}")
        if stage == Stage.FORMULATE and row["plan"]:
            raise ValueError(f"Formulate already has a plan in {row['id']}")
    return rows


def _normalized(question):
    return " ".join(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", question).casefold()))


def verify_splits(train, validation, held_out):
    train_groups = {row["group"] for row in train}
    validation_groups = {row["group"] for row in validation}
    train_questions = {_normalized(row["question"]) for row in train}
    validation_questions = {_normalized(row["question"]) for row in validation}
    held_questions = {_normalized(row["question"]) for row in held_out}
    if (train_groups & validation_groups or train_questions & validation_questions
            or (train_questions | validation_questions) & held_questions):
        raise ValueError("Training, validation and held-out questions must be disjoint")


def encode_example(row, capacity, format_name=SUPERVISED_FORMAT):
    stage = Stage(row["stage"])
    request = ProposalRequest("research", "grounded_analysis", stage, Research.grounded_procedures[stage],
                              row["question"], {"evidence": [{"text": text} for text in row["passages"]]},
                              row["plan"], row["candidate"], row["feedback"])
    if format_name == SPAN_FORMAT:
        return _encode_span(row, request, capacity)
    prefix = supervised_prefix(request, capacity)
    answer = list(row["answer"].encode("utf-8"))
    if len(prefix) + len(answer) > capacity:
        raise ValueError(f"Example {row['id']} exceeds checkpoint context; shorten its target")
    if stage != Stage.FORMULATE:
        displayed = {}
        for line in bytes(prefix).splitlines():
            if match := re.fullmatch(rb"\[([1-9][0-9]?)\] (.*)", line):
                displayed[int(match[1])] = match[2]
        premises = row["answer"].split("\nInference:", 1)[0].splitlines()[1:]
        for line in premises:
            if not line.strip():
                continue
            match = re.fullmatch(r"- \[([1-9][0-9]?)\] (.+)", line)
            if match is None or match[2].strip().encode() not in displayed.get(int(match[1]), b""):
                raise ValueError(f"Example {row['id']} quotes evidence outside the model prompt")
    return torch.tensor(prefix + answer), torch.tensor([-100] * (len(prefix) - 1) + answer + [END_PROPOSAL])


def _encode_span(row, request, capacity):
    prefix, displayed = span_prompt(request, capacity)
    answer = row["answer"]
    if request.stage != Stage.FORMULATE:
        numbers, inference, error = grounded_evidence(answer, request.context["evidence"])
        if error or len(numbers) != 1 or not inference.endswith((".", "?", "!")) or "\n" in inference:
            raise ValueError(f"Example {row['id']} requires one verified premise and a complete inference")
        number, = numbers
        if number not in displayed:
            raise ValueError(f"Example {row['id']} selects a passage outside the model prompt")
        answer = f"{number} {inference}"
    tokens = list(answer.encode("utf-8"))
    if len(prefix) + len(tokens) > capacity:
        raise ValueError(f"Example {row['id']} exceeds checkpoint context; shorten its target")
    return torch.tensor(prefix + tokens), torch.tensor([-100] * (len(prefix) - 1) + tokens + [END_PROPOSAL])


def _batch(samples):
    pad = torch.nn.utils.rnn.pad_sequence
    return pad([pair[0] for pair in samples], batch_first=True), pad(
        [pair[1] for pair in samples], batch_first=True, padding_value=-100)


def _loss(model, batch):
    inputs, targets = batch
    logits, _ = model(inputs)
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100)


@torch.no_grad()
def _measure(model, rows):
    model.eval()
    losses = [float(_loss(model, _batch([row]))) for row in rows]
    return sum(losses) / len(losses)


def train_proposals(checkpoint, examples, validation, output, *, held_out=None, steps=200,
                    batch_size=4, learning_rate=.001, seed=42, threads=4, format_name=SUPERVISED_FORMAT):
    sources = [checkpoint, examples, validation] + ([held_out] if held_out else [])
    if Path(output).resolve() in {Path(source).resolve() for source in sources}:
        raise ValueError("Proposal training requires a separate output checkpoint")
    if format_name not in (SUPERVISED_FORMAT, SPAN_FORMAT):
        raise ValueError("Unsupported proposal format")
    if min(steps, batch_size, threads) < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Invalid proposal training settings")
    train_rows, validation_rows = load_examples(examples), load_examples(validation)
    verify_splits(train_rows, validation_rows, load_cases(held_out) if held_out else [])
    torch.set_num_threads(min(threads, hardware_metrics()["available_cpus"]))
    torch.manual_seed(seed)
    model = load_model(checkpoint)
    minimum_context = 512 if format_name == SPAN_FORMAT else 256
    if model.config.context_size < minimum_context:
        raise ValueError(f"Proposal training requires a model context of at least {minimum_context} bytes")
    train = [encode_example(row, model.config.context_size, format_name) for row in train_rows]
    valid = [encode_example(row, model.config.context_size, format_name) for row in validation_rows]
    sampler = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    started = perf_counter()
    initial = _measure(model, valid)
    model.train()
    for _ in range(steps):
        indices = torch.randint(len(train), (batch_size,), generator=sampler)
        loss = _loss(model, _batch([train[i] for i in indices]))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    final = _measure(model, valid)
    if not math.isfinite(final):
        raise ValueError("Proposal training produced non-finite validation loss")
    report = {"train_examples": len(train), "validation_examples": len(valid), "steps": steps,
              "initial_validation_loss": initial, "validation_loss": final,
              "seconds": perf_counter() - started, "hardware": hardware_metrics()}
    _atomic_save({"format_version": 1, "model_config": asdict(model.config), "state_dict": model.state_dict(),
                  "proposal_format": format_name,
                  "examples_sha256": hashlib.sha256(encode_json(train_rows)).hexdigest(),
                  "training_report": report}, output)
    return report
