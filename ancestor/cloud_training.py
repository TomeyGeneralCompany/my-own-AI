"""Bounded, resumable cloud training using only approved owner-provided sources.

This is foundational paragraph-conditioned neural training, not intellectual
certification. Input axioms are immutable and loaded only from the approved
governance manifest. No pretrained language-model weights are used.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import torch
import random

from .checkpoints import save_checkpoint, restore_checkpoint
from .corpus import Corpus
from .governance import Constitution
from .model import AncestorModel, ModelConfig
from .tokenizer import ByteBPE
from .training import TrainConfig, Trainer, controller_completion_batch, masked_next_token_loss


def hash_data(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_identity(corpus: Corpus, heldout: bytes) -> str:
    documents = corpus.db.execute(
        "SELECT id,original_sha,exact_text_sha FROM documents ORDER BY id").fetchall()
    if not documents:
        raise ValueError("Private corpus is empty")
    payload = json.dumps(
        {"documents": documents, "heldout_sha256": hash_data(heldout)},
        separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hash_data(payload)


def source_records(corpus: Corpus):
    ids = corpus.db.execute(
        """SELECT id,document_id FROM records
           WHERE kind IN ('paragraph','pdf_page')
           ORDER BY document_id,paragraph_number,id""").fetchall()
    if not ids:
        raise ValueError("No approved learning records in private corpus")
    return [corpus.get(record_id) for record_id, _ in ids]


def safe_token_segments(tokenizer: ByteBPE, text: str, capacity: int) -> list[str]:
    if capacity < 1:
        raise ValueError("Mandatory axiom/context header leaves no target room")
    tokens = tokenizer.encode(text)
    if not tokens:
        raise ValueError("Empty source lesson")
    segments = []
    i = 0
    while i < len(tokens):
        end = min(len(tokens), i + capacity)
        while end > i:
            try:
                segment = tokenizer.decode(tokens[i:end])
                break
            except UnicodeDecodeError:
                end -= 1
        else:
            raise ValueError("Could not segment source at UTF-8 boundary")
        segments.append(segment)
        i = end
    if "".join(segments) != text:
        raise ValueError("Source segmentation altered manuscript text")
    return segments


def make_lesson(tokenizer: ByteBPE, constitution: Constitution,
                record, previous, capacity: int, cycle: int):
    # The owner's canonical principles govern all lessons. The previous
    # approved passage supplies local argument continuity, not a replacement
    # philosophical foundation.
    axioms = constitution.axiom_prefix()
    prompt = "Continue the author's own passage, retaining its terminology and reasoning."
    local_context = previous.exact_text if previous and previous.document_id == record.document_id else ""
    # Use a deterministic bounded suffix of the immediate prior paragraph.
    # If the axioms + task cannot fit by themselves, fail rather than drop one.
    base = len(tokenizer.encode_structured([
        ("<|AXIOM|>", axioms),
        ("<|CONTEXT|>", ""),
        ("<|DERIVE|>", prompt + "\n"),
    ]))
    available = capacity - base - 1
    if available < 16:
        raise ValueError("Mandatory canonical axioms and task exceed model context")
    # Let the local premise use up to one quarter of available tokens; the
    # actual paragraph target always receives the remainder.
    prior_ids = tokenizer.encode(local_context)
    keep = min(len(prior_ids), available // 4)
    while keep:
        try:
            local_context = tokenizer.decode(prior_ids[-keep:])
            break
        except UnicodeDecodeError:
            keep -= 1
    if not keep:
        local_context = ""
    limit = capacity - 1 - len(tokenizer.encode_structured([
        ("<|AXIOM|>", axioms),
        ("<|CONTEXT|>", local_context),
        ("<|DERIVE|>", prompt + "\n"),
    ]))
    segments = safe_token_segments(tokenizer, record.exact_text, limit)
    segment = segments[cycle % len(segments)]
    batch = controller_completion_batch(tokenizer, axioms, local_context,
                                        prompt, segment, capacity)
    return batch, len(segments), cycle % len(segments)


def heldout_validation(trainer: Trainer, tokenizer: ByteBPE,
                       constitution: Constitution, heldout: str):
    # This measures out-of-sample token prediction only; it is NOT an
    # examination of philosophical, theological, or mathematical competence.
    prefix = tokenizer.encode_structured([
        ("<|AXIOM|>", constitution.axiom_prefix()),
        ("<|CONTEXT|>", ""),
        ("<|DERIVE|>", "Continue a previously unseen English passage.\n"),
    ])
    available = trainer.config.block_size - len(prefix) - 1
    if available < 16:
        raise ValueError("Held-out evaluation does not fit")
    segment = safe_token_segments(tokenizer, heldout, available)[0]
    batch = controller_completion_batch(
        tokenizer, constitution.axiom_prefix(), "",
        "Continue a previously unseen English passage.", segment,
        trainer.config.block_size)
    return trainer.validate([batch])


def run(args):
    if args.updates < 1 or args.max_minutes < 1 or args.block not in (1024, 2048):
        raise ValueError("Invalid cloud training limits")
    corpus = Corpus(args.data)
    records = source_records(corpus)
    tokenizer = ByteBPE.load(args.tokenizer)
    constitution = Constitution.load(args.governance)
    heldout = Path(args.heldout).read_bytes()
    text = heldout.decode("utf-8")
    if not text.strip():
        raise ValueError("Missing independent held-out English text")
    if any(text == record.exact_text for record in records):
        raise ValueError("Held-out sample is identical to a training record")
    source_hash = source_identity(corpus, heldout)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = AncestorModel(ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=args.block,
        width=args.width, layers=args.layers, heads=args.heads,
        ffn_width=args.ffn))
    trainer = Trainer(model, TrainConfig(
        block_size=args.block, accumulation=1, microbatch=1))
    if args.resume:
        restored = restore_checkpoint(args.resume, trainer,
                                      tokenizer_hash=tokenizer.digest(),
                                      constitution=constitution,
                                      corpus_manifest_hash=source_hash)
        print("Resumed validated neural updates:", restored["completed_updates"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    trained = 0
    while trained < args.updates:
        # Leave a cushion for evaluation, checkpoint write, and private upload.
        if trained and (time.monotonic() - start) >= (args.max_minutes * 60):
            break
        current_idx = trainer.completed_updates % len(records)
        cycle = trainer.completed_updates // len(records)
        current = records[current_idx]
        prior = records[current_idx-1] if current_idx else None
        batch, count, index = make_lesson(
            tokenizer, constitution, current, prior, args.block, cycle)
        loss = trainer.step([batch])
        trained += 1
        if trained == 1 or trained % 10 == 0:
            print(json.dumps({
                "completed_updates": trainer.completed_updates,
                "source_id": current.id,
                "source_segment": index+1,
                "total_segments": count,
                "training_loss": round(loss, 6),
            }))
    if trained == 0:
        raise RuntimeError("No training update completed")
    validation = heldout_validation(trainer, tokenizer, constitution, text)
    checkpoint = save_checkpoint(
        output, trainer,
        tokenizer_hash=tokenizer.digest(), constitution=constitution,
        corpus_manifest_hash=source_hash, stage="FOUNDATIONAL_MANUSCRIPT_EXPOSURE",
        objective_id="CLOUD_PARAGRAPH_CONDITIONED_TRAINING")
    # The .pt and .json are uploaded to encrypted/restricted remote storage
    # by the hosting workflow. They are never GitHub Actions public artifacts.
    shutil.copy2(checkpoint, output/"latest.pt")
    shutil.copy2(checkpoint.with_suffix(".json"), output/"latest.json")
    summary = {
        "completed_updates": trainer.completed_updates,
        "updates_this_run": trained,
        "private_source_count": len(corpus.db.execute(
            "SELECT id FROM documents").fetchall()),
        "learning_units": len(records),
        "heldout_loss": validation,
        "governance_sha256": constitution.digest(),
        "tokenizer_sha256": tokenizer.digest(),
        "corpus_identity": source_hash,
        "training_status": "FOUNDATIONAL_EXPOSURE_NOT_CERTIFIED",
    }
    (output/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--governance", required=True)
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--max-minutes", type=int, default=180)
    parser.add_argument("--block", type=int, default=1024)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--ffn", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=1729)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
