"""Progressive reading invariants and cumulative-training regression."""
import json
from pathlib import Path

import pytest
import torch

from my_own_ai.corpus import collect_documents
from my_own_ai.progressive_reading import overlapping_windows, train_progressive, _reading_documents
from my_own_ai.tokenizer import ByteBPETokenizer
from my_own_ai.language_training import load_language, train_language


def test_overlap_keeps_context_but_supervises_each_target_once():
    tok = ByteBPETokenizer()
    text = ("A sentence about hierarchy and communication. " * 5).strip()
    docs = [{"id": "source-a", "text": text}]
    windows = overlapping_windows(docs, tok, context=32, overlap=8)
    expected = [257] + tok.encode(text) + [258]
    positions = [index for row in windows for index in row[3]]
    assert positions == list(range(1, len(expected)))
    assert len(positions) == len(set(positions))
    assert any(-100 in row[2] for row in windows[1:])
    assert all(len(row[1]) <= 32 and len(row[1]) == len(row[2]) for row in windows)
    # A loss mask must never add duplicate targets from overlap tokens.
    assert sum(sum(x != -100 for x in row[2]) for row in windows) == len(expected) - 1


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError, match="Overlap"):
        overlapping_windows([{"id": "x", "text": "Enough text to tokenize."}],
                            ByteBPETokenizer(), 32, 32)


def _setup(tmp_path):
    train = tmp_path / "train"
    validation = tmp_path / "validation"
    train.mkdir()
    validation.mkdir()
    for index in range(2):
        (train / f"book-{index}.txt").write_text(
            f"Merchant {index} reports changes. The clerk transmits messages.\n\n"
            f"The supervisor examines reports and revises the schedule {index}.",
            encoding="utf-8",
        )
    (validation / "holdout.txt").write_text(
        "A separate communication channel carries facts. The manager responds to news.",
        encoding="utf-8",
    )
    tok = ByteBPETokenizer.train(
        [d["text"] for d in collect_documents(train)], vocab_size=330)
    tok_path = tmp_path / "tok.json"
    tok.save(tok_path)
    opts = dict(tokenizer_path=tok_path, training=train, validation=validation,
                context=32, overlap=8, embed=16, heads=4, layers=1,
                batch_size=2, threads=1, lr=.0005, seed=13, checkpoint_every=2)
    return opts, tok_path


def test_full_passes_resume_exactly_and_feed_language_instruction(tmp_path):
    opts, tokenizer = _setup(tmp_path)
    short = tmp_path / "short.pt"
    full = tmp_path / "full.pt"
    first = train_progressive(**opts, output=short, passes=1)
    assert first["completed_passes"] == 1
    assert first["passes"][-1]["coverage"] == 1.0
    assert first["unique_supervised_tokens_per_pass"] > 0
    resumed = train_progressive(**opts, output=short, passes=2, resume=True)
    uninterrupted = train_progressive(**opts, output=full, passes=2)
    assert resumed["completed_passes"] == uninterrupted["completed_passes"] == 2
    assert resumed["last_validation_loss"] == uninterrupted["last_validation_loss"]
    s = torch.load(short, map_location="cpu", weights_only=True)
    f = torch.load(full, map_location="cpu", weights_only=True)
    assert s["reading_pass"] == 2 and s["next_window"] == 0
    assert [row["pass"] for row in s["reading_history"]] == [0, 1, 2]
    assert [row["validation_loss"] for row in s["reading_history"]] == [
        row["validation_loss"] for row in f["reading_history"]]
    assert s["completed_steps"] == f["completed_steps"]
    assert all(torch.equal(s["state_dict"][key], f["state_dict"][key])
               for key in s["state_dict"])
    model, tok = load_language(resumed["best_checkpoint"], tokenizer)
    assert model.config.vocab_size == tok.vocab_size

    # The same trained weights, same tokenizer, same architecture are accepted
    # by the existing PR #14/15 instruction-learning path.
    instruction_train = tmp_path / "instruction_train.json"
    instruction_valid = tmp_path / "instruction_valid.json"
    instruction_train.write_text(json.dumps([
        {"id": "a", "question": "What does the clerk carry?", "answer": "Messages."}
    ]))
    instruction_valid.write_text(json.dumps([
        {"id": "b", "question": "Who receives the report?", "answer": "The supervisor."}
    ]))
    tuned = train_language(
        tokenizer_path=tokenizer, training=instruction_train,
        validation=instruction_valid, output=tmp_path / "instruct.pt",
        phase="instruction", steps=1,
        init_checkpoint=resumed["best_checkpoint"],
        embed=16, heads=4, layers=1, context=32,
        batch_size=1, threads=1, checkpoint_every=1
    )
    assert tuned["completed_this_run"] == 1


def test_resume_rejects_changed_reading_settings_or_training_corpus(tmp_path):
    opts, _ = _setup(tmp_path)
    output = tmp_path / "model.pt"
    train_progressive(**opts, output=output, passes=1)
    with pytest.raises(ValueError, match="matching corpus"):
        train_progressive(**{**opts, "overlap": 4},
                          output=output, passes=2, resume=True)
    (opts["training"] / "book-0.txt").write_text(
        "The author altered the training source after checkpoint.", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="matching corpus"):
        train_progressive(**opts, output=output, passes=2, resume=True)



def test_approved_book_sections_are_joined_only_within_each_split(tmp_path):
    root = tmp_path / "prepared"
    for split in ("train", "validation", "test"):
        (root / split).mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({
        "format": "approved_books_v1",
        "sources": [{"id": "commerce_book"}],
    }))
    (root / "train" / "commerce_book_00000.txt").write_text(
        "Section zero ends with the word communication.", encoding="utf-8")
    (root / "train" / "commerce_book_00001.txt").write_text(
        "Section one resumes the argument about authority.", encoding="utf-8")
    (root / "validation" / "commerce_book_00002.txt").write_text(
        "This paragraph is exclusively validation material.", encoding="utf-8")
    (root / "test" / "commerce_book_00003.txt").write_text(
        "The held-out section must never reach training.", encoding="utf-8")
    train = _reading_documents(root / "train")
    val = _reading_documents(root / "validation")
    assert len(train) == len(val) == 1
    assert "Section zero" in train[0]["text"]
    assert train[0]["text"].index("Section zero") < train[0]["text"].index("Section one")
    assert "exclusively validation" not in train[0]["text"]
    assert "held-out section" not in train[0]["text"]
    assert "exclusively validation" in val[0]["text"]
    tokenizer = ByteBPETokenizer()
    windows = overlapping_windows(train, tokenizer, context=32, overlap=8)
    rebuilt = [257] + tokenizer.encode(train[0]["text"]) + [258]
    assert [p for w in windows for p in w[3]] == list(range(1, len(rebuilt)))
