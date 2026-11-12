"""Treatise reader and autonomous controller: equation fidelity and continuity."""
import json
from hashlib import sha256
from pathlib import Path

import pytest
import torch

from my_own_ai.numbered_paragraphs import numbered_paragraphs
from my_own_ai.published_corpus import prepare_published_corpus
from my_own_ai.progressive_reading import _reading_documents, overlapping_windows, train_progressive
from my_own_ai.tokenizer import ByteBPETokenizer


ROOT = Path(__file__).resolve().parents[1]


def test_actual_published_manuscript_has_537_lossless_numbered_units():
    source = (ROOT / "published_books/communal-nature-of-commerce-revised-second-edition.txt").read_text(encoding="utf-8")
    units = numbered_paragraphs(source)
    numbers = [u["number"] for u in units if u["number"] is not None]
    assert numbers == list(range(537))
    assert "".join(u["text"] for u in units) == source
    fifth = next(u["text"] for u in units if u["number"] == 5)
    assert "First Axiom" in fifth
    assert "(1)" in fifth  # standalone equation number stays within [5]
    assert "Xi,t" in fifth
    assert "[6] Terms" not in fifth
    sixth = next(u["text"] for u in units if u["number"] == 6)
    assert "Terms." in sixth


def test_numbering_gaps_duplicates_and_math_preservation():
    text = "PREFACE\n\n[0] Axiom.\n\n x = α + β\n\n(1)\n \t continuation.\n\n[1] Terms.\n"
    chunks = numbered_paragraphs(text)
    assert len(chunks) == 3
    assert "x = α + β" in chunks[1]["text"]
    assert "(1)" in chunks[1]["text"]
    assert "continuation" in chunks[1]["text"]
    assert "[1]" not in chunks[1]["text"]
    assert "".join(row["text"] for row in chunks) == text
    with pytest.raises(ValueError, match="Nonconsecutive"):
        numbered_paragraphs("[0] A\n[2] C")
    with pytest.raises(ValueError, match="Repeated"):
        numbered_paragraphs("[0] A\n[0] C")


def test_published_book_splits_by_whole_numbered_paragraph(tmp_path):
    data = prepare_published_corpus(ROOT, ROOT / "published_books/manifest.json",
                                    tmp_path / "corpus")
    assert data["chunk_unit"] == "numbered_paragraph"
    assert data["reading_ledgers"]["commerce"]["numbered_paragraphs"] == 537
    count = sum(data["sections"].values())
    assert count == 538  # 537 numbered units and frontmatter
    ids = []
    for split in ("train", "validation", "test"):
        docs = _reading_documents(tmp_path / "corpus" / split)
        assert docs
        assert all("[536]" not in row["text"] for row in docs if split == "train")
        ids.extend(row["id"] for row in docs)
    assert len(set(ids)) == len(ids)
    fifth = (tmp_path / "corpus/train/commerce_p000005.txt").read_text()
    assert "(1)" in fifth and "[6]" not in fifth


def test_complete_single_paragraph_is_windowed_without_dropping_math():
    text = "[0] " + "one two three four " * 100 + "Equation: A = B + C (1)\n"
    tok = ByteBPETokenizer.train([text], vocab_size=300)
    docs = [{"id":sha256(text.encode()).hexdigest(),"text":text}]
    windows = overlapping_windows(docs, tok, context=64, overlap=16)
    covered = [i for row in windows for i in row[3]]
    assert covered == list(range(1, 2 + len(tok.encode(text))))


def test_old_reading_checkpoint_keeps_weights_optimizer_when_resegmented(tmp_path):
    train_old, val_old, train_new, val_new = [tmp_path / x for x in
        ("old_train", "old_val", "new_train", "new_val")]
    for d in (train_old, val_old, train_new, val_new):
        d.mkdir()
    source = "A merchant receives reports from many clerks and adjusts prices. " * 10
    (train_old / "old.txt").write_text(source * 2)
    (val_old / "validation.txt").write_text("Validation clerks report new information. " * 8)
    for i, chunk in enumerate((source, source)):
        (train_new / f"p{i}.txt").write_text(chunk)
    (val_new / "validation.txt").write_text("Validation clerks report new information. " * 8)
    tok = ByteBPETokenizer.train([source], vocab_size=280)
    tokenizer = tmp_path / "tokenizer.json"
    tok.save(tokenizer)
    checkpoint = tmp_path / "reader.pt"
    old = train_progressive(tokenizer_path=tokenizer, training=train_old, validation=val_old,
                            output=checkpoint, passes=2, overlap=8,
                            context=64, embed=16, layers=1, heads=4,
                            batch_size=1, threads=1, seed=12,
                            max_updates=2, checkpoint_every=1)
    before = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert old["completed_passes"] == 0
    with pytest.raises(ValueError, match="matching corpus"):
        train_progressive(tokenizer_path=tokenizer, training=train_new, validation=val_new,
                          output=checkpoint, passes=2, overlap=8,
                          context=64, embed=16, layers=1, heads=4,
                          batch_size=1, threads=1, seed=12,
                          max_updates=1, checkpoint_every=1, resume=True)
    result = train_progressive(tokenizer_path=tokenizer, training=train_new, validation=val_new,
                               output=checkpoint, passes=2, overlap=8,
                               context=64, embed=16, layers=1, heads=4,
                               batch_size=1, threads=1, seed=12,
                               max_updates=1, checkpoint_every=1,
                               resume=True, allow_resegment=True)
    after = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert result["paragraph_resegmented"] is True
    assert after["completed_steps"] == before["completed_steps"] + 1
    assert any(row.get("event") == "paragraph_resegmentation"
               for row in after["reading_history"])
    assert after["optimizer"]["state"]
    assert before["optimizer"]["state"]
    assert after["training_sha256"] != before["training_sha256"]


def test_autonomous_workflow_resumes_via_dispatch_after_persisting():
    source = (ROOT / ".github/workflows/ancestor-train.yml").read_text()
    assert "gh release upload" in source
    assert "gh workflow run ancestor-train.yml --ref main" in source
    assert source.index("gh release upload") < source.index("gh workflow run ancestor-train.yml")
    assert "--max-session-seconds" in source
    assert "actions: write" in source
    assert "cancel-in-progress: false" in source
    assert "--max-reading-updates" not in source
    assert "auto_continue:" in source
    assert "session_seconds:" in source
