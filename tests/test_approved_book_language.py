"""Owner-approved source to local English model regression tests."""
from pathlib import Path
import json

import pytest
from my_own_ai.book_language import prepare_approved_books
from my_own_ai.book_instruction import create_book_continuations
from my_own_ai.intellectual_ledger import register, decide
from my_own_ai.corpus import collect_documents
from my_own_ai.tokenizer import ByteBPETokenizer
from my_own_ai.language_training import train_language, load_language
from my_own_ai.language_inference import answer_question


def make_book(path, count=42):
    paragraphs = []
    for number in range(count):
        paragraphs.append(
            f"A trading house {number} receives information from its agents. "
            f"The manager in section {number} reads the latest report before choosing an order. "
            f"Each clerk communicates a distinct change in demand to the management office. "
            f"After receiving the message, the manager adjusts the planned delivery. "
            f"Without timely information the manager cannot distinguish yesterday's demand "
            f"from today's demand in section {number}. "
            f"This reasoning shows how information and authority combine to organize a trading house."
        )
    path.write_text("\n\n".join(paragraphs), encoding="utf-8")
    return path


def approve(tmp_path):
    ledger = tmp_path / "approved.json"
    source = make_book(tmp_path / "commerce.txt")
    register(ledger, "commerce_book", "authored_work", source, origin="author manuscript")
    return ledger, source


def test_only_approved_unchanged_sources_can_form_local_book_corpus(tmp_path):
    ledger, original = approve(tmp_path)
    with pytest.raises(ValueError, match="No approved"):
        prepare_approved_books(ledger, tmp_path / "reject", target_words=128)
    decide(ledger, "commerce_book", "approved")
    generated = prepare_approved_books(ledger, tmp_path / "private", target_words=128)
    assert generated["policy"] == "section_disjoint_same_book"
    assert generated["total_words"] > 2000
    assert generated["sections"]["train"] > 0
    assert generated["sections"]["validation"] > 0
    assert generated["sections"]["test"] > 0
    assert "The manager" not in (tmp_path / "private" / "manifest.json").read_text()
    assert original.read_text() == original.read_text()
    assert set(r["source_sha256"] for r in generated["sources"]) != set()
    with pytest.raises(ValueError, match="new private"):
        prepare_approved_books(ledger, tmp_path / "private", target_words=128)
    original.write_text(original.read_text() + "\nAn additional unauthorized edit.")
    with pytest.raises(ValueError, match="changed"):
        prepare_approved_books(ledger, tmp_path / "after_mutation", target_words=128)


def test_training_sees_no_book_test_material_and_model_generates(tmp_path):
    ledger, _ = approve(tmp_path)
    decide(ledger, "commerce_book", "approved")
    book_dir = tmp_path / "private"
    prepare_approved_books(ledger, book_dir, target_words=128)
    exercises = tmp_path / "exercises"
    outcome = create_book_continuations(book_dir, exercises, limit_per_split=12)
    assert outcome["splits"]["train"] > 0
    assert outcome["splits"]["validation"] > 0
    assert outcome["splits"]["test"] > 0
    test_questions = {row["question"] for row in json.loads((exercises / "test.json").read_text())}
    train_questions = {row["question"] for row in json.loads((exercises / "train.json").read_text())}
    assert not test_questions & train_questions
    training_docs = collect_documents(book_dir / "train")
    tokenizer = ByteBPETokenizer.train([doc["text"] for doc in training_docs], vocab_size=360)
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    base = tmp_path / "base.pt"
    setup = dict(tokenizer_path=tokenizer_path, embed=16, heads=4, layers=1,
                 context=384, batch_size=1, threads=1, seed=42, checkpoint_every=1)
    previous = train_language(**setup, phase="pretrain", training=book_dir / "train",
                              validation=book_dir / "validation", output=base,
                              steps=2)
    assert previous["completed_this_run"] == 2
    next_model = tmp_path / "answer.pt"
    result = train_language(**setup, phase="instruction",
                            training=exercises / "train.json",
                            validation=exercises / "validation.json",
                            output=next_model, steps=2,
                            init_checkpoint=previous["best_checkpoint"])
    assert result["completed_this_run"] == 2
    model, token = load_language(result["best_checkpoint"], tokenizer_path)
    assert model.config.vocab_size == token.vocab_size
    answer = answer_question(result["best_checkpoint"], tokenizer_path,
                             "What does the clerk communicate?", threads=1, max_tokens=4)
    assert "answer" in answer and answer["model_generated"] is True
