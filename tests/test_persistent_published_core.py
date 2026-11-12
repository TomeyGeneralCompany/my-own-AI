"""Persistent author-book corpus, verified archives and instruction-only handoff."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import zipfile

import pytest
import torch

from my_own_ai.published_corpus import validate_published_manifest, prepare_published_corpus
from my_own_ai.persistent_registry import (
    package_generation, unpack_generation, load_active_pointer,
)
from my_own_ai.tokenizer import ByteBPETokenizer
from my_own_ai.language_training import load_language
from scripts.ancestor_core import train, ask


def sample_repo(tmp_path):
    root = tmp_path / "repo"
    (root / "published_books").mkdir(parents=True)
    lines = []
    for i in range(60):
        lines.append(
            f"[{i}] A merchant and a clerk exchange reports about trade in town {i}. "
            "The clerk reports changing prices. The manager adjusts the buying schedule. "
            "Each message informs a decision, and each office has duties. "
            "The company coordinates its work through communication and accountability."
        )
    body = "\n\n".join(lines)
    book = root / "published_books/commerce.txt"
    book.write_text(body, encoding="utf-8")
    manifest = {"format": "ancestor_published_corpus_v1",
                "corpus_id": "synthetic-book-test",
                "works": [{"id": "commerce", "path": "published_books/commerce.txt",
                           "sha256": sha256(book.read_bytes()).hexdigest(),
                           "approved_for_training": True}]}
    (root / "published_books/manifest.json").write_text(json.dumps(manifest))
    return root


def config(repo, tmp_path, state=None):
    return argparse.Namespace(
        repository=repo, state=state or (tmp_path / "state"),
        scratch=tmp_path / "scratch", generation="g0001", parent_state=None,
        section_words=128, max_exercises=6, vocab=320, context=256,
        embed=16, heads=4, layers=1, batch=1, seed=9, threads=1,
        passes=1, overlap=16, instruction_steps=2, max_reading_updates=5,
        max_instruction_updates=1, reading_lr=.0005, instruction_lr=.0003,
    )


def test_published_corpus_checks_hash_and_holdout(tmp_path):
    root = sample_repo(tmp_path)
    assert len(validate_published_manifest(root, root / "published_books/manifest.json")["sources"]) == 1
    made = prepare_published_corpus(root, root / "published_books/manifest.json",
                                    tmp_path / "prepared", target_words=128)
    assert made["policy"] == "section_disjoint_same_book"
    assert all(made["sections"][part] for part in ("train", "validation", "test"))
    with pytest.raises(ValueError):
        prepare_published_corpus(root, root / "published_books/manifest.json",
                                 tmp_path / "prepared")
    (root / "published_books/commerce.txt").write_text("Changed source", encoding="utf-8")
    with pytest.raises(ValueError, match="hash differs"):
        validate_published_manifest(root, root / "published_books/manifest.json")


def test_archive_rejects_corrupt_content_and_wrong_generation(tmp_path):
    state = tmp_path / "trained"
    state.mkdir()
    (state / "generation.json").write_text(json.dumps(
        {"format": "ancestor_generation_v1", "generation": "g0001", "phase": "reading"}))
    for name in ("tokenizer.json", "reading.pt", "reading.best.pt"):
        (state / name).write_bytes(name.encode())
    archive = tmp_path / "model.zip"
    made = package_generation(state, archive)
    assert made["generation"] == "g0001"
    loaded = unpack_generation(archive, tmp_path / "restored",
                               expected_generation="g0001",
                               expected_archive_sha256=made["bundle_sha256"])
    assert loaded["phase"] == "reading"
    with pytest.raises(ValueError, match="another generation"):
        unpack_generation(archive, tmp_path / "wrong", expected_generation="g0002")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        unpack_generation(archive, tmp_path / "wrong2", expected_archive_sha256="0"*64)
    with zipfile.ZipFile(archive, "a") as f:
        f.writestr("unexpected.txt", "attack")
    with pytest.raises(ValueError, match="Unexpected"):
        unpack_generation(archive, tmp_path / "wrong3")


def test_progress_saved_across_distinct_training_invocations_and_inference_is_read_only(tmp_path):
    repo = sample_repo(tmp_path)
    args = config(repo, tmp_path)
    # Read across many distinct invocations. The partial state is packed,
    # unpacked into a new working directory, then resumed.
    first = train(args)
    assert first["phase"] == "reading"
    archive = tmp_path / "generation.zip"
    package_generation(args.state, archive)
    second_state = tmp_path / "restored"
    unpack_generation(archive, second_state, expected_generation="g0001")
    args.state = second_state
    args.scratch = tmp_path / "scratch2"
    for n in range(100):
        current = train(args)
        if current["phase"] == "candidate":
            break
        args.scratch = tmp_path / f"scratch-more-{n}"
    assert current["phase"] == "candidate"
    assert current["completed_instruction_steps"] == 2
    assert current["completed_reading_passes"] == 1
    model, tokenizer = load_language(second_state / "assistant.best.pt",
                                     second_state / "tokenizer.json")
    assert model.config.vocab_size == tokenizer.vocab_size
    before = sha256((second_state / "assistant.best.pt").read_bytes()).hexdigest()
    result = ask(second_state, "What did the clerk report?", "",
                 tmp_path / "answer.json", threads=1, tokens=4)
    assert result["training_executed"] is False
    assert result["checkpoint_sha256"] == before
    assert sha256((second_state / "assistant.best.pt").read_bytes()).hexdigest() == before


def test_active_release_pointer_validation():
    pointer = {"format": "ancestor_active_v1", "generation": "g0001",
               "candidate_tag": "ancestor-g0001-candidate",
               "bundle_sha256": "a"*64, "status": "experimental"}
    assert load_active_pointer(json.dumps(pointer))["generation"] == "g0001"
    pointer["candidate_tag"] = "other-model"
    with pytest.raises(ValueError):
        load_active_pointer(json.dumps(pointer))
