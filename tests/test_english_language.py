"""Regression tests for original English tokenizer, training, inference and gauntlet."""
from pathlib import Path
import json

import pytest
import torch

from my_own_ai.tokenizer import ByteBPETokenizer, BOS, EOS, USER, ASSISTANT
from my_own_ai.language_training import _example, load_language, train_language
from my_own_ai.language_inference import answer_question, prompt_tokens
from my_own_ai.language_evaluation import load_gauntlet, repetition_rate
from my_own_ai.language_corpus import prepare_english_corpus, near_duplicate
from my_own_ai import language_regression
from my_own_ai.train import load_model

ROOT = Path(__file__).resolve().parents[1] / "examples" / "english"


def tiny_tokenizer(tmp_path):
    vocab = ByteBPETokenizer.train(
        ["The engineer explains the result.", "The operator sends the result.",
         "English includes punctuation and words."], vocab_size=300)
    location = tmp_path / "tokenizer.json"
    vocab.save(location)
    return vocab, location


def test_bpe_roundtrip_unicode_punctuation_and_unseen_input(tmp_path):
    tokenizer, path = tiny_tokenizer(tmp_path)
    for phrase in ("Hello, world!", "Résumé — Ελληνικά — 日本語", "new\nlines\nhere",
                   "completely unfamiliar terminology 12345", "A word and another word."):
        assert tokenizer.decode(tokenizer.encode(phrase)) == phrase
    assert ByteBPETokenizer.load(path).fingerprint() == tokenizer.fingerprint()
    with pytest.raises(ValueError, match="Special tokens"):
        tokenizer.decode([BOS, EOS])


def test_prompts_and_supervised_targets_preserve_role_and_loss_boundaries(tmp_path):
    tokenizer, _ = tiny_tokenizer(tmp_path)
    row = {"id": "sample", "question": "What happened?", "answer": "A message arrived."}
    inputs, labels = _example(row, tokenizer, context=128)
    prefix = [BOS, USER] + tokenizer.encode(row["question"]) + [EOS, ASSISTANT]
    assert inputs[:len(prefix)] == prefix
    assert labels[:len(prefix)-1] == [-100] * (len(prefix)-1)
    assert labels[len(prefix)-1:] == tokenizer.encode(row["answer"]) + [EOS]
    assert prompt_tokens("What happened?", tokenizer, 128) == prefix
    with pytest.raises(ValueError, match="context"):
        prompt_tokens("What happened? " * 50, tokenizer, 16)


def test_original_cpu_train_and_resume_match_uninterrupted(tmp_path):
    torch.set_num_threads(1)
    _, tokenizer = tiny_tokenizer(tmp_path)
    root = tmp_path / "data"
    root.mkdir()
    train_file = root / "train.json"
    valid_file = root / "valid.json"
    train_file.write_text(json.dumps([
        {"id": "t1", "question": "What is the first step?", "answer": "Read the evidence."},
        {"id": "t2", "question": "Where is the machine?", "answer": "The machine is inside."},
    ]))
    valid_file.write_text(json.dumps([
        {"id": "v1", "question": "What changed?", "answer": "The signal stopped."}
    ]))
    common = dict(tokenizer_path=tokenizer, training=train_file, validation=valid_file,
                  phase="instruction", embed=16, heads=4, layers=1, context=128,
                  batch_size=1, seed=7, threads=1, lr=.001, checkpoint_every=1)
    original = root / "whole.pt"
    resumed = root / "resumed.pt"
    train_language(**common, output=original, steps=4)
    train_language(**common, output=resumed, steps=2)
    train_language(**common, output=resumed, steps=4, resume=True)
    m1, _ = load_language(original, tokenizer)
    m2, _ = load_language(resumed, tokenizer)
    assert all(torch.equal(m1.state_dict()[key], m2.state_dict()[key])
               for key in m1.state_dict())
    response = answer_question(resumed, tokenizer, "What changed?", max_tokens=12, threads=1)
    assert response["model_generated"] and response["prompt_token_ids"][1] == USER
    assert response["stopped_by"] in ("eos", "max_tokens")
    assert isinstance(response["answer"], str)
    with pytest.raises(ValueError, match="Language checkpoint"):
        load_model(resumed).generate(b"Hi", 1)


def test_incompatible_tokenizer_and_corpus_revision_are_rejected(tmp_path):
    _, tokenizer = tiny_tokenizer(tmp_path)
    training = tmp_path / "train.json"
    validation = tmp_path / "validation.json"
    training.write_text(json.dumps([
        {"id": "t", "question": "Who arrived?", "answer": "The engineer."}
    ]))
    validation.write_text(json.dumps([
        {"id": "v", "question": "What moved?", "answer": "The machine."}
    ]))
    args = dict(tokenizer_path=tokenizer, training=training, validation=validation,
                output=tmp_path / "trained.pt", phase="instruction",
                steps=2, embed=16, heads=4, layers=1, context=128,
                batch_size=1, checkpoint_every=1, threads=1)
    train_language(**args)
    mismatch = tmp_path / "wrong-tokenizer.json"
    ByteBPETokenizer().save(mismatch)
    with pytest.raises(ValueError, match="Tokenizer mismatch"):
        load_language(args["output"], mismatch)
    training.write_text(training.read_text().replace("engineer", "operator"))
    with pytest.raises(ValueError, match="changed"):
        train_language(**{**args, "steps": 3, "resume": True})


def test_fixed_gauntlet_counts_and_no_training_question_overlap():
    cases = load_gauntlet(ROOT / "gauntlet.json")
    training = json.loads((ROOT / "dialogue/train.json").read_text())
    validation = json.loads((ROOT / "dialogue/validation.json").read_text())
    prompts = {case["question"].casefold() for case in cases}
    assert len(prompts) == 200
    assert not prompts.intersection(row["question"].casefold() for row in training + validation)
    assert repetition_rate([1, 2, 3, 1, 2, 3, 1, 2, 3]) > 0


def test_english_corpus_uses_document_disjoint_splits_and_deduplicates(tmp_path):
    source = tmp_path / "sources"
    source.mkdir()
    for i, statement in enumerate((
        "The mechanic checked the gearbox before making a repair.",
        "The gardener watered the trees during the early morning.",
        "The researcher observed a signal moving through the laboratory.",
        "The supervisor sent a report to the accounting department.",
        "The teacher discussed a poem with the class after lunch.",
        "The courier carried a letter to the distant library.",
    )):
        (source / f"document{i}.txt").write_text(statement * 4, encoding="utf-8")
    (source / "duplicate.txt").write_text((source / "document0.txt").read_text())
    result = prepare_english_corpus(source, tmp_path / "prepared")
    assert sum(result[k] for k in ("train", "validation", "test")) == 6
    assert sum(len(list((tmp_path / "prepared" / k).glob("*.txt")))
               for k in ("train", "validation", "test")) == 6
    assert near_duplicate("The reader wrote a new report.", "The reader wrote a new report.")
    with pytest.raises(ValueError, match="must not exist"):
        prepare_english_corpus(source, tmp_path / "prepared")


def test_english_release_gate_rejects_forgetting_and_requires_human_review(monkeypatch, tmp_path):
    previous, candidate = tmp_path / "previous.pt", tmp_path / "candidate.pt"
    previous.write_bytes(b"previous-model")
    candidate.write_bytes(b"candidate-model")
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}")
    gauntlet = ROOT / "gauntlet.json"
    all_cases = load_gauntlet(gauntlet)
    old = [{"id": "a", "exact_match": True}, {"id": "b", "exact_match": False}]
    improved = [{"id": "a", "exact_match": True}, {"id": "b", "exact_match": True}]
    category = {
        "syntax": {"auto_scored": 40, "exact": 35},
        "reading": {"auto_scored": 40, "exact": 35},
        "qa": {"auto_scored": 30, "exact": 27},
        "conditional": {"auto_scored": 20, "exact": 16},
    }
    def fake(checkpoint, *args, **kwargs):
        return {"exact_correct": 105 if Path(checkpoint) == previous else 113,
                "automatically_scored": 130, "conditional_pairs_passed": 8,
                "categories": category,
                "cases": old if Path(checkpoint) == previous else improved}
    monkeypatch.setattr(language_regression, "evaluate_english", fake)
    report = language_regression.compare_english(previous, candidate, tokenizer, gauntlet)
    assert report["automatic_improvement"] and not report["promotable"]
    assert not report["manual_review"]["passed"]

    from hashlib import sha256
    expected = [item["id"] for item in all_cases if item["requires_manual_review"]]
    grades = {identifier: {"grammar": 5, "coherence": 5, "relevance": 5}
              for identifier in expected}
    review = tmp_path / "review.json"
    review.write_text(json.dumps({
        "reviewer_approved": True,
        "previous_sha256": sha256(previous.read_bytes()).hexdigest(),
        "candidate_sha256": sha256(candidate.read_bytes()).hexdigest(),
        "gauntlet_sha256": sha256(gauntlet.read_bytes()).hexdigest(),
        "previous": grades, "candidate": grades,
    }))
    accepted = language_regression.promote_english(
        previous, candidate, tokenizer, gauntlet, tmp_path / "report.json",
        destination=tmp_path / "released.pt", review_path=review)
    assert accepted["promoted"]
    assert (tmp_path / "released.pt").read_bytes() == candidate.read_bytes()

    def regress(checkpoint, *args, **kwargs):
        result = fake(checkpoint, *args, **kwargs)
        if Path(checkpoint) == candidate:
            result["cases"] = [
                {"id": "a", "exact_match": False},
                {"id": "b", "exact_match": True}
            ]
        return result
    monkeypatch.setattr(language_regression, "evaluate_english", regress)
    denied = language_regression.compare_english(
        previous, candidate, tokenizer, gauntlet, review_path=review)
    assert denied["lost_previously_correct"] == ["a"] and not denied["promotable"]
