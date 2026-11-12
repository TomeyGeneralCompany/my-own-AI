"""Regression tests for original English tokenizer, training, inference and gauntlet."""
from pathlib import Path
import json

import pytest
import torch

from my_own_ai.tokenizer import ByteBPETokenizer, BOS, EOS, USER, ASSISTANT
from my_own_ai.language_training import _example, load_language, train_language
from my_own_ai.language_inference import answer_question, prompt_tokens
from my_own_ai.language_evaluation import load_gauntlet, repetition_rate
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
