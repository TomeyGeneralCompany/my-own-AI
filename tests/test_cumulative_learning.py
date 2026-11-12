"""Regression tests for intergenerational accumulation and gated learning."""
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from my_own_ai.cumulative_learning import prepare_replay, gate_and_promote
from my_own_ai.intellectual_ledger import approved_sources, decide, register, stage_approved_corpus
from my_own_ai.model import LocalTransformer, ModelConfig
from my_own_ai.proposal_training import train_proposals
from my_own_ai.train import load_model
from my_own_ai import cumulative_learning


def row(identifier, group, question):
    return {"id": identifier, "group": group, "stage": "construct",
            "question": question, "passages": ["The operator must receive the notice before acting."],
            "plan": "Track the message channel.", "candidate": "", "feedback": "",
            "answer": "Premises:\n- [1] The operator must receive the notice\nInference: Delaying the notice delays the action."}


def save(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_provenance_needs_approval_and_source_integrity(tmp_path):
    ledger = tmp_path / "ledger.json"
    source = save(tmp_path / "work.json", [row("w1", "new", "What follows from delayed notice?")])
    record = register(ledger, "work1", "derived_work", source,
                      origin="AI draft awaiting human review", evidence=["local:source-1"])
    assert record["status"] == "pending"
    assert approved_sources(ledger) == []
    with pytest.raises(ValueError, match="already exists"):
        register(ledger, "work1", "derived_work", source, origin="duplicate")
    decide(ledger, "work1", "approved")
    assert len(approved_sources(ledger)) == 1
    with pytest.raises(ValueError, match="pending"):
        decide(ledger, "work1", "approved")
    source.write_text("altered", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        approved_sources(ledger)


def test_approved_replay_preserves_old_examples_and_rejects_unapproved(tmp_path):
    ledger = tmp_path / "ledger.json"
    retained = save(tmp_path / "retained.json", [row("old", "old_group", "What delays an operator?")])
    valid = save(tmp_path / "valid.json", [row("validation", "validation_group", "Why does a relay wait?")])
    addition = save(tmp_path / "new.json", [row("new", "new_group", "What delays the mechanic?")])
    register(ledger, "new_work", "reasoning", addition, origin="owner-verified exercise")
    output = tmp_path / "replay.json"
    with pytest.raises(ValueError, match="explicitly approved"):
        prepare_replay(ledger, retained, valid, output, accepted_ids=["new_work"])
    decide(ledger, "new_work", "approved")
    report = prepare_replay(ledger, retained, valid, output, accepted_ids=["new_work"])
    assert report["retained"] == 1 and report["added"] == 1
    assert [r["id"] for r in json.loads(output.read_text())] == ["old", "new"]
    with pytest.raises(ValueError, match="distinct approved"):
        prepare_replay(ledger, retained, valid, output, accepted_ids=["new_work", "new_work"])


def _report(rows):
    return {"cases": [{"id": key, "group": group, "exact_consequence": passed}
                      for key, group, passed in rows],
            "exact_consequences": sum(passed for _, _, passed in rows),
            "counterfactual_passed": sum(all(passed for _, g, passed in rows if g == group)
                                         for group in {g for _, g, _ in rows})}


def test_gate_promotes_only_improvements_without_forgetting(monkeypatch, tmp_path):
    old = save(tmp_path / "old.json", [
        {"id": "a", "group": "g1", "domain": "old", "question": "What happens in the old system?"},
        {"id": "b", "group": "g1", "domain": "old", "question": "What happens in the old system?"}])
    new = save(tmp_path / "new.json", [
        {"id": "c", "group": "g2", "domain": "new", "question": "What happens in the new system?"},
        {"id": "d", "group": "g2", "domain": "new", "question": "What happens in the new system?"}])
    baseline = tmp_path / "baseline.pt"
    candidate = tmp_path / "candidate.pt"
    baseline.write_bytes(b"old")
    candidate.write_bytes(b"new")
    monkeypatch.setattr(cumulative_learning, "load_relational_cases", lambda path: json.loads(Path(path).read_text()))
    reports = {
        ("baseline.pt", "old.json"): _report([("a", "g1", True), ("b", "g1", True)]),
        ("candidate.pt", "old.json"): _report([("a", "g1", True), ("b", "g1", True)]),
        ("baseline.pt", "new.json"): _report([("c", "g2", False), ("d", "g2", False)]),
        ("candidate.pt", "new.json"): _report([("c", "g2", True), ("d", "g2", True)]),
    }
    monkeypatch.setattr(cumulative_learning, "_score", lambda cases, checkpoint, threads, tokens:
                        reports[(Path(checkpoint).name, "old.json" if cases[0]["domain"] == "old" else "new.json")])
    destination = tmp_path / "release.pt"
    training = save(tmp_path / "training.json", [row("training", "training_group", "How is a new signal delivered?")])
    validation = save(tmp_path / "validation.json", [row("validation", "validation_group", "When does response occur?")])
    result = gate_and_promote(baseline, candidate, old, new, tmp_path / "report.json",
                              training_examples=training, validation_examples=validation,
                              destination=destination)
    assert result["promotable"] and result["promoted"]
    assert destination.read_bytes() == b"new"
    reports[("candidate.pt", "old.json")] = _report([("a", "g1", True), ("b", "g1", False)])
    candidate.write_bytes(b"new-second")
    result = gate_and_promote(baseline, candidate, old, new, tmp_path / "report.json",
                              training_examples=training, validation_examples=validation,
                              destination=destination)
    assert not result["promotable"] and not result["promoted"]
    assert destination.read_bytes() == b"new"


def test_proposal_resume_restores_optimizer_rng_and_rejects_data_change(tmp_path):
    torch.set_num_threads(1)
    cfg = ModelConfig(context_size=384, embed_dim=16, heads=4, layers=1)
    torch.manual_seed(77)
    model = LocalTransformer(cfg)
    parent = tmp_path / "parent.pt"
    torch.save({"model_config": asdict(cfg), "state_dict": model.state_dict()}, parent)
    training = save(tmp_path / "training.json", [row("a", "a", "How does a delayed notice affect action?")])
    valid = save(tmp_path / "validation.json", [row("b", "b", "What follows when notice arrives late?")])
    continuous = tmp_path / "continuous.pt"
    interrupted = tmp_path / "interrupted.pt"
    train_proposals(parent, training, valid, continuous, steps=4, threads=1, checkpoint_every=2)
    train_proposals(parent, training, valid, interrupted, steps=2, threads=1, checkpoint_every=1)
    train_proposals(interrupted, training, valid, interrupted, steps=4, threads=1,
                    checkpoint_every=1, resume=True)
    state_full = load_model(continuous).state_dict()
    state_resumed = load_model(interrupted).state_dict()
    assert all(torch.equal(state_full[key], state_resumed[key]) for key in state_full)
    training.write_text(training.read_text().replace("delayed notice", "missing notice"))
    with pytest.raises(ValueError, match="Resume requires matching"):
        train_proposals(interrupted, training, valid, interrupted, steps=5, threads=1, resume=True)


def test_gate_blocks_training_answer_leakage(monkeypatch, tmp_path):
    baseline, candidate = tmp_path / "baseline.pt", tmp_path / "candidate.pt"
    baseline.write_bytes(b"previous")
    candidate.write_bytes(b"next")
    old = save(tmp_path / "old.json", [{"id": "old", "group": "old_group", "domain": "old_domain",
                                        "question": "What happens to the old system?"}])
    new = save(tmp_path / "new.json", [{"id": "new", "group": "new_group", "domain": "new_domain",
                                        "question": "What happens to the new system?"}])
    training = save(tmp_path / "training.json", [row("training", "new_group", "What happens to the new system?")])
    validation = save(tmp_path / "validation.json", [row("validation", "validation_group", "When does the valve respond?")])
    monkeypatch.setattr(cumulative_learning, "load_relational_cases", lambda path: json.loads(Path(path).read_text()))
    with pytest.raises(ValueError, match="absent from training"):
        gate_and_promote(baseline, candidate, old, new, tmp_path / "report.json",
                         training_examples=training, validation_examples=validation)


def test_only_approved_manuscripts_enter_staged_corpus(tmp_path):
    ledger = tmp_path / "ledger.json"
    approved = tmp_path / "original.md"
    approved.write_text("Original authored philosophical text.", encoding="utf-8")
    pending = tmp_path / "draft.txt"
    pending.write_text("Unapproved neural draft.", encoding="utf-8")
    register(ledger, "authored", "authored_work", approved, origin="author manuscript")
    register(ledger, "unreviewed", "derived_work", pending, origin="AI draft pending review")
    decide(ledger, "authored", "approved")
    target = tmp_path / "training-corpus"
    report = stage_approved_corpus(ledger, target)
    assert report["documents"] == 1
    assert (target / "authored.md").read_text() == approved.read_text()
    assert not (target / "unreviewed.txt").exists()
    with pytest.raises(ValueError, match="new directory"):
        stage_approved_corpus(ledger, target)
    assert (target / "source-manifest.json").is_file()
