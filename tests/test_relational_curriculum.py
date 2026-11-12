"""Counterfactual curriculum validation and honest transfer measurement."""
import json
from pathlib import Path

import pytest

from my_own_ai.cli import main
from my_own_ai.relational_curriculum import (
    FACETS, _training_rows, check_relational_splits, evaluate_relational,
    load_relational_cases, prepare_curriculum,
)
from my_own_ai.proposal_training import load_examples
from my_own_ai.reasoning import Stage


ROOT = Path(__file__).resolve().parents[1] / "examples" / "relational"


def datasets():
    return tuple(load_relational_cases(ROOT / f"{name}.json")
                 for name in ("train", "validation", "held"))


def test_synthetic_curriculum_preserves_counterfactuals_and_domain_holdout(tmp_path):
    train, valid, held = datasets()
    check_relational_splits(train, valid, held)
    result = prepare_curriculum(ROOT / "train.json", ROOT / "validation.json",
                                ROOT / "held.json", tmp_path / "prepared")
    assert result["training_cases"] == 4
    assert result["held_out_domains"] == ["hunting"]
    worked = load_examples(result["training"])
    assert len(worked) == 8
    for row in worked:
        assert row["stage"] in (Stage.FORMULATE, Stage.CONSTRUCT)
        assert "Identity is a unity of self and other" not in row["answer"]
    assert not (tmp_path / "prepared" / "relational_held.json").exists()


def test_model_input_excludes_gold_analysis_and_conclusion():
    train, _, _ = datasets()
    rows = _training_rows(train)
    for original, formulate, construct in zip(train, rows[::2], rows[1::2]):
        assert formulate["question"] == construct["question"] == original["question"]
        assert formulate["passages"] == construct["passages"] == original["evidence"]
        assert formulate["candidate"] == construct["candidate"] == ""
        assert original["conclusion"] not in formulate["question"]
        assert all(original["analysis"][facet] not in formulate["question"] for facet in FACETS)
        assert original["conclusion"] in construct["answer"]


def test_rejects_holdout_domain_leak_and_invalid_counterfactuals():
    train, valid, held = datasets()
    with pytest.raises(ValueError, match="Held-out disciplines"):
        check_relational_splits(train, valid, [{**row, "domain": "commerce"} for row in held])
    with pytest.raises(ValueError, match="change evidence AND consequence"):
        duplicate = [dict(x) for x in held]
        duplicate[1]["evidence"] = duplicate[0]["evidence"]
        check_relational_splits(train, valid, duplicate)
    with pytest.raises(ValueError, match="changes the question or domain"):
        duplicate = [dict(x) for x in held]
        duplicate[1]["question"] = "An unrelated question?"
        check_relational_splits(train, valid, duplicate)


def test_rejects_incomplete_ontological_analysis(tmp_path):
    train, _, _ = datasets()
    row = dict(train[0])
    row["analysis"] = {**row["analysis"]}
    row["analysis"].pop("hierarchy")
    path = tmp_path / "missing.json"
    path.write_text(json.dumps([row]), encoding="utf-8")
    with pytest.raises(ValueError, match="Every relational operation"):
        load_relational_cases(path)


def test_evaluation_requires_changed_generated_consequence():
    _, _, held = datasets()

    class Scripted:
        def propose(self, request):
            if request.stage == Stage.FORMULATE:
                return "identity: dynamic relationship; hierarchy: directional dependence."
            original = next(row for row in held if request.context["evidence"][0]["text"] == row["evidence"][0])
            return (f"Premises:\n- [1] {original['evidence'][0]}\n"
                    f"Inference: {original['conclusion']}")

    report = evaluate_relational(held, Scripted())
    assert report["exact_consequences"] == report["total"] == 2
    assert report["counterfactual_passed"] == report["counterfactual_families"] == 1
    assert report["domains"]["hunting"]["exact"] == 2
    assert report["semantic_entailment_verified"] is False

    class Constant(Scripted):
        def propose(self, request):
            if request.stage == Stage.FORMULATE:
                return super().propose(request)
            source = request.context["evidence"][0]["text"]
            return f"Premises:\n- [1] {source}\nInference: {held[0]['conclusion']}"

    fail = evaluate_relational(held, Constant())
    assert fail["exact_consequences"] == 1
    assert fail["counterfactual_passed"] == 0


def test_cli_prepares_curriculum(tmp_path, capsys):
    assert main(["prepare-relational", "--training", str(ROOT / "train.json"),
                 "--validation", str(ROOT / "validation.json"),
                 "--held-out", str(ROOT / "held.json"),
                 "--output", str(tmp_path / "processed")]) is None
    report = json.loads(capsys.readouterr().out)
    assert report["held_out_domains"] == ["hunting"]
    assert Path(report["training"]).exists()
