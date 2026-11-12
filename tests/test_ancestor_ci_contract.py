"""Reject incomplete, fabricated or mislabeled 200-question CI outputs."""
from copy import deepcopy
from pathlib import Path

import pytest

from my_own_ai.language_evaluation import load_gauntlet
from scripts.ancestor_ci_contract import verify_report, verify_lab

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "examples/english/gauntlet.json"


def fake_report():
    cases = load_gauntlet(CASES)
    rows = [
        {"id": case["id"], "category": case["category"],
         "question": case["question"], "expected_answer": case["expected_answer"],
         "actual": "The creature and He.",
         "manual_review_pending": case["requires_manual_review"],
         "exact_match": None if case["requires_manual_review"] else False,
         "utf8_valid": True, "eos": True,
         "raw_token_ids": [5, 7], "prompt_token_ids": [2, 3],
         "settings": {"temperature": 0.0, "seed": 42},
         "checkpoint": "verified.pt"} for case in cases
    ]
    from collections import Counter
    counts = Counter(c["category"] for c in cases)
    categories = {}
    for category, count in counts.items():
        values = [x for x in rows if x["category"] == category]
        categories[category] = {
            "items": count,
            "auto_scored": sum(x["exact_match"] is not None for x in values),
            "exact": 0,
            "manual_review_pending": sum(x["manual_review_pending"] for x in values)}
    return {
        "format": "ancestor_evaluation_v1", "generation": "g0001",
        "evaluated": 200, "automatically_scored": 130,
        "human_review_pending": 70, "promotable": False,
        "reason": "human review pending",
        "exact_correct": 0,
        "checkpoint_sha256": "a" * 64, "gauntlet_sha256": "b" * 64,
        "category_scores": categories,
        "raw_results": {
            "total": 200, "automatically_scored": 130, "manual_review_pending": 70,
            "valid_utf8": 200, "exact_correct": 0, "conditional_pairs_passed": 0,
            "natural_eos": 200, "mean_trigram_repetition": 0.0,
            "english_milestone_met": False,
            "categories": categories, "cases": rows}}


def test_zero_english_score_does_not_mask_correct_ci_wiring():
    outcome = verify_report(fake_report(), CASES, require_generation="g0001")
    assert outcome["infrastructure_ok"] is True
    assert outcome["exact_correct"] == 0
    assert outcome["language_milestone"] is False


@pytest.mark.parametrize("mutate", [
    lambda r: r["raw_results"]["cases"].pop(),
    lambda r: r["raw_results"]["cases"].__setitem__(0, {
        **r["raw_results"]["cases"][0], "id": r["raw_results"]["cases"][1]["id"]}),
    lambda r: r["raw_results"]["cases"][0].__setitem__("question", "tainted question"),
    lambda r: r["raw_results"]["cases"][0].__setitem__("utf8_valid", False),
    lambda r: r["raw_results"].__setitem__("valid_utf8", 199),
    lambda r: r.__setitem__("automatically_scored", 129),
    lambda r: r.__setitem__("promotable", True),
    lambda r: r.__setitem__("exact_correct", 1),
])
def test_failing_ci_contracts_are_not_passed(mutate):
    report = deepcopy(fake_report())
    mutate(report)
    with pytest.raises((AssertionError, KeyError)):
        verify_report(report, CASES, require_generation="g0001")


def test_lab_contract_enforces_read_only_determinism(tmp_path):
    import json
    metadata = {"phase": "candidate", "generation": "g0001",
                "completed_reading_passes": 1, "target_reading_passes": 1,
                "completed_instruction_steps": 2, "target_instruction_steps": 2,
                "chunk_unit": "numbered_paragraph", "tokenizer_sha256": "ab",
                "reading_progress": {"windows_per_pass": 5, "completed_steps": 2}}
    ans = {"generation": "g0001", "training_executed": False,
           "tokenizer_sha256": "ab", "checkpoint_sha256": "abc",
           "answer": {"valid_utf8": True, "raw_token_ids": [1,2,3]}}
    files = [tmp_path / f for f in ("state.json", "a.json", "b.json")]
    for f, value in zip(files, (metadata, ans, ans)):
        f.write_text(json.dumps(value))
    assert verify_lab(*files)["lab_integrated"]
    ans["answer"]["raw_token_ids"] = [7]
    files[-1].write_text(json.dumps(ans))
    with pytest.raises(AssertionError):
        verify_lab(*files)
