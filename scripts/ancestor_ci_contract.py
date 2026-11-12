"""CI contract for all 200 held-out questions and complete neural lab chain.

A failure means broken wiring, invalid evaluation coverage, corrupted checkpoint,
non-deterministic inference or missing case data. A score below the *learning*
milestone is diagnostic, NOT a CI infrastructure failure.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from my_own_ai.language_evaluation import AUTO, MANUAL, SIZES, load_gauntlet


def verify_report(report, cases_path, *, require_generation=None):
    cases = load_gauntlet(cases_path)
    assert len(cases) == 200
    assert report["format"] == "ancestor_evaluation_v1"
    assert report["evaluated"] == len(cases)
    if require_generation is not None:
        assert report["generation"] == require_generation
    assert report["automatically_scored"] == sum(SIZES[name] for name in AUTO)
    assert report["human_review_pending"] == sum(SIZES[name] for name in MANUAL)
    assert report["promotable"] is False
    assert report["reason"]
    results = report["raw_results"]
    assert results["total"] == 200
    assert results["automatically_scored"] == 130
    assert results["manual_review_pending"] == 70
    assert results["valid_utf8"] == 200
    assert len(results["cases"]) == 200
    assert 0 <= results["exact_correct"] <= 130
    assert 0 <= results["conditional_pairs_passed"] <= 10
    assert 0 <= results["natural_eos"] <= 200
    assert set(results["categories"]) == set(SIZES)
    assert len({row["id"] for row in results["cases"]}) == 200

    source = {row["id"]: row for row in cases}
    counts = Counter()
    for row in results["cases"]:
        reference = source[row["id"]]
        counts[row["category"]] += 1
        assert row["question"] == reference["question"]
        assert row["category"] == reference["category"]
        assert row["expected_answer"] == reference["expected_answer"]
        assert row["manual_review_pending"] is reference["requires_manual_review"]
        assert isinstance(row["actual"], str)
        assert row["utf8_valid"] is True
        assert row["exact_match"] is None if row["manual_review_pending"] else isinstance(row["exact_match"], bool)
        assert isinstance(row["raw_token_ids"], list)
        assert isinstance(row["prompt_token_ids"], list)
        assert isinstance(row["settings"], dict)
        assert row["settings"]["temperature"] == 0.0
        assert row["settings"]["seed"] == 42
        assert row["checkpoint"]
    assert dict(counts) == SIZES
    assert results["exact_correct"] == sum(row["exact_match"] is True for row in results["cases"])
    assert report["exact_correct"] == results["exact_correct"]
    assert report["category_scores"] == results["categories"]
    return {
        "generation": report["generation"],
        "total": 200,
        "automatic": 130,
        "manual_pending": 70,
        "exact_correct": results["exact_correct"],
        "conditional_pairs_passed": results["conditional_pairs_passed"],
        "natural_eos": results["natural_eos"],
        "mean_trigram_repetition": results["mean_trigram_repetition"],
        "model_checkpoint_sha256": report["checkpoint_sha256"],
        "gauntlet_sha256": report["gauntlet_sha256"],
        "infrastructure_ok": True,
        "language_milestone": results["english_milestone_met"],
    }


def verify_lab(state_path, answer_a, answer_b):
    state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    a, b = [json.loads(Path(path).read_text(encoding="utf-8")) for path in (answer_a, answer_b)]
    assert state["phase"] == "candidate"
    assert state["completed_reading_passes"] == state["target_reading_passes"]
    assert state["completed_instruction_steps"] == state["target_instruction_steps"]
    assert state["chunk_unit"] == "numbered_paragraph"
    assert state["reading_progress"]["windows_per_pass"] > 0
    for response in (a, b):
        assert response["training_executed"] is False
        assert response["generation"] == state["generation"]
        assert response["tokenizer_sha256"] == state["tokenizer_sha256"]
        assert response["answer"]["valid_utf8"] is True
        assert isinstance(response["answer"]["raw_token_ids"], list)
    assert a["checkpoint_sha256"] == b["checkpoint_sha256"]
    assert a["answer"]["raw_token_ids"] == b["answer"]["raw_token_ids"]
    return {"lab_integrated": True, "generation": state["generation"],
            "model_checkpoint_sha256": a["checkpoint_sha256"],
            "reading_updates": state["reading_progress"]["completed_steps"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report")
    parser.add_argument("--cases", default="examples/english/gauntlet.json")
    parser.add_argument("--generation")
    parser.add_argument("--lab-state")
    parser.add_argument("--lab-answer-a")
    parser.add_argument("--lab-answer-b")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.report:
        result = verify_report(json.loads(Path(args.report).read_text(encoding="utf-8")),
                               args.cases, require_generation=args.generation)
    else:
        if not all((args.lab_state, args.lab_answer_a, args.lab_answer_b)):
            parser.error("Give --report or all three lab paths")
        result = verify_lab(args.lab_state, args.lab_answer_a, args.lab_answer_b)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
