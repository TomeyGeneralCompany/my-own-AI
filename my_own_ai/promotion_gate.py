"""Independent human-reviewed, frozen-suite promotion gate for original Core."""
import json
from collections import defaultdict
from pathlib import Path
from .persistent_registry import digest_file

MANUAL_CATEGORIES = {"paraphrase", "coherence", "followup"}
AUTO_CATEGORIES = {"syntax", "reading", "qa", "conditional"}


def assess_promotion(evaluation_file, review_file, gauntlet_file,
                     previous_evaluation=None):
    report = json.loads(Path(evaluation_file).read_text(encoding="utf-8"))
    review = json.loads(Path(review_file).read_text(encoding="utf-8"))
    if report.get("format") != "ancestor_evaluation_v1":
        raise ValueError("Invalid candidate evaluation")
    if report.get("evaluated") != 200 or report.get("automatically_scored") != 130:
        raise ValueError("Promotion requires the complete 200-item suite")
    if report.get("gauntlet_sha256") != digest_file(gauntlet_file):
        raise ValueError("Evaluation does not match the frozen English gauntlet")
    if report["exact_correct"] / 130 < .8:
        raise ValueError("Automatic English score is below 80 percent")
    category = report.get("category_scores", {})
    for name in AUTO_CATEGORIES:
        row = category.get(name, {})
        if row.get("auto_scored", 0) == 0 or row.get("exact", 0) / row["auto_scored"] < .7:
            raise ValueError("Automatic category below 70 percent: " + name)
    pairs = report.get("raw_results", {}).get("conditional_pairs_passed", 0)
    if pairs < 7:
        raise ValueError("Fewer than seven counterfactual pairs passed")
    if (review.get("approved") is not True
            or review.get("evaluation_sha256") != digest_file(evaluation_file)
            or review.get("checkpoint_sha256") != report.get("checkpoint_sha256")):
        raise ValueError("Independent review missing or bound to wrong candidate")
    raw = report["raw_results"]["cases"]
    targets = {item["id"]: item for item in raw if item["category"] in MANUAL_CATEGORIES}
    score_rows = review.get("cases")
    if not isinstance(score_rows, list) or len(score_rows) != len(targets) or len(targets) != 70:
        raise ValueError("All 70 open-ended cases require independent review")
    seen, by_category = set(), defaultdict(list)
    for row in score_rows:
        if not isinstance(row, dict) or row.get("id") not in targets or row["id"] in seen:
            raise ValueError("Duplicate or unknown human-review case")
        seen.add(row["id"])
        scores = [row.get(name) for name in ("grammar", "coherence", "relevance")]
        if any(type(score) is not int or not 1 <= score <= 5 for score in scores):
            raise ValueError("Human review requires 1–5 integer scores")
        by_category[targets[row["id"]]["category"]].append(scores)
    if seen != set(targets):
        raise ValueError("Missing human review entries")
    for name in MANUAL_CATEGORIES:
        values = by_category[name]
        if not values:
            raise ValueError("Missing open-ended category")
        if sum(sum(x) for x in values) / (3 * len(values)) < 4:
            raise ValueError("Human rating average below 4/5: " + name)
        if sum(all(score >= 4 for score in x) for x in values) / len(values) < .7:
            raise ValueError("Human passing answers below 70 percent: " + name)
    if previous_evaluation is not None:
        prev = json.loads(Path(previous_evaluation).read_text(encoding="utf-8"))
        if prev.get("gauntlet_sha256") != report.get("gauntlet_sha256"):
            raise ValueError("Previous and candidate use different held-out test sets")
        before = {r["id"]: r for r in prev["raw_results"]["cases"]
                  if r.get("exact_match") is True}
        after = {r["id"]: r for r in raw}
        if any(after.get(key, {}).get("exact_match") is not True for key in before):
            raise ValueError("Regression: a previously correct automatic answer was lost")
    return {"promotable": True, "generation": report["generation"],
            "checkpoint_sha256": report["checkpoint_sha256"],
            "automatic_exact": report["exact_correct"],
            "manual_review_count": len(targets)}
