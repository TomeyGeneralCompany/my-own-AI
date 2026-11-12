"""English checkpoint promotion only after measured automatic and human gates."""
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile

from .language_evaluation import load_gauntlet, evaluate_english, MANUAL


def _sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _human_review(path, *, prior_digest, candidate_digest, gauntlet_digest, cases):
    if path is None:
        return {"passed": False, "reason": "Independent reviewer assessment missing"}
    review = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(review, dict) or set(review) != {
        "reviewer_approved", "previous_sha256", "candidate_sha256",
        "gauntlet_sha256", "previous", "candidate",
    } or review["reviewer_approved"] is not True:
        raise ValueError("Human review must contain explicit approval and complete scores")
    if (review["previous_sha256"] != prior_digest or
            review["candidate_sha256"] != candidate_digest or
            review["gauntlet_sha256"] != gauntlet_digest):
        raise ValueError("Human review binds a different model or evaluation set")
    required = {row["id"]: row["category"] for row in cases
                if row["category"] in MANUAL}
    if (not isinstance(review["previous"], dict) or
            not isinstance(review["candidate"], dict) or
            set(review["previous"]) != set(required) or
            set(review["candidate"]) != set(required)):
        raise ValueError("All 70 open-response cases must have independent review scores")

    def ratings(which):
        values = {}
        for identifier, grades in review[which].items():
            if not isinstance(grades, dict) or set(grades) != {
                "grammar", "coherence", "relevance",
            } or any(type(score) is not int or not 1 <= score <= 5
                     for score in grades.values()):
                raise ValueError(f"Invalid human review grades for {identifier}")
            values[identifier] = grades
        return values

    before, after = ratings("previous"), ratings("candidate")
    lost = sorted(item for item in required if
                  all(before[item][metric] >= 4 for metric in
                      ("grammar", "coherence", "relevance"))
                  and any(after[item][metric] < 4 for metric in
                          ("grammar", "coherence", "relevance")))
    metrics = {
        metric: sum(grades[metric] for grades in after.values()) / len(after)
        for metric in ("grammar", "coherence", "relevance")
    }
    category_rates = {}
    for category in sorted(MANUAL):
        items = [item for item in required if required[item] == category]
        category_rates[category] = sum(all(after[item][metric] >= 4 for metric in metrics)
                                       for item in items) / len(items)
    passed = (not lost and all(average >= 4 for average in metrics.values())
              and all(rate >= .70 for rate in category_rates.values()))
    return {"passed": passed, "lost_prior_manual_cases": lost,
            "averages": metrics, "category_success": category_rates,
            "reviewed": len(after)}


def compare_english(previous, candidate, tokenizer, gauntlet, *,
                    review_path=None, tokens=80, threads=2):
    if Path(previous).resolve() == Path(candidate).resolve():
        raise ValueError("Compare different immutable checkpoints")
    cases = load_gauntlet(gauntlet)
    baseline = evaluate_english(previous, tokenizer, cases, max_tokens=tokens, threads=threads)
    trial = evaluate_english(candidate, tokenizer, cases, max_tokens=tokens, threads=threads)
    old_by_id = {row["id"]: row for row in baseline["cases"]}
    next_by_id = {row["id"]: row for row in trial["cases"]}
    if old_by_id.keys() != next_by_id.keys():
        raise ValueError("Baseline and candidate must be evaluated on identical cases")
    lost = sorted(case for case in old_by_id if old_by_id[case]["exact_match"] is True
                  and next_by_id[case]["exact_match"] is not True)
    automatic_improvement = trial["exact_correct"] > baseline["exact_correct"]
    auto_accuracy = (trial["exact_correct"] / trial["automatically_scored"]
                     if trial["automatically_scored"] else 0)
    category_rates = {
        name: part["exact"] / part["auto_scored"]
        for name, part in trial["categories"].items() if part["auto_scored"]
    }
    sufficient_auto = (
        auto_accuracy >= .80 and all(rate >= .70 for rate in category_rates.values())
        and trial["conditional_pairs_passed"] >= 7
    )
    human = _human_review(
        review_path, prior_digest=_sha(previous), candidate_digest=_sha(candidate),
        gauntlet_digest=_sha(gauntlet), cases=cases
    )
    accepted = not lost and automatic_improvement and sufficient_auto and human["passed"]
    return {
        "promotable": accepted, "lost_previously_correct": lost,
        "previous_exact": baseline["exact_correct"],
        "candidate_exact": trial["exact_correct"],
        "auto_accuracy": auto_accuracy,
        "category_auto_accuracy": category_rates,
        "automatic_improvement": automatic_improvement,
        "automatic_threshold_passed": sufficient_auto,
        "manual_review": human,
        "candidate_sha256": _sha(candidate),
        "previous_sha256": _sha(previous),
        "gauntlet_sha256": _sha(gauntlet),
        "baseline": baseline, "candidate": trial,
    }


def promote_english(previous, candidate, tokenizer, gauntlet, report_path, *,
                    destination=None, review_path=None, tokens=80, threads=2):
    paths = [Path(p).resolve() for p in (previous, candidate, tokenizer, gauntlet, report_path)]
    if len(set(paths)) != len(paths):
        raise ValueError("Review report cannot replace any input")
    if destination is not None and Path(destination).resolve() in paths:
        raise ValueError("Release destination must not overwrite source checkpoints")
    report = compare_english(previous, candidate, tokenizer, gauntlet,
                             review_path=review_path, tokens=tokens, threads=threads)
    report["promoted"] = False
    if destination is not None and report["promotable"]:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            shutil.copyfile(candidate, temporary)
            if _sha(temporary) != report["candidate_sha256"]:
                raise ValueError("Candidate changed during release")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        report["promoted"] = True
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=Path(report_path).parent,
                            delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(report, stream, indent=2)
    os.replace(temporary, report_path)
    return report
