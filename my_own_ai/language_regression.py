"""English checkpoint regression gate with immutable previous release.

A checkpoint is never promoted based on validation loss alone. Previously
demonstrated exact successes must remain successes and independent human
review must be recorded for all open-ended dimensions before promotion.
"""
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import shutil
from .language_evaluation import load_gauntlet, evaluate_english


def compare_english(previous, candidate, tokenizer, gauntlet, *,
                    old_report=None, review_path=None, tokens=80, threads=2):
    if Path(previous).resolve() == Path(candidate).resolve():
        raise ValueError("Compare different immutable checkpoints")
    cases = load_gauntlet(gauntlet)
    baseline = evaluate_english(previous, tokenizer, cases, max_tokens=tokens, threads=threads)
    trial = evaluate_english(candidate, tokenizer, cases, max_tokens=tokens, threads=threads)
    old_by_id = {row["id"]: row for row in baseline["cases"]}
    new_by_id = {row["id"]: row for row in trial["cases"]}
    lost = sorted(case for case in old_by_id if old_by_id[case]["exact_match"] is True
                  and new_by_id[case]["exact_match"] is not True)
    growth = trial["exact_correct"] > baseline["exact_correct"]
    review_passed = False
    if review_path is not None:
        review = json.loads(Path(review_path).read_text(encoding="utf-8"))
        review_passed = (isinstance(review, dict) and review.get("reviewer_approved") is True
                         and review.get("candidate_sha256") == sha256(Path(candidate).read_bytes()).hexdigest()
                         and review.get("gauntlet_sha256") == sha256(Path(gauntlet).read_bytes()).hexdigest()
                         and review.get("manual_cases_reviewed") == 70
                         and review.get("grammar_average", 0) >= 4
                         and review.get("coherence_average", 0) >= 4)
    permitted = not lost and growth and review_passed
    return {
        "promotable": permitted, "lost_previously_correct": lost,
        "previous_exact": baseline["exact_correct"], "candidate_exact": trial["exact_correct"],
        "automatic_improvement": growth, "manual_review_passed": review_passed,
        "candidate_sha256": sha256(Path(candidate).read_bytes()).hexdigest(),
        "previous_sha256": sha256(Path(previous).read_bytes()).hexdigest(),
        "gauntlet_sha256": sha256(Path(gauntlet).read_bytes()).hexdigest(),
        "baseline": baseline, "candidate": trial,
    }


def promote_english(previous, candidate, tokenizer, gauntlet, report_path, *,
                    destination=None, review_path=None, tokens=80, threads=2):
    paths = [Path(p).resolve() for p in (previous, candidate, tokenizer, gauntlet, report_path)]
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
            if sha256(temporary.read_bytes()).hexdigest() != report["candidate_sha256"]:
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
