"""Controlled cumulative learning with rehearsal and regression-gated promotion."""
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile

from .intellectual_ledger import approved_sources
from .module_memory import encode_json
from .proposal_training import load_examples, verify_splits
from .relational_curriculum import load_relational_cases, evaluate_relational, _normalized
from .session import load_proposer


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(data, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _digest_file(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def prepare_replay(ledger, retained, validation, output, *, accepted_ids):
    """Mix all retained worked examples with explicitly approved new examples."""
    if not accepted_ids or len(set(accepted_ids)) != len(accepted_ids):
        raise ValueError("Choose distinct approved new work identifiers")
    entries = {row["id"]: row for row in approved_sources(ledger)}
    if any(identifier not in entries for identifier in accepted_ids):
        raise ValueError("New training work must be explicitly approved and unchanged")
    paths = [Path(retained).resolve(), Path(validation).resolve(), Path(ledger).resolve()]
    paths.extend(Path(entries[identifier]["source"]).resolve() for identifier in accepted_ids)
    if Path(output).resolve() in paths:
        raise ValueError("Replay output may not replace its input")
    previous = load_examples(retained)
    additions = []
    for identifier in accepted_ids:
        additions.extend(load_examples(entries[identifier]["source"]))
    combined = previous + additions
    if len({row["id"] for row in combined}) != len(combined):
        raise ValueError("Duplicate worked example ID across rehearsal and approved additions")
    validation_rows = load_examples(validation)
    verify_splits(combined, validation_rows, [])
    # Ensure previously accepted experiences remain present unchanged.
    if any(row not in combined for row in previous):
        raise ValueError("Rehearsal unexpectedly lost retained worked examples")
    _write_json(output, combined)
    return {"retained": len(previous), "added": len(additions), "total": len(combined),
            "fingerprint": sha256(encode_json(combined)).hexdigest(), "output": str(Path(output))}


def _score(cases, checkpoint, threads, tokens):
    return evaluate_relational(cases, load_proposer(checkpoint, threads, tokens))


def compare_models(baseline, candidate, old_cases, new_cases, *, training_examples,
                   validation_examples, threads=4, tokens=512):
    """Test baseline and candidate on identical untouched cases; no answer keys in model input."""
    old, new = load_relational_cases(old_cases), load_relational_cases(new_cases)
    old_ids = {row["id"] for row in old}
    new_ids = {row["id"] for row in new}
    old_groups = {row["group"] for row in old}
    new_groups = {row["group"] for row in new}
    if old_ids & new_ids or old_groups & new_groups:
        raise ValueError("Old and new evaluation sets must be independent")
    training = load_examples(training_examples)
    validation = load_examples(validation_examples)
    verify_splits(training, validation, [])
    seen_groups = {row["group"] for row in training + validation}
    seen_questions = {_normalized(row["question"]) for row in training + validation}
    eval_groups = old_groups | new_groups
    eval_questions = {_normalized(row["question"]) for row in old + new}
    if eval_groups & seen_groups or eval_questions & seen_questions:
        raise ValueError("Evaluation questions and groups must be absent from training and validation")
    old_domains = {row["domain"] for row in old}
    new_domains = {row["domain"] for row in new}
    if old_domains & new_domains:
        raise ValueError("New evaluation must use unseen disciplines")
    if Path(baseline).resolve() == Path(candidate).resolve():
        raise ValueError("Comparison requires distinct immutable model checkpoints")
    prior_old = _score(old, baseline, threads, tokens)
    prior_new = _score(new, baseline, threads, tokens)
    next_old = _score(old, candidate, threads, tokens)
    next_new = _score(new, candidate, threads, tokens)
    prior_successes = {row["id"] for row in prior_old["cases"] if row["exact_consequence"]}
    next_successes = {row["id"] for row in next_old["cases"] if row["exact_consequence"]}
    # No earlier demonstrated correct case may be lost, even if total score rises.
    retained = prior_successes <= next_successes
    prior_old_pairs = {row["group"] for row in old if row["group"] in
                       {group for group in {r["group"] for r in old}
                        if all(p["exact_consequence"] for p in prior_old["cases"]
                               if p["group"] == group)}}
    next_old_pairs = {group for group in prior_old_pairs if all(
                      row["exact_consequence"] for row in next_old["cases"]
                      if row["group"] == group)}
    retained_pairs = prior_old_pairs <= next_old_pairs
    new_better = (next_new["exact_consequences"] > prior_new["exact_consequences"]
                  and next_new["counterfactual_passed"] >= prior_new["counterfactual_passed"])
    accepted = retained and retained_pairs and new_better
    return {
        "promotable": accepted,
        "retained_previous_successes": retained,
        "retained_counterfactual_families": retained_pairs,
        "new_material_improved": new_better,
        "old": {"baseline": {"exact": prior_old["exact_consequences"],
                              "pairs": prior_old["counterfactual_passed"]},
                "candidate": {"exact": next_old["exact_consequences"],
                              "pairs": next_old["counterfactual_passed"]}},
        "new": {"baseline": {"exact": prior_new["exact_consequences"],
                              "pairs": prior_new["counterfactual_passed"]},
                "candidate": {"exact": next_new["exact_consequences"],
                              "pairs": next_new["counterfactual_passed"]}},
        "baseline_sha256": _digest_file(baseline),
        "candidate_sha256": _digest_file(candidate),
        "old_cases_sha256": _digest_file(old_cases),
        "new_cases_sha256": _digest_file(new_cases),
        "semantics_verified": False
    }


def gate_and_promote(baseline, candidate, old_cases, new_cases, report,
                     *, training_examples, validation_examples,
                     destination=None, threads=4, tokens=512):
    """Keep current release untouched unless candidate improves and retains all prior wins."""
    paths = [Path(p).resolve() for p in (baseline, candidate, old_cases, new_cases, report,
                                         training_examples, validation_examples)]
    if len(set(paths)) != len(paths):
        raise ValueError("Model, rubric and report files must not overlap")
    if destination is not None and Path(destination).resolve() in paths:
        raise ValueError("Promotion must use a separate release path")
    result = compare_models(baseline, candidate, old_cases, new_cases,
                            training_examples=training_examples, validation_examples=validation_examples,
                            threads=threads, tokens=tokens)
    result["training_sha256"] = _digest_file(training_examples)
    result["validation_sha256"] = _digest_file(validation_examples)
    result["promoted"] = False
    if destination is not None and result["promotable"]:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            shutil.copyfile(candidate, temporary)
            if _digest_file(temporary) != result["candidate_sha256"]:
                raise ValueError("Promoted candidate digest mismatch")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        result["promoted"] = True
        result["release_path"] = str(target)
    _write_json(report, result)
    return result
