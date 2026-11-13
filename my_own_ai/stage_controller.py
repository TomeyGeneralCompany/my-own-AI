"""Controller-owned competence stages. No training counter may unlock a gate.

This module is independent of the neural optimizer: an exam must provide
actual answers made by the precise checkpoint under evaluation.
"""
from hashlib import sha256
import json
from pathlib import Path

ENGLISH = "english"
READ = "read"
COMPREHEND = "comprehend"
SYNTHESIS = "synthesis"
REVIEW = "awaiting_owner_review"
COMPLETE = "qualified"
AUTO_MIN = .80
PER_CATEGORY_MIN = .70
BOOK_MIN = .80


def fingerprint(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def initial_ledger(manifest_sha256, books, parent, source_hashes=None):
    if not books or len(books) != len(set(books)):
        raise ValueError("Approved works must have unique ordered IDs")
    return {"format": "competency_locked_v1", "manifest_sha256": manifest_sha256,
            "parent_generation": parent, "books": list(books),
            "source_hashes": source_hashes or {},
            "stage": ENGLISH, "english_fundamentals_passed": False,
            "book_index": 0, "attempts": {},
            "evidence": [], "last_checkpoint_sha256": None}


def verify_ledger(ledger, manifest_sha256, books):
    if (ledger.get("format") != "competency_locked_v1" or
            ledger.get("manifest_sha256") != manifest_sha256 or
            ledger.get("books") != list(books) or
            not 0 <= ledger.get("book_index", -1) <= len(books) or
            ledger.get("stage") not in {ENGLISH, READ, COMPREHEND, SYNTHESIS, REVIEW, COMPLETE}):
        raise ValueError("Changed corpus, book order, or invalid stage record")
    if ledger["stage"] in (READ, COMPREHEND) and ledger["book_index"] >= len(books):
        raise ValueError("Book cursor has escaped the approved corpus")
    return ledger


def english_pass(report):
    # All categories, including paired changed-premise tests, are hard gates.
    if report.get("evaluated") != 200 or report.get("automatically_scored") != 130:
        return False
    if report.get("exact_correct", 0) < 104:
        return False
    for name in ("syntax", "reading", "qa", "conditional"):
        row = report.get("category_scores", {}).get(name, {})
        if row.get("auto_scored", 0) == 0 or row.get("exact", 0) / row["auto_scored"] < PER_CATEGORY_MIN:
            return False
    if report.get("raw_results", {}).get("conditional_pairs_passed", 0) < 7:
        return False
    return True


def book_pass(report):
    return (report.get("total", 0) >= 10 and
            report.get("correct", 0) / max(report.get("total", 1), 1) >= BOOK_MIN and
            report.get("checkpoint_sha256") and report.get("source_sha256") and
            report.get("questions_sha256"))


def advance_fundamentals(ledger, report, *, checkpoint_sha256):
    """Unseen word and reference extraction must be learned before grammar."""
    if ledger.get("stage") != ENGLISH or ledger.get("english_fundamentals_passed"):
        raise ValueError("Foundation exam is not the current unlocked stage")
    if report.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("Foundation exam belongs to different learned weights")
    total, correct = report.get("total", 0), report.get("correct", 0)
    passed = total >= 24 and correct / total >= .80 and report.get("exam_sha256")
    ledger["attempts"]["english_fundamentals"] = ledger["attempts"].get("english_fundamentals", 0) + 1
    ledger["evidence"].append({"stage": "english_fundamentals",
        "checkpoint_sha256":checkpoint_sha256, "passed":bool(passed),
        "report_sha256":sha256(json.dumps(report,sort_keys=True).encode()).hexdigest()})
    ledger["last_checkpoint_sha256"] = checkpoint_sha256
    if passed:
        ledger["english_fundamentals_passed"] = True
    return bool(passed)


def advance(ledger, kind, report, *, checkpoint_sha256):
    """Apply a verified result; failures retain the exact locked stage.

    'read' means all supervised tokens in the particular book were covered
    in source order, with no pending cursor. No lower stage can be skipped.
    """
    if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
        raise ValueError("Missing trained-checkpoint fingerprint")
    stage = ledger["stage"]
    expected = {ENGLISH: "english_exam", READ: "reading_coverage",
                COMPREHEND: "book_exam", SYNTHESIS: "synthesis_exam"}
    if stage not in expected or kind != expected[stage]:
        raise ValueError("Out-of-order stage transition blocked")
    if report.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("Exam does not belong to this trained checkpoint")
    if stage == ENGLISH:
        if ledger.get("english_fundamentals_passed") is not True:
            raise ValueError("Vocabulary/reference foundation exam must pass before English graduation")
        passed = english_pass(report)
    elif stage == READ:
        passed = (report.get("book") == ledger["books"][ledger["book_index"]] and
                  report.get("read_complete") is True and
                  report.get("next_window") == 0 and
                  report.get("covered_targets", 0) > 0 and
                  report.get("last_read_unit") is not None and
                  report.get("ordered_manifest_sha256") == ledger["manifest_sha256"] and
                  report.get("read_units_expected", 0) >= 10 and
                  report.get("read_units_seen") == report.get("read_units_expected") and
                  report.get("source_sha256") == ledger["source_hashes"].get(report["book"], report.get("source_sha256")) )
    elif stage == COMPREHEND:
        passed = (report.get("book") == ledger["books"][ledger["book_index"]] and
                  bool(book_pass(report)) and
                  report.get("source_sha256") == ledger["source_hashes"].get(report["book"], report.get("source_sha256")) )
    else:
        passed = (report.get("passes_logic") is True and
                  report.get("passes_mathematics") is True and
                  report.get("passes_transfer") is True and
                  report.get("human_review_required") is True)
    ledger["attempts"][stage] = ledger["attempts"].get(stage, 0) + 1
    ledger["evidence"].append({
        "stage": stage, "kind": kind, "passed": bool(passed),
        "checkpoint_sha256": checkpoint_sha256,
        "report_sha256": sha256(json.dumps(report, sort_keys=True).encode()).hexdigest(),
        "book_index": ledger["book_index"]})
    ledger["last_checkpoint_sha256"] = checkpoint_sha256
    if not passed:
        return False
    if stage == ENGLISH:
        ledger["stage"] = READ
    elif stage == READ:
        ledger["stage"] = COMPREHEND
    elif stage == COMPREHEND:
        ledger["book_index"] += 1
        ledger["stage"] = READ if ledger["book_index"] < len(ledger["books"]) else SYNTHESIS
    elif stage == SYNTHESIS:
        ledger["stage"] = REVIEW
    return True


def approve(ledger, *, evidence_sha256, checkpoint_sha256):
    if ledger["stage"] != REVIEW:
        raise ValueError("Owner approval before all neural exams is forbidden")
    if ledger["last_checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("Checkpoint changed after exam")
    if evidence_sha256 != sha256(json.dumps(ledger["evidence"], sort_keys=True).encode()).hexdigest():
        raise ValueError("Approval is not bound to the full assessment ledger")
    ledger["stage"] = COMPLETE
    return ledger
