"""Hard fail-closed curriculum transitions independent of training counters."""
from hashlib import sha256
import json
import pytest

from my_own_ai.stage_controller import (
    initial_ledger, verify_ledger, advance, approve, english_pass, book_pass,
    ENGLISH, READ, COMPREHEND, SYNTHESIS, REVIEW, COMPLETE)
from scripts.ancestor_staged import package, restore


SHA = "a" * 64
MANIFEST = "b" * 64


def english_report(correct=104, paired=7):
    return {"evaluated": 200, "automatically_scored": 130,
            "exact_correct": correct, "checkpoint_sha256": SHA,
            "category_scores": {name: {"auto_scored": count, "exact": count}
                                for name, count in (("syntax",40), ("reading",40),
                                                    ("qa",30), ("conditional",20))},
            "raw_results": {"conditional_pairs_passed": paired}}


def book_report(book="one", correct=8):
    return {"book":book, "total":10, "correct":correct, "source_sha256":SHA,
            "questions_sha256":SHA, "checkpoint_sha256":SHA}


def coverage(book="one", done=True):
    return {"book":book, "read_complete":done,
            "next_window":0 if done else 1, "covered_targets":100 if done else 0,
            "last_read_unit":"source-unit-42", "ordered_manifest_sha256":MANIFEST,
            "read_units_expected":42, "read_units_seen":42 if done else 2,
            "source_sha256":SHA,
            "checkpoint_sha256":SHA}


def test_failed_english_cannot_unlock_anything():
    state = initial_ledger(MANIFEST, ["one", "two"], "g0002")
    assert not advance(state, "english_exam", english_report(correct=0), checkpoint_sha256=SHA)
    assert state["stage"] == ENGLISH and state["book_index"] == 0
    assert not advance(state, "english_exam", english_report(correct=130, paired=0), checkpoint_sha256=SHA)
    assert state["stage"] == ENGLISH


def test_no_book_skips_and_no_step_count_promotions():
    state = initial_ledger(MANIFEST, ["one", "two"], "g0002")
    with pytest.raises(ValueError, match="Out-of-order"):
        advance(state, "reading_coverage", coverage(), checkpoint_sha256=SHA)
    assert advance(state, "english_exam", english_report(), checkpoint_sha256=SHA)
    assert state["stage"] == READ
    assert not advance(state, "reading_coverage", coverage(done=False), checkpoint_sha256=SHA)
    assert state["stage"] == READ
    assert advance(state, "reading_coverage", coverage(), checkpoint_sha256=SHA)
    assert state["stage"] == COMPREHEND
    assert not advance(state, "book_exam", book_report(correct=0), checkpoint_sha256=SHA)
    assert state["book_index"] == 0
    assert advance(state, "book_exam", book_report(), checkpoint_sha256=SHA)
    assert state["stage"] == READ and state["book_index"] == 1
    assert advance(state, "reading_coverage", coverage("two"), checkpoint_sha256=SHA)
    assert advance(state, "book_exam", book_report("two"), checkpoint_sha256=SHA)
    assert state["stage"] == SYNTHESIS and state["book_index"] == 2
    assert len(state["evidence"]) == 7


def test_reject_changed_corpus_order_or_checkpoint():
    state = initial_ledger(MANIFEST, ["one","two"], "g0002")
    with pytest.raises(ValueError, match="Changed corpus"):
        verify_ledger(state, "d" * 64, ["one","two"])
    with pytest.raises(ValueError, match="Changed corpus"):
        verify_ledger(state, MANIFEST, ["two","one"])
    with pytest.raises(ValueError, match="checkpoint"):
        advance(state, "english_exam", english_report(),
                checkpoint_sha256="c"*64)


def test_owner_cannot_approve_before_synthesis():
    state = initial_ledger(MANIFEST, ["one"], "g0002")
    with pytest.raises(ValueError, match="before"):
        approve(state, evidence_sha256=SHA, checkpoint_sha256=SHA)


def test_synthesis_still_requires_independent_verified_result():
    state = initial_ledger(MANIFEST, ["one"], "g0002")
    advance(state, "english_exam", english_report(), checkpoint_sha256=SHA)
    advance(state, "reading_coverage", coverage(), checkpoint_sha256=SHA)
    advance(state, "book_exam", book_report(), checkpoint_sha256=SHA)
    report = {"checkpoint_sha256":SHA, "passes_logic":False,
              "passes_mathematics":False, "passes_transfer":False,
              "human_review_required":True}
    assert not advance(state, "synthesis_exam", report, checkpoint_sha256=SHA)
    assert state["stage"] == SYNTHESIS
    with pytest.raises(ValueError):
        approve(state, evidence_sha256=SHA, checkpoint_sha256=SHA)


def test_safe_checkpoint_bundle_roundtrip(tmp_path):
    source = tmp_path/"source"
    source.mkdir()
    (source/"stage-ledger.json").write_text("{}")
    (source/"tokenizer.json").write_text("{}")
    (source/"english.pt").write_bytes(b"real-neural-checkpoint")
    archive = tmp_path/"state.zip"
    package(source,archive)
    dest = tmp_path/"restored"
    restore(archive,dest)
    assert (dest/"english.pt").read_bytes() == b"real-neural-checkpoint"
    with pytest.raises(ValueError):
        restore(archive,dest)


def test_legacy_workflow_cannot_certify_new_generations():
    from pathlib import Path
    script = Path(__file__).resolve().parents[1]/".github/workflows/ancestor-train.yml"
    assert 'int(os.environ["GENERATION"][1:]) <= 2' in script.read_text()
    auto = (script.parent/"ancestor-auto-acquire.yml").read_text()
    assert "ancestor-staged.yml" in auto
