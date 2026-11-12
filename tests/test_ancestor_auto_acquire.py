"""Automatic acquisition launches only justified, inherited training campaigns."""
import hashlib
import subprocess
from pathlib import Path

from scripts.ancestor_auto_acquire import plan


M = hashlib.sha256(b"seven-approved-works").hexdigest()


def test_starts_from_real_completed_parent():
    p = plan({"ancestor-g0001-candidate"}, M,
             {"published_manifest_sha256": "prior", "curriculum": "continuation"})
    assert p == {"action": "start", "generation": "g0002", "parent": "g0001"}


def test_does_not_start_same_corpus_twice():
    p = plan({"ancestor-g0001-candidate", "ancestor-g0002-candidate"}, M,
             {"published_manifest_sha256": M, "curriculum": "english_grounded"})
    assert p == {"action": "current", "generation": "g0002"}


def test_resumes_interrupted_progress_without_changing_parent():
    tags = {"ancestor-g0001-candidate", "ancestor-progress-g0002"}
    args = {"published_manifest_sha256": "old", "curriculum": "continuation"}
    assert plan(tags, M, args)["action"] == "resume"
    assert plan(tags, M, args, active=True)["action"] == "running"


def test_new_approved_source_creates_new_generation():
    tags = {"ancestor-g0001-candidate", "ancestor-g0002-candidate"}
    previous = {"published_manifest_sha256": "older", "curriculum": "english_grounded"}
    p = plan(tags, M, previous)
    assert p == {"action": "start", "generation": "g0003", "parent": "g0002"}


def test_no_fabricated_ancestral_weights():
    assert plan(set(), M)["action"] == "missing_parent"


def test_current_workflow_triggers_on_main_and_daily():
    workflow = (Path(__file__).resolve().parents[1] /
                ".github/workflows/ancestor-auto-acquire.yml").read_text()
    assert "branches: [main]" in workflow
    assert "schedule:" in workflow
    assert "scripts/ancestor_auto_acquire.py" in workflow
    assert "actions: write" in workflow


def test_no_automatic_activation_in_scheduler():
    source = (Path(__file__).resolve().parents[1] /
              "scripts/ancestor_auto_acquire.py").read_text()
    assert "ancestor-activate" not in source
    assert "english_grounded" in source
