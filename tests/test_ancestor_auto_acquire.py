"""Automatic acquisition launches only justified, inherited training campaigns."""
import hashlib
from datetime import datetime, timezone, timedelta
import subprocess
from pathlib import Path

from scripts.ancestor_auto_acquire import plan, stale_jobless_queues


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


def test_old_unstarted_queue_is_recoverable_but_active_work_is_not():
    now = datetime(2026, 11, 17, 15, 0, tzinfo=timezone.utc)
    old = (now - timedelta(hours=7)).isoformat()
    fresh = (now - timedelta(minutes=15)).isoformat()
    runs = [
        {"databaseId": 1, "headBranch": "main", "status": "queued", "createdAt": old},
        {"databaseId": 2, "headBranch": "main", "status": "in_progress", "createdAt": old},
        {"databaseId": 3, "headBranch": "main", "status": "waiting", "createdAt": old},
        {"databaseId": 4, "headBranch": "main", "status": "queued", "createdAt": fresh},
        {"databaseId": 5, "headBranch": "other", "status": "queued", "createdAt": old},
        {"databaseId": 6, "headBranch": "main", "status": "queued"},
    ]
    assert stale_jobless_queues(runs, now) == [1]


def test_scheduler_wakes_for_english_mastery_merges():
    workflow = (Path(__file__).resolve().parents[1] /
                ".github/workflows/ancestor-auto-acquire.yml").read_text()
    assert "my_own_ai/english_mastery.py" in workflow
    assert "my_own_ai/english_mastery_curriculum.py" in workflow
