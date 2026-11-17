"""Start/resume original-model training without manual workflow dispatch.

Every campaign retains all approved author sources, and keeps the owner's
source-grounded reasoning curriculum. No activation is performed.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile

TAG = re.compile(r"^ancestor-(g[0-9]{4})-candidate$")
PROGRESS = re.compile(r"^ancestor-progress-(g[0-9]{4})$")
CURRICULUM = "english_grounded"
PASSES = 5
STEPS = 6000
SESSION = 2100


def generation_number(generation):
    if not re.fullmatch(r"g[0-9]{4}", generation):
        raise ValueError("Invalid generation")
    return int(generation[1:])


def plan(tags, current_manifest, latest_metadata=None, active=False):
    """Pure scheduler decision; source identity is never overridden."""
    candidates = sorted((generation_number(match.group(1)), match.group(1))
                        for tag in tags if (match := TAG.fullmatch(tag)))
    if not candidates:
        return {"action": "missing_parent", "reason": "No completed ancestral candidate (g0001 required)"}
    latest_num, parent = candidates[-1]
    if latest_metadata and latest_metadata.get("published_manifest_sha256") == current_manifest and latest_metadata.get("curriculum") == CURRICULUM:
        return {"action": "current", "generation": parent}
    generation = f"g{latest_num + 1:04d}"
    if any(PROGRESS.fullmatch(tag) and tag == "ancestor-progress-" + generation for tag in tags):
        return {"action": "resume" if not active else "running", "generation": generation, "parent": parent}
    if active:
        return {"action": "running", "generation": generation, "parent": parent}
    return {"action": "start", "generation": generation, "parent": parent}


def gh(*args):
    result = subprocess.run(["gh", *args], capture_output=True, text=True, check=True)
    return result.stdout


def candidate_metadata(generation):
    with tempfile.TemporaryDirectory() as work:
        gh("release", "download", f"ancestor-{generation}-candidate",
           "--pattern", "core-generation.zip", "--dir", work)
        with zipfile.ZipFile(Path(work) / "core-generation.zip") as archive:
            if "generation.json" not in archive.namelist():
                raise ValueError("Candidate has no generation metadata")
            meta = json.loads(archive.read("generation.json"))
        if meta.get("generation") != generation or meta.get("phase") != "candidate":
            raise ValueError("Candidate release lineage is invalid")
        return meta


def stale_jobless_queues(runs, now=None):
    """Only old unstarted queues are candidates; completed/active jobs are safe."""
    from datetime import datetime, timezone, timedelta
    if now is None:
        now = datetime.now(timezone.utc)
    stale = []
    for run in runs:
        if run.get("headBranch") != "main" or run.get("status") != "queued":
            continue
        created = run.get("createdAt")
        if not created or not run.get("databaseId"):
            continue  # Missing metadata must not authorize cancellation.
        timestamp = datetime.fromisoformat(created.replace("Z", "+00:00"))
        if now - timestamp > timedelta(hours=6):
            stale.append(run["databaseId"])
    return stale


def main():
    """Start/resume the *staged* controller; never call the step-count candidate path."""
    tags = {row["tagName"] for row in json.loads(
        gh("release", "list", "--limit", "1000", "--json", "tagName"))}
    latest = max(((generation_number(match.group(1)), match.group(1))
                  for tag in tags if (match := TAG.fullmatch(tag))), default=None)
    if latest is None:
        raise RuntimeError("A trained ancestral checkpoint is required; cannot fabricate weights")
    corpus_digest = hashlib.sha256(Path("published_books/manifest.json").read_bytes()).hexdigest()[:12]
    if "ancestor-stage-review-" + corpus_digest in tags:
        print("Locked reasoning examination awaits owner review; no auto-promotion")
        return 0
    runs = json.loads(gh("run", "list", "--workflow", "ancestor-staged.yml",
                         "--limit", "30", "--json", "databaseId,status,headBranch,createdAt"))
    # A run can remain queued without ever obtaining a runner. The former
    # "queued means training" test permanently masked that failure.
    # Reconcile only old, jobless QUEUED runs. Never cancel a running job,
    # an approval wait, or anything with evidence of runner execution.
    stale = stale_jobless_queues(runs)
    for run_id in stale:
        detail = json.loads(gh("run", "view", str(run_id), "--json", "status,jobs"))
        if detail["status"] != "queued" or detail.get("jobs"):
            print("Queue changed; preserving run", run_id)
            continue
        print("Cancelling stale jobless queue", run_id, flush=True)
        gh("run", "cancel", str(run_id))
        # Do not dispatch replacement until GitHub confirms terminal state.
        import time
        for attempt in range(12):
            status = json.loads(gh("run", "view", str(run_id), "--json", "status"))["status"]
            if status in ("cancelled", "completed"):
                break
            time.sleep(5)
        else:
            raise RuntimeError(f"Stale queued run {run_id} did not cancel; refusing duplicate dispatch")
    runs = json.loads(gh("run", "list", "--workflow", "ancestor-staged.yml",
                         "--limit", "30", "--json", "databaseId,status,headBranch"))
    active = [run for run in runs if run.get("headBranch") == "main"
              and run["status"] in ("queued", "in_progress", "waiting", "requested", "pending")]
    if active:
        print("Competency-locked learning blocked or active:", json.dumps(active))
        return 0
    gh("workflow", "run", "ancestor-staged.yml", "--ref", "main",
       "-f", f"parent_generation={latest[1]}")
    print("Dispatched competency-locked original neural training from", latest[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
