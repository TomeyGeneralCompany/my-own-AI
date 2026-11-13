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


def main():
    """Start/resume the *staged* controller; never call the step-count candidate path."""
    tags = {row["tagName"] for row in json.loads(
        gh("release", "list", "--limit", "1000", "--json", "tagName"))}
    latest = max(((generation_number(match.group(1)), match.group(1))
                  for tag in tags if (match := TAG.fullmatch(tag))), default=None)
    if latest is None:
        raise RuntimeError("A trained ancestral checkpoint is required; cannot fabricate weights")
    if "ancestor-stage-review" in tags:
        print("Locked reasoning examination awaits owner review; no auto-promotion")
        return 0
    runs = json.loads(gh("run", "list", "--workflow", "ancestor-staged.yml",
                         "--limit", "30", "--json", "status,headBranch"))
    active = any(run["status"] in ("queued", "in_progress", "waiting", "requested", "pending")
                 and run.get("headBranch") == "main" for run in runs)
    if active:
        print("Competency-locked training already active")
        return 0
    gh("workflow", "run", "ancestor-staged.yml", "--ref", "main",
       "-f", f"parent_generation={latest[1]}")
    print("Dispatched competency-locked original neural training from", latest[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
