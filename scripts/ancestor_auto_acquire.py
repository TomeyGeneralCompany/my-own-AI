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
    manifest = hashlib.sha256(Path("published_books/manifest.json").read_bytes()).hexdigest()
    tags = {row["tagName"] for row in json.loads(gh("release", "list", "--limit", "1000", "--json", "tagName"))}
    latest = max((generation_number(match.group(1)), match.group(1))
                 for tag in tags if (match := TAG.fullmatch(tag)), default=None)
    metadata = candidate_metadata(latest[1]) if latest else None
    # The training workflow itself automatically hands off sessions. Only
    # restart interrupted campaigns when no job is active.
    runs = json.loads(gh("run", "list", "--workflow", "ancestor-train.yml",
                         "--limit", "30", "--json", "status,headBranch"))
    active = any(run["status"] in ("queued", "in_progress", "waiting", "requested", "pending")
                 and run.get("headBranch") == "main" for run in runs)
    decision = plan(tags, manifest, metadata, active)
    print(json.dumps(decision, sort_keys=True))
    if decision["action"] in ("start", "resume"):
        gh("workflow", "run", "ancestor-train.yml", "--ref", "main",
           "-f", f"generation={decision['generation']}",
           "-f", f"parent_generation={decision['parent']}",
           "-f", f"curriculum={CURRICULUM}",
           "-f", f"reading_passes={PASSES}",
           "-f", f"instruction_steps={STEPS}",
           "-f", f"session_seconds={SESSION}",
           "-f", "auto_continue=true")
        print("Dispatched original-weight training", decision["generation"])
    elif decision["action"] == "missing_parent":
        raise RuntimeError("No g0001 candidate exists; recover trained parent before scheduling")
    return 0


if __name__ == "__main__":
    sys.exit(main())
