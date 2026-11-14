"""Bounded storage for staged neural checkpoints; no training or stage logic."""
import argparse
import json
import re
import subprocess

CAPACITY = 1000
HIGH_WATER = 990
TARGET = 980
NAME = re.compile(r"stage-state-[0-9]+-[0-9]+\.zip\Z")


def api(path, *, delete=False):
    cmd = ["gh", "api", path]
    if delete:
        cmd.extend(["--method", "DELETE"])
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return None if delete else json.loads(result.stdout)


def inventory(repo, tag, request=api):
    release = request(f"repos/{repo}/releases/tags/{tag}")
    release_id = int(release["id"])
    assets = []
    page = 1
    while True:
        batch = request(f"repos/{repo}/releases/{release_id}/assets?per_page=100&page={page}")
        if not isinstance(batch, list):
            raise ValueError("Invalid checkpoint inventory")
        assets.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    ids = [int(asset["id"]) for asset in assets]
    if len(set(ids)) != len(ids):
        raise ValueError("Inconsistent release inventory; refusing to rotate")
    return release_id, assets


def uploaded_checkpoints(assets):
    return sorted((a for a in assets if a.get("state") == "uploaded"
                   and a.get("size", 0) > 0 and NAME.fullmatch(a.get("name", ""))),
                  key=lambda a: (a["created_at"], int(a["id"])))


def latest(repo, tag, request=api):
    _, assets = inventory(repo, tag, request)
    candidates = uploaded_checkpoints(assets)
    if not candidates:
        raise ValueError("Existing checkpoint release contains no usable checkpoints")
    return candidates[-1]["name"]


def rotate(repo, tag, request=api):
    """Make room only at high-water mark; retain newest recovery snapshots.

    Never delete a non-checkpoint asset or the newest successful checkpoint.
    If insufficient eligible old snapshots exist, fail before deleting any.
    The caller has already restored/trained from an integrity-verified state.
    """
    _, assets = inventory(repo, tag, request)
    if len(assets) < HIGH_WATER:
        return 0
    candidates = uploaded_checkpoints(assets)
    count = len(assets) - TARGET
    if count > len(candidates) - 2:
        raise ValueError("Insufficient old checkpoints to rotate safely")
    for asset in candidates[:count]:
        request(f"repos/{repo}/releases/assets/{asset['id']}", delete=True)
    # Confirm sufficient headroom, including any unrelated release assets.
    _, remaining = inventory(repo, tag, request)
    if len(remaining) >= CAPACITY:
        raise ValueError("Checkpoint release remains full after rotation")
    if latest_name(remaining) != candidates[-1]["name"]:
        raise ValueError("Newest recovery checkpoint unexpectedly changed")
    return count


def latest_name(assets):
    candidates = uploaded_checkpoints(assets)
    if not candidates:
        raise ValueError("No retained checkpoint")
    return candidates[-1]["name"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("latest", "rotate"))
    parser.add_argument("--repo", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    if args.operation == "latest":
        print(latest(args.repo, args.tag))
    else:
        print(f"Rotated {rotate(args.repo, args.tag)} oldest checkpoint snapshots")


if __name__ == "__main__":
    main()
