"""Safe, versioned transport of original-weight model state between Actions runs.

GitHub Release assets transport packages; Python owns validation, provenance,
and checkpoint selection. A release name is NOT a checkpoint digest.
"""
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile
import zipfile

FILES = ("tokenizer.json", "reading.pt", "reading.best.pt",
         "assistant.pt", "assistant.best.pt", "generation.json")
MAX_MEMBER = 300_000_000
FORMAT = "ancestor_checkpoint_bundle_v1"


def digest_bytes(payload):
    return sha256(payload).hexdigest()


def digest_file(path):
    h = sha256()
    with open(path, "rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_generation(generation):
    if not isinstance(generation, str) or not re.fullmatch(r"g[0-9]{4}", generation):
        raise ValueError("Generation must be g0001-style identifier")
    return generation


def package_generation(workspace, destination):
    root = Path(workspace)
    meta_file = root / "generation.json"
    if not meta_file.is_file():
        raise ValueError("Generation metadata is required before checkpoint export")
    record = json.loads(meta_file.read_text(encoding="utf-8"))
    validate_generation(record.get("generation"))
    if record.get("format") != "ancestor_generation_v1":
        raise ValueError("Invalid generation metadata")
    if record.get("phase") not in ("reading", "instruction", "candidate"):
        raise ValueError("Invalid training phase")
    required = {"tokenizer.json", "reading.pt", "reading.best.pt", "generation.json"}
    if record["phase"] == "candidate":
        required.update(("assistant.pt", "assistant.best.pt"))
    if not all((root / name).is_file() for name in required):
        raise ValueError("Missing required learned weights or tokenizer")
    names = [name for name in FILES if (root / name).is_file()]
    manifest = {"format": FORMAT, "generation": record["generation"],
                "phase": record["phase"],
                "members": {name: {"sha256": digest_file(root / name),
                                   "bytes": (root / name).stat().st_size} for name in names}}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".ancestor-", suffix=".zip",
                                dir=destination.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6) as archive:
            for name in sorted(names):
                archive.write(root / name, arcname=name)
            archive.writestr("package-manifest.json", json.dumps(manifest, sort_keys=True))
        os.replace(temp, destination)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return {"bundle": str(destination), "bundle_sha256": digest_file(destination),
            "generation": record["generation"], "phase": record["phase"],
            "members": sorted(names)}


def unpack_generation(archive_path, workspace, *, expected_generation=None,
                      expected_archive_sha256=None):
    archive_path, root = Path(archive_path), Path(workspace)
    if expected_archive_sha256 is not None and digest_file(archive_path) != expected_archive_sha256:
        raise ValueError("Downloaded checkpoint package SHA-256 mismatch")
    if root.exists() and any(root.iterdir()):
        raise ValueError("Checkpoint unpack destination must be empty")
    with zipfile.ZipFile(archive_path) as archive:
        filenames = archive.namelist()
        if len(filenames) != len(set(filenames)):
            raise ValueError("Duplicate archive member name")
        if any(name not in (*FILES, "package-manifest.json") for name in filenames):
            raise ValueError("Unexpected checkpoint archive member")
        if "package-manifest.json" not in filenames:
            raise ValueError("Missing integrity manifest")
        for info in archive.infolist():
            if info.file_size > MAX_MEMBER:
                raise ValueError("Checkpoint archive member exceeds size limit")
        manifest = json.loads(archive.read("package-manifest.json"))
        if manifest.get("format") != FORMAT:
            raise ValueError("Invalid package manifest")
        validate_generation(manifest.get("generation"))
        if expected_generation is not None and manifest["generation"] != expected_generation:
            raise ValueError("Checkpoint belongs to another generation")
        names = set(manifest.get("members", {}))
        if names != set(filenames) - {"package-manifest.json"}:
            raise ValueError("Checkpoint archive manifest/member disagreement")
        payloads = {}
        for name in names:
            raw = archive.read(name)
            entry = manifest["members"][name]
            if len(raw) != entry.get("bytes") or digest_bytes(raw) != entry.get("sha256"):
                raise ValueError("Corrupt checkpoint member: " + name)
            payloads[name] = raw
        if "generation.json" not in payloads or "tokenizer.json" not in payloads:
            raise ValueError("Incomplete checkpoint bundle")
        meta = json.loads(payloads["generation.json"])
        if (meta.get("generation") != manifest["generation"]
                or meta.get("phase") != manifest["phase"]):
            raise ValueError("Generation metadata does not match package")
    root.mkdir(parents=True, exist_ok=True)
    try:
        for name, raw in payloads.items():
            (root / name).write_bytes(raw)
    except BaseException:
        for name in payloads:
            (root / name).unlink(missing_ok=True)
        raise
    return {"generation": manifest["generation"], "phase": manifest["phase"],
            "archive_sha256": digest_file(archive_path)}


def load_active_pointer(body):
    try:
        payload = json.loads(body)
    except (TypeError, ValueError) as exc:
        raise ValueError("No valid active-Core release pointer") from exc
    if payload.get("format") != "ancestor_active_v1":
        raise ValueError("Unrecognized active-Core pointer")
    validate_generation(payload.get("generation"))
    tag = payload.get("candidate_tag")
    if tag != "ancestor-" + payload["generation"] + "-candidate":
        raise ValueError("Active release tag does not match generation")
    if not re.fullmatch(r"[a-f0-9]{64}", str(payload.get("bundle_sha256"))):
        raise ValueError("Active checkpoint lacks verified SHA-256")
    if payload.get("status") not in ("experimental", "evaluated"):
        raise ValueError("Invalid checkpoint activation status")
    return payload
