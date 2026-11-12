"""Local, append-only intellectual inheritance ledger with explicit owner approval.

Knowledge is not automatically promoted to training. Records bind an immutable
source digest, provenance and owner approval; verified research can be stored
as evidence without being treated as a correct training label.
"""
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from .module_memory import IDENTIFIER

FORMAT = "intellectual_ledger_v1"
KINDS = {"authored_work", "derived_work", "reasoning"}
STATUSES = {"pending", "approved", "rejected"}
FIELDS = {"id", "kind", "source", "sha256", "origin", "status", "evidence"}


def _digest(path):
    hash_value = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            hash_value.update(chunk)
    return hash_value.hexdigest()


def _read(path):
    path = Path(path)
    if not path.exists():
        return {"format": FORMAT, "records": []}
    with path.open(encoding="utf-8") as stream:
        state = json.load(stream)
    if not isinstance(state, dict) or set(state) != {"format", "records"} or state["format"] != FORMAT:
        raise ValueError("Invalid intellectual ledger")
    if not isinstance(state["records"], list) or len(state["records"]) > 10000:
        raise ValueError("Invalid intellectual ledger records")
    seen = set()
    for row in state["records"]:
        if not isinstance(row, dict) or set(row) != FIELDS:
            raise ValueError("Invalid intellectual entry schema")
        if (not isinstance(row["id"], str) or not IDENTIFIER.fullmatch(row["id"])
                or row["id"] in seen or row["kind"] not in KINDS or row["status"] not in STATUSES
                or not isinstance(row["source"], str) or not row["source"]
                or not isinstance(row["origin"], str) or not row["origin"]
                or not isinstance(row["sha256"], str) or len(row["sha256"]) != 64
                or any(ch not in "0123456789abcdef" for ch in row["sha256"])
                or not isinstance(row["evidence"], list)
                or any(not isinstance(item, str) or not item for item in row["evidence"])):
            raise ValueError("Invalid intellectual entry")
        seen.add(row["id"])
    return state


def _save(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(state, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def register(ledger, identifier, kind, source, *, origin, evidence=None):
    """Register an immutable source snapshot. Registration never grants approval."""
    if not isinstance(identifier, str) or not IDENTIFIER.fullmatch(identifier):
        raise ValueError("Invalid intellectual work identifier")
    if kind not in KINDS or not isinstance(origin, str) or not origin.strip():
        raise ValueError("Invalid intellectual provenance")
    if evidence is None:
        evidence = []
    if not isinstance(evidence, list) or any(not isinstance(x, str) or not x.strip() for x in evidence):
        raise ValueError("Invalid evidence references")
    path = Path(source).resolve()
    if not path.is_file() or path.resolve() == Path(ledger).resolve():
        raise ValueError("Source must be an existing separate file")
    state = _read(ledger)
    if any(row["id"] == identifier for row in state["records"]):
        raise ValueError("Intellectual entry already exists")
    record = {"id": identifier, "kind": kind, "source": str(path), "sha256": _digest(path),
              "origin": origin, "status": "pending", "evidence": evidence}
    state["records"].append(record)
    _save(ledger, state)
    return record


def decide(ledger, identifier, decision):
    """Explicit owner action; never auto-approve AI-drafted or retrieved text."""
    if decision not in ("approved", "rejected"):
        raise ValueError("An explicit approval or rejection is required")
    state = _read(ledger)
    rows = [row for row in state["records"] if row["id"] == identifier]
    if len(rows) != 1 or rows[0]["status"] != "pending":
        raise ValueError("Only a pending registered entry can be decided")
    row = rows[0]
    if _digest(row["source"]) != row["sha256"]:
        raise ValueError("Source has changed since registration")
    row["status"] = decision
    _save(ledger, state)
    return row


def approved_sources(ledger, *, kind=None):
    state = _read(ledger)
    if kind is not None and kind not in KINDS:
        raise ValueError("Unknown work kind")
    approved = [row for row in state["records"]
                if row["status"] == "approved" and (kind is None or row["kind"] == kind)]
    for row in approved:
        if _digest(row["source"]) != row["sha256"]:
            raise ValueError(f"Approved source changed: {row['id']}")
    return approved


def export_manifest(ledger, destination, *, kind=None):
    """Export approved provenance only; don't put private text into Git."""
    approved = approved_sources(ledger, kind=kind)
    payload = {"format": FORMAT, "approved": approved}
    if Path(destination).resolve() == Path(ledger).resolve():
        raise ValueError("Manifest cannot overwrite ledger")
    _save(destination, payload)
    return {"count": len(approved), "manifest": str(Path(destination))}


def stage_approved_corpus(ledger, destination):
    """Stage only approved manuscript files in a fresh local training directory."""
    import shutil
    from tempfile import TemporaryDirectory

    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("Corpus destination must be a new directory; existing material is never overwritten")
    works = [row for row in approved_sources(ledger)
             if row["kind"] in ("authored_work", "derived_work")]
    if not works:
        raise ValueError("No approved manuscript sources")
    allowed = {".txt", ".md", ".pdf", ".docx"}
    if any(Path(row["source"]).suffix.casefold() not in allowed for row in works):
        raise ValueError("Approved manuscripts require TXT, MD, text-PDF or DOCX format")
    for row in works:
        source = Path(row["source"]).resolve()
        if destination == source or destination in source.parents:
            raise ValueError("Corpus destination must not be inside an approved source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".approved-corpus-", dir=destination.parent) as tmp:
        staged = Path(tmp) / "corpus"
        staged.mkdir()
        manifests = []
        for row in works:
            source = Path(row["source"])
            target = staged / (row["id"] + source.suffix.lower())
            shutil.copyfile(source, target)
            if _digest(target) != row["sha256"]:
                raise ValueError("Approved manuscript changed while staging")
            manifests.append({"id": row["id"], "origin": row["origin"],
                              "sha256": row["sha256"], "file": target.name})
        with (staged / "source-manifest.json").open("w", encoding="utf-8") as stream:
            json.dump({"format": FORMAT, "sources": manifests}, stream, indent=2)
        os.replace(staged, destination)
    return {"documents": len(works), "corpus": str(destination)}
