"""Approved, pinned publication inputs for the persistent original-weight Core."""
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil

from .book_language import _sections
from .language_corpus import near_duplicate


def _digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def validate_published_manifest(repository_root, manifest_path):
    root = Path(repository_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_relative_to(root):
        raise ValueError("Publication manifest must belong to repository root")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (data.get("format") != "ancestor_published_corpus_v1"
            or not isinstance(data.get("works"), list) or not data["works"]):
        raise ValueError("Unrecognized or empty approved publication manifest")
    accepted, ids, texts = [], set(), []
    for row in data["works"]:
        if not isinstance(row, dict) or not row.get("approved_for_training"):
            raise ValueError("Every corpus entry must be explicitly approved")
        ident = row.get("id")
        relative = row.get("path")
        expected = row.get("sha256")
        if not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", ident):
            raise ValueError("Invalid published work identifier")
        if ident in ids:
            raise ValueError("Duplicate publication ID")
        ids.add(ident)
        if not isinstance(relative, str) or not relative.startswith("published_books/"):
            raise ValueError("Published sources must be inside published_books/")
        path = (root / relative).resolve()
        if not path.is_relative_to(root / "published_books") or path.suffix.lower() != ".txt":
            raise ValueError("Invalid or escaping publication path")
        if not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
            raise ValueError("Missing pinned SHA-256 for published source")
        if not path.is_file() or _digest(path) != expected:
            raise ValueError("Published source hash differs from approved edition")
        txt = path.read_text(encoding="utf-8")
        if len(txt.split()) < 1000:
            raise ValueError("Insufficient usable text in approved work")
        if any(near_duplicate(txt, other) for other in texts):
            raise ValueError("Overlapping publication edition detected")
        texts.append(txt)
        accepted.append({"id": ident, "text": txt, "sha256": expected,
                         "relative_path": relative,
                         "edition": row.get("edition", "")})
    return {"corpus_id": data.get("corpus_id"),
            "manifest_sha256": _digest(manifest_path),
            "sources": accepted}


def prepare_published_corpus(repository_root, manifest_path, destination,
                            *, target_words=512):
    """Deterministic, disjoint splits; never ingest test words into training.

    PR #16 reassembles adjacent *training* sections for progressive reading.
    If there are at least three books, entire books are reserved for validation
    and test; otherwise use chronological section holdouts, labeled as such.
    """
    if type(target_words) is not int or not 128 <= target_words <= 4096:
        raise ValueError("Invalid section target")
    parsed = validate_published_manifest(repository_root, manifest_path)
    dest = Path(destination).resolve()
    root = Path(repository_root).resolve()
    if dest.exists() or dest.is_relative_to(root / "published_books"):
        raise ValueError("Prepared destination must be new and separate from the source")
    sources = parsed["sources"]
    parts = {"train": [], "validation": [], "test": []}
    if len(sources) >= 3:
        selected = sorted(sources, key=lambda row: row["sha256"])
        grouping = {"train": selected[:-2], "validation": [selected[-2]],
                    "test": [selected[-1]]}
        for split, books in grouping.items():
            for book in books:
                for index, section in enumerate(_sections(book["text"], target_words=target_words)):
                    parts[split].append((book["id"], index, section))
        policy = "document_disjoint"
    else:
        for book in sources:
            sections = _sections(book["text"], target_words=target_words)
            if len(sections) < 10:
                raise ValueError("Book too short for independent held-out sections")
            n = len(sections)
            val = max(1, round(n * .1))
            test = max(1, round(n * .1))
            for idx, section in enumerate(sections):
                split = "train" if idx < n-val-test else (
                    "validation" if idx < n-test else "test")
                parts[split].append((book["id"], idx, section))
        policy = "section_disjoint_same_book"
    if any(not rows for rows in parts.values()):
        raise ValueError("Corpus needs training, validation and test material")
    seen = {}
    for split, rows in parts.items():
        for name, idx, text in rows:
            fingerprint = sha256(" ".join(text.casefold().split()).encode()).hexdigest()
            if fingerprint in seen and seen[fingerprint] != split:
                raise ValueError("Near-identical section across training/test boundary")
            seen[fingerprint] = split
    dest.mkdir(parents=True)
    manifest = {
        "format": "approved_books_v1", "corpus_id": parsed["corpus_id"],
        "policy": policy, "published_manifest_sha256": parsed["manifest_sha256"],
        "sources": [{"id": book["id"], "source_sha256": book["sha256"],
                     "edition": book["edition"]} for book in sources],
        "sections": {k: len(v) for k,v in parts.items()},
    }
    try:
        for split, rows in parts.items():
            folder = dest / split
            folder.mkdir()
            for name, idx, text in rows:
                (folder / f"{name}_{idx:05}.txt").write_text(text + "\n", encoding="utf-8")
        (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                            encoding="utf-8")
    except BaseException:
        shutil.rmtree(dest)
        raise
    return manifest
