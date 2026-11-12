"""Approved, pinned publication inputs for the persistent original-weight Core."""
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil

from .book_language import _sections
from .numbered_paragraphs import numbered_paragraphs, paragraph_ledger
from .structured_books import structured_units
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
        expected = row.get("sha256")
        if ident in ids:
            raise ValueError("Duplicate publication ID")
        ids.add(ident)
        shard_rows = row.get("parts")
        if shard_rows is None:
            shard_rows = [{"path": row.get("path"), "sha256": expected}]
        if not isinstance(shard_rows, list) or not shard_rows:
            raise ValueError("Publication requires nonempty source parts")
        recovered = []
        paths = []
        for part in shard_rows:
            if not isinstance(part, dict):
                raise ValueError("Invalid source part")
            relative, part_hash = part.get("path"), part.get("sha256")
            if not isinstance(relative, str) or not relative.startswith("published_books/"):
                raise ValueError("Published sources must be inside published_books/")
            path = (root / relative).resolve()
            if not path.is_relative_to(root / "published_books") or path.suffix.lower() != ".txt":
                raise ValueError("Invalid or escaping publication path")
            if not isinstance(part_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", part_hash):
                raise ValueError("Missing pinned SHA-256 for published source part")
            if not path.is_file() or _digest(path) != part_hash:
                raise ValueError("Published source hash differs from approved edition")
            paths.append(relative)
            recovered.append(path.read_text(encoding="utf-8"))
        joiner = row.get("part_joiner", "\n")
        if joiner not in ("", "\n"):
            raise ValueError("Unsupported source-part join rule")
        txt = joiner.join(recovered)
        aggregate = sha256(txt.encode("utf-8")).hexdigest()
        if expected is not None and expected != aggregate:
            raise ValueError("Combined published work hash mismatch")
        relative = paths[0]
        # For one supplied DOCX, the generic page extraction omitted Word
        # OMML objects. A separately SHA-pinned, source-anchored transcription
        # restores those formulas directly beside the corresponding argument.
        overlay = row.get("math_overlay")
        if overlay is not None:
            overlay_path = overlay.get("path")
            if (not isinstance(overlay_path, str) or
                    not overlay_path.startswith("published_books/library/") or
                    not overlay_path.endswith(".json")):
                raise ValueError("Invalid mathematical overlay path")
            resolved = (root / overlay_path).resolve()
            if not resolved.is_relative_to(root / "published_books"):
                raise ValueError("Mathematical overlay escapes approved source directory")
            if _digest(resolved) != overlay.get("sha256"):
                raise ValueError("Mathematical overlay source hash differs")
            math_source = json.loads(resolved.read_text(encoding="utf-8"))
            if (math_source.get("format") != "omml_text_overlay_v1" or
                    math_source.get("work_id") != ident):
                raise ValueError("Mathematical overlay belongs to another publication")
            for item in math_source["insertions"]:
                anchor, math = item["after"], item["text"]
                expression = r"\\s+".join(re.escape(part) for part in anchor.split())
                matches = list(re.finditer(expression, txt))
                if len(matches) != 1 or not math.strip():
                    raise ValueError("Mathematical insertion anchor is missing or ambiguous")
                idx = matches[0].end()
                txt = txt[:idx] + "\n\n[Linearized source equation]\n" + math + "\n\n" + txt[idx:]
            aggregate = sha256(txt.encode("utf-8")).hexdigest()
        if len(txt.split()) < 1000:
            raise ValueError("Insufficient usable text in approved work")
        if any(near_duplicate(txt, other) for other in texts):
            raise ValueError("Overlapping publication edition detected")
        texts.append(txt)
        chunking = row.get("chunking", "legacy_sections")
        if chunking not in ("legacy_sections", "numbered_paragraph", "structured_units"):
            raise ValueError("Unrecognized approved book chunking method")
        accepted.append({"id": ident, "text": txt, "sha256": aggregate,
                         "relative_path": relative, "source_parts": paths, "chunking": chunking,
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
    reading_ledgers = {}
    grouped_units = {}
    for book in sources:
        if book["chunking"] == "numbered_paragraph":
            units = numbered_paragraphs(book["text"])
            reading_ledgers[book["id"]] = paragraph_ledger(units)
            grouped_units[book["id"]] = [
                (book["id"], u["id"], u["text"]) for u in units]
        elif book["chunking"] == "structured_units":
            units = structured_units(book["text"])
            grouped_units[book["id"]] = [
                (book["id"], u["id"], u["text"]) for u in units]
        else:
            sections = _sections(book["text"], target_words=target_words)
            grouped_units[book["id"]] = [
                (book["id"], "s%06d" % index, text)
                for index, text in enumerate(sections)]
    # Every approved work must participate in TRAINING. A whole-book holdout
    # discards two of seven authored works; keep unit-disjoint per-work splits
    # instead, with immutable source hashes and no overlapping logical units.
    for book in sources:
        units = grouped_units[book["id"]]
        if len(units) < 10:
            raise ValueError("Book too short for independent train/validation/test units")
        n = len(units)
        valid = max(1, round(n * .1))
        test = max(1, round(n * .1))
        parts["train"].extend(units[:n-valid-test])
        parts["validation"].extend(units[n-valid-test:n-test])
        parts["test"].extend(units[n-test:])
    policy = "per_work_unit_disjoint"
    if any(not rows for rows in parts.values()):
        raise ValueError("Corpus needs training, validation and test material")
    # Repeated running headers, citations and page boilerplate are common in
    # PDFs. Assign identical units to their first partition, never silently
    # train on the same text also held out as validation/test.
    seen = {}
    dropped_duplicates = {"train": 0, "validation": 0, "test": 0}
    for split in ("train", "validation", "test"):
        unique = []
        for row in parts[split]:
            fingerprint = sha256(" ".join(row[2].casefold().split()).encode()).hexdigest()
            if fingerprint in seen:
                dropped_duplicates[split] += 1
                continue
            seen[fingerprint] = split
            unique.append(row)
        parts[split] = unique
    if any(not parts[split] for split in parts):
        raise ValueError("Duplicate elimination exhausted an evaluation partition")
    dest.mkdir(parents=True)
    manifest = {
        "format": "approved_books_v1", "corpus_id": parsed["corpus_id"],
        "policy": policy, "published_manifest_sha256": parsed["manifest_sha256"],
        "sources": [{"id": book["id"], "source_sha256": book["sha256"],
                     "edition": book["edition"]} for book in sources],
        "sections": {k: len(v) for k,v in parts.items()},
        "chunk_unit": ("mixed_structured" if any(b["chunking"]=="structured_units" for b in sources)
                       else "numbered_paragraph" if reading_ledgers else "legacy_sections"),
        "chunking_by_work": {b["id"]: b["chunking"] for b in sources},
        "reading_ledgers": reading_ledgers,
        "duplicate_source_units_excluded": dropped_duplicates,
    }
    try:
        for split, rows in parts.items():
            folder = dest / split
            folder.mkdir()
            for name, idx, text in rows:
                (folder / f"{name}_{idx}.txt").write_text(text, encoding="utf-8")
        (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                            encoding="utf-8")
    except BaseException:
        shutil.rmtree(dest)
        raise
    return manifest
