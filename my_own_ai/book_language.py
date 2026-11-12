"""Private, approved-book English corpus preparation.

The output is local-only. For one book, partitions are contiguous, disjoint
sections of paragraphs; a section split is NOT an unseen-document test.
All source text remains in local training directories and is never copied to
GitHub by this module.
"""
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

from .corpus import extract_text
from .intellectual_ledger import approved_sources
from .language_corpus import near_duplicate


def _words(text):
    return re.findall(r"\S+", text)


def _split_long_paragraph(paragraph, maximum):
    words = _words(paragraph)
    if len(words) <= maximum:
        return [paragraph]
    return [" ".join(words[start:start + maximum])
            for start in range(0, len(words), maximum)]


def _paragraphs(text, maximum_words):
    # Preserve sentence structure and paragraph breaks; don't train on
    # entire page headers as repeated independent "language samples".
    items = re.split(r"\n\s*\n+", text.replace("\r\n", "\n").replace("\r", "\n"))
    if len(items) < 3:
        items = text.splitlines()
    for item in items:
        item = " ".join(item.split())
        if not item:
            continue
        yield from _split_long_paragraph(item, maximum_words)


def _sections(text, *, target_words=512):
    paragraphs = list(_paragraphs(text, target_words))
    sections, pending, count = [], [], 0
    for paragraph in paragraphs:
        length = len(_words(paragraph))
        if pending and count + length > target_words:
            sections.append("\n\n".join(pending))
            pending, count = [], 0
        pending.append(paragraph)
        count += length
    if pending:
        if sections and count < target_words // 3:
            sections[-1] += "\n\n" + "\n\n".join(pending)
        else:
            sections.append("\n\n".join(pending))
    return sections


def _fingerprint(text):
    return sha256(text.encode("utf-8")).hexdigest()


def _write_json(path, payload):
    path = Path(path)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                            delete=False) as handle:
        tmp = Path(handle.name)
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def prepare_approved_books(ledger, destination, *, target_words=512,
                           min_total_words=2000, validation_ratio=.1,
                           test_ratio=.1):
    """Only the owner's unchanged, approved authored/derived works are admitted.

    Last held-out sections are reserved rather than using training-time
    validation questions as targets. A book-level split is preferred if at
    least three independent books are provided; otherwise the same-book
    section split is explicitly identified as a weaker measure.
    """
    if type(target_words) is not int or not 128 <= target_words <= 4096:
        raise ValueError("Target section size must be 128..4096 words")
    if type(min_total_words) is not int or min_total_words < 100:
        raise ValueError("Minimum word count must be at least 100")
    if not (0 < validation_ratio < .3 and 0 < test_ratio < .3 and
            validation_ratio + test_ratio < .5):
        raise ValueError("Invalid book partition ratios")
    output = Path(destination).resolve()
    if output.exists():
        raise ValueError("Output must be a new private directory")
    entries = [r for r in approved_sources(ledger)
               if r["kind"] in ("authored_work", "derived_work")]
    if not entries:
        raise ValueError("No approved authored or derived books in ledger")
    for entry in entries:
        source = Path(entry["source"]).resolve()
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("Output must not overlap approved source files")
    documents = []
    for entry in entries:
        source = Path(entry["source"])
        if source.suffix.lower() not in {".txt", ".md", ".docx", ".pdf"}:
            raise ValueError("Unsupported approved book format")
        contents = extract_text(source)
        if not contents.strip():
            raise ValueError("Approved book has no readable text")
        # Hash extracted text as well as original approved bytes so the
        # dataset's provenance is reproducible.
        documents.append({
            "id": entry["id"],
            "source_sha256": entry["sha256"],
            "text_sha256": _fingerprint(contents),
            "text": contents,
            "words": len(_words(contents)),
        })
    total_words = sum(doc["words"] for doc in documents)
    if total_words < min_total_words:
        raise ValueError("Approved books provide too little text for the requested run")

    # Prefer document-disjoint holdouts with 3+ sizeable, distinct books.
    distinct = []
    for document in documents:
        if any(near_duplicate(document["text"], prior["text"]) for prior in distinct):
            raise ValueError("Approved sources contain near-duplicate book editions")
        distinct.append(document)
    partitions = {"train": [], "validation": [], "test": []}
    if len(documents) >= 3 and all(doc["words"] >= 1000 for doc in documents):
        # Deterministic assignment by source identity; all sections from
        # a book remain within exactly one split.
        indexed = sorted(documents, key=lambda row: row["source_sha256"])
        n = len(indexed)
        test_n = max(1, round(n * test_ratio))
        valid_n = max(1, round(n * validation_ratio))
        grouped = {
            "train": indexed[:n-test_n-valid_n],
            "validation": indexed[n-test_n-valid_n:n-test_n],
            "test": indexed[n-test_n:],
        }
        for split, works in grouped.items():
            for doc in works:
                for section in _sections(doc["text"], target_words=target_words):
                    partitions[split].append((doc["id"], section))
        policy = "document_disjoint"
    else:
        # A book may span thousands of paragraphs but still be only one
        # independent work. Don't pretend section-disjoint means
        # cross-book or cross-domain generalization.
        for doc in documents:
            sections = _sections(doc["text"], target_words=target_words)
            if len(sections) < 10:
                raise ValueError("Each book needs 10+ coherent sections for section holdouts")
            n = len(sections)
            test_n = max(1, round(n * test_ratio))
            valid_n = max(1, round(n * validation_ratio))
            for idx, section in enumerate(sections):
                split = ("test" if idx >= n - test_n else
                         "validation" if idx >= n - test_n - valid_n else "train")
                partitions[split].append((doc["id"], section))
        policy = "section_disjoint_same_book"

    if any(not rows for rows in partitions.values()):
        raise ValueError("Book split left an empty partition")
    # Reject entire prepared splits if identical passages leak through
    # training, validation, or held-out text. Never silently drop a test.
    hashes = {}
    for split, rows in partitions.items():
        for work_id, section in rows:
            digest = _fingerprint(" ".join(section.split()).casefold())
            prior = hashes.setdefault(digest, split)
            if prior != split:
                raise ValueError("Repeated passage crosses book split boundary")
    output.mkdir(parents=True)
    summary = {"format": "approved_books_v1", "policy": policy,
               "total_words": total_words, "approved_books": len(documents),
               "sections": {}, "sources": [
                   {k: row[k] for k in ("id", "source_sha256", "text_sha256", "words")}
                   for row in documents]}
    try:
        for split, rows in partitions.items():
            folder = output / split
            folder.mkdir()
            summary["sections"][split] = len(rows)
            for idx, (identifier, section) in enumerate(rows):
                safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", identifier)
                filename = f"{safe_id}_{idx:05d}.txt"
                (folder / filename).write_text(section + "\n", encoding="utf-8")
        _write_json(output / "manifest.json", summary)
    except BaseException:
        # Never leave a plausible but partially prepared corpus behind.
        import shutil
        shutil.rmtree(output)
        raise
    return {"directory": str(output), **summary}
