"""Load all eleven owner-authorized published manuscript text extractions.

Unlike the owner's original DOCX/PDF files, these UTF-8 texts are *extractions*.
They are made publicly available by explicit authorization. This loader checks
every expected Git blob, indexes every nonempty block, and never treats a
repeated bracketed number or an unnumbered passage as a reason to discard it.

No trained model, approved axiom or external source is synthesized here.
"""
from __future__ import annotations
from hashlib import sha1, sha256
from pathlib import Path
import json
import re
from .corpus import Corpus

# SHA-1 values refer to the actual, publicly uploaded Git blob objects.
# Source hashes and paragraph hashes are separately SHA-256 in SQLite.
PUBLISHED_SOURCES = (
    ("WORK-001", "7 petitions of the Lord's Prayer",
     ("WORK-001.txt",), ("31cbf38e14bcffbb97ed47a7beb779aee7bab29b",)),
    ("WORK-002", "A Mathematical and Logical Defense of the True Doctrine of Election",
     ("WORK-002.txt",), ("852b8e5f9f0a44cac310c789058b20cee5b7f0e0",)),
    ("WORK-003", "Biospiritual Realism",
     ("WORK-003.txt",), ("195d2bee5b75e1a8aa520747c6c1d9e349388dd9",)),
    ("WORK-004", "Calvin Commentary, 635-page interior",
     tuple(f"WORK-004.part{i:02d}.txt" for i in range(1,6)),
     ("77be223c8da781ee772aec547ea5ee1ed756a2a5",
      "4ea8930796f0c2d975732addd00b749cd60cfd36",
      "36421967cc51fc656236e866687a3263d72a8dcb",
      "55889fab95e4877462cbf0ec4f3fd316f9d9bcef",
      "4e405aba275923978527ebd9c0f71df8af352063")),
    ("WORK-005", "Inane Electromagneticum: Theoria Cybernetica",
     ("WORK-005.txt",), ("3948152880cd7f7bbdad69592bb01fa722ad90a5",)),
    ("WORK-006", "Mechanical Acoustic Voice Transmission Machine",
     ("WORK-006.txt",), ("b0df000a3134b14feddb0236ff7e5772f2867223",)),
    ("WORK-007", "Noetic Cybernetics, Reformed Platonism, and Tomey's Law of Aletheic Fidelity",
     ("WORK-007.txt",), ("ac8a0f5e499b0545ae43328fb2ce75492eb54753",)),
    ("WORK-008", "Noetic Cybernetics and Melito of Sardis on the Old Testament Canon",
     ("WORK-008.txt",), ("d9490095d2c1a1735d2a8dec0cbc4e5f259862a9",)),
    ("WORK-009", "Roman Era Television Design",
     ("WORK-009.txt",), ("31a52758a05d4d80204ffd2f6451add5ac000a42",)),
    ("WORK-010", "The Electromagnetic Void: A Theory of Energetic Disequilibrium",
     ("WORK-010.txt",), ("804ae1953d0574ee022cfa21e52acfe5dbdf58f8",)),
    ("WORK-011", "The Patristic and Cybernetic Case for Sola Fide",
     ("WORK-011.txt",), ("642e06fde53b00535286f1ecddeea3c846ccb91a",)),
)

def git_blob_sha(raw: bytes) -> str:
    return sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()


def import_published_library(source_root: str | Path, data_root: str | Path) -> dict:
    """Import all eleven published extractions; fail without their full exact content."""
    source_root, data_root = Path(source_root), Path(data_root)
    if len(PUBLISHED_SOURCES) != 11 or len({s[0] for s in PUBLISHED_SOURCES}) != 11:
        raise ValueError("Published library manifest is incomplete or duplicated")
    verified = []
    for document_id, title, parts, expected_blobs in PUBLISHED_SOURCES:
        if len(parts) != len(expected_blobs):
            raise ValueError(f"Bad part manifest: {document_id}")
        pieces = []
        for name, expected_sha in zip(parts, expected_blobs):
            path = source_root / name
            raw = path.read_bytes()
            if git_blob_sha(raw) != expected_sha:
                raise ValueError(f"Source blob differs from published owner text: {name}")
            pieces.append(raw.decode("utf-8"))
        # Long commentary is stored as consecutive Git blobs to support
        # complete, lossless source-text transfer between connected services.
        exact = "\n\n".join(pieces)
        if not exact.strip():
            raise ValueError("Empty owner source: " + document_id)
        blocks = [b.strip() for b in re.split(r"\n[ \t]*\n+", exact) if b.strip()]
        if not blocks:
            raise ValueError("No content blocks: " + document_id)
        verified.append((document_id, title, exact, blocks))

    corpus = Corpus(data_root)
    # Preflight immutability and conflicting source identities.
    for document_id, title, exact, blocks in verified:
        digest = sha256(exact.encode("utf-8")).hexdigest()
        row = corpus.db.execute(
            "SELECT original_sha FROM documents WHERE id=?", (document_id,)).fetchone()
        if row and row[0] != digest:
            raise ValueError("Existing owner record has a different source: " + document_id)
        dest = data_root / "originals" / (document_id + ".txt")
        if dest.exists() and sha256(dest.read_bytes()).hexdigest() != digest:
            raise ValueError("Existing stored text differs: " + document_id)
    (data_root / "originals").mkdir(parents=True, exist_ok=True)
    (data_root / "extracted").mkdir(parents=True, exist_ok=True)

    with corpus.db:
        for document_id, title, exact, blocks in verified:
            raw = exact.encode("utf-8")
            digest = sha256(raw).hexdigest()
            for folder in ("originals", "extracted"):
                dest = data_root / folder / (document_id + ".txt")
                if not dest.exists():
                    dest.write_bytes(raw)
                elif sha256(dest.read_bytes()).hexdigest() != digest:
                    raise ValueError("Local source conflicts with imported version")
            corpus.db.execute(
                "INSERT OR IGNORE INTO documents VALUES(?,?,?,?,?)",
                (document_id, title, digest, digest, "published_library/" + document_id))
            for ordinal, block in enumerate(blocks, 1):
                record_id = f"{document_id}:EXTRACT-{ordinal:06d}"
                text_sha = sha256(block.encode("utf-8")).hexdigest()
                existing = corpus.db.execute(
                    "SELECT text_sha FROM records WHERE id=?", (record_id,)).fetchone()
                if existing and existing[0] != text_sha:
                    raise ValueError("Conflicting extracted block ID: " + record_id)
                corpus.db.execute(
                    "INSERT OR IGNORE INTO records VALUES(?,?,?,?,?,?,?)",
                    (record_id, document_id, ordinal, "paragraph", None, block, text_sha))
    report = {
        "owner_published_sources": len(verified),
        "indexed_extracted_text_blocks": sum(len(blocks) for _,_,_,blocks in verified),
        "source_sha256": {
            document_id: sha256(exact.encode("utf-8")).hexdigest()
            for document_id,_,exact,_ in verified
        },
        "provenance": "published-UTF8-text-extractions; original DOCX/PDF binaries not in Git",
    }
    (data_root / "published-library-manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(import_published_library(args.source,args.data), indent=2))


if __name__ == "__main__":
    main()
