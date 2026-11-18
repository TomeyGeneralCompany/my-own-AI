"""Immutable source import, paragraph/block indexing and Webster headword parsing."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3

# Canonical text must be provided by the owner. No fabricated corpus is shipped.
PARA = re.compile(r"(?m)^\s*(?:§\s*)?(\d+)[.)]?\s+(.+?)(?=^\s*(?:§\s*)?\d+[.)]?\s+|\Z)", re.S)
BLOCK = re.compile(r"(?mi)^\s*(Terms\.|Consequence\.|Proposition\.|Equation\.)\s*")

@dataclass(frozen=True)
class SourceRecord:
    id: str
    document_id: str
    paragraph_number: int
    exact_text: str
    kind: str = "paragraph"
    parent_id: str = ""
    source_hash: str = ""

class Corpus:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "index.sqlite3")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
          id TEXT PRIMARY KEY, title TEXT NOT NULL, original_sha TEXT NOT NULL,
          exact_text_sha TEXT NOT NULL, source_name TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS records (
          id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id),
          paragraph_number INTEGER NOT NULL, kind TEXT NOT NULL,
          parent_id TEXT, exact_text TEXT NOT NULL, text_sha TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS dependencies (
          src TEXT NOT NULL REFERENCES records(id),
          dst TEXT NOT NULL REFERENCES records(id),
          relation TEXT NOT NULL, PRIMARY KEY(src,dst,relation)
        );
        CREATE TABLE IF NOT EXISTS lexicon (
          headword TEXT NOT NULL, sequence INTEGER NOT NULL, entry TEXT NOT NULL,
          PRIMARY KEY (headword, sequence)
        );
        """)
        self.db.commit()

    def ingest_text(self, path: str | Path, document_id: str, title: str,
                    *, require_numbered: bool = True) -> int:
        path = Path(path)
        raw = path.read_bytes()
        exact = raw.decode("utf-8")
        source_hash = sha256(raw).hexdigest()
        matches = list(PARA.finditer(exact))
        if require_numbered and not matches:
            raise ValueError("No numbered paragraphs; no silent fallback parsing")
        if require_numbered and exact[:matches[0].start()].strip():
            # Front matter is preserved in original bytes, but not silently treated as a paragraph.
            pass
        if not require_numbered:
            matches = []
        items = []
        seen = set()
        for match in matches:
            num = int(match.group(1))
            if num in seen:
                raise ValueError(f"Duplicate paragraph number {num}")
            seen.add(num)
            content = match.group(2).strip()
            if not content:
                raise ValueError(f"Empty paragraph {num}")
            items.append(SourceRecord(f"{document_id}:P-{num:06d}", document_id, num, content))
        dest = self.root / "originals" / (document_id + ".txt")
        dest.parent.mkdir(exist_ok=True)
        if dest.exists() and sha256(dest.read_bytes()).hexdigest() != source_hash:
            raise ValueError("Document ID already mapped to different source version")
        with self.db:
            existing = self.db.execute("SELECT original_sha FROM documents WHERE id=?", (document_id,)).fetchone()
            if existing and existing[0] != source_hash:
                raise ValueError("Cannot silently replace an approved source")
            self.db.execute("INSERT OR IGNORE INTO documents VALUES(?,?,?,?,?)",
                            (document_id,title,source_hash,sha256(exact.encode()).hexdigest(),path.name))
            for item in items:
                self.db.execute("INSERT OR IGNORE INTO records VALUES(?,?,?,?,?,?,?)",
                    (item.id,document_id,item.paragraph_number,item.kind,None,item.exact_text,
                     sha256(item.exact_text.encode()).hexdigest()))
                markers = list(BLOCK.finditer(item.exact_text))
                for i, marker in enumerate(markers, start=1):
                    label = marker.group(1).lower().rstrip(".")
                    end = markers[i].start() if i < len(markers) else len(item.exact_text)
                    exact_block = item.exact_text[marker.start():end].strip()
                    block_id = f"{item.id}:{label.upper()}-{i:02d}"
                    self.db.execute("INSERT OR IGNORE INTO records VALUES(?,?,?,?,?,?,?)",
                        (block_id, document_id, item.paragraph_number, label, item.id,
                         exact_block, sha256(exact_block.encode()).hexdigest()))
        if not dest.exists():
            dest.write_bytes(raw)
        return len(items)

    def get(self, record_id: str) -> SourceRecord:
        row = self.db.execute("SELECT id,document_id,paragraph_number,exact_text,kind,coalesce(parent_id,''),text_sha FROM records WHERE id=?",
                              (record_id,)).fetchone()
        if row is None:
            raise KeyError(record_id)
        if sha256(row[3].encode()).hexdigest() != row[6]:
            raise ValueError("Source record integrity failure")
        return SourceRecord(*row)

    def list_paragraphs(self, document_id: str) -> list[SourceRecord]:
        ids = self.db.execute("SELECT id FROM records WHERE document_id=? AND kind='paragraph' ORDER BY paragraph_number", (document_id,))
        return [self.get(r[0]) for r in ids]

    def subblocks(self, record_id: str) -> list[SourceRecord]:
        ids = self.db.execute("SELECT id FROM records WHERE parent_id=? ORDER BY id", (record_id,))
        return [self.get(row[0]) for row in ids]

    def add_dependency(self, src: str, dst: str, relation: str) -> None:
        if relation not in {"DEFINES","USES_VARIABLE","DEPENDS_ON","DERIVES_FROM","SUPPORTS",
                            "QUALIFIES","CITES","CONTINUES","CONTRASTS_WITH"}:
            raise ValueError("Unapproved relationship type")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO dependencies VALUES(?,?,?)", (src,dst,relation))

    def prerequisites(self, record_id: str) -> list[SourceRecord]:
        ids = self.db.execute("SELECT dst FROM dependencies WHERE src=? ORDER BY dst", (record_id,))
        return [self.get(row[0]) for row in ids]

    def import_dictionary_tsv(self, path: str | Path) -> int:
        """Import explicitly delimited headword<TAB>definition lines.

        Reject malformed rows and metadata instead of promoting them to lessons.
        Official Webster data must be supplied and verified by the owner.
        """
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        results = []
        for n, line in enumerate(lines, 1):
            if not line.strip():
                continue
            if "\t" not in line:
                raise ValueError(f"Missing headword separator at line {n}")
            head, definition = line.split("\t",1)
            head, definition = head.strip(), definition.strip()
            if not head or not definition:
                raise ValueError(f"Empty headword/definition at line {n}")
            if head.startswith(("00-","__")) or head.lower() in {"00-cide-info","00-database-info"}:
                raise ValueError(f"Dictionary metadata in lexical input at line {n}: {head}")
            results.append((head,definition))
        def key(item):
            head = item[0].casefold()
            return (re.sub(r"^[^a-z0-9]+", "", head), head)
        results.sort(key=key)
        with self.db:
            self.db.execute("DELETE FROM lexicon")
            self.db.executemany("INSERT INTO lexicon VALUES(?,?,?)",
                                [(head,i,definition) for i,(head,definition) in enumerate(results)])
        return len(results)

    def dictionary(self) -> list[tuple[str,str]]:
        return list(self.db.execute("SELECT headword,entry FROM lexicon ORDER BY sequence"))
