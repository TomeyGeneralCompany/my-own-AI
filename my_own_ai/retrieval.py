"""Local SQLite passage index with deterministic ranking and source locators."""
from contextlib import closing
from hashlib import sha256
import os
from pathlib import Path
import re
import sqlite3
from tempfile import NamedTemporaryFile

from .corpus import corpus_files, source_sections

PASSAGE_LIMIT = 720
SCHEMA = 1
WORDS = re.compile(r"[^\W_]{2,}", re.UNICODE)
STOP = frozenset("about after also and are can could does for from have how into more most only "
                 "over should than that the their them then there these they this those through "
                 "under was were what when where which who will with would you your".split())


def digest(text):
    return sha256(text.encode("utf-8")).hexdigest()


def passages(text):
    remaining = " ".join(text.split())
    while len(remaining) > PASSAGE_LIMIT:
        cut = remaining.rfind(" ", 0, PASSAGE_LIMIT + 1)
        cut = cut if cut > 0 else PASSAGE_LIMIT
        yield remaining[:cut]
        remaining = remaining[cut:].lstrip()
    if remaining:
        yield remaining


def build_index(folder, output):
    folder, output = Path(folder).resolve(), Path(output).absolute()
    files = corpus_files(folder)
    if output.resolve() in files:
        raise ValueError("Index destination overlaps a corpus source")
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, prefix=".index-", delete=False) as handle:
        temp = Path(handle.name)
    documents = count = 0
    try:
        with closing(sqlite3.connect(temp)) as db, db:
            db.execute(f"PRAGMA user_version = {SCHEMA}")
            db.execute("CREATE TABLE passages (id INTEGER PRIMARY KEY, source TEXT, locator TEXT, "
                       "body TEXT, sha256 TEXT)")
            db.execute("CREATE VIRTUAL TABLE terms USING fts5(body, source, content='passages', "
                       "content_rowid='id', tokenize='unicode61 remove_diacritics 2')")
            for path in files:
                source = path.relative_to(folder).as_posix()
                before = count
                for locator, text in source_sections(path):
                    for number, body in enumerate(passages(text), 1):
                        db.execute("INSERT INTO passages(source, locator, body, sha256) VALUES (?, ?, ?, ?)",
                                   (source, f"{locator}, passage {number}", body, digest(body)))
                        count += 1
                documents += count > before
            if not count:
                raise ValueError("Corpus contains no indexable text")
            db.execute("INSERT INTO terms(terms) VALUES ('rebuild')")
            db.execute("INSERT INTO terms(terms, rank) VALUES ('integrity-check', 1)")
        os.replace(temp, output)
        directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temp.unlink(missing_ok=True)
    return {"documents": documents, "passages": count}


class CorpusIndex:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.db = None

    def __enter__(self):
        db = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, check_same_thread=False)
        try:
            if db.execute("PRAGMA user_version").fetchone()[0] != SCHEMA:
                raise ValueError("Unsupported corpus index version")
            db.execute("SELECT id, source, locator, body, sha256 FROM passages LIMIT 0")
            db.execute("SELECT rowid FROM terms LIMIT 0")
        except BaseException:
            db.close()
            raise
        self.db = db
        return self

    def __exit__(self, *exc):
        self.db.close()
        self.db = None

    def search(self, query, limit=5):
        if self.db is None:
            raise RuntimeError("Corpus index is closed")
        if not isinstance(query, str) or not 1 <= len(query) <= 4000:
            raise ValueError("Expected a query of 1–4,000 characters")
        if type(limit) is not int or not 1 <= limit <= 12:
            raise ValueError("Search limit must be between 1 and 12")
        tokens = list(dict.fromkeys(word for word in WORDS.findall(query.casefold()) if word not in STOP))[:24]
        if not tokens:
            return []
        pattern = " OR ".join(f'"{token}"' for token in tokens)
        rows = self.db.execute(
            "SELECT p.source, p.locator, p.body, p.sha256 FROM terms "
            "JOIN passages p ON p.id = terms.rowid WHERE terms MATCH ? "
            "ORDER BY bm25(terms, 1.0, 0.35), p.id LIMIT ?", (pattern, limit))
        return [dict(zip(("source", "locator", "text", "sha256"), row)) for row in rows]
