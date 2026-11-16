"""Conventional webpage acquisition; the Core alone interprets source passages."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

from .retrieval import build_index, CorpusIndex
from .corpus import source_sections


def render_page(url, destination):
    """Use installed Linux renderer; no account or external inference API."""
    executable = shutil.which("wkhtmltopdf")
    if executable is None:
        raise RuntimeError("Install wkhtmltopdf to acquire webpages")
    subprocess.run([executable, "--quiet", "--disable-local-file-access",
                    "--disable-javascript", url, str(destination)], check=True,
                   stdin=subprocess.DEVNULL)


class WebLibrary:
    def __init__(self, directory, renderer=None):
        self.root = Path(directory).resolve()
        self.documents = self.root / "documents"
        self.index = self.root / "passages.sqlite"
        self.renderer = renderer or render_page

    def acquire(self, url):
        if not isinstance(url, str) or any(c.isspace() or ord(c)<32 for c in url):
            raise ValueError("Expected an HTTP(S) webpage URL")
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
            raise ValueError("Expected an HTTP(S) webpage URL without credentials")
        self.documents.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=self.root) as directory:
            temp = Path(directory)/"page.pdf"
            self.renderer(url, temp)
            raw = temp.read_bytes()
            if not raw.startswith(b"%PDF-"):
                raise ValueError("Renderer did not produce a PDF")
            readable = any(body.strip() for _, body in source_sections(temp))
            identity = sha256(raw).hexdigest()
            captured = datetime.now(timezone.utc).isoformat()
            record_id = sha256((url+"\n"+identity).encode()).hexdigest()
            name = record_id+".pdf"
            target = self.documents/name
            metadata = self.documents/(record_id+".json")
            record = {"url":url,"pdf":name,"sha256":identity,"captured_at":captured,
                      "readable_text":readable}
            if metadata.exists():
                record = json.loads(metadata.read_text(encoding="utf-8"))
            else:
                temp.replace(target)
                pending=Path(directory)/"record.json"
                pending.write_text(json.dumps(record),encoding="utf-8")
                pending.replace(metadata)
            if readable:
                build_index(self.documents,self.index)
            return record

    def search(self, query, limit=3):
        if not self.index.exists():
            return []
        with CorpusIndex(self.index) as index:
            rows=index.search(query,limit)
        for row in rows:
            record=json.loads((self.documents/Path(row["source"]).with_suffix(".json")).read_text(encoding="utf-8"))
            row["url"]=record["url"]
            row["pdf_sha256"]=record["sha256"]
            row["captured_at"]=record["captured_at"]
        return rows
