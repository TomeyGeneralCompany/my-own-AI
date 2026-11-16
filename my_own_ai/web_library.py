"""Conventional webpage acquisition; the Core alone interprets source passages."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit, urlencode
from contextlib import contextmanager

from .retrieval import build_index, CorpusIndex
from .corpus import source_sections


@contextmanager
def browser_page():
    """One ephemeral Chromium session per operation, closed even on failure."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("Install the web extra and run playwright install chromium") from exc
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            context = browser.new_context(accept_downloads=False)
            try:
                page = context.new_page()
                yield page
            finally:
                context.close()
        finally:
            browser.close()


def open_page(page, url):
    response = page.goto(url, wait_until="domcontentloaded", timeout=60000)
    if response is not None and response.status >= 400:
        raise RuntimeError(f"Webpage returned HTTP {response.status}")
    # Allow ordinary asynchronous content to arrive; continuously connected
    # pages need not become network-idle before they can be captured.
    try:
        page.wait_for_load_state("networkidle", timeout=5000)
    except Exception as exc:
        from playwright.sync_api import TimeoutError
        if not isinstance(exc, TimeoutError):
            raise


def render_page(url, destination):
    """Print a JavaScript-enabled page using Chromium's PDF engine."""
    with browser_page() as page:
        open_page(page, url)
        page.emulate_media(media="screen")
        page.pdf(path=str(destination), print_background=True)
        return {"final_url": page.url, "title": page.title()}


def discover_pages(query, limit):
    """Return ordinary search links, not AI-generated summaries."""
    url = "https://www.google.com/search?" + urlencode({"q": query})
    with browser_page() as page:
        open_page(page, url)
        rows = page.locator("a:has(h3)").evaluate_all(
            "(links) => links.map(a => ({url:a.href,title:a.innerText.trim()}))")
    results, seen = [], set()
    for row in rows:
        target = row["url"]
        parts = urlsplit(target)
        if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
            continue
        if target in seen or not row["title"]:
            continue
        seen.add(target)
        results.append({"url": target, "title": row["title"]})
        if len(results) == limit:
            break
    if not results:
        raise RuntimeError("Search provider returned no usable links (empty results or challenge)")
    return results


class WebLibrary:
    def __init__(self, directory, renderer=None, discoverer=None):
        self.root = Path(directory).resolve()
        self.documents = self.root / "documents"
        self.index = self.root / "passages.sqlite"
        self.renderer = renderer or render_page
        self.discoverer = discoverer or discover_pages

    def discover(self, query, limit=5):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Expected a nonempty search query")
        if type(limit) is not int or limit < 1:
            raise ValueError("Expected a positive result count")
        return self.discoverer(query, limit)

    def acquire(self, url):
        if not isinstance(url, str) or any(c.isspace() or ord(c)<32 for c in url):
            raise ValueError("Expected an HTTP(S) webpage URL")
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
            raise ValueError("Expected an HTTP(S) webpage URL without credentials")
        self.documents.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=self.root) as directory:
            temp = Path(directory)/"page.pdf"
            rendered = self.renderer(url, temp) or {}
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
                      "readable_text":readable, "final_url":rendered.get("final_url",url),
                      "title":rendered.get("title","")}
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
