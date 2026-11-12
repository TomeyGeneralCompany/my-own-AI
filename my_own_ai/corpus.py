"""Offline book ingestion: TXT, PDF and DOCX, with stable metadata and duplicate checks."""
import hashlib
import json
from pathlib import Path

SUPPORTED = {".txt", ".md", ".pdf", ".docx"}

def source_sections(path):
    """Yield stable document locators and extracted text in source order."""
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            yield f"line {number}", line
    elif suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF ingestion requires pip install -e '.[books]'") from exc
        for number, page in enumerate(PdfReader(str(path)).pages, 1):
            yield f"PDF page {number}", page.extract_text() or ""
    elif suffix == ".docx":
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError("DOCX ingestion requires pip install -e '.[books]'") from exc
        for number, paragraph in enumerate(Document(str(path)).paragraphs, 1):
            yield f"paragraph {number}", paragraph.text
    else:
        raise ValueError(f"Unsupported format: {suffix}")


def extract_text(path):
    if path.suffix.lower() in (".txt", ".md"):
        return path.read_text(encoding="utf-8-sig")
    return "\n".join(text for _, text in source_sections(path))


def corpus_files(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError("Missing corpus directory")
    return [path for path in sorted(folder.rglob("*"))
            if not path.is_symlink() and path.is_file() and path.suffix.lower() in SUPPORTED]


def collect_documents(folder):
    folder = Path(folder)
    docs, hashes = [], set()
    for path in corpus_files(folder):
        text = extract_text(path).replace("\r\n", "\n").replace("\r", "\n").strip()
        if not text:
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in hashes:
            continue
        hashes.add(digest)
        docs.append({"id": digest, "source": str(path.relative_to(folder)),
                     "title": path.stem, "format": path.suffix.lower(),
                     "characters": len(text), "text": text})
    if not docs:
        raise ValueError("Corpus has no .txt or other nonempty supported documents")
    return docs

def corpus_fingerprint(documents):
    hashes = sorted(d["id"] for d in documents)
    return hashlib.sha256("\n".join(hashes).encode("ascii")).hexdigest()

def build_manifest(folder, output):
    docs = collect_documents(folder)
    manifest = {"schema": 1, "fingerprint": corpus_fingerprint(docs),
                "documents": [{k: v for k, v in d.items() if k != "text"} for d in docs]}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest

def split_documents(docs, context_size, seed=42):
    """Document-disjoint split when possible; single-document fallback explicitly marked."""
    import random
    ordered = sorted(docs, key=lambda d: d["id"])
    random.Random(seed).shuffle(ordered)
    usable = [d for d in ordered if len(d["text"].encode("utf-8")) > context_size + 1]
    if not usable:
        raise ValueError("No document exceeds context size")
    if len(usable) == 1:
        raw = usable[0]["text"].encode("utf-8")
        # Small single-file datasets cannot offer a document-disjoint holdout.
        cut = max(context_size + 2, int(len(raw) * .8))
        if len(raw) - cut <= context_size + 1:
            return [raw], [], "no-validation-single-document"
        return [raw[:cut]], [raw[cut:]], "single-document-byte-split"
    n_valid = max(1, round(len(usable) * .2))
    valid = [d["text"].encode("utf-8") for d in ordered[:n_valid] if d in usable]
    train = [d["text"].encode("utf-8") for d in ordered[n_valid:] if d in usable]
    if not train or not valid:
        raise ValueError("Could not construct document-disjoint split")
    return train, valid, "document-disjoint"
