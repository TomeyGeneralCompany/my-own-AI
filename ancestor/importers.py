"""Owner-reviewed extraction adapters for local DOCX/PDF/TXT sources.

Extraction is preparation, NOT silent approval of a reconstructed manuscript.
Original artifacts and extracted text are retained separately. Numbered paragraph
verification and owner approval are necessary before production curriculum use.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import json
import shutil
from tempfile import NamedTemporaryFile
from zipfile import ZipFile
import xml.etree.ElementTree as ET
from .corpus import Corpus

WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

def extract_docx(path: str | Path) -> str:
    """Read Word paragraphs in document order; preserve literal text, no paraphrase."""
    with ZipFile(path) as archive:
        root=ET.fromstring(archive.read("word/document.xml"))
    paragraphs=[]
    for para in root.iter(WORD_NS+"p"):
        runs=[]
        for node in para.iter():
            if node.tag==WORD_NS+"t":
                runs.append(node.text or "")
            elif node.tag==WORD_NS+"tab":
                runs.append("\t")
            elif node.tag==WORD_NS+"br":
                runs.append("\n")
        if runs:
            paragraphs.append("".join(runs))
    return "\n".join(paragraphs)

def extract_pdf(path: str | Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise RuntimeError("PDF extraction requires optional pypdf installation") from e
    pages=[]
    for i,page in enumerate(PdfReader(path).pages):
        text=page.extract_text() or ""
        if not text.strip():
            raise ValueError(f"PDF page {i+1} has no extractable text; requires review")
        pages.append(text)
    return "\n".join(pages)

def prepare_document(source: str | Path, output_directory: str | Path) -> dict:
    source=Path(source)
    output_directory=Path(output_directory)
    output_directory.mkdir(parents=True,exist_ok=True)
    original_sha=sha256(source.read_bytes()).hexdigest()
    original=output_directory / ("original-"+original_sha+source.suffix.lower())
    if not original.exists():
        shutil.copyfile(source,original)
    if source.suffix.lower()==".txt":
        extracted=source.read_text(encoding="utf-8")
    elif source.suffix.lower()==".docx":
        extracted=extract_docx(source)
    elif source.suffix.lower()==".pdf":
        extracted=extract_pdf(source)
    else:
        raise ValueError("Unsupported manuscript format")
    if not extracted.strip():
        raise ValueError("Empty extracted document")
    text_path=output_directory / ("extracted-"+original_sha+".txt")
    if text_path.exists() and text_path.read_text(encoding="utf-8")!=extracted:
        raise ValueError("Extracted text mismatch for same original artifact")
    text_path.write_text(extracted,encoding="utf-8")
    manifest={"source_filename":source.name,"original_sha256":original_sha,
              "original_path":str(original),"extracted_path":str(text_path),
              "extracted_sha256":sha256(extracted.encode()).hexdigest(),
              "approval_status":"PENDING_OWNER_REVIEW"}
    (output_directory / ("manifest-"+original_sha+".json")).write_text(
        json.dumps(manifest,indent=2),encoding="utf-8")
    return manifest

def commit_reviewed_text(corpus: Corpus, manifest_path: str | Path,
                         *, document_id: str, title: str, approved_hash: str) -> int:
    """Exact owner-approved extracted hash is mandatory for indexing."""
    manifest=json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest["extracted_sha256"]!=approved_hash:
        raise PermissionError("Approved text hash does not match extraction")
    path=Path(manifest["extracted_path"])
    if sha256(path.read_bytes()).hexdigest()!=approved_hash:
        raise ValueError("Extracted content changed since review")
    return corpus.ingest_text(path,document_id,title,require_numbered=True)
