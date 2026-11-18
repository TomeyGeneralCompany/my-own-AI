"""Import owner-supplied private library bundles without publishing manuscript contents.

Only a locally supplied bundle is accepted. Canonical originals and extracted text
are copied into local data storage, verified by SHA-256, and indexed as traceable
Word paragraphs or PDF page learning units. No missing paragraph is invented;
PDF page units remain explicitly typed and eligible for later review.
"""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import os
import re
import shutil
import sqlite3
import tempfile
from zipfile import ZipFile
from .corpus import Corpus

def digest(content: bytes) -> str:
    return sha256(content).hexdigest()

def _safe_member(name: str) -> None:
    name_path = Path(name)
    if name_path.is_absolute() or ".." in name_path.parts or "\\" in name:
        raise ValueError("Invalid archive path")

def import_private_library(bundle: str | Path, data_directory: str | Path) -> dict:
    """Fully validate before changing the corpus; import records transactionally.

    The return value describes completed registration, not verified neural
    comprehension or final PDF paragraph reconstruction.
    """
    bundle=Path(bundle)
    data=Path(data_directory)
    with ZipFile(bundle,"r") as archive:
        names=set(archive.namelist())
        for name in names:
            _safe_member(name)
        if "manifest.json" not in names:
            raise ValueError("Missing library manifest")
        manifest=json.loads(archive.read("manifest.json"))
        sources=manifest.get("sources",[])
        if manifest.get("version")!=2 or not sources:
            raise ValueError("Unsupported or empty private library format")
        if manifest.get("source_count")!=len(sources):
            raise ValueError("Library source count mismatch")
        verified=[]
        ids=set()
        for source in sources:
            document_id=source["id"]
            if not re.fullmatch(r"[A-Za-z0-9_-]+",document_id) or document_id in ids:
                raise ValueError("Invalid or duplicate document ID")
            ids.add(document_id)
            ext=source["format"]
            if ext not in {"docx","pdf","txt"}:
                raise ValueError("Unsupported private source type")
            original_path=f"originals/{document_id}.{ext}"
            extracted_path=f"extracted/{document_id}.txt"
            records_path=f"index/{document_id}.jsonl"
            if not {original_path,extracted_path,records_path}<=names:
                raise ValueError(f"Missing source parts: {document_id}")
            original=archive.read(original_path)
            text_bytes=archive.read(extracted_path)
            index_bytes=archive.read(records_path)
            if digest(original)!=source["original_sha256"]:
                raise ValueError(f"Original hash mismatch: {document_id}")
            if digest(text_bytes)!=source["extracted_sha256"]:
                raise ValueError(f"Extracted text mismatch: {document_id}")
            if digest(index_bytes)!=source["records_sha256"]:
                raise ValueError(f"Index hash mismatch: {document_id}")
            exact_text=text_bytes.decode("utf-8")
            index=[json.loads(line) for line in index_bytes.decode("utf-8").splitlines() if line]
            if len(index)!=source["record_count"]:
                raise ValueError(f"Index length mismatch: {document_id}")
            previous_end=0
            source_ids=set()
            for entry in index:
                if entry["document_id"]!=document_id or entry["id"] in source_ids:
                    raise ValueError("Cross-document or duplicate index identifier")
                source_ids.add(entry["id"])
                start,end=entry["start_char"],entry["end_char"]
                if not (0<=previous_end<=start<end<=len(exact_text)):
                    raise ValueError("Invalid source character boundaries")
                text=exact_text[start:end]
                if text!=entry["text"] or digest(text.encode("utf-8"))!=entry["sha256"]:
                    raise ValueError(f"Indexed source mismatch: {entry['id']}")
                previous_end=end
            verified.append((source,original,text_bytes,index))
    corpus=Corpus(data)
    # Preflight existing source identities so one incompatible item cannot cause
    # partial replacement of a previously imported library.
    for source,original,text_bytes,index in verified:
        identity=corpus.db.execute("SELECT original_sha FROM documents WHERE id=?",
                (source["id"],)).fetchone()
        if identity and identity[0]!=source["original_sha256"]:
            raise ValueError("Conflicting source document ID: "+source["id"])
        for row in index:
            previous=corpus.db.execute("SELECT text_sha FROM records WHERE id=?",(row["id"],)).fetchone()
            if previous and previous[0]!=row["sha256"]:
                raise ValueError("Conflicting source record: "+row["id"])
    # Local-only copies. Never adds private source material to git.
    (data/"originals").mkdir(parents=True,exist_ok=True)
    (data/"extracted").mkdir(parents=True,exist_ok=True)
    for source,original,text_bytes,index in verified:
        document_id=source["id"]
        for dest,raw in (
            (data/"originals"/(document_id+"."+source["format"]),original),
            (data/"extracted"/(document_id+".txt"),text_bytes),
        ):
            if dest.exists():
                if digest(dest.read_bytes())!=digest(raw):
                    raise ValueError("Local source file conflicts with imported version")
            else:
                fd,temp=tempfile.mkstemp(prefix=".ancestor-",dir=dest.parent)
                try:
                    with os.fdopen(fd,"wb") as fh:
                        fh.write(raw)
                        fh.flush()
                        os.fsync(fh.fileno())
                    os.replace(temp,dest)
                finally:
                    if os.path.exists(temp):
                        os.unlink(temp)
    with corpus.db:
        for source,original,text_bytes,index in verified:
            document_id=source["id"]
            corpus.db.execute("INSERT OR IGNORE INTO documents VALUES(?,?,?,?,?)",
              (document_id,source["filename"],source["original_sha256"],
               source["extracted_sha256"],source["filename"]))
            for record in index:
                # Use 'paragraph' for original Word paragraph units and 'pdf_page'
                # for page-scoped text, which is never misrepresented as a paragraph.
                kind=("paragraph" if record["kind"]=="word_paragraph"
                      else "pdf_page")
                corpus.db.execute("INSERT OR IGNORE INTO records VALUES(?,?,?,?,?,?,?)",
                  (record["id"],document_id,record["ordinal"],kind,None,
                   record["text"],record["sha256"]))
    report={
      "imported_source_count":len(verified),
      "indexed_learning_units":sum(len(rows) for _,_,_,rows in verified),
      "private_corpus_directory":str(data),
      "source_manifest":{source["id"]:source["original_sha256"] for source,_,_,_ in verified},
      "pdf_page_units_not_certified_as_paragraphs":sum(
          len(rows) for source,_,_,rows in verified if source["format"]=="pdf"),
      "status":"READY_FOR_APPROVED_TRAINING_SETUP",
    }
    (data/"private-library-manifest.json").write_text(json.dumps(
        report,ensure_ascii=False,indent=2),encoding="utf-8")
    return report

def export_training_text(data_directory: str | Path, output: str | Path) -> dict:
    """Export full text of registered private sources in deterministic source order.

    Full-corpus export is intended for tokenizer/learning coverage. A held-out
    validation set must be supplied independently to measure generalization.
    """
    data=Path(data_directory)
    corpus=Corpus(data)
    documents=corpus.db.execute(
        "SELECT id,exact_text_sha FROM documents ORDER BY id").fetchall()
    if not documents:
        raise ValueError("No registered documents")
    texts=[]
    for document_id,expected_sha in documents:
        path=data/"extracted"/(document_id+".txt")
        if not path.exists():
            raise ValueError("Missing extracted source: "+document_id)
        raw=path.read_bytes()
        if digest(raw)!=expected_sha:
            raise ValueError("Source extraction changed: "+document_id)
        texts.append(raw.decode("utf-8"))
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text("\n\n".join(texts),encoding="utf-8")
    return {"document_count":len(texts),"output":str(output),
            "sha256":digest(output.read_bytes())}
