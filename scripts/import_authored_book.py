"""Import an author-approved PDF/DOCX/TXT into the seven-work library.

Use python scripts/import_authored_book.py --input path/to/book.pdf --id slug
--title Title --approve to create source text, pinned SHA-256 parts and a
manifest entry. Existing editions/checkpoints are never overwritten.

Equation text in text-based PDFs and OMML DOCX is retained on a best-effort
basis; scanned graphics and two-dimensional equation layout require a source
quality review rather than silent OCR.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from my_own_ai.structured_books import structured_units
from my_own_ai.numbered_paragraphs import numbered_paragraphs

NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS_M = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def extract_input(path):
    path = Path(path)
    kind = path.suffix.lower()
    if kind in (".txt", ".md"):
        return path.read_text(encoding="utf-8-sig"), []
    if kind == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        pieces = [page.extract_text() or "" for page in reader.pages]
        if any(not page.strip() for page in pieces):
            raise ValueError("PDF contains blank/unreadable pages; review original before training")
        return "\n\n".join(pieces), ["Review PDF diagrams and display equations against original artwork"]
    if kind == ".docx":
        with ZipFile(path) as archive:
            root = ET.fromstring(archive.read("word/document.xml"))
        paragraphs = []
        for p in root.iter(f"{{{NS_W}}}p"):
            pieces = []
            for node in p.iter():
                if node.tag in (f"{{{NS_W}}}t", f"{{{NS_M}}}t") and node.text:
                    pieces.append(node.text)
                elif node.tag == f"{{{NS_W}}}tab":
                    pieces.append("\t")
                elif node.tag in (f"{{{NS_W}}}br",f"{{{NS_W}}}cr"):
                    pieces.append("\n")
            if pieces:
                paragraphs.append("".join(pieces))
        if not paragraphs:
            raise ValueError("No DOCX text or OMML math tokens available")
        return "\n\n".join(paragraphs), ["Check spatial arrangement of OMML fractions and matrices"]
    raise ValueError("Supported source extensions are PDF, DOCX, TXT and MD")


def split_transport(text, limit=170000):
    if limit < 1024:
        raise ValueError("Transport part budget too small")
    result = []
    while text:
        if len(text) <= limit:
            result.append(text)
            break
        cut = text.rfind("\n\n", int(limit*.6), limit)
        if cut < 0:
            cut = text.rfind("\n", int(limit*.6), limit)
        if cut < 0:
            cut = limit
        else:
            cut += 1
        result.append(text[:cut])
        text = text[cut:]
    assert "".join(result)
    return result


def prepare_new_book(*, input_path, repository, ident, title, author,
                     chunking="structured_units", approve=False, dry_run=False):
    root = Path(repository).resolve()
    source = Path(input_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not re.fullmatch(r"[a-z0-9-]{3,70}",ident):
        raise ValueError("Invalid work ID")
    manifest_path = root/"published_books/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if any(x["id"]==ident for x in manifest["works"]):
        raise ValueError("An existing work cannot be overwritten")
    text, warnings=extract_input(source)
    if len(text.split())<1000:
        raise ValueError("Input is too short to train from")
    if chunking=="numbered_paragraph":
        units=numbered_paragraphs(text)
    elif chunking=="structured_units":
        units=structured_units(text)
    else:
        raise ValueError("Unknown unit strategy")
    if len(units)<10:
        raise ValueError("Insufficient logical units for disjoint holdouts")
    parts=split_transport(text)
    outputs=[]
    for i,part in enumerate(parts,1):
        relative=f"published_books/library/{ident}-part-{i:02d}.txt"
        dest=root/relative
        if dest.exists():
            raise ValueError(f"Transport file already exists: {relative}")
        outputs.append((relative,part))
    row={"id":ident,"title":title,"author":author,
         "source_format":source.suffix.lower()[1:]+"_text_extraction",
         "source_status":"author-approved source text" if approve else "pending author approval",
         "approved_for_training":bool(approve),"chunking":chunking,
         "parts":[{"path":path,"sha256":sha256(part.encode()).hexdigest()}
                  for path,part in outputs]}
    report={"id":ident,"source_file":str(source),"source_characters":len(text),
            "logical_units":len(units),"parts":len(outputs),
            "source_sha256":sha256(text.encode()).hexdigest(),
            "warnings":warnings,"approved_for_training":bool(approve),
            "dry_run":bool(dry_run)}
    if not dry_run:
        for relative,part in outputs:
            dest=root/relative
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_text(part,encoding="utf-8")
        manifest["works"].append(row)
        manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",
                                 encoding="utf-8")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True)
    parser.add_argument("--repository",default=".")
    parser.add_argument("--id",required=True)
    parser.add_argument("--title",required=True)
    parser.add_argument("--author",default="Isaiz James Tomey")
    parser.add_argument("--chunking",choices=["structured_units","numbered_paragraph"],
                        default="structured_units")
    parser.add_argument("--approve",action="store_true")
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args()
    print(json.dumps(prepare_new_book(input_path=args.input,repository=args.repository,
                   ident=args.id,title=args.title,author=args.author,
                   chunking=args.chunking,approve=args.approve,dry_run=args.dry_run),indent=2))


if __name__=="__main__":
    main()
