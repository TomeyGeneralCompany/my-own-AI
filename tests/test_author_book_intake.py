"""Safe repeatable author-source intake with original equation tokens."""
from hashlib import sha256
import json
from pathlib import Path

import pytest
from docx import Document
from pypdf import PdfWriter

from scripts.import_authored_book import extract_input,prepare_new_book,split_transport
from my_own_ai.published_corpus import validate_published_manifest


def setup_repo(tmp_path):
    repo=tmp_path/"repo"
    (repo/"published_books").mkdir(parents=True)
    (repo/"published_books/manifest.json").write_text(json.dumps({
        "format":"ancestor_published_corpus_v1",
        "corpus_id":"fixture-library","works":[]}))
    return repo


def synthetic_text():
    return "\n\n".join(
       f"Section {i}. The system receives a message from the operator and "
       "responds according to the feedback conditions; A = B + C and "
       "state X(t) depends upon the preceding state of the system. "
       "The resulting operation feeds back into the following cycle."
       for i in range(90))


def test_import_txt_is_lossless_versioned_and_requires_author_approval(tmp_path):
    repo=setup_repo(tmp_path)
    original=synthetic_text()
    inp=tmp_path/"source.txt"
    inp.write_text(original)
    with pytest.raises(ValueError,match="approve"):
        prepare_new_book(input_path=inp,repository=repo,ident="new-essay",
                         title="New Essay",author="Author")
    preview=prepare_new_book(input_path=inp,repository=repo,ident="new-essay",
                             title="New Essay",author="Author",dry_run=True)
    assert preview["dry_run"] and preview["source_characters"]==len(original)
    assert not list((repo/"published_books").glob("library/*.txt"))
    accepted=prepare_new_book(input_path=inp,repository=repo,ident="new-essay",
                              title="New Essay",author="Author",approve=True)
    assert accepted["approved_for_training"] is True
    manifest=json.loads((repo/"published_books/manifest.json").read_text())
    assert manifest["works"][0]["part_joiner"]==""
    approved=validate_published_manifest(repo,repo/"published_books/manifest.json")
    assert approved["sources"][0]["text"]==original
    assert approved["sources"][0]["sha256"]==sha256(original.encode()).hexdigest()
    with pytest.raises(ValueError,match="overwritten"):
        prepare_new_book(input_path=inp,repository=repo,ident="new-essay",
                         title="New Essay",author="Author",approve=True)


def test_docx_math_and_paragraphs_are_extracted_without_dropping_symbols(tmp_path):
    document=Document()
    for i in range(60):
        document.add_paragraph(f"Paragraph {i}. The recurrence is X(t+1) = A + βG(t). "
                               "This shows how the system changes its state by feedback.")
    inp=tmp_path/"with-math.docx"
    document.save(inp)
    text,warnings=extract_input(inp)
    assert "X(t+1) = A + βG(t)" in text
    assert "Paragraph 59" in text
    assert "Check spatial arrangement" in warnings[0]
    assert len(text.split("\n\n"))==60


def test_importer_rejects_blank_scanned_pdf_instead_of_inventing_ocr(tmp_path):
    out=tmp_path/"blank.pdf"
    writer=PdfWriter()
    writer.add_blank_page(width=300,height=300)
    with out.open("wb") as target:
        writer.write(target)
    with pytest.raises(ValueError,match="unreadable"):
        extract_input(out)


def test_transport_split_never_drops_utf8_equations():
    original=synthetic_text()+"\n\nΦ∑β = ∫0→∞"
    parts=split_transport(original,limit=2200)
    assert len(parts)>1
    assert "".join(parts)==original
