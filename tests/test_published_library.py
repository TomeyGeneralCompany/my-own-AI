"""Real eleven-work integration: source identity, indexing, reproducible inputs."""
from pathlib import Path
import json
import pytest
from ancestor.public_library import import_published_library, PUBLISHED_SOURCES
from ancestor.public_training_setup import prepare, OWNER_THREE_AXIOMS
from ancestor.corpus import Corpus
from ancestor.tokenizer import ByteBPE
from ancestor.governance import Constitution

ROOT=Path(__file__).resolve().parents[1]

def test_all_eleven_real_sources_preserved_in_training_index(tmp_path):
    report=import_published_library(ROOT/"published_library",tmp_path/"corpus")
    assert report["owner_published_sources"]==11
    assert report["indexed_extracted_text_blocks"]>=11
    corpus=Corpus(tmp_path/"corpus")
    for number in range(1,12):
        id=f"WORK-{number:03d}"
        assert corpus.list_paragraphs(id),id
        assert corpus.db.execute("SELECT count(*) FROM records WHERE document_id=?",(id,)).fetchone()[0]>0
        assert (tmp_path/"corpus"/"originals"/(id+".txt")).is_file()
        assert (tmp_path/"corpus"/"extracted"/(id+".txt")).is_file()
    # Bracketed paragraph numbering belongs to the author's text, not an
    # accidental SQL uniqueness condition.
    election=(tmp_path/"corpus"/"originals"/"WORK-002.txt").read_text(encoding="utf-8")
    latin=(tmp_path/"corpus"/"originals"/"WORK-005.txt").read_text(encoding="utf-8")
    assert "[1138]" in election
    assert "[190]" in latin
    again=import_published_library(ROOT/"published_library",tmp_path/"corpus")
    assert again["source_sha256"]==report["source_sha256"]

def test_fresh_lexicon_and_owner_axioms_derive_from_uploaded_books(tmp_path):
    report=import_published_library(ROOT/"published_library",tmp_path/"corpus")
    summary=prepare(tmp_path/"corpus",tmp_path/"inputs")
    assert summary["documents"]==11
    assert summary["vocab_size"]>256
    c=Constitution.load(tmp_path/"inputs"/"governance.json")
    assert c.axioms==OWNER_THREE_AXIOMS
    tokenizer=ByteBPE.load(tmp_path/"inputs"/"tokenizer.json")
    assert tokenizer.digest()==summary["tokenizer_sha256"]
    assert len((tmp_path/"inputs"/"heldout.txt").read_text(encoding="utf-8"))>100
