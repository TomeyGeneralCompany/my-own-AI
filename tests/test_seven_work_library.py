"""Real seven-work library integrity, equation-aware segmentation and held-out test."""
from pathlib import Path
import json
from hashlib import sha256

from my_own_ai.published_corpus import validate_published_manifest,prepare_published_corpus
from my_own_ai.structured_books import structured_units
from my_own_ai.reasoning_curriculum import create_reasoning_lessons
from scripts.ancestor_cross_work import make_suite

ROOT=Path(__file__).resolve().parents[1]


def test_all_seven_authored_works_are_verified_and_losslessly_segmented():
    manifest=validate_published_manifest(ROOT,ROOT/"published_books/manifest.json")
    assert len(manifest["sources"])==7
    assert {s["id"] for s in manifest["sources"]} == {
        "commerce","calvin-commentary","electromagnetic-void",
        "biospiritual-realism","patristic-cybernetic-sola-fide",
        "ten-commandments-exposition","noetic-cybernetics-melito"}
    for source in manifest["sources"]:
        assert len(source["text"].split())>=1000
        assert sha256(source["text"].encode()).hexdigest()==source["sha256"]
        assert source["source_parts"]
        if source["chunking"]=="structured_units":
            parts=structured_units(source["text"])
            assert len(parts)>=10
            assert "".join(part["text"] for part in parts)==source["text"]


def test_equations_adhere_to_adjacent_prose_not_separate_empty_units():
    text=("I. The System\n\nThe state is formally defined in this passage:\n\n"
          "X(i,t) = A + B\n\n(1)\n\n"
          "The next state is computed from the preceding state.\n\n"
          "II. Conclusions\n\nThe relation is operational.")
    units=structured_units(text)
    assert "".join(u["text"] for u in units)==text
    assert any("X(i,t) = A + B" in u["text"] and "(1)" in u["text"] for u in units)
    assert all(u["text"].strip() for u in units)


def test_all_seven_works_train_and_hold_out_independently(tmp_path):
    prepared=tmp_path/"prepared"
    result=prepare_published_corpus(ROOT,ROOT/"published_books/manifest.json",prepared)
    assert result["policy"]=="per_work_unit_disjoint"
    assert result["chunk_unit"]=="mixed_structured"
    assert len(result["sources"])==7
    for split in ("train","validation","test"):
        files=[p.name for p in (prepared/split).glob("*.txt")]
        for source in result["sources"]:
            assert any(f.startswith(source["id"]+"_") for f in files),(split,source["id"])
    # Complete logical units of original [0]..[536] still remain available.
    assert result["reading_ledgers"]["commerce"]["numbered_paragraphs"]==537
    hashes={}
    for split in ("train","validation","test"):
        for p in (prepared/split).glob("*.txt"):
            txt=" ".join(p.read_text().casefold().split())
            assert sha256(txt.encode()).hexdigest() not in hashes
            hashes[sha256(txt.encode()).hexdigest()]=split
    assert sum(result["duplicate_source_units_excluded"].values())>=0


def test_cross_work_heldout_questions_have_provenance_for_every_book(tmp_path):
    prepared=tmp_path/"prepared"
    prepare_published_corpus(ROOT,ROOT/"published_books/manifest.json",prepared)
    lessons=tmp_path/"lessons"
    result=create_reasoning_lessons(prepared,lessons,limit_per_split=5000)
    assert all(result["splits"][s]["total"]>0 for s in ("train","validation","test"))
    cases=make_suite(lessons,{
        "commerce","calvin-commentary","electromagnetic-void",
        "biospiritual-realism","patristic-cybernetic-sola-fide",
        "ten-commandments-exposition","noetic-cybernetics-melito"},questions_per_book=2)
    assert len(cases)==14
    assert len({x["id"] for x in cases})==14
    assert all(x["source_unit"].startswith(x["book"]+"_") for x in cases)
    trains=json.loads((lessons/"train.json").read_text())
    tests=json.loads((lessons/"test.json").read_text())
    assert not {r["question"] for r in trains}&{r["question"] for r in tests}
