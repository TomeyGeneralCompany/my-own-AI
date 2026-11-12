"""PR21: authentic numbered-paragraph instruction and immutable ancestry."""
import json
from pathlib import Path
import pytest
from my_own_ai.reasoning_curriculum import extract_lessons, create_reasoning_lessons
from my_own_ai.published_corpus import prepare_published_corpus

ROOT=Path(__file__).resolve().parents[1]


def test_grounded_definitions_consequences_and_attached_math():
    passage=("[5] Identity is a unity of self and other. "
             "The relation is internal to identity. Therefore relation changes the actuality of self.\n"
             "A = B + C\n\n(1)\n")
    rows=extract_lessons(passage,source_id="p000005")
    assert {x["kind"] for x in rows} >= {"definition","derivation","mathematical","proposition"}
    normalized=" ".join(passage.split())
    for row in rows:
        assert row["answer"] in normalized
        assert row["paragraph"]==5
        assert row["source"]=="p000005"
    assert any(x["answer"]=="A = B + C" for x in rows)


def test_source_mathematical_term_definitions_are_retrieved_not_invented():
    source=(ROOT/"published_books/communal-nature-of-commerce-revised-second-edition.txt").read_text()
    from my_own_ai.numbered_paragraphs import numbered_paragraphs
    terms=next(x["text"] for x in numbered_paragraphs(source) if x["number"]==6)
    rows=extract_lessons(terms,source_id="p000006")
    math=[x for x in rows if x["kind"]=="mathematical_interpretation"]
    assert math
    assert any("X(i,t)" in x["question"] for x in math)
    assert all(x["answer"] in " ".join(terms.split()) for x in math)


def test_full_author_manuscript_yields_disjoint_real_reasoning_lessons(tmp_path):
    corpus=tmp_path/"prepared"
    summary=prepare_published_corpus(ROOT,ROOT/"published_books/manifest.json",corpus)
    assert summary["chunk_unit"]=="mixed_structured"
    out=tmp_path/"lessons"
    info=create_reasoning_lessons(corpus,out,limit_per_split=1000)
    assert info["format"]=="numbered_reasoning_lessons_v1"
    split_questions=[]
    for split in ("train","validation","test"):
        rows=json.loads((out/f"{split}.json").read_text())
        assert rows
        assert all(set(r)=={"id","question","answer"} for r in rows)
        assert len({r["id"] for r in rows})==len(rows)
        split_questions.append({r["question"].casefold() for r in rows})
    assert not (split_questions[0]&split_questions[1])
    assert not (split_questions[0]&split_questions[2])
    assert not (split_questions[1]&split_questions[2])
    assert all(info["splits"][k]["total"]>0 for k in ("train","validation","test"))


def test_no_source_text_no_manufactured_answers(tmp_path):
    assert extract_lessons("[2] hello.",source_id="source") == []
    with pytest.raises((ValueError,FileNotFoundError)):
        create_reasoning_lessons(ROOT,tmp_path/"unused")


def test_legacy_training_unaffected_and_new_curriculum_explicit():
    source=(ROOT/"scripts/ancestor_core.py").read_text()
    assert 'choices=("continuation", "reasoning")' in source
    workflow=(ROOT/".github/workflows/ancestor-train.yml").read_text()
    assert 'parent_generation:' in workflow
    assert 'curriculum:' in workflow
    assert 'ANCESTOR_TAG="ancestor-' in workflow
    assert '--curriculum "$CURRICULUM"' in workflow
