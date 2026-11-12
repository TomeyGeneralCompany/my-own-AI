import sqlite3

import pytest

from my_own_ai.functional_modules import grounded_evidence
from my_own_ai.corpus_eval import evaluate_cases, validate_cases
from my_own_ai.modules import CheckpointStore, ModuleController, default_modules
from my_own_ai.reasoning import ProposalRequest, Stage, Status, _grounded_prompt
from my_own_ai.retrieval import CorpusIndex, PASSAGE_LIMIT, build_index, digest


class Engine:
    def __init__(self, *proposals):
        self.proposals = iter(proposals)
        self.requests = []

    def propose(self, request):
        self.requests.append(request)
        return next(self.proposals)


@pytest.fixture
def index(tmp_path):
    folder = tmp_path / "corpus"
    folder.mkdir()
    (folder / "beacon.txt").write_text("The beacon keeper signals hazards before clearance.\n")
    (folder / "dispatch.md").write_text("The ferry dispatcher records a reason for every reroute.\n")
    path = tmp_path / "corpus.sqlite"
    assert build_index(folder, path) == {"documents": 2, "passages": 2}
    return path


def test_index_is_deterministic_read_only_and_recovers_from_failed_rebuild(index, tmp_path):
    with CorpusIndex(index) as corpus:
        results = corpus.search('beacon (unknown: " OR * clearance')
        assert results == corpus.search('beacon (unknown: " OR * clearance')
        assert results[0]["source"] == "beacon.txt"
        assert results[0]["locator"] == "line 1, passage 1"
        assert results[0]["sha256"] == digest(results[0]["text"])
        assert corpus.search("the") == []
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            corpus.db.execute("DELETE FROM passages")
    with pytest.raises(RuntimeError, match="closed"):
        corpus.search("beacon")
    with pytest.raises(ValueError, match="overlaps"):
        build_index(tmp_path / "corpus", tmp_path / "corpus/beacon.txt")
    (tmp_path / "corpus/broken.md").write_bytes(b"\xff")
    with pytest.raises(UnicodeError):
        build_index(tmp_path / "corpus", index)
    with CorpusIndex(index) as reopened:
        assert reopened.search("beacon") == results
    assert not list(tmp_path.glob(".index-*"))


def test_docx_locators_and_chunking(tmp_path):
    from docx import Document

    folder = tmp_path / "books"
    folder.mkdir()
    document = Document()
    document.add_paragraph()
    document.add_paragraph("accordion " * 200)
    document.save(folder / "manual.docx")
    path = tmp_path / "catalog.sqlite"
    assert build_index(folder, path) == {"documents": 1, "passages": 3}
    with CorpusIndex(path) as corpus:
        hits = corpus.search("accordion")
    assert {hit["locator"] for hit in hits} == {f"paragraph 2, passage {n}" for n in (1, 2, 3)}
    assert all(len(hit["text"]) <= PASSAGE_LIMIT for hit in hits)


def test_grounded_operation_preserves_retrieved_evidence_and_requires_exact_quote(index, tmp_path):
    engine = Engine("Plan", "Premises:\n- [1] I invented this quote.\nInference: Incorrect inference text.",
                    "Premises:\n- [1] The beacon keeper signals hazards before clearance.\n"
                    "Inference: Waiting for clearance would delay the warning.")
    with ModuleController(CheckpointStore(tmp_path / "state"),
                          default_modules(corpus_index=index), engine) as core:
        core.switch("research")
        core.start("grounded_analysis", "Would clearance delay the beacon hazard warning?", inputs={"limit": 1})
        formulated = core.step()
        resource = core.active.index
        core.step()
        run = core.step()
        assert run["status"] == Status.REQUIRES_CORRECTION
        assert "Unverified premise quote" in run["feedback"]
        core.switch("writing")
        assert resource.db is None
        index.unlink()
        # Restore without a configured index. The checkpoint already contains the retrieved evidence.
    with ModuleController(CheckpointStore(tmp_path / "state"), default_modules(), engine) as core:
        core.switch("research")
        assert core.progress()["context"] == formulated["context"]
        core.step()
        assert core.step()["status"] == Status.COMPLETED
        assert engine.requests[-1].context == formulated["context"]


def test_held_out_rubric_stays_outside_model_input_and_can_fail_completed_operation(index):
    cases = [{"id": "unseen", "question": "Does beacon clearance risk changing when hazards are signaled?",
              "expected_sources": ["beacon.txt"], "inference_terms": ["acceptance_rubric_only"]}]
    engine = Engine("Plan", "Premises:\n- [1] The beacon keeper signals hazards before clearance.\n"
                    "Inference: A clearance requirement would delay the hazard warning.")
    report = evaluate_cases(index, cases, engine, limit=1)
    row = report["cases"][0]
    assert row["status"] == Status.COMPLETED and row["verified_citations"] == 1
    assert row["retrieved_expected"] == row["cited_expected"] == 1
    assert report["passed"] == 0
    assert all("acceptance_rubric_only" not in str(request) for request in engine.requests)
    cases[0]["inference_terms"] = ["delay", "warning"]
    passing = evaluate_cases(index, cases, Engine("Plan", "Premises:\n- [1] The beacon keeper signals "
                                                 "hazards before clearance.\nInference: Clearance "
                                                 "would delay the hazard warning."), limit=1)
    assert passing["passed"] == 1
    with pytest.raises(ValueError, match="duplicate"):
        validate_cases(cases * 2)


def test_grounded_prompt_contains_numbered_evidence():
    request = ProposalRequest("research", "grounded_analysis", Stage.CORRECT, "Use Premises and Inference.",
                              "Would clearance delay the warning?",
                              {"evidence": [{"text": "The beacon keeper signals hazards before clearance."}]},
                              "Plan", "Incorrect answer", "Quote was wrong")
    prompt = bytes(_grounded_prompt(request, 384))
    assert b"[1] The beacon keeper signals hazards before clearance." in prompt
    assert b"Quote was wrong" in prompt
    assert prompt.endswith(b"Correct: ")


def test_pdf_page_locator(tmp_path):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    folder = tmp_path / "corpus"
    folder.mkdir()
    writer = PdfWriter()
    writer.add_blank_page(200, 200)
    page = writer.add_blank_page(200, 200)
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): DictionaryObject({
            NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })})
    })
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 20 100 Td (Signal lattice) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    writer.write(folder / "reference.pdf")
    path = tmp_path / "catalog.sqlite"
    assert build_index(folder, path) == {"documents": 1, "passages": 1}
    with CorpusIndex(path) as corpus:
        match, = corpus.search("lattice")
    assert match["locator"] == "PDF page 2, passage 1"
    assert match["text"] == "Signal lattice"


def test_inference_cannot_add_unverified_reference():
    candidate = "Premises:\n- [1] A documented premise.\nInference: This conclusion relies on [2]."
    assert "without a verified premise" in grounded_evidence(candidate, [{"text": "A documented premise."}])[2]
