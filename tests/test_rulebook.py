from pathlib import Path
import json

import pytest

from my_own_ai.cli import main
from my_own_ai.corpus_eval import evaluate_cases
from my_own_ai.modules import CheckpointStore, ModuleController, default_modules
from my_own_ai.reasoning import Stage, Status
from my_own_ai.retrieval import build_index
from my_own_ai.rulebook import NO_PROOF, Rulebook, RulebookProposer

EXAMPLE_BOOK = Path(__file__).parents[1] / "examples" / "inference_rules.json"


def make_index(root, documents):
    root.mkdir()
    for name, text in documents.items():
        (root / name).write_text(text, encoding="utf-8")
    index = root / "corpus.sqlite"
    build_index(root, index)
    return index


def run(book, index, question, terms, sources=("reference.txt",)):
    cases = [{"id": "sample", "question": question, "expected_sources": list(sources),
              "inference_terms": terms}]
    return evaluate_cases(index, cases, RulebookProposer(book), rulebook=book,
                          include_answers=True)["cases"][0]


@pytest.mark.parametrize(("premise", "consequence"), [
    ("The yard dispatcher acts only after reading the notice.", "notice delays"),
    ("The yard dispatcher responds immediately to a lamp and uses the notice only to log the response.",
     "notice does not delay"),
])
def test_same_question_follows_changed_evidence(tmp_path, premise, consequence):
    book = Rulebook.load(EXAMPLE_BOOK)
    index = make_index(tmp_path / "input", {"reference.txt": premise})
    result = run(book, index, "Will a late notice postpone the dispatcher’s response?", [consequence])
    assert result["passed"]
    assert premise in result["answer"]
    assert result["verified_citations"] == 1


def test_contradictory_evidence_abstains(tmp_path):
    book = Rulebook.load(EXAMPLE_BOOK)
    index = make_index(tmp_path / "input", {
        "reference.txt": "The dispatcher acts only after reading the notice.",
        "conflict.txt": "The dispatcher responds immediately to a lamp and uses the notice only to log the response.",
    })
    result = run(book, index, "Will a late notice postpone the dispatcher’s response?", ["notice"])
    assert result["status"] == Status.UNSUCCESSFUL
    assert result["verified_citations"] == 0
    assert result["evaluation_feedback"] == NO_PROOF


def test_unification_limits_authority_to_named_person_and_scope(tmp_path):
    book = Rulebook.load(EXAMPLE_BOOK)
    index = make_index(tmp_path / "input", {
        "reference.txt": "The parish steward has authority to issue market permits under the charter.",
    })
    assert run(book, index, "May the parish steward grant a market permit?", ["charter"])["passed"]
    assert not run(book, index, "May the parish steward grant a fishing permit?", ["charter"])["passed"]
    assert not run(book, index, "May the road inspector grant a market permit?", ["charter"])["passed"]


def test_deposit_without_attribution_does_not_establish_authorship(tmp_path):
    book = Rulebook.load(EXAMPLE_BOOK)
    index = make_index(tmp_path / "input", {
        "reference.txt": "The archive distributes files deposited by the archivist.",
    })
    result = run(book, index, "Who created the files that the archive distributes?", ["archivist"])
    assert not result["passed"] and result["evaluation_feedback"] == NO_PROOF


def test_source_scoped_premise_search_and_multiple_rule_steps(tmp_path):
    book = Rulebook({
        "version": 1,
        "extractors": [
            {"input": "question", "pattern": r"\bcontinuity\b", "fact": ["asks_continuity"]},
            {"input": "evidence", "pattern": "The left channel supplies the signal", "fact": ["has_feed"],
             "source": "architecture.txt"},
            {"input": "evidence", "pattern": "The spare channel is powered", "fact": ["spare_is_ready"],
             "source": "operations.txt"},
        ],
        "rules": [
            {"id": "coverage", "requires": [["has_feed"], ["spare_is_ready"]], "produces": ["redundant"]},
            {"id": "application", "requires": [["asks_continuity"], ["redundant"]], "produces": ["covered"],
             "answer": "Continuity is possible because the spare channel can carry the signal."},
        ],
    })
    index = make_index(tmp_path / "input", {
        "architecture.txt": "The left channel supplies the signal to the receiver.",
        "operations.txt": "The spare channel is powered during the handover.",
    })
    result = run(book, index, "Is continuity possible?", ["spare channel"],
                 sources=("architecture.txt", "operations.txt"))
    assert result["passed"] and result["verified_citations"] == 2
    assert result["evaluation_feedback"] == "Verified rules: coverage, application"
    no_goal = run(book, index, "Who paid for the spare channel?", ["spare channel"])
    assert not no_goal["passed"]


def test_controller_rechecks_proposed_claim_and_resumes_with_original_book(tmp_path):
    book = Rulebook.load(EXAMPLE_BOOK)
    index = make_index(tmp_path / "input", {
        "reference.txt": "The yard dispatcher acts only after reading the notice.",
    })

    class WrongFirst(RulebookProposer):
        def propose(self, request):
            proposal = super().propose(request)
            if request.stage == Stage.CONSTRUCT:
                return proposal.replace("delays the response", "speeds the response")
            return proposal

    directory = tmp_path / "state"
    with ModuleController(CheckpointStore(directory), default_modules(corpus_index=index, rulebook=book),
                          WrongFirst(book)) as core:
        core.switch("research")
        core.start("grounded_analysis", "Will a late notice postpone the dispatcher’s response?")
        core.step()
        core.step()
        failed = core.step()
        assert failed["status"] == Status.REQUIRES_CORRECTION
        assert failed["feedback"] == "The proposal differs from the verified rule derivation"

    with ModuleController(CheckpointStore(directory), default_modules(corpus_index=index)) as core:
        core.switch("research")
        with pytest.raises(ValueError, match="original rulebook"):
            core.step()
        assert core.progress() == failed

    with ModuleController(CheckpointStore(directory), default_modules(corpus_index=index, rulebook=book),
                          RulebookProposer(book)) as core:
        core.switch("research")
        core.step()
        assert core.step()["status"] == Status.COMPLETED

    with ModuleController(CheckpointStore(directory), default_modules()) as core:
        core.switch("research")
        assert core.progress()["status"] == Status.COMPLETED


def test_invalid_or_circular_rulebook_rejected():
    with pytest.raises(ValueError, match="circular"):
        Rulebook({"version": 1,
                  "extractors": [{"input": "question", "pattern": "x", "fact": ["observed"]}],
                  "rules": [
                      {"id": "first", "requires": [["other"]], "produces": ["derived"]},
                      {"id": "second", "requires": [["derived"]], "produces": ["other"]},
                  ]})
    with pytest.raises(ValueError, match="bound variables"):
        Rulebook({"version": 1,
                  "extractors": [{"input": "evidence", "pattern": "x", "fact": ["observed"]}],
                  "rules": [{"id": "out", "requires": [["observed"]], "produces": ["derived"],
                             "answer": "The answer would expose {unknown}."}]})


def test_cli_runs_rulebook_without_checkpoint(tmp_path, capsys):
    index = make_index(tmp_path / "input", {
        "reference.txt": "The yard dispatcher acts only after reading the notice.",
    })
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{"id": "scenario", "question": "Will a late notice delay a response?",
                                  "expected_sources": ["reference.txt"], "inference_terms": ["delay"]}]))
    status = main(["evaluate-corpus", "--index", str(index), "--cases", str(cases),
                   "--rulebook", str(EXAMPLE_BOOK), "--include-answers"])
    assert status == 0
    assert json.loads(capsys.readouterr().out)["passed"] == 1
