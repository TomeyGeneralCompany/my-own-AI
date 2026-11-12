import json

import pytest

from my_own_ai.cli import main
from my_own_ai.neural_judge import NeuralJudge, check_split, evaluate_judge, load_cases, parse_cases, train_judge
from my_own_ai.retrieval import CorpusIndex, build_index


CANDIDATES = [
    "The late report delays the keeper's response because the keeper must wait for it.",
    "An immediate signal triggers the keeper's response, so the report does not delay it.",
]
PREMISES = [
    "The keeper acts only after reading the report. A late report delays the keeper's response.",
    "The keeper responds immediately to a separate signal. The report is only a record of the response.",
    "The administrator listed file dates on a schedule. The list supplies no decision.",
]


def examples(partition, roles):
    rows = []
    for index, role in enumerate(roles):
        group = f"{partition}_{index}"
        question = f"Will a late report postpone the response of the {role}?"
        for number, evidence in enumerate(PREMISES):
            rows.append({"id": f"{group}_{number}", "group": group, "question": question,
                         "evidence": [evidence], "candidates": CANDIDATES,
                         "accepted": number if number < 2 else None})
    return rows


def save(root, filename, rows):
    path = root / filename
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


@pytest.fixture
def trained(tmp_path):
    train = save(tmp_path, "train.json", examples("training", ["lock keeper", "yard attendant", "depot steward"]))
    validation = save(tmp_path, "validation.json", examples("validation", ["station recorder"]))
    test = save(tmp_path, "test.json", examples("testing", ["dock manager"]))
    model = tmp_path / "judge.pt"
    report = train_judge(train, validation, model, held_out=test, steps=100, threads=1)
    assert report["hardware"]["device"] == "cpu"
    return model, test


def test_evidence_changes_neural_choice_and_unresolved_example(trained):
    path, test = trained
    judge = NeuralJudge.load(path, threads=1)
    cases = load_cases(test)
    report = evaluate_judge(cases, judge, include_scores=True)
    assert report["passed"] == 3
    assert report["counterfactual_passed"] == report["counterfactual_total"] == 1
    assert report["unresolved_correct"] == report["unresolved_total"] == 1
    for case in cases[:2]:
        forward = judge.rank(case.question, case.evidence, case.candidates)
        reverse = judge.rank(case.question, case.evidence, tuple(reversed(case.candidates)))
        assert forward["selected"] == 1 - reverse["selected"]
    assert judge.rank(cases[0].question, PREMISES[:2], CANDIDATES)["selected"] is None


def test_cli_retrieves_evidence_and_writes_an_outline(trained, tmp_path, capsys):
    path, _ = trained
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "record.txt").write_text(PREMISES[0], encoding="utf-8")
    index = tmp_path / "index.sqlite"
    build_index(corpus, index)
    brief = save(tmp_path, "brief.json", {"question": "Will a late report postpone the keeper's response?",
                                         "candidates": CANDIDATES})
    status = main(["draft-neural-outline", "--checkpoint", str(path), "--index", str(index),
                   "--brief", str(brief), "--threads", "1"])
    result = json.loads(capsys.readouterr().out)
    assert status == 0
    assert result["status"] == "draft_outline" and result["thesis"] == CANDIDATES[0]
    with CorpusIndex(index) as db:
        assert result["evidence"] == db.search("keeper report")[0]


def test_rejects_bad_labels_and_leaking_splits(tmp_path):
    source = examples("original", ["depot steward"])
    with pytest.raises(ValueError, match="out of range"):
        parse_cases([{**source[0], "accepted": True}])
    with pytest.raises(ValueError, match="duplicate"):
        parse_cases([{**source[0], "candidates": ["same words", "Same words!"]}])
    with pytest.raises(ValueError, match="disjoint"):
        check_split(parse_cases(source), parse_cases(examples("other", ["depot steward"])))
    first = save(tmp_path, "first.json", source)
    second = save(tmp_path, "second.json", examples("other", ["another steward"]))
    with pytest.raises(ValueError, match="separate"):
        train_judge(first, second, first, steps=1)


def test_counterfactual_metric_requires_identical_task_and_changed_evidence():
    """Different questions or candidate orders must not inflate premise-switch coverage."""
    class StubJudge:
        def rank(self, question, evidence, candidates):
            return {"selected": 0, "scores": [1, 0], "evidence_positions": [0, 0]}

    base = examples("paired", ["lock keeper"])[:2]
    valid = parse_cases(base)
    assert evaluate_judge(valid, StubJudge())["counterfactual_total"] == 1

    changed_question = [dict(base[0]), dict(base[1])]
    changed_question[1]["question"] = "Does a completely different subject respond?"
    assert evaluate_judge(parse_cases(changed_question), StubJudge())["counterfactual_total"] == 0

    changed_candidates = [dict(base[0]), dict(base[1])]
    changed_candidates[1]["candidates"] = list(reversed(base[1]["candidates"]))
    changed_candidates[1]["accepted"] = 1
    assert evaluate_judge(parse_cases(changed_candidates), StubJudge())["counterfactual_total"] == 0

    same_evidence = [dict(base[0]), dict(base[1])]
    same_evidence[1]["evidence"] = list(base[0]["evidence"])
    assert evaluate_judge(parse_cases(same_evidence), StubJudge())["counterfactual_total"] == 0
