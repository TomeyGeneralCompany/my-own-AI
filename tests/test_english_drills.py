"""English drill integrity and authored-reasoning preservation."""
import json
from pathlib import Path
import pytest

from my_own_ai.english_drills import english_drills, augment_grounded_curriculum


def test_drills_are_distinct_and_grammatical():
    train, validation = english_drills("train"), english_drills("validation")
    assert len(train) == 80 and len(validation) == 40
    assert len({row["id"] for row in train + validation}) == 120
    assert {r["question"].casefold() for r in train}.isdisjoint(
        {r["question"].casefold() for r in validation})
    assert {r["answer"] for r in train}.isdisjoint(
        {r["answer"] for r in validation})
    assert all(set(row) == {"id", "question", "answer"} for row in train + validation)


def test_authored_material_retained(tmp_path):
    examples = {
        "train": [{"id":"author-train", "question":"What is constitutive of identity?",
                   "answer":"The unity of self and other."}],
        "validation": [{"id":"author-val", "question":"What is the order of communion?",
                        "answer":"Communion is hierarchical."}],
    }
    for split, rows in examples.items():
        (tmp_path / f"{split}.json").write_text(json.dumps(rows))
    before = {k: v[0].copy() for k,v in examples.items()}
    result = augment_grounded_curriculum(tmp_path)
    for split in examples:
        merged = json.loads((tmp_path / f"{split}.json").read_text())
        assert merged[0] == before[split]
        assert result[split]["authored_lessons"] == 1
        assert result[split]["english_drills"] == len(english_drills(split))


def test_no_silent_removal_of_reasoning_lessons(tmp_path):
    for split in ("train", "validation"):
        (tmp_path / f"{split}.json").write_text("[]")
    with pytest.raises(ValueError, match="owner"):
        augment_grounded_curriculum(tmp_path)


def test_unknown_partition_fails():
    with pytest.raises(ValueError):
        english_drills("test")
