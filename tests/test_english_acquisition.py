"""General English training is broad, deterministic and isolated from exams."""
from collections import Counter
from my_own_ai.english_acquisition import drills


def test_broad_original_english_teaching_has_all_exam_skills():
    train = drills("train")
    val = drills("validation")
    assert len(train) >= 5000
    assert len(val) >= 100
    assert len({x["id"] for x in train}) == len(train)
    assert len({x["id"] for x in val}) == len(val)
    assert {x["question"] for x in train}.isdisjoint({x["question"] for x in val})
    assert {x["answer"] for x in train if x["answer"] not in ("yes","no")}
    assert sum(x["answer"]=="yes" for x in train) > 50
    assert sum(x["answer"]=="no" for x in train) > 50
    assert all(set(row)=={"id","question","answer"} for row in train+val)


def test_stable_questions_not_imported_from_frozen_gauntlet():
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    exam=json.loads((root/"examples/english/gauntlet.json").read_text())
    frozen={x["question"].casefold() for x in exam}
    assert not ({x["question"].casefold() for x in drills("train")} & frozen)
    assert not ({x["question"].casefold() for x in drills("validation")} & frozen)
    assert drills("train")==drills("train")
