from my_own_ai.english_mastery import (
    record_partial, correction_sample, remediation_samples, transfer_question, correct,
)
from my_own_ai.language_training import _example
from my_own_ai.tokenizer import EOS


class Characters:
    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


def test_correction_uses_exam_not_teaching_prompt():
    t = Characters()
    unit = dict(id="dictionary-1-1", question="Write book", exam="Recall book", answer="v. t.")
    p = {}
    record_partial(p, unit, "v.", True, t)
    inputs, labels = correction_sample(p, unit, t, 256)
    exam_inputs, _ = _example(dict(unit, question=unit["exam"]), t, 256)
    teach_inputs, _ = _example(unit, t, 256)
    assert inputs == exam_inputs and inputs != teach_inputs
    assert any(x != -100 for x in labels)
    assert not correct(unit, "v.", True)


def test_insertion_is_corrected_even_when_all_targets_match():
    t = Characters()
    unit = dict(id="dictionary-1-1", question="Write book", exam="Recall book", answer="abc")
    p = {}
    record = record_partial(p, unit, "axbc", True, t)
    assert not record["missing"]
    assert record["distance"] == 1
    assert record["first_divergence"] == 1
    samples = remediation_samples(p, unit, t, 256)
    assert len(samples) == 2
    assert [x for x in samples[1][1] if x != -100] == [ord("b")]


def test_plateau_recovery_uses_wrong_prefix_without_training_transfer():
    t = Characters()
    unit = dict(id="dictionary-1-1", question="Write the text of Webster 1913 entry 2: Book, part 1 of 1.",
                exam="From memory, reproduce Webster 1913 entry 2: Book, section 1/1.", answer="abc")
    p = {}
    for _ in range(4):
        record = record_partial(p, unit, "axc", True, t)
    assert record["stagnant"] >= 3
    samples = remediation_samples(p, unit, t, 512)
    recovery = dict(unit, question=unit["exam"]+"\nYour attempt began: ax\nRestart and give the complete corrected answer.")
    assert _example(recovery, t, 512) in samples
    assert _example(dict(unit, question="Follow this instruction exactly: "+unit["question"]), t, 512) in samples
    held_out = _example(dict(unit, question=transfer_question(unit)), t, 512)
    assert held_out not in samples
    assert transfer_question(unit) != unit["exam"]
    assert transfer_question(unit) != unit["question"]
    assert correct(unit, "abc", True)
    assert not correct(unit, "abc", False)
    assert not correct(unit, "axc", True)


def test_migration_preserves_existing_partial_retention():
    import json
    t = Characters()
    unit = dict(id="dictionary-1-1", question="Write", exam="Recall", answer="abc")
    p = {}
    old = record_partial(p, unit, "ab", True, t)
    for k in ("distance","best_distance","stagnant","first_divergence","actual","ended"):
        old.pop(k)
    p = json.loads(json.dumps(p))
    new = record_partial(p, unit, "ac", True, t)
    assert set(old["retained"]) <= set(new["retained"])
    assert new["attempts"] == 2
    assert new["first_divergence"] == 1
    assert new["actual"] == "ac"
