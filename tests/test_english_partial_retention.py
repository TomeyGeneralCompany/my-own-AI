from my_own_ai.english_mastery import record_partial, correction_sample
from my_own_ai.tokenizer import EOS


class Characters:
    def encode(self, text):
        return [ord(c) for c in text]


def test_cumulative_retention_survives_new_error():
    tokenizer = Characters()
    unit = dict(id="example", question="Copy", exam="Copy", answer="v. t. p. p.")
    progress = {}
    first = record_partial(progress, unit, "v. t. p. b.", True, tokenizer)
    second = record_partial(progress, unit, "v. p. p.", True, tokenizer)
    assert set(first["retained"]) <= set(second["retained"])
    assert second["regressions"]
    assert set(second["regressions"]) <= set(second["missing"])
    assert progress["partial_regressions"] > 0
    sample = correction_sample(progress, unit, tokenizer, 128)
    assert sample is not None
    _, labels = sample
    assert sum(label != -100 for label in labels) == len(second["missing"])
    final = record_partial(progress, unit, unit["answer"], True, tokenizer)
    assert not final["missing"]
    assert not final["regressions"]
    assert correction_sample(progress, unit, tokenizer, 128) is None


def test_partial_records_are_checkpoint_serializable_and_question_specific():
    import json
    tokenizer = Characters()
    unit = dict(id="a", question="Copy A", exam="Copy A", answer="abc")
    progress = {}
    record_partial(progress, unit, "ab", False, tokenizer)
    resumed = json.loads(json.dumps(progress))
    record = record_partial(resumed, unit, "ac", True, tokenizer)
    assert 1 in record["regressions"]
    assert record["attempts"] == 2
    other = dict(unit, exam="Copy B")
    assert correction_sample(resumed, other, tokenizer, 128) is None
