"""Tiered memory must survive actual neural checkpoint continuation."""
import json
import torch
from my_own_ai import intellectual_memory as mem
from my_own_ai.comprehension_curriculum import GROUPS
from my_own_ai.language_training import train_language
from my_own_ai.tokenizer import ByteBPETokenizer


def test_consolidation_regression_and_context_budget():
    m = mem.new_memory([], 42, 10)
    for cycle in range(10, 13):
        mem.consolidate(m, dict.fromkeys(GROUPS, 0.2), cycle)
    assert all(r["tier"] == "established" for r in m["records"].values())
    losses = dict.fromkeys(GROUPS, 0.2)
    losses["conditional"] = 1.2
    mem.consolidate(m, losses, 13)
    assert m["records"]["conditional"]["tier"] == "consolidation"
    assert m["records"]["conditional"]["source"] == mem.source_examples("conditional", 42)
    assert "conditional" in {mem.replay_group(m, i) for i in range(16, 256, 16)}
    assert "negation" in mem.context_for(m, "Under this rule may work begin?")
    class Tokenizer:
        def encode(self, text):
            return list(text)
    q = "Under this rule may work begin?"
    assert mem.contextual_question(m, q, Tokenizer(), len(q)) == q
    assert mem.contextual_question(m, q, Tokenizer(), 512).endswith(q)
    mem.validate_memory(m, [], 42, 14)


def test_migration_and_resume_with_memory(tmp_path, monkeypatch):
    import my_own_ai.comprehension_curriculum as curriculum
    import my_own_ai.language_training as trainer
    monkeypatch.setattr(curriculum, "quotas", lambda rows: {
        "syntax":2,"reading":2,"qa":2,"conditional":2,
        "paraphrase":1,"coherence":1,"followup":1,"foundation":1})
    monkeypatch.setattr(trainer, "validation_lessons", lambda seed: [])
    tok = tmp_path / "tok.json"
    ByteBPETokenizer.train(["The porter carries a lantern."] * 4, vocab_size=280).save(tok)
    rows = [{"id":"a","question":"Correct this sentence: The porter carry a lantern.",
             "answer":"The porter carries a lantern."},
            {"id":"b","question":"Read: A lantern is here. What is here?","answer":"A lantern."}]
    tr, va = tmp_path/"train.json", tmp_path/"val.json"
    tr.write_text(json.dumps(rows))
    va.write_text(json.dumps([{"id":"v","question":"Name the object: a candle.","answer":"candle"}]))
    args = dict(tokenizer_path=tok, training=tr, validation=va, phase="instruction",
                embed=16, heads=4, layers=1, context=256, batch_size=2, threads=1,
                checkpoint_every=4, curriculum_cycles=True, comprehension_cycles=True)
    a, b = tmp_path/"a.pt", tmp_path/"b.pt"
    for p in (a,b):
        train_language(**args, output=p, steps=7)
    train_language(**args, output=a, steps=14, resume=True, tiered_memory=True)
    train_language(**args, output=a, steps=21, resume=True, tiered_memory=True)
    train_language(**args, output=b, steps=21, resume=True, tiered_memory=True)
    x,y = (torch.load(p, weights_only=True) for p in (a,b))
    assert x["completed_steps"] == y["completed_steps"] == 21
    assert x["curriculum_completed"] == y["curriculum_completed"] == 3
    assert x["intellectual_memory"] == y["intellectual_memory"]
    assert x["intellectual_memory"]["source_archive"] == rows
    assert x["intellectual_memory"]["first_cycle"] == 1
    assert x["intellectual_memory"]["covered_through"] == 2
    assert all(torch.equal(x["state_dict"][k], y["state_dict"][k]) for k in x["state_dict"])
