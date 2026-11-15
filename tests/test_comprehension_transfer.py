"""Transfer assessment and actual checkpoint upgrade coverage."""
import json
import torch
from my_own_ai.comprehension_transfer import exercise, author_examples
from my_own_ai import transfer_diagnostics
from my_own_ai.language_training import train_language
from my_own_ai.tokenizer import ByteBPETokenizer

def test_paired_questions_are_distinct_with_same_answer(monkeypatch):
    for g in ("syntax", "reading", "qa", "conditional"):
        for i in range(4):
            a,b = [exercise(g,i,42,0,"probe",v) for v in (0,1)]
            assert a["question"] != b["question"]
            assert a["answer"] == b["answer"]
    monkeypatch.setattr(transfer_diagnostics, "answer_question",
                        lambda *a, **kw: {"answer": "not a valid answer"})
    result = transfer_diagnostics.paired_transfer("checkpoint","tokenizer")
    assert result["pairs"] == 16 and result["both_correct"] == 0

def test_actual_transfer_resume(tmp_path, monkeypatch):
    import my_own_ai.comprehension_curriculum as curriculum
    import my_own_ai.language_training as trainer
    monkeypatch.setattr(curriculum,"quotas",lambda rows: {
        "syntax":2,"reading":2,"qa":2,"conditional":2,
        "paraphrase":1,"coherence":1,"followup":1,"foundation":1})
    monkeypatch.setattr(trainer,"validation_lessons",lambda seed: [])
    tok=tmp_path/"tok.json"
    ByteBPETokenizer.train(["The porter carries a lantern."]*4,vocab_size=280).save(tok)
    tr,va,ckpt=tmp_path/"train.json",tmp_path/"val.json",tmp_path/"english.pt"
    tr.write_text(json.dumps([{"id":"a","question":"Correct this sentence: The porter carry a lantern.","answer":"The porter carries a lantern."},
                              {"id":"b","question":"Read: A lantern is here. What is here?","answer":"A lantern."}]))
    va.write_text(json.dumps([{"id":"v","question":"Name the object: a candle.","answer":"candle"}]))
    owner=tmp_path/"owner"
    owner.mkdir()
    (owner/"unit.txt").write_text("The relationship determines the operation of the system.")
    assert author_examples(owner)
    args=dict(tokenizer_path=tok,training=tr,validation=va,output=ckpt,phase="instruction",
              embed=16,heads=4,layers=1,context=512,batch_size=2,threads=1,
              checkpoint_every=4,curriculum_cycles=True,comprehension_cycles=True,tiered_memory=True)
    train_language(**args,steps=7)
    old=torch.load(ckpt,weights_only=True)
    result=train_language(**args,steps=14,resume=True,transfer_learning=True,author_language_folder=owner)
    state=torch.load(ckpt,weights_only=True)
    assert result["transfer_batches"] > 0 and result["author_excerpt_count"] == 1
    assert len(result["transfer_probe_loss"]) == 8
    assert state["transfer_state"]["introduced_at_cycle"] == old["curriculum_completed"]
    assert state["completed_steps"] == 14 and state["curriculum_completed"] == 2
    result2=train_language(**args,steps=21,resume=True,transfer_learning=True,author_language_folder=owner)
    assert result2["transfer_batches"] > result["transfer_batches"]
