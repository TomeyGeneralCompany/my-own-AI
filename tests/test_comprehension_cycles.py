"""Comprehension-only tests: coverage, retention, and original neural resume."""
import json
import torch
from my_own_ai.comprehension_curriculum import (lesson_groups, cycle_size, fixed_group,
    validation_lessons, quotas, CORE, _row)
from my_own_ai.comprehension_diagnostics import retention
from my_own_ai.english_drills import english_drills
from my_own_ai.english_acquisition import drills
from my_own_ai.language_training import train_language
from my_own_ai.tokenizer import ByteBPETokenizer
from collections import Counter


def test_complete_balanced_novel_curriculum():
    fixed=english_drills("train")+drills("train")
    generated,order=lesson_groups(fixed,0,42)
    later,later_order=lesson_groups(fixed,1,42)
    assert sorted(order)==list(range(cycle_size(fixed)))
    assert (generated,order)==lesson_groups(fixed,0,42)
    assert generated!=later and order!=later_order
    counts=Counter(fixed_group(x) for x in fixed)
    counts.update(quotas(fixed))
    assert max(counts[g] for g in CORE)-min(counts[g] for g in CORE)<=1
    assert counts["foundation"]>=256
    assert {r["question"] for r in generated}.isdisjoint(r["question"] for r in validation_lessons(42))


def test_conditional_counterfactual_pairs():
    for i in range(0,40,2):
        yes=_row("conditional",0,i,42,"train")
        no=_row("conditional",0,i+1,42,"train")
        assert yes["answer"]=="yes" and no["answer"]=="no"
        prefix, fact, suffix = yes["question"].rsplit(" is ready.", 1)[0], " is ready.", yes["question"].rsplit(" is ready.", 1)[1]
        assert prefix + " is not ready." + suffix == no["question"]


def test_retention_distinguishes_forgetting_and_acquisition():
    def report(bits):
        return {"raw_results":{"cases":[{"id":str(i),"category":"syntax","exact_match":b}
                                          for i,b in enumerate(bits)]}}
    x=retention(report([True,True,False,False]),report([True,False,True,False]),3)
    assert (x["retained"],x["lost"],x["gained"],x["best_observed_correct"])==(1,1,1,3)


def test_comprehension_checkpoint_resume_preserves_optimizer_and_cycle(tmp_path,monkeypatch):
    import my_own_ai.comprehension_curriculum as curriculum
    import my_own_ai.language_training as trainer
    def small_quotas(rows):
        return {"syntax":2,"reading":2,"qa":2,"conditional":2,
                "paraphrase":1,"coherence":1,"followup":1,"foundation":1}
    monkeypatch.setattr(curriculum,"quotas",small_quotas)
    monkeypatch.setattr(trainer,"validation_lessons",lambda seed: [])
    tok=tmp_path/"tokenizer.json"
    ByteBPETokenizer.train(["The porter carries the lantern."]*4,vocab_size=280).save(tok)
    tr=tmp_path/"train.json"
    va=tmp_path/"validation.json"
    tr.write_text(json.dumps([{"id":"a","question":"Correct this sentence: The porter carry a lantern.","answer":"The porter carries a lantern."},
                              {"id":"b","question":"Read: A lantern is here. What is here?","answer":"A lantern."}]))
    va.write_text(json.dumps([{"id":"v","question":"Name the object: a candle.","answer":"candle"}]))
    args=dict(tokenizer_path=tok,training=tr,validation=va,phase="instruction",
              embed=16,heads=4,layers=1,context=256,batch_size=2,threads=1,
              checkpoint_every=4)
    a,b=tmp_path/"a.pt",tmp_path/"b.pt"
    for path in (a,b):
        train_language(**args,output=path,steps=2)
    old=torch.load(a,weights_only=True)
    kw=dict(curriculum_cycles=True,allow_curriculum_migration=True,comprehension_cycles=True)
    result=train_language(**args,**kw,output=a,steps=9,resume=True)
    assert result["curriculum_cycles_completed"]==1
    assert result["curriculum_cursor"]==0
    train_language(**args,**kw,output=a,steps=16,resume=True)
    train_language(**args,**kw,output=b,steps=16,resume=True)
    x,y=(torch.load(p,weights_only=True) for p in (a,b))
    assert x["completed_steps"]==y["completed_steps"]==16
    assert x["curriculum_lineage"][0]["migrated_at_update"]==old["completed_steps"]
    assert x["comprehension_schedule"]==y["comprehension_schedule"]
    assert all(torch.equal(x["state_dict"][k],y["state_dict"][k]) for k in x["state_dict"])
