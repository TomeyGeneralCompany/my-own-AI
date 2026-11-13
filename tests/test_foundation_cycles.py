"""Coverage, exact resume, migration and exam separation for foundation education."""
import json
import torch
import pytest
from my_own_ai.english_acquisition import fundamentals
from my_own_ai.tokenizer import ByteBPETokenizer
from my_own_ai.language_training import train_language


def fixtures(tmp_path):
    words = ("river", "stone", "tower", "garden", "lighthouse")
    tokenizer = ByteBPETokenizer.train(words * 4, vocab_size=280)
    tokenizer_path = tmp_path / "tokens.json"
    tokenizer.save(tokenizer_path)
    train = tmp_path / "train.json"
    valid = tmp_path / "valid.json"
    train.write_text(json.dumps([
        {"id":f"t{i}", "question":f"Copy: {word}", "answer":word}
        for i,word in enumerate(words[:4])]))
    valid.write_text(json.dumps([
        {"id":"v1", "question":"Copy: garden?", "answer":"garden"}]))
    common = dict(tokenizer_path=tokenizer_path, training=train, validation=valid,
                  phase="instruction", embed=16, heads=4, layers=1, context=96,
                  batch_size=2, seed=7, threads=1, lr=.001, checkpoint_every=1)
    return common,train,valid


def test_complete_coverage_and_exact_resume(tmp_path):
    torch.set_num_threads(1)
    args,_,_ = fixtures(tmp_path)
    whole = tmp_path / "whole.pt"
    resume = tmp_path / "resume.pt"
    a = train_language(**args, output=whole, steps=3, curriculum_cycles=True)
    assert a["curriculum_cycles_completed"] == 2
    assert a["curriculum_cursor"] == 0
    assert a["completed_steps"] == 4
    train_language(**args, output=resume, steps=2, curriculum_cycles=True)
    b = train_language(**args, output=resume, steps=3, resume=True, curriculum_cycles=True)
    assert b["completed_steps"] == 4
    one = torch.load(whole, weights_only=True)
    two = torch.load(resume, weights_only=True)
    assert one["curriculum_completed"] == two["curriculum_completed"]
    assert one["curriculum_order"] == two["curriculum_order"]
    assert all(torch.equal(one["state_dict"][name],two["state_dict"][name])
               for name in one["state_dict"])


def test_legacy_migration_is_explicit_and_preserves_optimizer(tmp_path):
    args,train,_ = fixtures(tmp_path)
    model = tmp_path / "model.pt"
    train_language(**args,output=model,steps=2)
    old = torch.load(model,weights_only=True)
    with pytest.raises(ValueError,match="migration"):
        train_language(**args,output=model,steps=4,resume=True,curriculum_cycles=True)
    # Add one new lesson; this MUST NOT silently invalidate the checkpoint.
    rows = json.loads(train.read_text())
    rows.append({"id":"added","question":"Copy: river now","answer":"river"})
    train.write_text(json.dumps(rows))
    updated = train_language(**args,output=model,steps=4,resume=True,
                             curriculum_cycles=True,allow_curriculum_migration=True)
    new = torch.load(model,weights_only=True)
    assert updated["curriculum_migrated"]
    assert new["curriculum_lineage"][0]["migrated_at_update"] == old["completed_steps"]
    assert new["completed_steps"] > old["completed_steps"]
    assert new["optimizer"]["state"]
    assert all(k in new["state_dict"] for k in old["state_dict"])
    assert new["curriculum_cursor"] == 0
    # A second migration is not silently accepted.
    rows.append({"id":"extra","question":"Copy: tower now","answer":"tower"})
    train.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match="migration"):
        train_language(**args,output=model,steps=new["completed_steps"]+2,
                       resume=True,curriculum_cycles=True,allow_curriculum_migration=True)


def test_compositional_curriculum_is_disjoint_from_unseen_examination():
    train=fundamentals("train")
    val=fundamentals("validation")
    exam=fundamentals("test")
    assert len(train)>288
    assert len(train)==len({x["question"] for x in train})
    assert {x["question"] for x in train}.isdisjoint(x["question"] for x in exam)
    assert {x["answer"] for x in train}.isdisjoint(x["answer"] for x in exam)
    assert {x["question"] for x in val}.isdisjoint(x["question"] for x in exam)


def test_dynamic_character_copy_generates_new_examples_each_cycle():
    from my_own_ai.copy_curriculum import synthetic_copy_rows
    first=synthetic_copy_rows(0, 50)
    second=synthetic_copy_rows(1, 50)
    assert first == synthetic_copy_rows(0, 50)
    assert {x["question"] for x in first}.isdisjoint(x["question"] for x in second)
    assert all(x["answer"] in x["question"] for x in first)
    heldout={word for triple in __import__("my_own_ai.english_acquisition",
                       fromlist=["FOUNDATION_TEST"]).FOUNDATION_TEST for word in triple}
    assert {x["answer"] for x in first+second}.isdisjoint(heldout)


def test_dynamic_curriculum_exact_resume(tmp_path):
    args,_,_=fixtures(tmp_path)
    single=tmp_path/"dynamic-single.pt"
    split=tmp_path/"dynamic-split.pt"
    a=train_language(**args,output=single,steps=5,curriculum_cycles=True,dynamic_copy=True)
    assert a["curriculum_examples_total"]==8
    assert a["curriculum_cycles_completed"]==2
    assert a["completed_steps"]==8
    train_language(**args,output=split,steps=4,curriculum_cycles=True,dynamic_copy=True)
    b=train_language(**args,output=split,steps=5,resume=True,
                     curriculum_cycles=True,dynamic_copy=True)
    assert b["completed_steps"]==8
    m1=torch.load(single,weights_only=True)
    m2=torch.load(split,weights_only=True)
    assert m1["copy_generator_version"]==m2["copy_generator_version"]
    assert m1["curriculum_order"]==m2["curriculum_order"]
    assert all(torch.equal(m1["state_dict"][k],m2["state_dict"][k])
               for k in m1["state_dict"])
