import json
import pytest
from ancestor.reasoning import load_reasoning_examples,reasoning_sample
from ancestor.corpus import Corpus
from ancestor.tokenizer import ByteBPE
from ancestor.governance import Constitution

def test_owner_approved_axiom_first_derivation(tmp_path):
    f=tmp_path/"manuscript.txt"
    f.write_text("1. Identity is a unity of self and other.",encoding="utf-8")
    corpus=Corpus(tmp_path/"data")
    corpus.ingest_text(f,"WORK","Test")
    examples=[{"id":"E","axiom_ids":[0],"premise_source_ids":["WORK:P-000001"],
        "question":"What follows?","derivation":"The stated principle is the premise.",
        "conclusion":"The conclusion follows from the stated premise.",
        "owner_approved":True}]
    path=tmp_path/"examples.json"
    path.write_text(json.dumps(examples))
    ex=load_reasoning_examples(path)[0]
    c=Constitution(("Identity is a unity of self and other.",))
    x,y,mask=reasoning_sample(ex,corpus,c,ByteBPE([]),context_length=1024)
    assert x.shape==y.shape==mask.shape
    assert mask.sum().item()>0
    examples[0]["owner_approved"]=False
    path.write_text(json.dumps(examples))
    with pytest.raises(PermissionError):
        load_reasoning_examples(path)
