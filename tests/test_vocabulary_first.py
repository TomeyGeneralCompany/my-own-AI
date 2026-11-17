from my_own_ai import english_mastery as m
from my_own_ai.english_mastery_curriculum import Curriculum, STAGES
from types import SimpleNamespace

class Tokens:
    def encode(self, s):
        return list(s)

def curriculum():
    return Curriculum([("zebra","A striped animal."),("apple",'A fruit. "The apple fell from the tree."'),("bear","An animal.")],Tokens(),512)

def test_alphabetical_order_and_per_word_sentence():
    c=curriculum()
    assert STAGES[0]=="dictionary"
    assert [w for w,_ in c.records]==["apple","bear","zebra"]
    units=c.units("dictionary",0)
    assert units[-1]["usage_kind"]=="source_example"
    assert units[-1]["answer"]=="The apple fell from the tree."
    assert c.next(["dictionary",0,len(units)-1])==["dictionary",1,0]
    assert c.units("dictionary",1)[-1]["usage_kind"]=="definition_sentence"

def test_sentence_failure_blocks_advancement(monkeypatch):
    c=curriculum()
    unit=c.units("dictionary",0)[-1]
    monkeypatch.setattr(m,"generate",lambda *args:("wrong",True))
    ok,rows=m.assess_independently(SimpleNamespace(config=SimpleNamespace(context_size=512)),Tokens(),unit,{})
    assert not ok and not rows[0]["correct"]

def test_migration_archives_legacy_and_keeps_update_accounting():
    c=curriculum()
    p=m.new_progress(c,42,"source")
    p.pop("curriculum_version")
    p.update(cursor=["grammar",3,0],training_updates=5,phase="remediate",remediation=[["grammar",3,0]])
    p["passed_items"]["grammar"]=3
    new=m.migrate_curriculum(p,c,"source",47)
    assert new["cursor"]==["dictionary",0,0]
    assert new["legacy_curriculum"]["passed_items"]["grammar"]==3
    assert new["legacy_curriculum"]["remediation"]==[["grammar",3,0]]
    assert new["initial_step"]+new["training_updates"]==47
    assert not any(new["passed_items"].values())
    assert m.migrate_curriculum(new,c,"source",47) is new

def test_remediation_does_not_follow_hallucinations_or_advance_word():
    c=curriculum()
    p=m.new_progress(c,0,"source")
    m.fail_root(p,c,c.unit(p["cursor"]),"zebra bear")
    assert p["remediation"]==[["dictionary",0,0]]


def test_alphabetical_headwords_ignore_leading_punctuation_and_accents():
    c=Curriculum([
        ("zebra","An animal."), ("'Ecart'e","A card game."),
        ("apple","A fruit."), ("\u00c9clair","A pastry."),
        ("'abacus","A counting frame."),
    ],Tokens(),512)
    assert [word for word,_ in c.records] == [
        "'abacus","apple","'Ecart'e","\u00c9clair","zebra",
    ]
    # Traversal order does not rewrite any source headword.
    assert c.records[2][0] == "'Ecart'e"


def test_small_context_dictionary_chunks_cover_every_character():
    # A character tokenizer makes it possible to verify lossless chunking.
    definition="one two three four five six seven eight nine ten"
    c=Curriculum([("apple",definition)],Tokens(),256)
    recall=c.units("dictionary",0)[:-1]
    assert len(recall)>1
    assert all(len(u["answer"])<=24 for u in recall)
    assert " ".join(u["answer"] for u in recall)==definition


def test_remediation_checks_soon_and_preserves_independent_transfer():
    from my_own_ai.english_mastery import (
        REPETITIONS, record_partial, remediation_samples, transfer_question
    )
    assert REPETITIONS[-1] <= 32
    unit=dict(id="dictionary-0-1",
              question="Write the source text.", exam="Reproduce the entry.",
              answer="A small entry.")
    class Characters:
        def encode(self,text):
            return [ord(c) for c in text]
        def decode(self,tokens):
            return "".join(chr(t) for t in tokens)
    t=Characters()
    p={}
    for _ in range(4):
        record_partial(p,unit,"A wrong entry.",True,t)
    # Regression: varied teaching prompts may be trained, but never the
    # independent held-out transfer wording itself.
    samples=remediation_samples(p,unit,t,512)
    assert len(samples)>3
    from my_own_ai.language_training import _example
    held_out=_example(dict(unit,question=transfer_question(unit)),t,512)
    assert held_out not in samples
