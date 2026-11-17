from types import SimpleNamespace
from my_own_ai import english_mastery as mastery
from my_own_ai.english_mastery_curriculum import Curriculum, STAGES
from my_own_ai.stage_controller import initial_ledger, advance_english_mastery, ENGLISH, READ
from my_own_ai.webster1913 import BLOB


class Characters:
    def encode(self, text):
        return list(text)


MODEL = SimpleNamespace(config=SimpleNamespace(context_size=512))
UNIT = dict(id="grammar-0", question="Repair the sentence", exam="Correct this sentence", answer="The dog runs.")


def test_failed_transfer_blocks_mastery(monkeypatch):
    outputs = iter([("The dog runs.", True), ("ter ter ter", False)])
    monkeypatch.setattr(mastery, "generate", lambda *args: next(outputs))
    p = {}
    ok, rows = mastery.assess_independently(MODEL, Characters(), UNIT, p)
    assert not ok and rows[0]["correct"] and not rows[1]["correct"]
    assert p["transfer_correct"] == 0


def test_failed_recall_cannot_be_replaced_with_memory(monkeypatch):
    monkeypatch.setattr(mastery, "generate", lambda *args: ("wrong", True))
    p = {"verified_memory": {UNIT["exam"]: UNIT["answer"]}}
    ok, rows = mastery.assess_independently(MODEL, Characters(), UNIT, p)
    assert not ok and len(rows) == 1 and rows[0]["origin"] == "neural"


def test_composition_requires_changed_facts(monkeypatch):
    c = Curriculum([("dog","animal")], Characters(), 512)
    unit = c.unit(["composition",0,0])
    assert unit["composition_practice"]["answer"] != unit["composition_transfer"]["answer"]
    outputs = iter([(unit["answer"],True),(unit["answer"],True),
                    (unit["composition_practice"]["answer"],True)])
    monkeypatch.setattr(mastery,"generate",lambda *args:next(outputs))
    ok, rows=mastery.assess_independently(MODEL,Characters(),unit,{})
    assert not ok and rows[-1]["kind"]=="composition"


def test_migration_preserves_pending_remediation_and_learned_credit():
    c=Curriculum([("dog","animal")],Characters(),512)
    p=mastery.new_progress(c,42,BLOB)
    p["phase"]="remediate"
    p["remediation"]=[["dictionary",0,0],["grammar",0,0]]
    p["remaining"]=19
    p["partial_answers"]={"old":{"retained":[0]}}
    mastery.migrate_assessment(p,42)
    assert p["assessment_version"]==2
    assert p["remaining"]==19 and len(p["remediation"])==2
    assert p["partial_answers"]["old"]["retained"]==[0]
    assert p["initial_step"]==42


def test_graduation_rejects_old_or_incomplete_independent_evidence():
    sha="a"*64
    ledger=initial_ledger("b"*64,["book"],"g0002")
    ledger["english_fundamentals_passed"]=True
    totals=dict(grammar=384,dictionary=100000,usage=100000,reading=256,composition=128)
    report=dict(format="english_mastery_v1",checkpoint_sha256=sha,source=BLOB,
                totals=totals,passed_items=totals,cumulative_items=totals,
                stage_passes={s:1 for s in STAGES},cumulative_started=42,
                completed_updates=42,qualified=True,phase="complete",cursor=None,
                remediation_pending=0)
    assert not advance_english_mastery(ledger,report,checkpoint_sha256=sha)
    report.update(assessment_version=2,independent_cumulative_units=0)
    assert not advance_english_mastery(ledger,report,checkpoint_sha256=sha)
    assert ledger["stage"]==ENGLISH
    report["independent_cumulative_units"]=sum(totals.values())
    report.update(curriculum_version=3,word_sentence_correct=totals["dictionary"])
    assert advance_english_mastery(ledger,report,checkpoint_sha256=sha)
    assert ledger["stage"]==READ
