from pathlib import Path
import pytest
from ancestor.tokenizer import ByteBPE
from ancestor.governance import Constitution,INTELLECTUAL_CONSTITUTION,REQUIREMENTS
from ancestor.context import ContextBuilder
from ancestor.corpus import Corpus
from ancestor.controller import Controller,Objective
from ancestor.ledger import LearningLedger,StageRequirement

def test_governance_manifest_and_authority(tmp_path):
    c=Constitution(("Identity is a unity of self and other.","Communion is hierarchical."))
    path=tmp_path/"governance.json"
    c.save_initial(path)
    assert Constitution.load(path).digest()==c.digest()
    assert "owner's intellectual system" in INTELLECTUAL_CONSTITUTION
    assert len(REQUIREMENTS)==6
    with pytest.raises(FileExistsError):
        c.save_initial(path)
    modified=path.read_text().replace("Communion is hierarchical.","Communion is optional.")
    path.write_text(modified)
    with pytest.raises(ValueError):
        Constitution.load(path)

def test_paragraph_and_exam_gate(tmp_path):
    source=tmp_path/"book.txt"
    source.write_text("1. First principle.\n2. Consequence follows.",encoding="utf-8")
    corpus=Corpus(tmp_path/"data")
    assert corpus.ingest_text(source,"BOOK1","Example")==2
    p=corpus.list_paragraphs("BOOK1")
    assert len(p)==2
    corpus.add_dependency(p[1].id,p[0].id,"DEPENDS_ON")
    c=Constitution(("Identity is a unity of self and other.",))
    ctx=ContextBuilder(ByteBPE([]),c,1024)
    ledger=LearningLedger(tmp_path/"ledger.jsonl")
    controller=Controller(corpus,ctx,ledger,c)
    objective=Objective("OBJECTIVE1",p[1].id,"Explain consequence.",(p[0].id,))
    window=controller.formulate(objective)
    assert set(window.source_ids)=={p[0].id,p[1].id}
    controller.record_coverage(objective)
    gate=StageRequirement("STAGE-1",(p[1].id,),("EXAM-1",))
    with pytest.raises(PermissionError):
        controller.stage_gate(gate)
    controller.evaluate("EXAM-1",passed=False,
                        evidence={"questions":2,"correct":0},independent=True)
    with pytest.raises(PermissionError):
        controller.stage_gate(gate)
    controller.evaluate("EXAM-1",passed=True,
                        evidence={"questions":2,"correct":2},independent=True)
    controller.stage_gate(gate)
    with pytest.raises(PermissionError):
        controller.protected_axiom_revision()
    with pytest.raises(ValueError):
        ctx.build([(p[0].id,"x"*2000)],"Task",reserve=128)

def test_dictionary_metadata_is_rejected(tmp_path):
    d=Corpus(tmp_path/"data")
    p=tmp_path/"dict.tsv"
    p.write_text("00-cide-info\t00-database-info\n")
    with pytest.raises(ValueError):
        d.import_dictionary_tsv(p)
    p.write_text("Apple\tfruit\nAbility\tcapacity\n")
    assert d.import_dictionary_tsv(p)==2
    assert [x[0] for x in d.dictionary()]==["Ability","Apple"]
    records=d.dictionary_records()
    assert [r.id for r in records]==["WEBSTER1913:W-000000","WEBSTER1913:W-000001"]
    assert records[0].exact_text=="Ability\ncapacity"
    p.write_text("Different\tchanged\n")
    with pytest.raises(ValueError):
        d.import_dictionary_tsv(p)


def test_owner_constitution_is_documented_verbatim():
    from pathlib import Path
    from ancestor.governance import INTELLECTUAL_CONSTITUTION
    doc=(Path(__file__).resolve().parents[1]/"docs"/"ARCHITECTURE.md").read_text(encoding="utf-8")
    required_passages=INTELLECTUAL_CONSTITUTION.split("\n\n")[1:]
    for passage in required_passages:
        assert passage in doc
    assert "Contradictions are treated as failures of the intended reasoning architecture" in doc
    assert "Other intellectual systems are objects of analysis" in doc
    assert "Stage gating must test whether he actually reasons according to your framework" in doc


def test_original_manuscript_numbering_formats(tmp_path):
    from ancestor.corpus import Corpus
    cases=[
        ("brackets","[0] First axiom.\n[1] Second consequence.",[0,1]),
        ("decimal","1. First axiom.\n2. Second consequence.",[1,2]),
        ("section","§ 3 First axiom.\n§ 4 Second consequence.",[3,4]),
    ]
    corpus=Corpus(tmp_path/"numbered")
    for ident,content,expected in cases:
        f=tmp_path/(ident+".txt")
        f.write_text(content,encoding="utf-8")
        assert corpus.ingest_text(f,ident,ident)==2
        assert [x.paragraph_number for x in corpus.list_paragraphs(ident)]==expected
