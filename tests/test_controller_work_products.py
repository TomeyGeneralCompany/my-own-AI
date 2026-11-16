from zipfile import ZipFile
import pytest
from my_own_ai.modules import CheckpointStore, ModuleController, default_modules
from my_own_ai.reasoning import Stage, Status


class Fixture:
    def __init__(self):
        self.requests=[]
    def propose(self, request):
        self.requests.append(request)
        if request.stage == Stage.FORMULATE:
            return "Fixture plan"
        return {"conversation":"A sufficiently long answer.",
                "writing":"# Section\nThis is the section body.",
                "coding":"answer = 42",
                "research":"Premises:\n- [1] Revenue was ten dollars.\nInference: The recorded revenue was ten dollars."}[request.module]


def complete(core, operation, objective, project="default", inputs=None):
    core.start(operation,objective,project=project,inputs=inputs)
    for _ in range(3):
        result=core.step()
    assert result["status"]==Status.COMPLETED
    return result


def test_dialogue_persists_across_switch_and_restart(tmp_path):
    store=CheckpointStore(tmp_path)
    fixture=Fixture()
    with ModuleController(store,default_modules(),fixture) as core:
        core.switch("conversation")
        complete(core,"reply","First question")
        assert core.dialogue_history()==[{"question":"First question","answer":"A sufficiently long answer."}]
        core.switch("writing")
        core.switch("conversation")
        complete(core,"reply","Second question")
        assert fixture.requests[-1].context["dialogue_history"][0]["question"]=="First question"
    with ModuleController(store,default_modules(),fixture) as core:
        core.switch("conversation")
        assert [x["question"] for x in core.dialogue_history()]==["First question","Second question"]


def test_manuscript_and_code_owner_instruments(tmp_path):
    with ModuleController(CheckpointStore(tmp_path/"memory"),default_modules(),Fixture()) as core:
        core.switch("writing")
        complete(core,"manuscript","Write first",project="first")
        complete(core,"manuscript","Write second",project="second")
        path=tmp_path/"book.docx"
        assert core.export_manuscript(path,["first","second"])["sections"]==2
        with ZipFile(path) as archive:
            assert archive.read("word/document.xml").count(b"This is the section body.")==2
        with pytest.raises(FileExistsError):
            core.export_manuscript(path,["first","second"])
        core.switch("coding")
        complete(core,"python_source","Set answer",inputs={"literals":{"answer":42}})
        result=core.work_product()
        assert result["compiled"] and not result["functional_tests_passed"]
        with pytest.raises(ValueError):
            core.export_manuscript(tmp_path/"wrong.docx",["first"])


def test_research_preserves_source_and_inference(tmp_path):
    from my_own_ai.retrieval import build_index
    corpus=tmp_path/"corpus"
    corpus.mkdir()
    (corpus/"report.txt").write_text("Revenue was ten dollars.")
    index=tmp_path/"index.sqlite"
    build_index(corpus,index)
    with ModuleController(CheckpointStore(tmp_path/"memory"),default_modules(corpus_index=index),Fixture()) as core:
        core.switch("research")
        complete(core,"grounded_analysis","Revenue")
        proof=core.work_product()
        assert proof["citation_numbers"]==[1]
        assert not proof["semantic_entailment_verified"]
