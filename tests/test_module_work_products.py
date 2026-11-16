import json
from zipfile import ZipFile
from xml.etree import ElementTree
import pytest
from my_own_ai.business_module import Business, calculate_scenario
from my_own_ai.module_work_products import append_exchange, assemble_manuscript, compare_observations, check_engineering_artifact
from my_own_ai.modules import ModuleController, CheckpointStore, default_modules
from my_own_ai.reasoning import Stage, Status


def test_business_math():
    assert calculate_scenario(100,500,200,1000)["operating_result_cents"]==29000
    with pytest.raises(ValueError):
        calculate_scenario(True,500,200,1000)


def test_business_controller(tmp_path):
    class Fixture:
        def propose(self, request):
            assert request.objective=="Owner objective"
            if request.stage==Stage.FORMULATE:
                return "Fixture"
            return '{"plan":"Fixture","units":100,"price_cents":500}'
    inputs=dict(variable_cents=200,fixed_cents=1000,target_cents=29000,max_units=100,max_price_cents=500)
    assert "business" not in {m.name for m in default_modules()}
    with ModuleController(CheckpointStore(tmp_path),[Business()],Fixture()) as core:
        core.switch("business")
        core.start("business_plan","Owner objective",inputs=inputs)
        for _ in range(3):
            result=core.step()
        assert result["status"]==Status.COMPLETED
        assert json.loads(result["feedback"])["operating_result_cents"]==29000
        assert json.loads(result["feedback"])["scenario_only"] is True


def test_work_products(tmp_path):
    history=append_exchange([], "Question", "Answer")
    assert append_exchange(history,"Next","Response")[0]==history[0]
    assert len(history)==1
    path=tmp_path/"book.docx"
    result=assemble_manuscript({"b":"Second","a":"First & foremost"},["a","b"],path)
    assert result["sections"]==2
    with ZipFile(path) as z:
        xml=ElementTree.fromstring(z.read("word/document.xml"))
        ns={"w":"http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        assert [x.text for x in xml.findall(".//w:t",ns)]==["First & foremost","Second"]
    with pytest.raises(FileExistsError):
        assemble_manuscript({"a":"New"},["a"],path)
    proof=check_engineering_artifact("x=2")
    assert proof["compiled"] and not proof["functional_tests_passed"]
    assert not check_engineering_artifact("x=")["compiled"]
    with pytest.raises(ValueError):
        check_engineering_artifact("x=2","wrong")
    result=compare_observations("Premises:\n- [1] The observed price was ten.\nInference: The observation supports this statement.",[{"text":"The observed price was ten."}])
    assert result["citation_numbers"]==[1]
    assert not result["semantic_entailment_verified"]
