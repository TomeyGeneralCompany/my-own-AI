from my_own_ai.modules import ModuleController, CheckpointStore, default_modules, BasicModule
from my_own_ai.plugin_catalog import standard_plugins
from my_own_ai.workbench import Workbench
from my_own_ai.reasoning import Stage
import pytest


class Engine:
    def propose(self, request):
        return "Fixture plan" if request.stage == Stage.FORMULATE else "A fixture response."


def test_six_modes_manual_switch_and_resume(tmp_path):
    modules = default_modules(tmp_path) + standard_plugins(tmp_path)
    with ModuleController(CheckpointStore(tmp_path/"state"), modules, Engine()) as controller:
        ui = Workbench(controller)
        assert controller.mode_catalog()["active"] is None
        assert len(controller.mode_catalog()["modules"]) == 6
        with pytest.raises(RuntimeError):
            ui.call("start", operation="reply", objective="Test")
        ui.call("switch", mode="conversation")
        ui.call("start", operation="reply", objective="Test")
        before=ui.call("step")["result"]
        for mode in ("writing","coding","research","finance","business"):
            reply=ui.call("switch",mode=mode)
            assert reply["catalog"]["active"]==mode
            assert sum(m["loaded"] for m in reply["catalog"]["modules"])==1
        ui.call("switch",mode="conversation")
        assert ui.call("progress")["result"]==before
    assert not any(m.loaded for m in modules)
    with ModuleController(CheckpointStore(tmp_path/"state"), default_modules(tmp_path)+standard_plugins(tmp_path)) as controller:
        assert controller.mode_catalog()["active"] is None
        controller.switch("conversation")
        assert controller.progress()==before


def test_extension_visible_without_autoactivation(tmp_path):
    with ModuleController(CheckpointStore(tmp_path), [BasicModule("custom")]) as controller:
        assert controller.mode_catalog()["modules"][0]["label"]=="custom"
        assert not controller.mode_catalog()["modules"][0]["loaded"]
