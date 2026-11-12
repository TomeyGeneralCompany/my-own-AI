import pytest
from my_own_ai.modules import BasicModule, CheckpointStore, ModuleController, default_modules

def test_switch_and_restore(tmp_path):
    c = ModuleController(CheckpointStore(tmp_path), default_modules())
    c.switch("writing")
    old = c.active
    old.state["chapter"] = 12
    c.switch("coding")
    assert c.active.name == "coding" and old.state is None and not old.loaded
    c.switch("writing")
    assert c.active.state == {"chapter": 12}
    assert sum(m.loaded for m in c.modules.values()) == 1
    c.close()
    assert not any(m.loaded for m in c.modules.values())

def test_failed_checkpoint_does_not_unload(tmp_path, monkeypatch):
    store = CheckpointStore(tmp_path)
    c = ModuleController(store, default_modules())
    c.switch("writing")
    def fail(*args):
        raise OSError("disk full")
    monkeypatch.setattr(store, "write", fail)
    with pytest.raises(OSError):
        c.switch("coding")
    assert c.active.name == "writing" and c.active.loaded

def test_failed_activation_recovers(tmp_path):
    class Broken(BasicModule):
        def activate(self, state):
            raise RuntimeError("broken")
    c = ModuleController(CheckpointStore(tmp_path), [BasicModule("writing"), Broken("broken")])
    c.switch("writing")
    c.active.state["page"] = 7
    with pytest.raises(RuntimeError):
        c.switch("broken")
    assert c.active.name == "writing" and c.active.state == {"page": 7}

def test_unsafe_module_name(tmp_path):
    with pytest.raises(ValueError):
        CheckpointStore(tmp_path).read("../escape")
