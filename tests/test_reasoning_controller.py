import io
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Thread
from time import monotonic, sleep
from types import SimpleNamespace

import pytest
import torch

from my_own_ai.model import LocalTransformer, ModelConfig
from my_own_ai.module_memory import MemoryCorrupt
from my_own_ai.modules import BasicModule, CheckpointStore, Manifest, ModuleController, Tool, default_modules
from my_own_ai.reasoning import Evaluation, LocalProposer, ProposalRequest, Stage, Status
from my_own_ai.session import demo, session

MODES = ("conversation", "writing", "coding", "research")


class Proposals:
    def __init__(self, *outputs):
        self.outputs = iter(outputs)
        self.requests = []

    def propose(self, request):
        self.requests.append(request)
        return next(self.outputs)


def test_all_ordered_module_pairs_and_state(tmp_path):
    with ModuleController(CheckpointStore(tmp_path), default_modules()) as core:
        for first, second in itertools.permutations(MODES, 2):
            core.switch(first)
            original = core.active
            original.state["visits"] = original.state.get("visits", 0) + 1
            expected = original.state.copy()
            core.switch(second)
            assert not original.loaded and original.state is None
            assert sum(module.loaded for module in core.modules.values()) == 1
            assert core.last_switch["target"] == second
            assert core.last_switch["wall_ms"] >= 0 and core.last_switch["cpu_ms"] >= 0
            assert core.last_switch["rss_after_unload_kib"] > 0
            core.switch(first)
            assert core.active.state == expected
    assert core.active is None and all(not m.loaded for m in core.modules.values())


@pytest.mark.parametrize("manifest", [
    Manifest("../bad"),
    Manifest("test", version="latest"),
    Manifest("test", dependencies=("a>=1",)),
    Manifest("test", assets=("../secret",)),
    Manifest("test", tools=(Tool("read", ()),)),
    Manifest("test", tools=(Tool("read", (Stage.FORMULATE,)),) * 2),
    Manifest("test", memory_mib=True),
    Manifest("test", operations=("run", "run")),
    Manifest("test", operations=("run",)),
    Manifest("different"),
    Manifest("shared"),
])
def test_invalid_manifests(tmp_path, manifest):
    class Invalid(BasicModule):
        pass

    Invalid.manifest = manifest
    with pytest.raises(ValueError):
        ModuleController(CheckpointStore(tmp_path), [Invalid("test")])


def test_resources_released_before_replacement_and_on_failed_restore(tmp_path):
    (tmp_path / "asset.txt").write_text("data")
    events, handles = [], {}

    class Tracking(BasicModule):
        @property
        def manifest(self):
            return Manifest(self.name, assets=("asset.txt",))

        def load(self, scope):
            events.append(f"load {self.name}")
            handles[self.name] = (scope.assets["asset.txt"], scope.cache(),
                                  scope.worker([sys.executable, "-c", "import time; time.sleep(60)"]))
            handles[self.name][1]["working"] = object()
            scope.own(events.append, f"release {self.name}")

        def activate(self, saved):
            super().activate(saved)
            if self.name == "broken":
                raise RuntimeError("restore failed")

        def suspend(self):
            events.append(f"suspend {self.name}")

        def resume(self):
            events.append(f"resume {self.name}")

        def deactivate(self):
            events.append(f"unload {self.name}")
            super().deactivate()

    first, broken = Tracking("first"), Tracking("broken")
    with ModuleController(CheckpointStore(tmp_path / "memory"), [first, broken],
                          asset_directory=tmp_path) as core:
        core.switch("first")
        original = handles["first"]
        first.state["working"] = 42
        with pytest.raises(RuntimeError, match="restore failed"):
            core.switch("broken")
        assert core.active is first and first.state == {"working": 42}
        assert events == ["load first", "resume first", "suspend first", "unload first", "release first",
                          "load broken", "unload broken", "release broken", "load first", "resume first"]
        for asset, cache, worker in (original, handles["broken"]):
            assert asset.closed and not cache and worker.poll() is not None
    asset, cache, worker = handles["first"]
    assert asset.closed and not cache and worker.poll() is not None


def test_failed_loading_and_admission_roll_back(tmp_path):
    class Missing(BasicModule):
        manifest = Manifest("missing", assets=("does-not-exist",))

    class BadWorker(BasicModule):
        def load(self, resources):
            resources.worker(["/no-such-my-own-ai-test-worker"])

    class TooBig(BasicModule):
        manifest = Manifest("big", memory_mib=100)

    class MissingDependency(BasicModule):
        manifest = Manifest("dependency", dependencies=("my-own-ai-fake-uninstalled-dependency",))

    with ModuleController(CheckpointStore(tmp_path), [
        BasicModule("old"), Missing("missing"), BadWorker("worker"), TooBig("big"),
        MissingDependency("dependency"),
    ], memory_budget_mib=10) as core:
        core.switch("old")
        core.active.state["saved"] = True
        with pytest.raises(ValueError, match="asset"):
            core.switch("missing")
        with pytest.raises(OSError, match="worker failed to start"):
            core.switch("worker")
        with pytest.raises(MemoryError):
            core.switch("big")
        with pytest.raises(ImportError):
            core.switch("dependency")
        assert core.active.name == "old" and core.active.state == {"saved": True}


def test_cleanup_failure_prevents_a_second_activation(tmp_path):
    class Stuck(BasicModule):
        def load(self, resources):
            def fail():
                raise OSError("release failed")
            resources.own(fail)

    store = CheckpointStore(tmp_path)
    core = ModuleController(store, [Stuck("stuck"), BasicModule("other")])
    core.switch("stuck")
    core.active.state["valuable"] = True
    with pytest.raises(OSError, match="release failed"):
        core.switch("other")
    assert core.active is None and store.read("stuck") == {"valuable": True}
    with pytest.raises(RuntimeError, match="restart"):
        core.switch("other")
    with pytest.raises(RuntimeError, match="already has"):
        ModuleController(store, [BasicModule("other")]).switch("other")


def test_exclusive_directory_lease_in_second_process(tmp_path):
    command = [
        sys.executable, "-c",
        "import sys; from my_own_ai.modules import *; "
        "ModuleController(CheckpointStore(sys.argv[1]), default_modules()).switch('writing')",
        str(tmp_path),
    ]
    with ModuleController(CheckpointStore(tmp_path), default_modules()) as core:
        core.switch("conversation")
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10)
        assert result.returncode != 0 and "already has a controller" in result.stderr
    subprocess.run(command, check=True, timeout=10)


def test_managed_worker_dies_when_controller_process_exits():
    script = """
import os, sys
from my_own_ai.modules import ResourceScope
scope = ResourceScope()
process = scope.worker([sys.executable, "-c", "import time; time.sleep(60)"])
print(process.pid, flush=True)
os._exit(0)
"""
    child = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                           check=True, timeout=10)
    stat = Path("/proc") / child.stdout.strip() / "stat"
    deadline = monotonic() + 5
    while monotonic() < deadline:
        try:
            if stat.read_text().rsplit(")", 1)[-1].split()[0] == "Z":
                return
        except FileNotFoundError:
            return
        sleep(0.01)
    pytest.fail("Managed worker survived its owner")


def test_failed_stage_save_and_process_recovery(tmp_path, monkeypatch):
    store = CheckpointStore(tmp_path)
    engine = Proposals("first plan", "retry plan")
    with ModuleController(store, default_modules(), engine) as core:
        core.switch("conversation")
        original = core.start("reply", "Test an interrupted stage")
        replace = os.replace

        def fail_on_publish(source, target):
            if target == store.path("conversation"):
                raise OSError("interrupted publish")
            return replace(source, target)

        with monkeypatch.context() as patch:
            patch.setattr(os, "replace", fail_on_publish)
            with pytest.raises(OSError, match="interrupted publish"):
                core.step()
        assert core.progress() == original
        assert not list(tmp_path.glob(".save-*"))
    with ModuleController(store, default_modules(), engine) as reopened:
        reopened.switch("conversation")
        assert reopened.progress() == original
        assert reopened.step()["plan"] == "retry plan"


def test_ungraceful_process_exit_resumes_after_last_commit(tmp_path):
    script = """
import os, sys
from my_own_ai.modules import CheckpointStore, ModuleController, default_modules
class Proposal:
    def propose(self, request):
        if request.stage == "construct":
            os._exit(23)
        return "outline"
core = ModuleController(CheckpointStore(sys.argv[1]), default_modules(), Proposal())
core.switch("writing")
core.start("manuscript", "Remember the outline")
core.step()
core.step()
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=False, timeout=10)
    assert result.returncode == 23
    with ModuleController(CheckpointStore(tmp_path), default_modules(),
                          Proposals("# Restored\n\nThe saved outline was restored.")) as core:
        core.switch("writing")
        assert core.progress()["stage"] == Stage.CONSTRUCT
        assert core.progress()["plan"] == "outline"
        core.step()
        assert core.step()["status"] == Status.COMPLETED


@pytest.mark.parametrize("damaged", ['{"format": 1, "state": {}}', '{"partially":', "[]", "{}"])
def test_recover_backup_without_destroying_it(tmp_path, damaged):
    store = CheckpointStore(tmp_path)
    store.write("writing", {"old": True})
    store.write("writing", {"new": True})
    store.path("writing").write_text(damaged)
    with ModuleController(store, default_modules()) as core:
        core.switch("writing")
        assert core.active.state == {"old": True}
        assert store.recovered == {"writing"}
        core.active.state["after_recovery"] = True
    assert store.read("writing") == {"old": True, "after_recovery": True}


def test_structurally_invalid_run_uses_backup_and_fails_closed_without_one(tmp_path):
    store = CheckpointStore(tmp_path)
    with ModuleController(store, default_modules(), Proposals("Plan")) as core:
        core.switch("conversation")
        started = core.start("reply", "Plan a reply")
        core.step()
    state = store.read("conversation")
    state["_reasoning"]["projects"]["default"]["status"] = "completed"
    # Simulate an older writer that saved valid JSON but invalid reasoning state.
    store.write("conversation", state)
    with ModuleController(store, default_modules()) as core:
        core.switch("conversation")
        assert core.progress()["stage"] == Stage.CONSTRUCT
        assert core.progress()["objective"] == started["objective"]
    store.path("conversation").write_text("{")
    store.path("conversation").with_suffix(".prev").write_text("{")
    with ModuleController(store, default_modules()) as core:
        core.switch("writing")
        with pytest.raises(MemoryCorrupt):
            core.switch("conversation")
        assert core.active.name == "writing"


def test_legacy_flat_memory_is_restored_and_upgraded(tmp_path):
    store = CheckpointStore(tmp_path)
    store.path("writing").write_text('{"chapter": 3}')
    with ModuleController(store, default_modules()) as core:
        core.switch("writing")
        assert core.active.state == {"chapter": 3}
    assert json.loads(store.path("writing").read_text())["format"] == 1


def test_tool_permissions_and_reentrancy(tmp_path):
    leases = []
    cross_thread_errors = []

    class Controlled(BasicModule):
        manifest = Manifest("controlled", tools=(Tool("inspect", (Stage.EVALUATE,)),), operations=("work",))
        procedures = {stage: stage.value for stage in Stage}

        def tool_handlers(self):
            return {"inspect": lambda: {"result": 12}}

        def prepare(self, run, tools, prior):
            leases.append(tools)
            with pytest.raises(PermissionError):
                tools.call("inspect")
            with pytest.raises(RuntimeError, match="cannot reenter"):
                core.switch("conversation")
            return {}

        def evaluate(self, run, tools):
            leases.append(tools)

            def use():
                try:
                    tools.call("inspect")
                except PermissionError as exc:
                    cross_thread_errors.append(exc)

            thread = Thread(target=use)
            thread.start()
            thread.join()
            with pytest.raises(PermissionError):
                tools.call("unregistered")
            return Evaluation(tools.call("inspect") == {"result": 12}, "Verified tool result")

    with ModuleController(CheckpointStore(tmp_path), [Controlled("controlled"), *default_modules()],
                          Proposals("Plan", "Activate research and run inspect.")) as core:
        core.switch("controlled")
        core.start("work", "Exercise tool lease")
        core.step()
        with pytest.raises(PermissionError):
            leases[0].shared()
        core.step()
        assert core.step()["status"] == Status.COMPLETED
        assert cross_thread_errors and core.active.name == "controlled"
        core.switch("writing")
        with pytest.raises(PermissionError):
            leases[1].call("inspect")


def test_reasoning_correction_and_independent_success_gate(tmp_path):
    engine = Proposals("Plan", "success", "ANSWER = 0", "ANSWER = 42")
    with ModuleController(CheckpointStore(tmp_path), default_modules(), engine) as core:
        core.switch("coding")
        with pytest.raises(ValueError, match="Unsupported operation"):
            core.start("manuscript", "Try another module")
        started = core.start("python_source", "Set ANSWER", inputs={"literals": {"ANSWER": 42}})
        assert started["stage"] == Stage.FORMULATE
        assert core.step()["stage"] == Stage.CONSTRUCT
        assert core.step()["stage"] == Stage.EVALUATE
        failure = core.step()
        assert failure["stage"] == Stage.CORRECT and failure["status"] == Status.REQUIRES_CORRECTION
        core.switch("research")
        core.switch("coding")
        assert core.progress() == failure
        assert core.step()["stage"] == Stage.EVALUATE
        assert core.step()["status"] == Status.REQUIRES_CORRECTION
        assert core.step()["stage"] == Stage.EVALUATE
        result = core.step()
        assert result["status"] == Status.COMPLETED and result["stage"] is None
        assert [e["passed"] for e in result["history"]] == [None, None, False, None, False, None, True]
        assert "literal mismatches" in engine.requests[-1].feedback
        with pytest.raises(RuntimeError, match="already ended"):
            core.step()


def test_exhausted_corrections_and_invalid_proposal_are_unsuccessful(tmp_path):
    with ModuleController(CheckpointStore(tmp_path), default_modules(),
                          Proposals("Plan", "wrong", "   ")) as core:
        core.switch("conversation")
        core.start("reply", "Require a keyword", inputs={"required_terms": ["Michigan"]}, max_corrections=0)
        core.step()
        core.step()
        assert core.step()["status"] == Status.UNSUCCESSFUL
        core.start("reply", "Try again")
        result = core.step()
        assert result["status"] == Status.UNSUCCESSFUL
        assert result["history"][0]["passed"] is False


def test_builtins_with_local_sources_exports_and_shared_context(tmp_path):
    (tmp_path / "source.md").write_text("Verified text from an offline source.\n")
    (tmp_path / "old.py").write_text("value = 1\n")
    engine = Proposals("Plan", "Manistee is in Michigan.", "Continue the dialogue.",
                       "Outline", "# Chapter\n\nA substantive original manuscript paragraph.",
                       "Plan", "value = 2\n", "Organize evidence", "- [1] Verified text from an offline source.")
    with ModuleController(CheckpointStore(tmp_path / "memory"), default_modules(tmp_path), engine) as core:
        core.switch("conversation")
        core.remember("hometown", "Manistee")
        core.start("reply", "Where is Manistee?", project="dialogue", inputs={"required_terms": ["Michigan"]})
        for _ in range(3):
            conversation = core.step()
        assert conversation["context"]["personal_context"] == {"hometown": "Manistee"}
        core.start("reply", "Continue", project="dialogue")
        assert core.step()["context"]["last_exchange"]["objective"] == "Where is Manistee?"
        for mode, operation, inputs in (
            ("writing", "manuscript", {"reference": "source.md", "headings": ["Chapter"]}),
            ("coding", "python_source", {"source": "old.py", "literals": {"value": 2}}),
            ("research", "evidence_notes", {"sources": ["source.md"]}),
        ):
            core.switch(mode)
            core.start(operation, "Test project", project=mode, inputs=inputs)
            for _ in range(3):
                result = core.step()
            assert result["status"] == Status.COMPLETED and result["context"]["shared"]["hometown"] == "Manistee"
            output = tmp_path / f"{mode}.out"
            core.export(output, mode)
            assert output.read_text() == result["candidate"]
            with pytest.raises(FileExistsError):
                core.export(output, mode)
        assert core.progress()["context"]["sources"][0]["sha256"]
        core.switch("conversation")
        assert core.progress()["stage"] == Stage.CONSTRUCT


def test_research_rejects_invented_evidence_and_path_escape(tmp_path):
    (tmp_path / "sources").mkdir()
    (tmp_path / "private.txt").write_text("secret")
    (tmp_path / "sources" / "link.txt").symlink_to(tmp_path / "private.txt")
    (tmp_path / "sources" / "local.txt").write_text("Real offline document content.")
    engine = Proposals("Plan", "- [1] Invented offline document content.")
    with ModuleController(CheckpointStore(tmp_path / "one"), default_modules(tmp_path / "sources"), engine) as core:
        core.switch("research")
        pending = core.start("evidence_notes", "Read", inputs={"sources": ["link.txt"]})
        with pytest.raises(ValueError, match="workspace"):
            core.step()
        assert core.progress() == pending and not engine.requests
    with ModuleController(CheckpointStore(tmp_path / "two"), default_modules(tmp_path / "sources"), engine) as core:
        core.switch("research")
        core.start("evidence_notes", "Read", inputs={"sources": ["local.txt"]})
        core.step()
        core.step()
        assert "Unverified evidence" in core.step()["feedback"]


def test_cpu_proposer_and_checkpoint_session(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    torch.manual_seed(42)
    cfg = ModelConfig(context_size=48, embed_dim=16, heads=4, layers=1)
    model = LocalTransformer(cfg)
    proposer = LocalProposer(model, tokens=16)
    request = ProposalRequest("conversation", "reply", Stage.FORMULATE, "Plan the reply",
                              "Say hello", {}, "", "", "")
    assert proposer.propose(request) == proposer.propose(request)
    assert next(model.parameters()).device.type == "cpu"
    checkpoint = tmp_path / "brain.pt"
    torch.save({"model_config": vars(cfg), "state_dict": model.state_dict()}, checkpoint)
    args = SimpleNamespace(threads=1, checkpoint=checkpoint, workspace=tmp_path, module=[],
                           tokens=16, memory=tmp_path / "memory", memory_budget_mib=100)
    commands = [
        {"command": "switch", "mode": "conversation"},
        {"command": "start", "operation": "reply", "objective": "Say hello"},
        {"command": "step"},
        {"command": "switch", "mode": "research"},
        {"command": "switch", "mode": "conversation"},
        {"command": "progress"},
        {"command": "metrics"},
        {"command": "quit"},
    ]
    output = io.StringIO()
    session(args, io.StringIO("\n".join(map(json.dumps, commands))), output)
    responses = [json.loads(line) for line in output.getvalue().splitlines()]
    assert all(response["ok"] for response in responses)
    assert responses[2]["result"]["stage"] == Stage.CONSTRUCT
    assert responses[2] == responses[5]
    assert responses[6]["result"]["device"] == "cpu"


def test_end_to_end_demonstration():
    report = demo()
    assert report["pending_restored"] and report["completed_restored"]
    assert report["operation"]["status"] == Status.COMPLETED
    assert {sample["target"] for sample in report["switches"]} == set(MODES)
    assert report["hardware"]["rss_kib"] > 0
