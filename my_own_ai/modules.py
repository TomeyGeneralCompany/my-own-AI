"""Exclusive Linux module lifecycle and controller-owned tool capabilities."""
from contextlib import ExitStack
from dataclasses import dataclass, field
from functools import wraps
from importlib.metadata import version
import os
from pathlib import Path
import re
import resource
import select
import signal
import subprocess
import sys
from threading import RLock, get_ident
from time import perf_counter, process_time

from .module_memory import CheckpointStore, IDENTIFIER, json_copy
from .reasoning import Operation, Stage, Status, advance

RUNTIME = "_reasoning"
SHARED = "shared"
_QUARANTINED_LEASES = []


@dataclass(frozen=True)
class Tool:
    name: str
    stages: tuple[Stage, ...]


@dataclass(frozen=True)
class Manifest:
    identifier: str
    version: str = "1.0.0"
    dependencies: tuple[str, ...] = ()
    assets: tuple[str, ...] = ()
    tools: tuple[Tool, ...] = ()
    memory_mib: int = 0
    operations: tuple[str, ...] = ()

    def validate(self):
        _names((self.identifier,))
        if not isinstance(self.version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", self.version):
            raise ValueError("Module version must be major.minor.patch")
        _names(self.operations)
        if type(self.memory_mib) is not int or self.memory_mib < 0:
            raise ValueError("Expected a nonnegative integer memory estimate")
        if (not isinstance(self.dependencies, tuple) or
                any(not isinstance(d, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", d)
                    for d in self.dependencies)
                or len(set(self.dependencies)) != len(self.dependencies)):
            raise ValueError("Invalid dependencies (expected installed distribution names)")
        if (not isinstance(self.assets, tuple) or
                any(not isinstance(a, str) or not a or "\0" in a or Path(a).is_absolute()
                    or ".." in Path(a).parts for a in self.assets)
                or len(set(self.assets)) != len(self.assets)):
            raise ValueError("Invalid local asset paths")
        if not isinstance(self.tools, tuple) or not all(isinstance(t, Tool) for t in self.tools):
            raise ValueError("Invalid tool declarations")
        _names(tuple(t.name for t in self.tools))
        for tool in self.tools:
            if (not isinstance(tool.stages, tuple) or not tool.stages
                    or any(not isinstance(s, Stage) for s in tool.stages)
                    or len(set(tool.stages)) != len(tool.stages)):
                raise ValueError("Invalid tool stages")


def _names(names):
    if (not isinstance(names, tuple)
            or any(not isinstance(n, str) or not IDENTIFIER.fullmatch(n) for n in names)
            or len(set(names)) != len(names)):
        raise ValueError("Invalid or duplicate identifiers")


class ResourceScope:
    """Every acquired module resource must register its cleanup immediately."""
    def __init__(self):
        self.assets = {}
        self._stack = ExitStack()
        self._closed = False

    def enter(self, manager):
        if self._closed:
            raise RuntimeError("Resource scope is closed")
        return self._stack.enter_context(manager)

    def own(self, cleanup, *args):
        if self._closed:
            raise RuntimeError("Resource scope is closed")
        self._stack.callback(cleanup, *args)

    def cache(self):
        cache = {}
        self.own(cache.clear)
        return cache

    def worker(self, argv):
        """Start a trusted extension worker in a process group owned by this scope."""
        if self._closed:
            raise RuntimeError("Resource scope is closed")
        if not isinstance(argv, (tuple, list)) or not argv or not all(isinstance(arg, str) for arg in argv):
            raise ValueError("Worker requires a nonempty argument vector")
        read_fd, ready_fd = os.pipe()
        try:
            command = [sys.executable, "-m", "my_own_ai.module_worker",
                       str(os.getpid()), str(ready_fd), *argv]
            try:
                process = subprocess.Popen(command, start_new_session=True, pass_fds=(ready_fd,),
                                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL)
                self.own(self._stop_worker, process)
            finally:
                os.close(ready_fd)
            if not select.select([read_fd], [], [], 5)[0]:
                raise TimeoutError("Module worker startup timed out")
            if error := os.read(read_fd, 4096):
                raise OSError(f"Module worker failed to start: {error.decode(errors='replace')}")
            return process
        finally:
            os.close(read_fd)

    @staticmethod
    def _stop_worker(process):
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)

    def close(self):
        self._closed = True
        try:
            self._stack.close()
        finally:
            self.assets.clear()


@dataclass
class BasicModule:
    """Extension base. Default lifecycle restores a JSON working state."""
    name: str
    state: dict | None = field(default=None, init=False)
    loaded: bool = field(default=False, init=False)
    procedures = {}
    exportable = False

    @property
    def manifest(self):
        return Manifest(self.name)

    def load(self, resources):
        """Acquire optional resources through the supplied scope."""

    def activate(self, previous_state):
        self.state = json_copy(previous_state)
        self.loaded = True

    def resume(self):
        """Resume after restore or an aborted switch."""

    def suspend(self):
        """Stop module activity before the controller takes a snapshot."""

    def snapshot(self):
        if not self.loaded:
            raise RuntimeError("Inactive module")
        return json_copy(self.state)

    def deactivate(self):
        self.state = None
        self.loaded = False

    def validate_state(self, state):
        """Validate extension-specific memory; accept legacy flat snapshots if applicable."""

    def validate_inputs(self, operation, inputs):
        if inputs:
            raise ValueError("This operation takes no options")

    def validate_context(self, operation, inputs, context):
        """Validate durable data returned by prepare."""

    def tool_handlers(self):
        return {}

    def prepare(self, run, tools, prior):
        return {"shared": tools.shared(), "prior": prior}

    def procedure(self, operation, stage):
        return self.procedures[stage]

    def evaluate(self, run, tools):
        raise NotImplementedError


class ToolLease:
    def __init__(self, module, manifest, stage, shared):
        self._handlers = dict(module.tool_handlers())
        self._permitted = {t.name for t in manifest.tools if stage in t.stages}
        self._shared = shared
        self._owner = get_ident()
        self._open = True

    def _check(self):
        if not self._open or get_ident() != self._owner:
            raise PermissionError("Tool access requires the current active stage")

    def call(self, name, /, **arguments):
        self._check()
        if name not in self._permitted:
            raise PermissionError(f"Tool {name!r} is unavailable in this stage")
        return json_copy(self._handlers[name](**json_copy(arguments)))

    def shared(self):
        self._check()
        return json_copy(self._shared)

    def revoke(self):
        self._open = False
        self._handlers.clear()
        self._shared = None


def _serialized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            if self._busy:
                raise RuntimeError("Controller callbacks cannot reenter the controller")
            self._busy = True
            try:
                return method(self, *args, **kwargs)
            finally:
                self._busy = False
    return call


def _proc_kib(path, field):
    with open(path, encoding="ascii") as handle:
        for line in handle:
            if line.startswith(field + ":"):
                return int(line.split()[1])
    raise RuntimeError(f"Linux did not report {field}")


def _rss():
    return _proc_kib("/proc/self/status", "VmRSS")


def hardware_metrics():
    rss = _rss()
    return {"device": "cpu", "available_cpus": len(os.sched_getaffinity(0)),
            "available_memory_kib": _proc_kib("/proc/meminfo", "MemAvailable"),
            "rss_kib": rss, "peak_rss_kib": max(rss, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "process_cpu_seconds": process_time()}


class ModuleController:
    def __init__(self, store, modules, engine=None, *, asset_directory=None, memory_budget_mib=None):
        self.store = store
        self.engine = engine
        self.asset_directory = Path(asset_directory or store.directory / "assets").resolve()
        if memory_budget_mib is not None and (type(memory_budget_mib) is not int or memory_budget_mib < 0):
            raise ValueError("Invalid module memory budget")
        self.memory_budget_mib = memory_budget_mib
        self._modules = {}
        self._manifests = {}
        for module in modules:
            manifest = module.manifest
            manifest.validate()
            if module.name != manifest.identifier or module.name in self._modules or module.loaded:
                raise ValueError("Expected distinct inactive modules with matching names")
            if module.name == SHARED:
                raise ValueError("Reserved module identifier")
            if manifest.operations and (set(module.procedures) != set(Stage)
                    or any(not isinstance(p, str) or not p.strip() for p in module.procedures.values())):
                raise ValueError("Module operations require procedures for all four stages")
            self._modules[module.name] = module
            self._manifests[module.name] = manifest
        self._active = self._scope = self._lease = None
        self._poisoned = False
        self._lock = RLock()
        self._busy = False
        self.last_switch = None

    @property
    def modules(self):
        return dict(self._modules)

    @property
    def active(self):
        return self._active

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _acquire(self):
        if self._poisoned:
            raise RuntimeError("Resource cleanup failed; restart the local controller")
        if self._lease is None:
            lease = self.store.lease()
            lease.__enter__()
            self._lease = lease

    def _release(self):
        if self._lease is not None:
            lease, self._lease = self._lease, None
            lease.__exit__(None, None, None)

    def _validate(self, module, state):
        module.validate_state(state)
        accepted = state.get("_accepted", {})
        if (not isinstance(accepted, dict)
                or any(not IDENTIFIER.fullmatch(key) or not isinstance(value, dict)
                       or set(value) != {"objective", "candidate"}
                       or not all(isinstance(text, str) for text in value.values())
                       for key, value in accepted.items())):
            raise ValueError("Invalid accepted work checkpoint")
        if RUNTIME not in state:
            return
        runtime = state[RUNTIME]
        if (not isinstance(runtime, dict) or set(runtime) != {"latest", "projects"}
                or not isinstance(runtime["latest"], str) or not isinstance(runtime["projects"], dict)
                or runtime["latest"] not in runtime["projects"]):
            raise ValueError("Incomplete reasoning memory")
        for project, raw in runtime["projects"].items():
            run = Operation.restore(raw)
            if run.project != project or run.operation not in self._manifests[module.name].operations:
                raise ValueError("Invalid project checkpoint")
            module.validate_inputs(run.operation, run.inputs)
            if run.history and run.history[0]["passed"] is None:
                module.validate_context(run.operation, run.inputs, run.context)
            if project != runtime["latest"] and run.stage is not None:
                raise ValueError("Only the latest operation may be pending")

    def _validator(self, module):
        return lambda state: self._validate(module, state)

    def _save(self, module, state):
        self.store.write(module.name, state, self._manifests[module.name].version, self._validator(module))

    def _admit(self, manifest):
        available = _proc_kib("/proc/meminfo", "MemAvailable") // 1024
        if self.memory_budget_mib is not None:
            available = min(available, self.memory_budget_mib)
        if manifest.memory_mib > available:
            raise MemoryError(f"{manifest.identifier} requires {manifest.memory_mib} MiB; {available} MiB available")
        for dependency in manifest.dependencies:
            version(dependency)

    def _unload(self, module, scope):
        self._active = self._scope = None
        try:
            module.deactivate()
        finally:
            # A faulty extension must not leave its controller-owned state accessible.
            module.state = None
            module.loaded = False
            try:
                scope.close()
            except BaseException:
                self._poisoned = True
                _QUARANTINED_LEASES.append(self._lease)
                raise

    def _load(self, module, saved):
        manifest = self._manifests[module.name]
        scope = ResourceScope()
        try:
            self._admit(manifest)
            for asset in manifest.assets:
                path = (self.asset_directory / asset).resolve()
                if not path.is_relative_to(self.asset_directory) or not path.is_file():
                    raise ValueError(f"Unavailable module asset: {asset}")
                scope.assets[asset] = scope.enter(path.open("rb"))
            module.load(scope)
            handlers = module.tool_handlers()
            if set(handlers) != {t.name for t in manifest.tools} or not all(map(callable, handlers.values())):
                raise ValueError("Module tool handlers must match its manifest")
            module.activate(saved)
            if not module.loaded:
                raise RuntimeError("Module failed to restore its state")
            self._validate(module, module.snapshot())
            module.resume()
        except BaseException:
            self._unload(module, scope)
            raise
        self._active, self._scope = module, scope

    def _suspend_and_save(self, module):
        try:
            module.suspend()
            state = module.snapshot()
            self._save(module, state)
        except BaseException:
            try:
                module.resume()
            except BaseException:
                self._unload(module, self._scope)
                raise
            raise
        return state

    @_serialized
    def switch(self, target):
        new = self._modules[target]
        old = self._active
        if new is old:
            rss = _rss()
            self.last_switch = {"target": target, "active": target, "wall_ms": 0, "cpu_ms": 0,
                                "rss_before_kib": rss, "rss_after_unload_kib": rss, "rss_after_kib": rss,
                                "recovered_checkpoint": False}
            return
        self._acquire()
        before, cpu, rss = perf_counter(), process_time(), _rss()
        released_rss = rss
        saved_old = None
        self.store.recovered.discard(target)
        try:
            saved = self.store.read(target, self._manifests[target].version, self._validator(new))
            if old is not None:
                saved_old = self._suspend_and_save(old)
                self._unload(old, self._scope)
                released_rss = _rss()
            self._load(new, saved)
        except BaseException as error:
            if saved_old is not None and self._active is None and not self._poisoned:
                try:
                    self._load(old, saved_old)
                except BaseException as rollback:
                    if not self._poisoned:
                        self._release()
                    raise RuntimeError(f"Switch failed ({error}); rollback failed ({rollback}). "
                                       "Previous checkpoint is preserved.") from rollback
            if self._active is None and not self._poisoned:
                self._release()
            raise
        finally:
            self.last_switch = {"target": target, "active": self._active.name if self._active else None,
                                "wall_ms": (perf_counter() - before) * 1000,
                                "cpu_ms": (process_time() - cpu) * 1000,
                                "rss_before_kib": rss, "rss_after_unload_kib": released_rss,
                                "rss_after_kib": _rss(),
                                "recovered_checkpoint": target in self.store.recovered}

    @_serialized
    def checkpoint(self):
        if self._active:
            self._save(self._active, self._active.snapshot())

    @_serialized
    def close(self):
        try:
            if self._active:
                self._suspend_and_save(self._active)
                self._unload(self._active, self._scope)
        finally:
            if self._active is None and not self._poisoned:
                self._release()

    def _require_active(self):
        if self._active is None:
            raise RuntimeError("Activate a module first")
        return self._active

    def _publish(self, module, state):
        self._save(module, state)
        module.state = state

    @_serialized
    def start(self, operation, objective, *, project="default", inputs=None, max_corrections=2):
        module = self._require_active()
        if operation not in self._manifests[module.name].operations:
            raise ValueError(f"Unsupported operation for {module.name}: {operation}")
        state = module.snapshot()
        run = Operation(project, operation, objective, {} if inputs is None else inputs, max_corrections)
        snapshot = run.snapshot()
        Operation.restore(snapshot)
        module.validate_inputs(operation, snapshot["inputs"])
        runtime = state.setdefault(RUNTIME, {"latest": project, "projects": {}})
        previous = runtime["projects"].get(runtime["latest"])
        if previous and previous["stage"] is not None:
            raise RuntimeError("Resume the pending operation before starting another")
        runtime["latest"] = project
        runtime["projects"][project] = snapshot
        self._publish(module, state)
        return json_copy(snapshot)

    def _current(self, state, project=None):
        runtime = state.get(RUNTIME)
        if runtime is None:
            raise RuntimeError("No reasoning operation has been started")
        return runtime["projects"][runtime["latest"] if project is None else project]

    @_serialized
    def progress(self, project=None):
        return json_copy(self._current(self._require_active().snapshot(), project))

    @_serialized
    def step(self):
        module = self._require_active()
        state = module.snapshot()
        run = Operation.restore(self._current(state))
        if run.stage is None:
            raise RuntimeError("Operation has already ended")
        if run.stage != Stage.EVALUATE and self.engine is None:
            raise RuntimeError("A local proposal engine is required for this stage")
        shared = self.store.read(SHARED, legacy=False)
        tools = ToolLease(module, self._manifests[module.name], run.stage, shared)
        try:
            advance(run, module, self.engine, tools, state.get("_accepted", {}).get(run.project, {}))
        finally:
            tools.revoke()
        snapshot = run.snapshot()
        state[RUNTIME]["projects"][run.project] = snapshot
        if run.status == Status.COMPLETED:
            state.setdefault("_accepted", {})[run.project] = {
                "objective": run.objective, "candidate": run.candidate
            }
        self._publish(module, state)
        return json_copy(snapshot)

    @_serialized
    def remember(self, key, value):
        self._require_active()
        _names((key,))
        shared = self.store.read(SHARED, legacy=False)
        shared[key] = json_copy(value)
        self.store.write(SHARED, shared)

    @_serialized
    def export(self, destination, project=None):
        module = self._require_active()
        if not module.exportable:
            raise ValueError("This module does not export documents")
        run = self._current(module.snapshot(), project)
        if run["status"] != Status.COMPLETED:
            raise ValueError("Only successfully evaluated projects can be exported")
        with Path(destination).open("x", encoding="utf-8") as handle:
            handle.write(run["candidate"])


def default_modules(workspace=None, *, corpus_index=None):
    from .functional_modules import Coding, Conversation, Research, Writing

    workspace = Path(workspace or Path.cwd()).resolve()
    return [Conversation(), Writing(workspace), Coding(workspace), Research(workspace, corpus_index)]
