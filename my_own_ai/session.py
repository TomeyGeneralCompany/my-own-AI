"""Offline session interface and a reproducible lifecycle demonstration."""
from importlib import import_module
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter, process_time

from .modules import CheckpointStore, ModuleController, default_modules, hardware_metrics
from .reasoning import LocalProposer, Stage, Status
from .rulebook import Rulebook, RulebookProposer


def configure_session(parser):
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--rulebook", type=Path, help="Opt in to verified Research derivations")
    parser.add_argument("--memory", type=Path, default=Path("data/module-memory"))
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--corpus-index", type=Path, help="Local passage index for Research")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=96)
    parser.add_argument("--memory-budget-mib", type=int)
    parser.add_argument("--module", action="append", default=[], metavar="PACKAGE:FACTORY",
                        help="Register a trusted installed Python module factory")


def _factory(specification):
    package, separator, function = specification.partition(":")
    if not separator or not package or not function.isidentifier():
        raise ValueError("Expected a trusted extension in PACKAGE:FACTORY form")
    return getattr(import_module(package), function)


def _switch(controller, mode):
    controller.switch(mode)
    return {"active": controller.active.name, "switch": controller.last_switch}


def _commands(controller):
    # The model never receives this owner command interface.
    return {"switch": lambda mode: _switch(controller, mode),
            "start": controller.start, "step": controller.step, "progress": controller.progress,
            "checkpoint": controller.checkpoint, "remember": controller.remember,
            "export": controller.export, "metrics": hardware_metrics}


def load_proposer(checkpoint, threads, tokens):
    import torch
    from .train import load_model

    if threads < 1:
        raise ValueError("Thread count must be positive")
    torch.set_num_threads(min(threads, hardware_metrics()["available_cpus"]))
    return LocalProposer(load_model(checkpoint), tokens)


def session(args, source, output):
    book_path = getattr(args, "rulebook", None)
    if not args.checkpoint and not book_path:
        raise ValueError("Specify a neural checkpoint or a rulebook")
    book = Rulebook.load(book_path) if book_path else None
    model = load_proposer(args.checkpoint, args.threads, args.tokens) if args.checkpoint else None
    proposer = RulebookProposer(book, model) if book is not None else model
    modules = default_modules(args.workspace, corpus_index=getattr(args, "corpus_index", None), rulebook=book)
    modules.extend(_factory(specification)(args.workspace) for specification in args.module)
    with ModuleController(CheckpointStore(args.memory), modules, proposer,
                          memory_budget_mib=args.memory_budget_mib) as controller:
        commands = _commands(controller)
        for line in source:
            if not line.strip():
                continue
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("Expected a JSON command object")
                command = request.pop("command", None)
                if command == "quit" and not request:
                    controller.close()
                    print(json.dumps({"ok": True, "result": "closed"}), file=output, flush=True)
                    break
                if not isinstance(command, str) or command not in commands:
                    raise ValueError(f"Unknown session command: {command!r}")
                result = commands[command](**request)
            except Exception as exc:
                print(json.dumps({"ok": False, "error": str(exc)}), file=output, flush=True)
            else:
                print(json.dumps({"ok": True, "result": result}), file=output, flush=True)


def demo():
    """Exercise all stages, switching and cold restore on isolated temporary memory."""
    import torch
    from .model import LocalTransformer, ModelConfig

    torch.set_num_threads(min(4, hardware_metrics()["available_cpus"]))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = LocalTransformer(ModelConfig(context_size=64, embed_dim=32, heads=4, layers=1))
    proposer = LocalProposer(model, tokens=16)
    switch_samples = []

    class DemonstrationProposals:
        def propose(self, request):
            if request.stage == Stage.FORMULATE:
                return proposer.propose(request)
            if request.stage == Stage.CONSTRUCT:
                return "answer ="
            return "answer = 42"

    with TemporaryDirectory(prefix="my-own-ai-demo-") as directory:
        store = CheckpointStore(Path(directory))
        with ModuleController(store, default_modules(), DemonstrationProposals()) as controller:
            switch_samples.append(_switch(controller, "coding")["switch"])
            controller.start("python_source", "Define answer as the integer 42.",
                             inputs={"literals": {"answer": 42}})
            neural_wall, neural_cpu = perf_counter(), process_time()
            controller.step()
            neural_wall, neural_cpu = perf_counter() - neural_wall, process_time() - neural_cpu
            controller.step()
            before = controller.step()
            controller.checkpoint()
            for target in ("conversation", "writing", "research", "coding"):
                switch_samples.append(_switch(controller, target)["switch"])
            if before["status"] != Status.REQUIRES_CORRECTION or before != controller.progress():
                raise RuntimeError("Pending coding operation did not restore")
            controller.step()
            completed = controller.step()
        with ModuleController(store, default_modules()) as controller:
            controller.switch("coding")
            restored = controller.progress()
        if completed != restored or restored["status"] != Status.COMPLETED:
            raise RuntimeError("Completed operation did not survive reopening")
    return {"proposals": "CPU Transformer formulation; scripted invalid source and correction",
            "pending_restored": True, "completed_restored": True, "operation": restored,
            "formulation_wall_ms": neural_wall * 1000, "formulation_cpu_ms": neural_cpu * 1000,
            "switches": switch_samples, "hardware": hardware_metrics()}
