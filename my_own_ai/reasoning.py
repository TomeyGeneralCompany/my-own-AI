"""Shared reasoning state machine. The proposal engine receives data only."""
from dataclasses import asdict, dataclass, field
from enum import Enum
import json

from .module_memory import IDENTIFIER, json_copy

from .grounding import GroundedProposal, SPAN_FORMAT, span_prompt


class Stage(str, Enum):
    FORMULATE = "formulate"
    CONSTRUCT = "construct"
    EVALUATE = "evaluate"
    CORRECT = "correct"


class Status(str, Enum):
    PENDING = "pending"
    REQUIRES_CORRECTION = "requires_correction"
    COMPLETED = "completed"
    UNSUCCESSFUL = "unsuccessful"


@dataclass(frozen=True)
class Evaluation:
    passed: bool
    evidence: str


@dataclass(frozen=True)
class ProposalRequest:
    module: str
    operation: str
    stage: Stage
    procedure: str
    objective: str
    context: dict
    plan: str
    candidate: str
    feedback: str


@dataclass
class Operation:
    project: str
    operation: str
    objective: str
    inputs: dict
    max_corrections: int
    stage: Stage | None = Stage.FORMULATE
    status: Status = Status.PENDING
    context: dict = field(default_factory=dict)
    plan: str = ""
    candidate: str = ""
    feedback: str = ""
    history: list = field(default_factory=list)

    def snapshot(self):
        return json_copy(asdict(self))

    @classmethod
    def restore(cls, data):
        if not isinstance(data, dict) or set(data) != set(cls.__dataclass_fields__):
            raise ValueError("Incomplete operation")
        run = cls(**data)
        if (not isinstance(run.project, str) or not IDENTIFIER.fullmatch(run.project)
                or not isinstance(run.operation, str) or not IDENTIFIER.fullmatch(run.operation)
                or not isinstance(run.inputs, dict) or not isinstance(run.context, dict)
                or not all(isinstance(x, str) for x in (run.objective, run.plan, run.candidate, run.feedback))
                or not run.objective.strip() or type(run.max_corrections) is not int
                or not 0 <= run.max_corrections <= 10 or not isinstance(run.history, list)):
            raise ValueError("Invalid operation data")
        stage, status, corrections = Stage.FORMULATE, Status.PENDING, 0
        for entry in run.history:
            if (not isinstance(entry, dict) or set(entry) != {"stage", "passed", "evidence"}
                    or stage is None or entry["stage"] != stage
                    or not isinstance(entry["evidence"], str) or not entry["evidence"].strip()
                    or (type(entry["passed"]) is not bool if stage == Stage.EVALUATE
                        else entry["passed"] is not None and entry["passed"] is not False)):
                raise ValueError("Invalid reasoning history")
            stage, status, corrections = transition(stage, entry["passed"], corrections, run.max_corrections)
        if run.stage != stage or run.status != status:
            raise ValueError("Reasoning checkpoint disagrees with recorded transitions")
        evaluations = [entry["evidence"] for entry in run.history if entry["stage"] == Stage.EVALUATE]
        if run.feedback != (evaluations[-1] if evaluations else ""):
            raise ValueError("Evaluation feedback disagrees with recorded history")
        proposals = {entry["stage"] for entry in run.history if entry["passed"] is None}
        if ((Stage.FORMULATE in proposals) != bool(run.plan.strip())
                or (Stage.FORMULATE not in proposals and run.context)
                or (Stage.CONSTRUCT in proposals) != bool(run.candidate.strip())):
            raise ValueError("Incomplete reasoning work product")
        run.stage, run.status = stage, status
        return run


def transition(stage, passed, corrections, limit):
    if stage != Stage.EVALUATE and passed is False:
        return None, Status.UNSUCCESSFUL, corrections
    if stage == Stage.FORMULATE:
        return Stage.CONSTRUCT, Status.PENDING, corrections
    if stage == Stage.CONSTRUCT:
        return Stage.EVALUATE, Status.PENDING, corrections
    if stage == Stage.EVALUATE:
        if passed:
            return None, Status.COMPLETED, corrections
        if corrections == limit:
            return None, Status.UNSUCCESSFUL, corrections
        return Stage.CORRECT, Status.REQUIRES_CORRECTION, corrections
    if stage == Stage.CORRECT:
        return Stage.EVALUATE, Status.PENDING, corrections + 1
    raise ValueError(f"Cannot transition from {stage}")


def advance(run, module, engine, tools, prior):
    """Mutate a detached operation. The caller publishes it only after a validated save."""
    stage = run.stage
    passed = None
    if stage == Stage.EVALUATE:
        result = module.evaluate(run.snapshot(), tools)
        if (not isinstance(result, Evaluation) or type(result.passed) is not bool
                or not isinstance(result.evidence, str) or not result.evidence.strip()):
            raise ValueError("Module returned invalid evaluation evidence")
        passed, evidence = result.passed, result.evidence
        run.feedback = evidence
    else:
        context = module.prepare(run.snapshot(), tools, prior) if stage == Stage.FORMULATE else run.context
        procedure = module.procedure(run.operation, stage)
        proposal = engine.propose(ProposalRequest(module.name, run.operation, stage, procedure,
                                                  run.objective, json_copy(context), run.plan,
                                                  run.candidate, run.feedback))
        if isinstance(proposal, GroundedProposal) and stage in (Stage.CONSTRUCT, Stage.CORRECT):
            proposal = module.render_grounded(proposal, context)
        if not isinstance(proposal, str) or not proposal.strip() or len(proposal) > 64_000:
            passed, evidence = False, "Proposal must be nonempty text up to 64,000 characters"
        else:
            if stage == Stage.FORMULATE:
                run.context = context
                run.plan = proposal
            else:
                run.candidate = proposal
            evidence = procedure
    corrections = sum(event["stage"] == Stage.CORRECT for event in run.history)
    run.stage, run.status, _ = transition(stage, passed, corrections, run.max_corrections)
    run.history.append({"stage": stage, "passed": passed, "evidence": evidence})


def _grounded_prompt(request, capacity):
    label = f"\n{request.stage.value.capitalize()}: ".encode()
    directive = request.procedure.encode()[:capacity // 5]
    question = request.objective.encode()[:capacity // 6]
    feedback = request.feedback.encode()[:capacity // 8]
    previous = (request.candidate or request.plan).encode()[:capacity // 12]
    prefix = b"\n".join((directive, question, feedback, previous)) + b"\n"
    remaining = max(0, capacity - len(prefix) - len(label))
    evidence = request.context["evidence"]
    shown = min(len(evidence), max(1, remaining // 90))
    allowance = remaining // max(1, shown)
    excerpts = b"".join(
        f"[{number}] ".encode() + source["text"].encode()[:max(0, allowance - len(f"[{number}] ") - 1)] + b"\n"
        for number, source in enumerate(evidence[:shown], 1)
        if allowance >= 6
    )
    return list((prefix + excerpts + label)[-capacity:])


SUPERVISED_FORMAT = "grounded_supervised_v1"
END_PROPOSAL = 0


def supervised_prefix(request, capacity):
    return _grounded_prompt(request, capacity // 2)


class LocalProposer:
    """Greedy CPU decoding using the existing byte-level Transformer's forward interface."""
    def __init__(self, model, tokens=96):
        if type(tokens) is not int or not 1 <= tokens <= 4096:
            raise ValueError("Expected a token limit between 1 and 4096")
        if any(p.device.type != "cpu" for p in model.parameters()):
            raise ValueError("The modular runtime requires a CPU model")
        self.model = model.eval()
        self.tokens = tokens

    def propose(self, request):
        import torch

        capacity = self.model.config.context_size
        grounded = "evidence" in request.context and request.operation == "grounded_analysis"
        format_name = getattr(self.model, "proposal_format", None)
        if grounded and format_name == SPAN_FORMAT:
            return self._propose_span(request, capacity)
        supervised = grounded and format_name == SUPERVISED_FORMAT
        if supervised:
            anchor = supervised_prefix(request, capacity)
        else:
            anchor = _grounded_prompt(request, max(1, capacity * 3 // 4)) if grounded else []
        if grounded:
            window = anchor
        else:
            label = f"\n{request.stage.value.capitalize()}: ".encode("utf-8")
            prompt = b"\n".join([
                request.procedure.encode("utf-8")[:capacity // 6],
                request.objective.encode("utf-8")[:capacity // 4],
                request.feedback.encode("utf-8")[:capacity // 4],
                request.plan.encode("utf-8")[:capacity // 8],
                request.candidate.encode("utf-8")[:capacity // 8],
            ])
            remaining = max(0, capacity - len(label) - len(prompt) - 1)
            context = json.dumps(request.context, ensure_ascii=False).encode("utf-8")[:remaining]
            window = list((context + b"\n" + prompt + label)[-capacity:])
        generated = bytearray()
        limit = min(self.tokens, capacity - len(anchor) + 1) if supervised else self.tokens
        with torch.inference_mode():
            for _ in range(limit):
                logits, _ = self.model(torch.tensor([window], dtype=torch.long, device="cpu"))
                token = int(logits[0, -1].argmax())
                if supervised and token == END_PROPOSAL:
                    break
                generated.append(token)
                if anchor and len(anchor) < capacity:
                    window = anchor + list(generated[-(capacity - len(anchor)):])
                else:
                    window = (window + [token])[-capacity:]
        return generated.decode("utf-8", errors="replace")


    def _propose_span(self, request, capacity):
        import torch

        prefix, excerpts = span_prompt(request, capacity)
        choices = tuple(excerpts) if request.stage != Stage.FORMULATE else ()
        if request.stage != Stage.FORMULATE and not choices:
            return GroundedProposal(0, "", "", False)
        output = bytearray()
        selected = None
        finished = False
        with torch.inference_mode():
            for step in range(min(self.tokens, capacity - len(prefix) + 1)):
                window = prefix + list(output)
                logits, _ = self.model(torch.tensor([window], dtype=torch.long, device="cpu"))
                scores = logits[0, -1]
                if step == 0 and choices:
                    allowed = [ord(str(choice)) for choice in choices]
                    token = allowed[int(scores[allowed].argmax())]
                    selected = int(chr(token))
                else:
                    token = int(scores.argmax())
                if token == END_PROPOSAL:
                    finished = True
                    break
                output.append(token)
        if selected is None:
            return output.decode("utf-8", errors="replace")
        return GroundedProposal(selected, excerpts[selected], output[1:].decode("utf-8", errors="replace"), finished)
