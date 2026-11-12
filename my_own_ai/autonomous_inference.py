"""Experimental autonomous hypothesis generation, neural ranking and Research verification.

All candidate inferences originate in the local trained Transformer. The neural judge
compares generated alternatives, but has no access to ground-truth answer labels.
The Research controller retains authority over evidence format and completion.
"""
from tempfile import TemporaryDirectory
from pathlib import Path

import torch

from .functional_modules import grounded_evidence
from .grounding import SPAN_FORMAT
from .modules import CheckpointStore, ModuleController, default_modules
from .reasoning import LocalProposer, Stage, SUPERVISED_FORMAT, END_PROPOSAL, supervised_prefix


class GeneratingProposer:
    """Use local neural samples, never owner-supplied candidate conclusions."""

    def __init__(self, model, judge, *, attempts=4, tokens=160, temperature=.85, seed=42):
        if getattr(model, "proposal_format", None) != SUPERVISED_FORMAT:
            raise ValueError("Train a grounded_supervised_v1 proposal checkpoint first")
        if type(attempts) is not int or not 2 <= attempts <= 8:
            raise ValueError("Generation attempts must be between 2 and 8")
        if type(tokens) is not int or not 32 <= tokens <= 4096:
            raise ValueError("Generation tokens must be between 32 and 4096")
        if not 0 < temperature <= 2:
            raise ValueError("Invalid generation temperature")
        if type(seed) is not int or seed < 0:
            raise ValueError("Invalid sampling seed")
        if any(param.device.type != "cpu" for param in model.parameters()):
            raise ValueError("The generating model must run on CPU")
        self.model = model.eval()
        self.judge = judge
        self.formulator = LocalProposer(model, tokens=tokens)
        self.attempts = attempts
        self.tokens = tokens
        self.temperature = temperature
        self.rng = torch.Generator(device="cpu").manual_seed(seed)
        self.last_generation = None

    @torch.inference_mode()
    def _sample(self, request):
        capacity = self.model.config.context_size
        prefix = supervised_prefix(request, capacity)
        if not prefix or len(prefix) >= capacity:
            return ""
        output = bytearray()
        for _ in range(min(self.tokens, capacity - len(prefix))):
            window = prefix + list(output)
            logits, _ = self.model(torch.tensor([window], dtype=torch.long))
            scores = logits[0, -1] / self.temperature
            token = int(torch.multinomial(torch.softmax(scores, dim=-1), 1, generator=self.rng))
            if token == END_PROPOSAL:
                break
            output.append(token)
        return output.decode("utf-8", errors="replace")

    def propose(self, request):
        if request.stage == Stage.FORMULATE:
            return self.formulator.propose(request)
        if request.stage not in (Stage.CONSTRUCT, Stage.CORRECT) or request.operation != "grounded_analysis":
            return self.formulator.propose(request)
        sources = request.context.get("evidence", [])
        if not sources:
            self.last_generation = {"generated": 0, "valid": 0, "selected": None}
            return "No evidence was retrieved."
        unique = {}
        raw = []
        for _ in range(self.attempts):
            candidate = self._sample(request)
            raw.append(candidate)
            citations, inference, error = grounded_evidence(candidate, sources)
            if not error and inference not in unique:
                unique[inference] = candidate
        alternatives = list(unique)
        selected = None
        if len(alternatives) >= 2:
            assessment = self.judge.rank(request.objective, [p["text"] for p in sources], alternatives)
            selected = assessment["selected"]
        self.last_generation = {"generated": len(raw), "valid": len(alternatives),
                                "selected": selected}
        if selected is not None:
            return unique[alternatives[selected]]
        # Refuse to manufacture a proof when the model and judge cannot agree.
        # A malformed first attempt is returned to the controller for feedback.
        return raw[0] if raw else ""


def infer(index_path, model, judge, question, *, limit=5, attempts=4, tokens=160,
          temperature=.85, seed=42, max_corrections=2):
    """Run the existing Research state machine without any supplied thesis."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("A nonempty question is required")
    engine = GeneratingProposer(model, judge, attempts=attempts, tokens=tokens,
                                temperature=temperature, seed=seed)
    with TemporaryDirectory(prefix="my-own-ai-neural-inference-") as directory:
        with ModuleController(CheckpointStore(Path(directory)),
                              default_modules(corpus_index=index_path), engine) as controller:
            controller.switch("research")
            run = controller.start("grounded_analysis", question, project="neural_inference",
                                   inputs={"limit": limit}, max_corrections=max_corrections)
            while run["stage"] is not None:
                run = controller.step()
    premises = run["context"].get("evidence", [])
    cited, inference, error = grounded_evidence(run["candidate"], premises)
    return {
        "status": run["status"],
        "candidate": run["candidate"],
        "inference": inference if not error else None,
        "citations": [{"source": premises[n - 1]["source"], "locator": premises[n - 1]["locator"]}
                      for n in sorted(cited)],
        "controller_feedback": run["feedback"],
        "generation": engine.last_generation,
        "stages": [event["stage"] for event in run["history"]],
        "semantic_entailment_verified": False,
    }
