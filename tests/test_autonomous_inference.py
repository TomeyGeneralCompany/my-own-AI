"""End-to-end tests for neural-origin hypotheses without supplied theses."""
import torch

from my_own_ai.autonomous_inference import GeneratingProposer, infer
from my_own_ai.model import LocalTransformer, ModelConfig
from my_own_ai.reasoning import ProposalRequest, Stage, SUPERVISED_FORMAT
from my_own_ai.retrieval import build_index


class Judge:
    def __init__(self, selected=0):
        self.selected = selected
        self.inputs = []

    def rank(self, question, evidence, candidates):
        self.inputs.append((question, evidence, candidates))
        return {"selected": self.selected}


def small_model():
    torch.set_num_threads(1)
    model = LocalTransformer(ModelConfig(context_size=256, embed_dim=16, heads=4, layers=1))
    model.proposal_format = SUPERVISED_FORMAT
    return model


def request(evidence):
    return ProposalRequest("research", "grounded_analysis", Stage.CONSTRUCT, "Use exact evidence",
                           "What happens when the signal arrives late?",
                           {"evidence": [{"text": evidence}]}, "Analyze the signals", "", "")


def test_generated_candidates_not_supplied_by_question(monkeypatch):
    source = "The keeper responds only after the signal arrives, so a late signal delays the response."
    first = f"Premises:\n- [1] {source}\nInference: The late signal delays the keeper's response."
    second = f"Premises:\n- [1] {source}\nInference: The keeper waits for the signal before acting."
    model = small_model()
    judge = Judge(selected=1)
    proposer = GeneratingProposer(model, judge, attempts=2, tokens=48)
    options = iter([first, second])
    monkeypatch.setattr(proposer, "_sample", lambda _: next(options))
    result = proposer.propose(request(source))
    assert result == second
    assert proposer.last_generation == {"generated": 2, "valid": 2, "selected": 1}
    assert judge.inputs[0][2] == [first.split("Inference: ")[1], second.split("Inference: ")[1]]


def test_no_fabricated_success_when_neural_candidates_invalid(monkeypatch):
    source = "The signal arrives after the keeper has left."
    proposer = GeneratingProposer(small_model(), Judge(), attempts=2, tokens=32)
    monkeypatch.setattr(proposer, "_sample", lambda _: "An unsupported claim.")
    assert proposer.propose(request(source)) == "An unsupported claim."
    assert proposer.last_generation["valid"] == 0


def test_research_cycle_uses_neural_origin_and_real_source(monkeypatch, tmp_path):
    source = "The keeper responds only after the signal arrives, so a late signal delays the response."
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "signals.txt").write_text(source, encoding="utf-8")
    index = tmp_path / "index.sqlite"
    build_index(corpus, index)
    first = f"Premises:\n- [1] {source}\nInference: The keeper's response follows the signal's arrival."
    second = f"Premises:\n- [1] {source}\nInference: A late signal delays the keeper's response."
    options = iter([first, second])
    monkeypatch.setattr(GeneratingProposer, "_sample", lambda self, _: next(options))
    # The formulation proposal remains neural; the hypothesis samples are replaced
    # in this deterministic integration test so controller boundaries can be tested.
    result = infer(index, small_model(), Judge(selected=1),
                   "What happens when the signal arrives late?", attempts=2, tokens=32)
    assert result["status"] == "completed"
    assert result["inference"] == "A late signal delays the keeper's response."
    assert len(result["citations"]) == 1
    assert result["stages"] == ["formulate", "construct", "evaluate"]
    assert result["semantic_entailment_verified"] is False


def test_real_random_weight_sampler_produces_neural_bytes():
    proposer = GeneratingProposer(small_model(), Judge(), attempts=2, tokens=32, seed=1)
    result = proposer._sample(request("The keeper responds only after the signal arrives."))
    assert isinstance(result, str)
    assert len(result.encode("utf-8", errors="replace")) <= 32 * 3
