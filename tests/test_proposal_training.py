from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
import torch

from my_own_ai.model import LocalTransformer, ModelConfig
from my_own_ai.proposal_training import encode_example, load_examples, train_proposals, verify_splits
from my_own_ai.reasoning import END_PROPOSAL, SUPERVISED_FORMAT, LocalProposer, ProposalRequest, Stage
from my_own_ai.train import load_model


def row(group="beacon", question="Would clearance delay a warning?", stage="construct"):
    return {"id": f"{group}_{stage}", "group": group, "stage": stage, "question": question,
            "passages": ["Send the hazard code before seeking clearance."],
            "plan": "Identify which step occurs first.", "candidate": "", "feedback": "",
            "answer": "Premises:\n- [1] Send the hazard code\nInference: Requiring clearance first can delay the warning."}


def write_rows(tmp_path, name, rows):
    path = tmp_path / name
    path.write_text(json.dumps(rows))
    return path


def test_targets_mask_prompt_and_use_only_displayed_evidence():
    example = row()
    inputs, targets = encode_example(example, 384)
    answer = list(example["answer"].encode())
    assert targets[:-(len(answer) + 1)].unique().tolist() == [-100]
    assert targets[-(len(answer) + 1):].tolist() == answer + [END_PROPOSAL]
    assert inputs[-len(answer):].tolist() == answer
    example["passages"] = ["X" * 250 + " Send the hazard code before seeking clearance."]
    with pytest.raises(ValueError, match="outside the model prompt"):
        encode_example(example, 384)


def test_leakage_and_invalid_stage_or_target_are_rejected(tmp_path):
    train = [row()]
    validation = [row("dispatcher", "What does the dispatcher record?")]
    verify_splits(train, validation, [{"question": "Different objective"}])
    with pytest.raises(ValueError, match="disjoint"):
        verify_splits(train, validation, [{"question": " WOULD clearance delay a warning!!! "}])
    with pytest.raises(ValueError, match="disjoint"):
        verify_splits(train, [row("beacon", "Unrelated question")], [])
    invalid = write_rows(tmp_path, "invalid.json", [row(stage="evaluate")])
    with pytest.raises(ValueError, match="controller-owned"):
        load_examples(invalid)
    bad_quote = row()
    bad_quote["answer"] = bad_quote["answer"].replace("Send the hazard code", "Invent the hazard code")
    with pytest.raises(ValueError, match="Unverified"):
        load_examples(write_rows(tmp_path, "invented.json", [bad_quote]))


def test_cpu_training_preserves_parent_and_saves_compatible_weights(tmp_path):
    torch.manual_seed(7)
    config = ModelConfig(context_size=384, embed_dim=16, heads=4, layers=1)
    parent_model = LocalTransformer(config)
    checkpoint = tmp_path / "parent.pt"
    torch.save({"model_config": asdict(config), "state_dict": parent_model.state_dict()}, checkpoint)
    before = checkpoint.read_bytes()
    train = write_rows(tmp_path, "train.json", [row()])
    valid = write_rows(tmp_path, "valid.json", [row("dispatcher", "What does the dispatcher record?")])
    output = tmp_path / "worked.pt"
    with pytest.raises(ValueError, match="separate output"):
        train_proposals(checkpoint, train, valid, checkpoint, steps=1)
    stats = train_proposals(checkpoint, train, valid, output, steps=3, threads=1)
    trained = load_model(output)
    assert trained.proposal_format == SUPERVISED_FORMAT
    assert checkpoint.read_bytes() == before
    assert stats["steps"] == 3 and stats["hardware"]["device"] == "cpu"
    assert any(not torch.equal(a, b) for a, b in zip(parent_model.parameters(), trained.parameters()))


def test_trained_proposer_stops_at_end_marker_only_for_grounded_operation():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(context_size=384)
            self.proposal_format = SUPERVISED_FORMAT
            self.inputs = []

        def forward(self, inputs):
            self.inputs.append(inputs)
            logits = torch.zeros((1, inputs.shape[1], 256))
            token = ord("A") if len(self.inputs) == 1 else END_PROPOSAL
            logits[0, -1, token] = 1
            return logits, None

    model = Toy()
    engine = LocalProposer(model, tokens=10)
    request = ProposalRequest("research", "grounded_analysis", Stage.CONSTRUCT, "Draft",
                              "What follows?", {"evidence": [{"text": "Displayed source"}]},
                              "Plan", "", "")
    assert engine.propose(request) == "A"
    assert bytes(model.inputs[0][0].tolist()) in bytes(model.inputs[1][0].tolist())
