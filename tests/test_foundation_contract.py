import pytest
import torch
from my_own_ai.foundation_contract import FoundationContract, immutable_inference, model_digest
from scripts.ask_modern_core import select_checkpoint


def test_owner_axioms_fixed():
    contract=FoundationContract()
    assert not contract.record()["operational_weight_updates"]
    with pytest.raises(ValueError):
        FoundationContract(("Replacement premise",))
    with pytest.raises(AttributeError):
        contract.axioms=()


def test_inference_does_not_mutate_weights():
    model=torch.nn.Linear(2,2)
    before=model_digest(model)
    with immutable_inference(model):
        assert not torch.is_grad_enabled()
        assert not model.weight.requires_grad
        model(torch.ones(1,2))
    assert model_digest(model)==before
    with pytest.raises(RuntimeError,match="alter foundational"):
        with immutable_inference(model):
            model.weight.add_(1)
    assert model_digest(model)==before


def test_modern_ask_selects_current_english(tmp_path):
    for name in ("inherited.pt","current.best.pt","english.pt","foundation.pt"):
        (tmp_path/name).touch()
    assert select_checkpoint(tmp_path,{"stage":"english","english_fundamentals_passed":True}).name=="english.pt"
    assert select_checkpoint(tmp_path,{"stage":"english","english_fundamentals_passed":False}).name=="foundation.pt"
    assert select_checkpoint(tmp_path,{"stage":"qualified"}).name=="current.best.pt"
