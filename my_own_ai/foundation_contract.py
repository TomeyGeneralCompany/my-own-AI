"""Owner-defined foundations; external references never become weight updates."""
from dataclasses import dataclass
from hashlib import sha256
from contextlib import contextmanager

AXIOMS = (
    "Identity is a unity of self and other.",
    "Communion is hierarchical.",
)


@dataclass(frozen=True)
class FoundationContract:
    axioms: tuple = AXIOMS

    def __post_init__(self):
        if self.axioms != AXIOMS:
            raise ValueError("Foundation changes require an owner-reviewed code revision")

    def record(self):
        return {"axioms": list(self.axioms), "authority": "owner-authored corpus",
                "external_material": "reference evidence, not replacement axioms",
                "derived_conclusions": "revisable", "operational_weight_updates": False}


def model_digest(model):
    h = sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(bytes(tensor.detach().cpu().contiguous().reshape(-1).view(__import__("torch").uint8).tolist()))
    return h.hexdigest()


@contextmanager
def immutable_inference(model):
    """Detect mutation, preserve the original weights and disable gradient updates."""
    import torch
    snapshot = {name: tensor.detach().clone() for name,tensor in model.state_dict().items()}
    before = model_digest(model)
    model.eval()
    model.requires_grad_(False)
    try:
        with torch.inference_mode():
            yield
    finally:
        if model_digest(model) != before:
            model.load_state_dict(snapshot)
            raise RuntimeError("Operational inference attempted to alter foundational weights")
