import torch
import pytest
from my_own_ai.model import LocalTransformer, ModelConfig, encode, decode
from my_own_ai.train import train, load_model

def test_unicode():
    assert decode(encode("An argument — Ελληνικά")) == "An argument — Ελληνικά"

def test_future_is_masked():
    torch.manual_seed(1)
    model = LocalTransformer(ModelConfig(context_size=8, embed_dim=16, heads=4, layers=1)).eval()
    a, _ = model(torch.tensor([[65, 66, 67, 68]]))
    b, _ = model(torch.tensor([[65, 66, 70, 72]]))
    assert torch.allclose(a[:, :2], b[:, :2], atol=1e-5)

def test_train_and_reload(tmp_path):
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "sample.txt").write_text("An original training sentence. " * 12)
    checkpoint = tmp_path / "model.pt"
    cfg = ModelConfig(context_size=8, embed_dim=16, heads=4, layers=1)
    stats = train(source, checkpoint, cfg, steps=2, batch_size=2)
    assert stats["parameters"] > 0
    assert len(load_model(checkpoint).generate(b"reason", 3)) == 9

def test_empty_corpus(tmp_path):
    with pytest.raises(ValueError, match="no .txt"):
        train(tmp_path, tmp_path / "out.pt", ModelConfig(context_size=8))
