import pytest
import torch
from ancestor.model import AncestorModel,ModelConfig
from ancestor.training import Trainer,TrainConfig,masked_next_token_loss,structured_prefix_batch
from ancestor.tokenizer import ByteBPE

def tiny_model():
    return AncestorModel(ModelConfig(vocab_size=266,context_length=32,
                                     width=24,layers=1,heads=4,ffn_width=48))

def test_forward_backward_and_future_mask():
    torch.manual_seed(2)
    model=tiny_model().eval()
    a=torch.tensor([[1,2,3,4]])
    b=torch.tensor([[1,2,3,5]])
    with torch.no_grad():
        l1,l2=model(a),model(b)
    assert torch.allclose(l1[:,:3],l2[:,:3],atol=1e-5)
    trainer=Trainer(model,TrainConfig(block_size=4))
    loss=trainer.step([(a,b,None)])
    assert loss>=0
    assert trainer.completed_updates==1

def test_masked_prefix_targets():
    t=ByteBPE([])
    x,y,m=structured_prefix_batch(t,[("<|AXIOM|>","x"),("<|CONTEXT|>","y"),
                                     ("<|DERIVE|>","z")],64)
    assert x.shape==y.shape==m.shape
    assert m.sum().item()>=2
    assert m[0,0].item()==0
    with pytest.raises(ValueError):
        structured_prefix_batch(t,[("<|DERIVE|>","x")],1)
