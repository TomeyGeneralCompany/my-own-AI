import random
import torch
import pytest
from ancestor.model import AncestorModel,ModelConfig
from ancestor.training import Trainer,TrainConfig
from ancestor.checkpoints import save_checkpoint,restore_checkpoint
from ancestor.governance import Constitution

def make_trainer():
    return Trainer(AncestorModel(ModelConfig(vocab_size=266,context_length=8,
                  width=24,layers=1,heads=4,ffn_width=48)),
                  TrainConfig(block_size=8))

def test_real_checkpoint_resume(tmp_path):
    random.seed(19);torch.manual_seed(19)
    trainer=make_trainer()
    batch=(torch.randint(0,266,(1,8)),torch.randint(0,266,(1,8)),None)
    trainer.step([batch])
    c=Constitution(("The foundational premise is fixed.",))
    cp=save_checkpoint(tmp_path,trainer,tokenizer_hash="tokenizer-1",
                       constitution=c,corpus_manifest_hash="source-1",
                       stage="ENGLISH",objective_id="L1")
    # The uninterrupted continuation is compared against restored continuation.
    random.seed(31)
    trainer.step([batch])
    expected={k:v.detach().clone() for k,v in trainer.model.state_dict().items()}
    restored=make_trainer()
    state=restore_checkpoint(cp,restored,tokenizer_hash="tokenizer-1",
                constitution=c,corpus_manifest_hash="source-1")
    assert state["completed_updates"]==1
    restored.step([batch])
    for k,v in expected.items():
        assert torch.allclose(v,restored.model.state_dict()[k],atol=1e-6)
    with pytest.raises(ValueError):
        restore_checkpoint(cp,make_trainer(),tokenizer_hash="wrong",constitution=c,
                           corpus_manifest_hash="source-1")
