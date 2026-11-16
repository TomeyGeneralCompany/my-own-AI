import pytest
import torch

from my_own_ai.finance_module import Finance
from my_own_ai.module_bpe import BPEProposer
from my_own_ai.model import LocalTransformer, ModelConfig
from my_own_ai.modules import ModuleController, CheckpointStore
from my_own_ai.reasoning import ProposalRequest, Stage, Status
from my_own_ai.retrieval import build_index
from my_own_ai.tokenizer import ByteBPETokenizer


def test_real_bpe_proposal_uses_original_model_without_mutation():
    tok = ByteBPETokenizer()
    model = LocalTransformer(ModelConfig(context_size=512, embed_dim=16, heads=2, layers=1, vocab_size=tok.vocab_size))
    proposer = BPEProposer(model, tok, tokens=1)
    request = ProposalRequest("finance","library_research",Stage.FORMULATE,"Review source.","Study prices.",{},"","","")
    before = {k:v.clone() for k,v in model.state_dict().items()}
    # A random checkpoint may produce an invalid UTF-8 byte: that is real output,
    # not grounds to replace it with a meaningful canned response.
    calls=[]
    hook=model.register_forward_hook(lambda *args: calls.append(True))
    try:
        try:
            result=proposer.propose(request)
            assert isinstance(result,str)
        except UnicodeDecodeError:
            pass
    finally:
        hook.remove()
    assert calls
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    with pytest.raises(ValueError,match="context window"):
        proposer.propose(ProposalRequest("finance","library_research",Stage.FORMULATE,"Review","x"*1000,{},"","",""))


class Fixture:
    def propose(self, request):
        if request.stage == Stage.FORMULATE:
            return "Read the supplied source."
        return "Premises:\n- [1] Prices convey information.\nInference: Prices can convey information to participants."


def test_finance_retrieves_and_preserves_source_through_restart(tmp_path):
    corpus=tmp_path/"books"
    corpus.mkdir()
    (corpus/"fixture.txt").write_text("Prices convey information.")
    index=tmp_path/"finance.sqlite"
    build_index(corpus,index)
    store=CheckpointStore(tmp_path/"memory")
    with ModuleController(store,[Finance(tmp_path)],Fixture()) as core:
        core.switch("finance")
        core.start("library_research","What information do prices convey?",inputs={"index":"finance.sqlite","limit":1})
        result=core.step()
        assert result["context"]["evidence"][0]["source"]=="fixture.txt"
    index.unlink()
    with ModuleController(store,[Finance(tmp_path)],Fixture()) as core:
        core.switch("finance")
        core.step()
        result=core.step()
        assert result["status"]==Status.COMPLETED
        assert "substantive inference requires independent review" in result["feedback"]
    with pytest.raises(ValueError):
        Finance(tmp_path).search_library("../outside.sqlite","prices",1)
