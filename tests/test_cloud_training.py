"""Real fresh-weight off-laptop training and stateful continuation."""
from argparse import Namespace
from ancestor.cloud_training import run, safe_token_segments
from ancestor.corpus import Corpus
from ancestor.governance import Constitution
from ancestor.tokenizer import ByteBPE

def test_utf8_segments_preserve_complete_original():
    t = ByteBPE([])
    text = "A great many ελληνικά characters—are retained. " * 20
    segments = safe_token_segments(t, text, 45)
    assert len(segments) > 1
    assert "".join(segments) == text

def test_remote_training_cycle_real_optimizer_and_resume(tmp_path):
    source = tmp_path/"work.txt"
    source.write_text("1. Truth proceeds from the governing principle.\n"
                      "2. The consequent conclusion follows the original premises.",
                      encoding="utf-8")
    data = tmp_path/"data"
    corpus = Corpus(data)
    assert corpus.ingest_text(source, "BOOK", "Test owner work") == 2
    tokenizer = ByteBPE([])
    tok = tmp_path/"tokenizer.json"
    tokenizer.save(tok)
    governance = tmp_path/"governance.json"
    Constitution(("Identity is a unity of self and other.",)).save_initial(governance)
    heldout = tmp_path/"heldout.txt"
    heldout.write_text("Previously unseen English text with different content.",
                       encoding="utf-8")
    params = dict(data=str(data),tokenizer=str(tok),governance=str(governance),
                  heldout=str(heldout),output=str(tmp_path/"checkpoints"),
                  block=1024,width=24,layers=1,heads=4,ffn=48,
                  updates=1,max_minutes=1,seed=5)
    summary = run(Namespace(**params, resume=None))
    assert summary["completed_updates"] == 1
    assert summary["learning_units"] == 2
    assert summary["best_improved_this_run"]
    assert summary["best_heldout_loss"] == summary["heldout_loss"]
    checkpoint = tmp_path/"checkpoints"/"latest.pt"
    assert checkpoint.exists()
    assert checkpoint.with_suffix(".json").exists()
    assert (tmp_path/"checkpoints"/"best.pt").exists()
    assert (tmp_path/"checkpoints"/"best.json").exists()
    again = run(Namespace(**params, resume=str(checkpoint)))
    assert again["completed_updates"] == 2
    assert (tmp_path/"checkpoints"/"best.pt").exists()

def test_controller_shuffles_and_replays_without_skipping_records(tmp_path):
    from ancestor.controller import Controller
    from ancestor.context import ContextBuilder
    from ancestor.ledger import LearningLedger
    source = tmp_path/"work.txt"
    source.write_text("\\n".join(f"{i}. Passage number {i} is approved." for i in range(1,9)))
    corpus = Corpus(tmp_path/"corpus")
    corpus.ingest_text(source,"BOOK","Source")
    constitution = Constitution(("Identity is a unity of self and other.",))
    controller = Controller(corpus, ContextBuilder(ByteBPE([]),constitution,1024),
                            LearningLedger(tmp_path/"events.jsonl"), constitution)
    from ancestor.cloud_training import source_records
    records = source_records(corpus)
    first = [controller.cloud_exposure_selection(records,i,1729)[0].id for i in range(len(records))]
    second = [controller.cloud_exposure_selection(records,i+len(records),1729)[0].id for i in range(len(records))]
    assert set(first) == set(second) == {r.id for r in records}
    assert len(first) == len(set(first))
    assert first != second
    assert not controller.ledger.events()

def test_heldout_validation_uses_all_segments(tmp_path):
    from ancestor.cloud_training import heldout_validation
    from ancestor.model import AncestorModel,ModelConfig
    from ancestor.training import Trainer,TrainConfig
    tok=ByteBPE([])
    const=Constitution(("Identity is a unity of self and other.",))
    model=AncestorModel(ModelConfig(tok.vocab_size,1024,24,1,4,48))
    trainer=Trainer(model,TrainConfig(block_size=1024))
    long_text="Independent phrase with consequences. " * 120
    score=heldout_validation(trainer,tok,const,long_text)
    assert score > 0
