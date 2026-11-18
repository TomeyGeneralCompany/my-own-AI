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
    assert summary["training_status"] == "FOUNDATIONAL_EXPOSURE_NOT_CERTIFIED"
    checkpoint = tmp_path/"checkpoints"/"latest.pt"
    assert checkpoint.exists()
    assert checkpoint.with_suffix(".json").exists()
    again = run(Namespace(**params, resume=str(checkpoint)))
    assert again["completed_updates"] == 2
