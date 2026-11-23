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

def test_remote_training_cycle_real_optimizer_and_resume(tmp_path, monkeypatch):
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
    from hashlib import sha256
    import ancestor.cloud_training as cloud
    preserved = sha256((tmp_path/"checkpoints"/"best.pt").read_bytes()).hexdigest()
    # Force a regression. Latest must advance, but the best weights must not.
    monkeypatch.setattr(cloud, "heldout_validation", lambda *a: 1000.0)
    again = run(Namespace(**params, resume=str(checkpoint)))
    assert again["completed_updates"] == 2
    assert not again["best_improved_this_run"]
    assert again["best_heldout_loss"] == summary["best_heldout_loss"]
    assert sha256((tmp_path/"checkpoints"/"best.pt").read_bytes()).hexdigest() == preserved

def test_regressed_latest_recovers_verified_best_without_rewinding_cursor(tmp_path, monkeypatch):
    from hashlib import sha256
    import ancestor.cloud_training as cloud
    source = tmp_path / "book.txt"
    source.write_text("1. First passage establishes the premise.\\n"
                      "2. Second passage develops the consequence.", encoding="utf-8")
    data = tmp_path / "corpus"
    assert Corpus(data).ingest_text(source, "BOOK", "Owner work") == 2
    tok = tmp_path / "tokens.json"
    ByteBPE([]).save(tok)
    governance = tmp_path / "axioms.json"
    Constitution(("Identity is a unity of self and other.",)).save_initial(governance)
    heldout = tmp_path / "heldout.txt"
    heldout.write_text("Some independent English diagnostic text.", encoding="utf-8")
    out = tmp_path / "out"
    args = dict(data=str(data), tokenizer=str(tok), governance=str(governance),
                heldout=str(heldout), output=str(out), block=1024,
                width=24, layers=1, heads=4, ffn=48,
                updates=1, max_minutes=1, seed=5)

    # The deliberately controlled diagnostic makes the second run regress.
    monkeypatch.setattr(cloud, "heldout_validation",
                        lambda trainer, *rest: float(trainer.completed_updates))
    first = run(Namespace(**args, resume=None))
    assert first["best_improved_this_run"]
    latest = out / "latest.pt"
    second = run(Namespace(**args, resume=str(latest)))
    assert second["completed_updates"] == 2
    assert second["heldout_loss"] > second["best_heldout_loss"]
    best_hash = sha256((out / "best.pt").read_bytes()).hexdigest()

    # Both artifacts must be independently scored; the newer, worse weights
    # are NOT the starting state even though the total update cursor advances.
    inspected = []
    original_restore = cloud.restore_checkpoint
    def observed_restore(path, trainer, **kwargs):
        result = original_restore(path, trainer, **kwargs)
        inspected.append((str(path), trainer.completed_updates,
                          sha256(next(trainer.model.parameters()).detach()
                                 .cpu().numpy().tobytes()).hexdigest()))
        return result
    monkeypatch.setattr(cloud, "restore_checkpoint", observed_restore)
    third = run(Namespace(**args, resume=str(latest)))
    assert third["resume_selection"] == "best"
    assert third["resume_latest_heldout_loss"] == 2.0
    assert third["resume_best_heldout_loss"] == 1.0
    assert third["completed_updates"] == 3
    assert inspected[0][1] == 2 and inspected[1][1] == 1
    assert inspected[0][2] != inspected[1][2]
    assert sha256((out / "best.pt").read_bytes()).hexdigest() == best_hash

def test_controller_shuffles_and_replays_without_skipping_records(tmp_path):
    from ancestor.controller import Controller
    from ancestor.context import ContextBuilder
    from ancestor.ledger import LearningLedger
    source = tmp_path/"work.txt"
    source.write_text("\n".join(f"{i}. Passage number {i} is approved." for i in range(1,9)))
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
