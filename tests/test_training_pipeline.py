import json
import torch
import pytest
from my_own_ai.corpus import collect_documents, build_manifest, split_documents
from my_own_ai.model import ModelConfig
from my_own_ai.train import train, load_model, benchmark

CFG = ModelConfig(context_size=8, embed_dim=16, heads=4, layers=1)

def corpus(tmp_path):
    folder = tmp_path / "books"
    folder.mkdir()
    for n in range(5):
        (folder / f"book{n}.txt").write_text(("chapter " + str(n) + " some prose and ideas. ") * 8)
    return folder

def test_ingestion_dedupe_and_manifest(tmp_path):
    folder = corpus(tmp_path)
    (folder / "duplicate.txt").write_text((folder / "book0.txt").read_text())
    docs = collect_documents(folder)
    assert len(docs) == 5
    manifest = build_manifest(folder, tmp_path / "manifest.json")
    assert len(manifest["documents"]) == 5
    assert "text" not in json.loads((tmp_path / "manifest.json").read_text())["documents"][0]

def test_document_disjoint_validation(tmp_path):
    docs = collect_documents(corpus(tmp_path))
    train_docs, valid_docs, method = split_documents(docs, 8)
    assert method == "document-disjoint"
    assert not set(train_docs).intersection(valid_docs)

def test_train_resume_preserves_training_state(tmp_path):
    folder = corpus(tmp_path)
    ck = tmp_path / "brain.pt"
    first = train(folder, ck, CFG, steps=2, batch_size=2, checkpoint_every=1)
    second = train(folder, ck, CFG, steps=4, batch_size=2, resume=True, checkpoint_every=1)
    assert second["steps_completed_this_run"] == 2
    assert second["validation_loss"] is not None
    assert second["documents"] == 5
    assert len(load_model(ck).generate(b"Hi", 2)) == 4
    assert benchmark(ck, iterations=1, generation_tokens=2)["generation_tokens_per_second"] > 0

def test_resume_matches_uninterrupted(tmp_path):
    folder = corpus(tmp_path)
    ck1, ck2 = tmp_path / "one.pt", tmp_path / "two.pt"
    train(folder, ck1, CFG, steps=4, batch_size=2, checkpoint_every=2)
    train(folder, ck2, CFG, steps=2, batch_size=2, checkpoint_every=2)
    train(folder, ck2, CFG, steps=4, batch_size=2, resume=True, checkpoint_every=2)
    a, b = load_model(ck1), load_model(ck2)
    for x, y in zip(a.parameters(), b.parameters()):
        assert torch.equal(x, y)

def test_resume_rejects_changed_corpus(tmp_path):
    folder = corpus(tmp_path)
    ck = tmp_path / "brain.pt"
    train(folder, ck, CFG, steps=1, batch_size=2)
    (folder / "book0.txt").write_text("a different corpus with enough bytes to be longer than the window")
    with pytest.raises(ValueError, match="Corpus or split"):
        train(folder, ck, CFG, steps=2, batch_size=2, resume=True)

def test_docx_ingestion(tmp_path):
    from docx import Document
    directory = tmp_path / "files"
    directory.mkdir()
    doc = Document()
    doc.add_paragraph("Original commentary and analysis.")
    doc.save(directory / "book.docx")
    assert "Original commentary" in collect_documents(directory)[0]["text"]

def test_single_document_fallback(tmp_path):
    folder = tmp_path / "one"
    folder.mkdir()
    (folder / "book.txt").write_text("A long chapter for training and validation. " * 15)
    training, validation, policy = split_documents(collect_documents(folder), 8)
    assert training and validation and policy == "single-document-byte-split"
