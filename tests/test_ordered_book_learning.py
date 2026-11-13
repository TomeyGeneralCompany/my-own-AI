"""Actual tiny neural reading regression: every source unit, zero skips, exact resume."""
from dataclasses import asdict
import torch
import pytest

from my_own_ai.model import ModelConfig, LocalTransformer
from my_own_ai.tokenizer import ByteBPETokenizer
from my_own_ai.ordered_book_learning import source_units, read_all_units


def test_identical_source_paragraphs_are_not_silently_dropped(tmp_path):
    text = "The relation constitutes identity and trust reduces friction."
    for name in ("book_p000001.txt", "book_p000002.txt", "book_p000003.txt"):
        (tmp_path/name).write_text(text)
    units = source_units(tmp_path)
    assert len(units) == 3
    assert len({u["id"] for u in units}) == 3
    assert [u["source"] for u in units] == sorted(u["source"] for u in units)


def test_real_neural_reader_covers_all_units_and_resumes_without_replay(tmp_path):
    folder = tmp_path/"book"
    folder.mkdir()
    for index, text in enumerate((
        "Identity is a unity of self and other. This relation is constitutive.",
        "Trust reduces coordination friction, making reliable cooperation possible.",
        "Virtue sustains the reliable conduct upon which commerce depends.",
    )):
        (folder/f"book_p{index+1:06d}.txt").write_text(text)
    tok = ByteBPETokenizer()
    tokenizer_path = tmp_path/"tokenizer.json"
    tok.save(tokenizer_path)
    cfg = ModelConfig(vocab_size=tok.vocab_size, context_size=64,
                      embed_dim=16, heads=4, layers=1)
    model = LocalTransformer(cfg)
    ancestor = tmp_path/"ancestor.pt"
    torch.save({"language_format":1, "model_config":asdict(cfg),
                "tokenizer_sha256":tok.fingerprint(),
                "state_dict":model.state_dict()},ancestor)
    out=tmp_path/"reading.pt"
    args=dict(tokenizer_path=tokenizer_path, source=folder,
              output=out, parent=ancestor, max_updates=1,
              context=64, overlap=8, embed=16, layers=1,
              heads=4, batch_size=1, threads=1)
    prior=0
    for _ in range(100):
        result=read_all_units(**args)
        record=torch.load(out,map_location="cpu",weights_only=True)
        assert record["supervised_tokens"] >= prior
        prior=record["supervised_tokens"]
        assert record["next_window"] <= 100
        if result["read_complete"]:
            break
    else:
        pytest.fail("Complete book did not finish in finite source windows")
    assert record["supervised_tokens"] == record["total_target_tokens"]
    assert result["source_units"] == 3
    assert result["next_window"] == 0
    with pytest.raises(ValueError,match="Reading resume changed"):
        (folder/"book_p000004.txt").write_text("A new, different approved unit.")
        read_all_units(**args)
