import tempfile
from pathlib import Path
import pytest
from ancestor.tokenizer import ByteBPE

def test_lossless_utf8_and_reserved_text():
    text="§ 42. Identity—κόσμος \n α≤β <|AXIOM|> 'œ'"
    t=ByteBPE.train([text]*3,320)
    assert t.decode(t.encode(text))==text
    assert t.encode("<|AXIOM|>") != [t.special_id("<|AXIOM|>")]
    assert t.encode(text)==t.encode(text)
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/"tokenizer.json"
        t.save(path)
        same=ByteBPE.load(path)
        assert same.digest()==t.digest()
        assert same.decode(same.encode(text))==text

def test_structured_markers_are_atomic():
    t=ByteBPE([])
    ids=t.encode_structured([("<|AXIOM|>","Principle"),("<|DERIVE|>","Result")])
    assert ids[0]==t.special_id("<|AXIOM|>")
    assert t.special_id("<|DERIVE|>") in ids
    with pytest.raises(ValueError):
        t.decode(ids)
