"""Private release upload interruption cannot erase last verified state."""
from hashlib import sha256
import json
from ancestor.remote_checkpoints import choose_recovery

def record(directory, step, *, corrupt=False):
    directory.mkdir(parents=True,exist_ok=True)
    data=("weights step "+str(step)).encode()
    (directory/"latest.pt").write_bytes(data)
    (directory/"latest.json").write_text(json.dumps({
        "completed_updates":step,
        "sha256":"bad" if corrupt else sha256(data).hexdigest(),
    }),encoding="utf-8")

def test_latest_valid_slot_wins(tmp_path):
    a,b=tmp_path/"a",tmp_path/"b"
    assert choose_recovery(a,b)["resume"]==""
    record(a,7)
    selection=choose_recovery(a,b)
    assert selection["completed_updates"]==7
    assert selection["next_slot"]=="b"
    record(b,11)
    selection=choose_recovery(a,b)
    assert selection["completed_updates"]==11
    assert selection["next_slot"]=="a"
    record(a,15,corrupt=True)
    selection=choose_recovery(a,b)
    assert selection["completed_updates"]==11
    assert selection["next_slot"]=="a"
