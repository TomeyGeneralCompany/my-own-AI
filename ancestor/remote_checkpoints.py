"""Select the latest valid private recovery slot without trusting incomplete uploads.

GitHub release assets are not uploaded as one database transaction. Two private
slots provide a prior validated recovery point if upload to the other slot is
interrupted.
"""
from __future__ import annotations
from hashlib import sha256
import argparse
import json
from pathlib import Path


def checkpoint_metadata(directory: str | Path) -> dict | None:
    directory=Path(directory)
    pt=directory/"latest.pt"
    manifest=directory/"latest.json"
    if not (pt.is_file() and manifest.is_file()):
        return None
    try:
        data=json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(data.get("completed_updates"),int) or data["completed_updates"]<0:
            return None
        if not isinstance(data.get("sha256"),str):
            return None
        h=sha256()
        with pt.open("rb") as source:
            for chunk in iter(lambda:source.read(1024*1024),b""):
                h.update(chunk)
        if h.hexdigest()!=data["sha256"]:
            return None
        return {"path":str(pt),"completed_updates":data["completed_updates"]}
    except (ValueError,OSError,KeyError,TypeError):
        return None


def choose_recovery(a: str | Path, b: str | Path) -> dict:
    candidates=[]
    for name,path in [("a",a),("b",b)]:
        valid=checkpoint_metadata(path)
        if valid is not None:
            candidates.append((valid["completed_updates"],name,valid["path"]))
    if not candidates:
        return {"resume":"","next_slot":"a","completed_updates":0}
    _,chosen,path=max(candidates)
    return {"resume":path,"next_slot":"b" if chosen=="a" else "a",
            "completed_updates":max(candidates)[0]}


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--slot-a",required=True)
    parser.add_argument("--slot-b",required=True)
    parser.add_argument("--github-env")
    args=parser.parse_args(argv)
    selection=choose_recovery(args.slot_a,args.slot_b)
    print(json.dumps(selection))
    if args.github_env:
        with Path(args.github_env).open("a",encoding="utf-8") as stream:
            stream.write("ANCESTOR_RESUME="+selection["resume"]+"\n")
            stream.write("ANCESTOR_NEXT_SLOT="+selection["next_slot"]+"\n")


if __name__=="__main__":
    main()
