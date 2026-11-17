"""Production checkpoint trial: no release writes, no promotion of live state."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

from my_own_ai.english_mastery import assess_independently
from my_own_ai.english_mastery_curriculum import Curriculum
from my_own_ai.language_inference import load_language
from my_own_ai.webster1913 import acquire as acquire_webster


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_continuation(before, after, result, report, checkpoint_hash):
    assert after["completed_steps"] >= before["completed_steps"]
    assert after["completed_steps"] > before["completed_steps"] or result["session_tests"] > 0
    assert after["tokenizer_sha256"] == before["tokenizer_sha256"]
    assert set(before.get("verified_memory", {})) <= set(after.get("verified_memory", {}))
    p = after["english_mastery"]
    assert p["curriculum_version"] == 2
    assert report["curriculum_version"] == 2
    assert p["assessment_version"] == 2
    assert report["assessment_version"] == 2
    assert report["checkpoint_sha256"] == checkpoint_hash
    assert result["assessment_version"] == 2
    assert result["completed_updates"] == after["completed_steps"]
    assert report["passed_items"] == p["passed_items"]
    assert report["cumulative_items"] == p["cumulative_items"]
    assert not report["qualified"] or (
        report["independent_cumulative_units"] >= sum(report["totals"].values())
        and report["cumulative_started"] == after["completed_steps"]
        and not report["remediation_pending"])
    for row in result.get("last_independent_exam") or []:
        assert row["origin"] == "neural"
        assert row["correct"] == (row["eos"] and " ".join(row["actual"].split()) ==
                                  " ".join(row["expected"].split()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state, out = args.state, args.output
    checkpoint = state/"english.pt"
    tok = state/"tokenizer.json"
    token_hash = digest(tok)
    before = torch.load(checkpoint, map_location="cpu", weights_only=True)
    parent = json.loads((state/"stage-ledger.json").read_text())["parent_generation"]
    for i in (1,2):
        scratch = out/f"trial-scratch-{i}"
        completed = subprocess.run([sys.executable,"-u","scripts/ancestor_staged.py","train",
            "--repository",str(args.repository),"--state",str(state),"--scratch",str(scratch),
            "--parent-generation",parent,"--english-updates","1"],check=True,
            stdout=subprocess.PIPE,text=True)
        result = json.loads(completed.stdout)
        report = json.loads((state/"last-exam.json").read_text())
        (out/f"trial-result-{i}.json").write_text(json.dumps(result,indent=2))
        (out/f"exam-{i}.json").write_text(json.dumps(report,indent=2))
        after = torch.load(checkpoint,map_location="cpu",weights_only=True)
        validate_continuation(before,after,result,report,digest(checkpoint))
        assert digest(tok)==token_hash
        # An explicit independent probe is required even when the two
        # continuation sessions are still in the middle of remediation.
        # A failed learner answer is recorded, not disguised as graduation.
        records=acquire_webster(scratch/"dictionary")
        model,tokenizer=load_language(checkpoint,tok)
        curriculum=Curriculum(records,tokenizer,model.config.context_size)
        progress={}
        probes=[]
        for key in (["dictionary",0,0],["dictionary",0,len(curriculum.units("dictionary",0))-1],["grammar",0,0],["composition",0,0]):
            ok,rows=assess_independently(model,tokenizer,curriculum.unit(key),progress)
            assert rows and rows[0]["kind"]=="recall"
            assert all(row["origin"]=="neural" for row in rows)
            assert ok==all(row["correct"] for row in rows)
            if ok:
                assert len(rows)==(3 if key[0]=="composition" else 2)
            probes.append({"unit":key,"passed":ok,"answers":rows})
        (out/f"retention-{i}.json").write_text(json.dumps(
            {"checkpoint_sha256":digest(checkpoint),"probes":probes},indent=2))
        print(json.dumps({"cycle":i,"updates":result["completed_updates"],
                          "probes":probes},ensure_ascii=False))
        before=after
        if report["qualified"]:
            break


if __name__=="__main__":
    main()
