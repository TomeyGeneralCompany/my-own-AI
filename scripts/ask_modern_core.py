"""Ask the actual current staged checkpoint without advancing or training it."""
import argparse
import json
import os
from pathlib import Path
from my_own_ai.foundation_contract import FoundationContract
from my_own_ai.language_inference import answer_question
from my_own_ai.persistent_registry import digest_file
from my_own_ai.stage_controller import verify_ledger


def select_checkpoint(state, ledger):
    stage = ledger["stage"]
    if stage == "english":
        name = "english.pt" if ledger.get("english_fundamentals_passed") else "foundation.pt"
    elif stage in ("read", "comprehend"):
        book = ledger["books"][ledger["book_index"]]
        name = ("read_" if stage=="read" else "learn_") + book + ".pt"
    else:
        name = "current.best.pt"
    if not (state/name).is_file():
        name = "current.best.pt" if (state/"current.best.pt").is_file() else "inherited.pt"
    if not (state/name).is_file():
        raise ValueError("No staged original checkpoint found")
    return state/name


def ask(state, question, evidence="", tokens=120, *, retrieve_memory=True):
    state=Path(state)
    ledger=json.loads((state/"stage-ledger.json").read_text())
    verify_ledger(ledger,ledger["manifest_sha256"],ledger["books"])
    checkpoint=select_checkpoint(state,ledger)
    tokenizer=state/"tokenizer.json"
    before=digest_file(checkpoint)
    tok_before=digest_file(tokenizer)
    prompt=question if not evidence else "External evidence: "+evidence+"\nQuestion: "+question
    response=answer_question(checkpoint,tokenizer,prompt,max_tokens=tokens,threads=2,
                             retrieve_memory=retrieve_memory)
    if digest_file(checkpoint)!=before or digest_file(tokenizer)!=tok_before:
        raise RuntimeError("Checkpoint or tokenizer mutated during inference")
    return {"format":"ancestor_staged_answer_v1","generation":ledger["parent_generation"],
            "stage":ledger["stage"],"checkpoint":checkpoint.name,"checkpoint_sha256":before,
            "tokenizer_sha256":response["tokenizer_sha256"],"question":question,
            "model_prompt":prompt,"evidence":evidence or None,"answer":response,
            "foundation_contract":FoundationContract().record(),
            "training_executed":False,"qualified":ledger["stage"]=="qualified"}


def regression(state,cases,tokens=120):
    """Collect paired raw outputs for independent owner review; no canned verdict."""
    results=[]
    for case in cases:
        baseline=ask(state,case["question"],tokens=tokens,retrieve_memory=False)
        challenged=ask(state,case["question"],case["contradictory_evidence"],
                       tokens=tokens,retrieve_memory=False)
        results.append({"id":case["id"],"baseline":baseline,"challenged":challenged,
                        "owner_review_required":True})
    return {"format":"foundation_invariance_review_v1","cases":results,
            "qualification_granted":False}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--state",required=True)
    p.add_argument("--question",default=os.environ.get("CORE_QUESTION",""))
    p.add_argument("--evidence",default=os.environ.get("CORE_EVIDENCE",""))
    p.add_argument("--tokens",type=int,default=120)
    p.add_argument("--output",required=True)
    p.add_argument("--regression-cases")
    a=p.parse_args()
    result=(regression(a.state,json.loads(Path(a.regression_cases).read_text()),a.tokens)
            if a.regression_cases else ask(a.state,a.question,a.evidence,a.tokens))
    output=Path(a.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
