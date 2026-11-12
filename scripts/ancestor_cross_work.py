"""Frozen-source cross-work evaluation, distinct from the 200 English cases.

Test prompts are derived from held-out logical units in EACH approved work.
This is an infrastructure-and-comparative-evaluation harness, not a claim of
semantic competence from simple exact matching.
"""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re

from my_own_ai.language_inference import answer_question
from my_own_ai.language_training import load_language


def make_suite(exercise_dir, expected_books, *, questions_per_book=2):
    base=Path(exercise_dir)
    examples=json.loads((base/"test.json").read_text(encoding="utf-8"))
    traces=json.loads((base/"test-provenance.json").read_text(encoding="utf-8"))
    groups=defaultdict(list)
    for item in examples:
        trace=traces[item["id"]]
        stem=Path(trace["source"]).stem
        book=stem.rsplit("_",1)[0]
        groups[book].append({**item,"book":book,"source_unit":stem,
                             "kind":trace["kind"]})
    required=set(expected_books)
    if set(groups)!=required:
        raise ValueError(f"Missing or unexpected held-out book coverage: {sorted(required-set(groups))}; {sorted(set(groups)-required)}")
    if any(len(groups[book])<questions_per_book for book in required):
        raise ValueError("Insufficient source-grounded independent cases for every book")
    return [row for book in sorted(required)
            for row in groups[book][:questions_per_book]]


def evaluate_cross_work(state, exercises, book_ids, *, per_book=2, max_tokens=40):
    suite=make_suite(exercises,book_ids,questions_per_book=per_book)
    checkpoint=Path(state)/"assistant.best.pt"
    tok=Path(state)/"tokenizer.json"
    load_language(checkpoint,tok)
    output=[]
    for row in suite:
        answer=answer_question(checkpoint,tok,row["question"],max_tokens=max_tokens)
        actual=answer["answer"].strip()
        expected=row["answer"].strip()
        normalized=lambda x:" ".join(re.findall(r"[^\W_]+",x.casefold()))
        output.append({**row,"actual":actual,
                       "exact_match":normalized(actual)==normalized(expected),
                       "utf8_valid":answer["valid_utf8"],
                       "stopped_by":answer["stopped_by"],
                       "raw_token_ids":answer["raw_token_ids"]})
    if not all(row["utf8_valid"] for row in output):
        raise ValueError("Invalid model output encoding")
    return {"format":"ancestor_cross_work_v1",
            "source_books":sorted(book_ids),
            "questions_per_book":per_book,
            "evaluated":len(output),
            "exact_correct":sum(row["exact_match"] for row in output),
            "questions":output,
            "checkpoint_sha256":sha256(checkpoint.read_bytes()).hexdigest(),
            "training_executed":False}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--state",required=True)
    ap.add_argument("--exercises",required=True)
    ap.add_argument("--prepared",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--questions-per-book",type=int,default=2)
    args=ap.parse_args()
    manifest=json.loads((Path(args.prepared)/"manifest.json").read_text())
    book_ids=[x["id"] for x in manifest["sources"]]
    report=evaluate_cross_work(args.state,args.exercises,book_ids,
                               per_book=args.questions_per_book)
    target=Path(args.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="questions"},indent=2))


if __name__=="__main__":
    main()
