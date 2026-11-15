"""Read-only paired language probes. Never import into training."""
from .comprehension_transfer import exercise
from .language_inference import answer_question
import re

def normalize(value):
    return " ".join(re.findall(r"[^\W_]+", value.casefold()))

def paired_transfer(checkpoint, tokenizer, *, seed=97413, threads=2, max_tokens=50):
    records = []
    for group in ("syntax", "reading", "qa", "conditional"):
        for i in range(4):
            pair = []
            for variant in (0, 1):
                row = exercise(group, i, seed, 0, "probe", variant)
                response = answer_question(checkpoint, tokenizer, row["question"],
                                           threads=threads, max_tokens=max_tokens)
                actual = response["answer"]
                pair.append(dict(question=row["question"], expected=row["answer"],
                                 actual=actual, correct=normalize(actual)==normalize(row["answer"])))
            records.append(dict(group=group, variants=pair,
                                both_correct=all(x["correct"] for x in pair),
                                one_correct=sum(x["correct"] for x in pair)==1))
    return {"pairs": len(records), "both_correct": sum(x["both_correct"] for x in records),
            "wording_sensitive": sum(x["one_correct"] for x in records), "records": records,
            "assessment": "unambiguous grammar repair and same-meaning wording transfer; separate from frozen exam"}
