"""Frozen, original-English gauntlet with honest automated and manual scoring.

Open-ended grammaticality and coherence cannot be fully established by simple
regex. Record raw responses and leave those scores pending independent review.
"""
from collections import Counter
import json
import re
from pathlib import Path
from .language_inference import answer_question

SIZES = {"syntax": 40, "reading": 40, "paraphrase": 30, "qa": 30,
         "coherence": 20, "conditional": 20, "followup": 20}
MANUAL = {"paraphrase", "coherence", "followup"}
AUTO = set(SIZES) - MANUAL


def load_gauntlet(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) != 200:
        raise ValueError("English gauntlet must contain exactly 200 tasks")
    ids, count = set(), Counter()
    for item in data:
        if (not isinstance(item, dict) or set(item) !=
                {"id", "category", "question", "expected_answer", "requires_manual_review"}
                or item["category"] not in SIZES or
                not isinstance(item["id"], str) or item["id"] in ids or
                not isinstance(item["question"], str) or not item["question"].strip()
                or item["requires_manual_review"] != (item["category"] in MANUAL)):
            raise ValueError("Invalid English gauntlet case")
        if item["category"] in AUTO and (not isinstance(item["expected_answer"], str)
                                        or not item["expected_answer"].strip()):
            raise ValueError("Auto-scored cases require reference answers")
        if item["category"] in MANUAL and item["expected_answer"] is not None:
            raise ValueError("Manually reviewed cases cannot embed model answers")
        ids.add(item["id"])
        count[item["category"]] += 1
    if dict(count) != SIZES:
        raise ValueError("English gauntlet has incorrect category sizes")
    if len({row["question"].casefold().strip() for row in data}) != 200:
        raise ValueError("English gauntlet questions must be distinct")
    switches = sorted((row for row in data if row["category"] == "conditional"),
                      key=lambda row: row["id"])
    for first, second in zip(switches[::2], switches[1::2]):
        if (first["expected_answer"], second["expected_answer"]) != ("yes", "no"):
            raise ValueError("Conditional pair must have opposite correct conclusions")
    return data


def repetition_rate(tokens):
    if len(tokens) < 4:
        return 0.
    chunks = [tuple(tokens[i:i+3]) for i in range(len(tokens)-2)]
    return 1 - len(set(chunks)) / len(chunks)


def evaluate_english(checkpoint, tokenizer_path, cases, *,
                     max_tokens=120, limit=None, threads=2):
    if limit is not None and not 1 <= limit <= len(cases):
        raise ValueError("Invalid evaluation case limit")
    subset = cases if limit is None else cases[:limit]
    results = []
    for case in subset:
        response = answer_question(checkpoint, tokenizer_path, case["question"],
                                   max_tokens=max_tokens, threads=threads)
        actual = response["answer"].strip()
        reference = case["expected_answer"]
        # Exact metric is deliberately strict and should never be presented
        # as a general proof of comprehension or grammatical correctness.
        exact = None if reference is None else (
            " ".join(re.findall(r"[^\W_]+", actual.casefold())) ==
            " ".join(re.findall(r"[^\W_]+", reference.casefold()))
        )
        results.append({
            "id": case["id"], "category": case["category"], "question": case["question"],
            "actual": actual, "expected_answer": reference,
            "exact_match": exact, "manual_review_pending": case["requires_manual_review"],
            "utf8_valid": response["valid_utf8"],
            "eos": response["stopped_by"] == "eos",
            "repetition_3gram": repetition_rate(response["raw_token_ids"]),
            "raw_token_ids": response["raw_token_ids"],
            "prompt_token_ids": response["prompt_token_ids"],
            "checkpoint": str(checkpoint), "settings": response["settings"],
        })
    category = {}
    for name in SIZES:
        values = [x for x in results if x["category"] == name]
        if not values:
            continue
        scored = [x for x in values if x["exact_match"] is not None]
        category[name] = {
            "items": len(values), "auto_scored": len(scored),
            "exact": sum(x["exact_match"] for x in scored),
            "manual_review_pending": sum(x["manual_review_pending"] for x in values),
        }
    switches = sorted((item for item in results if item["category"] == "conditional"),
                      key=lambda item: item["id"])
    pair_total = len(switches) // 2
    pair_passed = sum(first["exact_match"] is True and second["exact_match"] is True
                      for first, second in zip(switches[::2], switches[1::2]))
    return {
        "conditional_pairs": pair_total,
        "conditional_pairs_passed": pair_passed,
        "total": len(results), "automatically_scored": sum(x["exact_match"] is not None for x in results),
        "exact_correct": sum(x["exact_match"] is True for x in results),
        "manual_review_pending": sum(x["manual_review_pending"] for x in results),
        "valid_utf8": sum(x["utf8_valid"] for x in results),
        "natural_eos": sum(x["eos"] for x in results),
        "mean_trigram_repetition": sum(x["repetition_3gram"] for x in results) / len(results),
        "english_milestone_met": False,
        "reason": "Independent human grammaticality and coherence review pending; exact answers alone are insufficient",
        "categories": category, "cases": results,
    }
