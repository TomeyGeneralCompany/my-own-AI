"""Keep the author's relational curriculum trainable in the same English Core.

This is supervised data conversion, not an ontology-driven response generator:
at inference time the neural model receives only the question and premises.
"""
import json
from pathlib import Path
import re
from .relational_curriculum import (
    FACETS, load_relational_cases, check_relational_splits,
)
from .language_inference import answer_question


def relational_prompt(row):
    return (
        "Analyze the following situation through relationships and their "
        "consequences. Question: " + row["question"] +
        "\nEvidence:\n" + "\n".join("- " + source for source in row["evidence"]) +
        "\nExplain the relationships, communication, internal states, "
        "feedback, and the consequence supported by the evidence."
    )


def relational_target(row):
    return "\n".join(
        [f"{name.capitalize()}: {row['analysis'][name]}" for name in FACETS]
        + ["Conclusion: " + row["conclusion"]]
    )


def prepare_language_relational(training, validation, held_out, output):
    train = load_relational_cases(training)
    valid = load_relational_cases(validation)
    held = load_relational_cases(held_out)
    check_relational_splits(train, valid, held)
    destination = Path(output)
    if destination.exists():
        raise ValueError("New relational training destination must not exist")
    inputs = [Path(path).resolve() for path in (training, validation, held_out)]
    if destination.resolve() in inputs:
        raise ValueError("Cannot overwrite source data")
    destination.mkdir(parents=True)
    def convert(rows):
        return [{"id": row["id"], "question": relational_prompt(row),
                 "answer": relational_target(row)} for row in rows]
    for name, rows in (("train", train), ("validation", valid)):
        (destination / (name + ".json")).write_text(
            json.dumps(convert(rows), ensure_ascii=False, indent=2),
            encoding="utf-8")
    return {"train": len(train), "validation": len(valid), "heldout": len(held),
            "heldout_labels_written": False, "output": str(destination)}


def evaluate_language_relational(checkpoint, tokenizer, held_out, *,
                                 max_tokens=300, threads=2):
    cases = load_relational_cases(held_out)
    output = []
    for case in cases:
        result = answer_question(checkpoint, tokenizer, relational_prompt(case),
                                 max_tokens=max_tokens, threads=threads)
        expected = " ".join(re.findall(r"[^\W_]+", case["conclusion"].casefold()))
        actual = " ".join(re.findall(r"[^\W_]+", result["answer"].casefold()))
        matched = bool(expected and expected in actual)
        output.append({"id": case["id"], "domain": case["domain"],
                       "group": case["group"], "exact_conclusion_present": matched,
                       "generated": result["answer"], "eos": result["stopped_by"] == "eos"})
    pairs = {}
    for item in output:
        pairs.setdefault(item["group"], []).append(item)
    return {"total": len(output),
            "exact_conclusions": sum(item["exact_conclusion_present"] for item in output),
            "counterfactual_families": len(pairs),
            "counterfactual_passed": sum(all(item["exact_conclusion_present"] for item in family)
                                         for family in pairs.values()),
            "targets_supplied_at_inference": False, "semantic_entailment_verified": False,
            "cases": output}
