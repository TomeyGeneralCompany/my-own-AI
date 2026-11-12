"""Author-supervised relational reasoning curriculum and held-out neural assessment.

This is supervision for the existing, original-weight Transformer, not a
deterministic ontology engine. The analysis/answer appear only in training
targets; inference sees the question and evidence, without answer keys.
"""
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import unicodedata

from .functional_modules import grounded_evidence
from .module_memory import IDENTIFIER, encode_json
from .proposal_training import load_examples, verify_splits
from .reasoning import LocalProposer, ProposalRequest, Stage
from .functional_modules import Research

FACETS = ("identity", "hierarchy", "channels", "states", "disturbance", "feedback")
FIELDS = {"id", "group", "domain", "question", "evidence", "analysis", "conclusion"}
MAX_CASES = 2000


def _normalized(value):
    return " ".join(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", value).casefold()))


def load_relational_cases(path):
    with Path(path).open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, list) or not 1 <= len(data) <= MAX_CASES:
        raise ValueError("Expected between one and 2000 relational examples")
    ids, cases = set(), []
    for row in data:
        if not isinstance(row, dict) or set(row) != FIELDS:
            raise ValueError("Invalid relational case fields")
        if any(not isinstance(row[key], str) or not IDENTIFIER.fullmatch(row[key])
               for key in ("id", "group", "domain")):
            raise ValueError("Invalid relational identifier, group or domain")
        if row["id"] in ids:
            raise ValueError("Duplicate relational case ID")
        ids.add(row["id"])
        if not isinstance(row["question"], str) or not 1 <= len(row["question"].strip()) <= 1500:
            raise ValueError("Invalid relational question")
        if (not isinstance(row["evidence"], list) or not 1 <= len(row["evidence"]) <= 6
                or any(not isinstance(value, str) or not 8 <= len(value.strip()) <= 1400
                       or "\0" in value for value in row["evidence"])):
            raise ValueError("Invalid relational evidence")
        if (not isinstance(row["analysis"], dict) or set(row["analysis"]) != set(FACETS)
                or any(not isinstance(row["analysis"][facet], str)
                       or not 8 <= len(row["analysis"][facet].strip()) <= 700
                       or "\0" in row["analysis"][facet] for facet in FACETS)):
            raise ValueError("Every relational operation requires a worked analysis")
        if (not isinstance(row["conclusion"], str) or not 16 <= len(row["conclusion"].strip()) <= 650
                or "\0" in row["conclusion"]):
            raise ValueError("Invalid derived consequence")
        cases.append(row)
    return cases


def check_relational_splits(train, validation, held_out):
    """Disallow overlapping families/questions, and isolate every held-out domain."""
    for name, rows in (("train", train), ("validation", validation), ("held-out", held_out)):
        if not rows:
            raise ValueError(f"Empty {name} relational split")
    partitions = (train, validation, held_out)
    ids = [set(row["id"] for row in cases) for cases in partitions]
    groups = [set(row["group"] for row in cases) for cases in partitions]
    questions = [set(_normalized(row["question"]) for row in cases) for cases in partitions]
    for left in range(3):
        for right in range(left + 1, 3):
            if (ids[left] & ids[right] or groups[left] & groups[right]
                    or questions[left] & questions[right]):
                raise ValueError("Relational identifiers, groups and questions must be split-disjoint")
    held_domains = {row["domain"] for row in held_out}
    if held_domains & {row["domain"] for row in train + validation}:
        raise ValueError("Held-out disciplines must be absent from train and validation")
    for name, cases in (("train", train), ("validation", validation), ("held-out", held_out)):
        for group in {row["group"] for row in cases}:
            family = [row for row in cases if row["group"] == group]
            tasks = {(_normalized(row["question"]), row["domain"]) for row in family}
            if len(tasks) != 1:
                raise ValueError(f"Counterfactual family {group} changes the question or domain")
            if len(family) < 2:
                raise ValueError(f"Counterfactual family {group} needs changed premises")
            if (len({tuple(_normalized(p) for p in row["evidence"]) for row in family}) < 2
                    or len({_normalized(row["conclusion"]) for row in family}) < 2):
                raise ValueError(f"Counterfactual family {group} must change evidence AND consequence")


def _training_rows(cases):
    rows = []
    for row in cases:
        # The inputs to both model stages contain the question and evidence ONLY.
        # Facet-specific worked analysis and conclusion live in target tokens.
        plan = "\n".join(f"{facet}: {row['analysis'][facet]}" for facet in FACETS)
        quote = row["evidence"][0].strip()
        answer = f"Premises:\n- [1] {quote}\nInference: {row['conclusion'].strip()}"
        common = {"group": row["group"], "question": row["question"],
                  "passages": row["evidence"], "candidate": "", "feedback": ""}
        rows.append({**common, "id": row["id"] + "_formulate", "stage": "formulate",
                     "plan": "", "answer": plan})
        rows.append({**common, "id": row["id"] + "_construct", "stage": "construct",
                     "plan": plan, "answer": answer})
    return rows


def prepare_curriculum(training, validation, held_out, output):
    """Create private, trainable examples. Never copy held-out answers to outputs."""
    paths = [Path(p).resolve() for p in (training, validation, held_out)]
    destination = Path(output).resolve()
    if destination in paths or any(destination == path.parent for path in paths):
        raise ValueError("Curriculum output must be separate from input data")
    train = load_relational_cases(training)
    valid = load_relational_cases(validation)
    test = load_relational_cases(held_out)
    check_relational_splits(train, valid, test)
    worked_train, worked_valid = _training_rows(train), _training_rows(valid)
    # Reuse the production proposal-training validator before writing artifacts.
    # Write atomically so interrupted preparation does not expose half-written files.
    destination.mkdir(parents=True, exist_ok=True)
    from tempfile import NamedTemporaryFile
    import os
    def save(filename, payload):
        target = destination / filename
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        return str(target)
    # Check target format with same loader used by actual training, after write.
    train_file = save("relational_train.json", worked_train)
    valid_file = save("relational_validation.json", worked_valid)
    load_examples(train_file)
    load_examples(valid_file)
    verify_splits(worked_train, worked_valid, [])
    return {"training": train_file, "validation": valid_file, "held_out": str(Path(held_out)),
            "training_cases": len(train), "validation_cases": len(valid),
            "held_out_cases": len(test), "held_out_domains": sorted({r["domain"] for r in test}),
            "training_fingerprint": sha256(encode_json(worked_train)).hexdigest()}


def _request(row, stage, plan=""):
    return ProposalRequest("research", "grounded_analysis", stage,
                           Research.grounded_procedures[stage], row["question"],
                           {"evidence": [{"text": text} for text in row["evidence"]]},
                           plan, "", "")


def evaluate_relational(cases, proposer):
    """Test genuine model generations; no source-answer or facet labels enter prompts."""
    records = []
    for row in cases:
        plan = proposer.propose(_request(row, Stage.FORMULATE))
        proposal = proposer.propose(_request(row, Stage.CONSTRUCT, plan=plan))
        citations, inferred, format_error = grounded_evidence(
            proposal, [{"text": text} for text in row["evidence"]])
        exact = (not format_error and
                 _normalized(inferred) == _normalized(row["conclusion"]))
        records.append({"id": row["id"], "group": row["group"], "domain": row["domain"],
                        "cited_evidence": bool(citations), "exact_consequence": bool(exact),
                        "generated_consequence": inferred if not format_error else None,
                        "plan_mentions_facets": sum(facet in plan.casefold() for facet in FACETS)})
    families = defaultdict(list)
    for result in records:
        families[result["group"]].append(result)
    pairs_passed = sum(all(item["exact_consequence"] for item in family)
                       and len({item["generated_consequence"] for item in family}) == len(family)
                       for family in families.values())
    domains = sorted({r["domain"] for r in records})
    return {"total": len(records), "exact_consequences": sum(r["exact_consequence"] for r in records),
            "verified_quotes": sum(r["cited_evidence"] for r in records),
            "counterfactual_families": len(families), "counterfactual_passed": pairs_passed,
            "domains": {domain: {"total": sum(r["domain"] == domain for r in records),
                                 "exact": sum(r["domain"] == domain and r["exact_consequence"] for r in records)}
                        for domain in domains},
            "model_received_answer_labels": False, "semantic_entailment_verified": False,
            "cases": records}
