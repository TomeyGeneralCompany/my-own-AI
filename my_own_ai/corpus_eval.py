"""Evaluate retrieval, citations and held-out answer criteria independently."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from .functional_modules import grounded_evidence
from .module_memory import IDENTIFIER
from .modules import CheckpointStore, ModuleController, default_modules, hardware_metrics
from .reasoning import Status


def load_cases(path):
    with Path(path).open(encoding="utf-8") as handle:
        return validate_cases(json.load(handle))


def validate_cases(cases):
    if not isinstance(cases, list) or not 1 <= len(cases) <= 100:
        raise ValueError("Expected between 1 and 100 held-out cases")
    ids = set()
    for case in cases:
        if (not isinstance(case, dict) or set(case) != {"id", "question", "expected_sources", "inference_terms"}
                or not isinstance(case["id"], str) or not IDENTIFIER.fullmatch(case["id"])
                or case["id"] in ids or not isinstance(case["question"], str)
                or not 1 <= len(case["question"].strip()) <= 4000):
            raise ValueError("Invalid or duplicate held-out case")
        for field in ("expected_sources", "inference_terms"):
            values = case[field]
            if (not isinstance(values, list) or not 1 <= len(values) <= 12
                    or any(not isinstance(value, str) or not value.strip() for value in values)
                    or len(set(values)) != len(values)):
                raise ValueError(f"Case {case['id']} requires distinct values for {field}")
        ids.add(case["id"])
    return cases


def evaluate_cases(index, cases, engine, *, limit=5, max_corrections=2):
    """Run the production Research operation with rubrics outside the model input."""
    validate_cases(cases)
    results = []
    with TemporaryDirectory(prefix="my-own-ai-eval-") as directory:
        with ModuleController(CheckpointStore(directory), default_modules(corpus_index=index), engine) as core:
            core.switch("research")
            for case in cases:
                start = perf_counter()
                run = core.start("grounded_analysis", case["question"], project=case["id"],
                                 inputs={"limit": limit}, max_corrections=max_corrections)
                while run["stage"] is not None:
                    run = core.step()
                passages = run["context"].get("evidence", [])
                cited, inference, _ = grounded_evidence(run["candidate"], passages)
                expected = set(case["expected_sources"])
                retrieved = {passage["source"] for passage in passages}
                used = {passages[number - 1]["source"] for number in cited}
                terms = [term for term in case["inference_terms"] if term.casefold() in inference.casefold()]
                passed = (run["status"] == Status.COMPLETED and expected <= used
                          and len(terms) == len(case["inference_terms"]))
                results.append({"id": case["id"], "passed": passed, "status": run["status"],
                                "retrieved_expected": len(expected & retrieved), "expected_sources": len(expected),
                                "cited_expected": len(expected & used), "verified_citations": len(cited),
                                "matched_inference_terms": len(terms),
                                "required_inference_terms": len(case["inference_terms"]),
                                "stages": [event["stage"] for event in run["history"]],
                                "wall_seconds": perf_counter() - start})
    return {"cases": results, "passed": sum(result["passed"] for result in results),
            "total": len(results), "hardware": hardware_metrics()}
