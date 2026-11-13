"""Evidence-only diagnosis: frozen examination answers are never training inputs."""
import json
from pathlib import Path
import torch
from .language_inference import answer_question


def diagnose_foundation(checkpoint, tokenizer, train_file, validation_file, *, sample=16):
    """Probe known and held-out instruction forms using actual current weights."""
    if sample < 1:
        raise ValueError("Diagnostic sample must be positive")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    results = {}
    for name, path in (("training", train_file), ("validation", validation_file)):
        rows = json.loads(Path(path).read_text(encoding="utf-8"))[:sample]
        answers = []
        for row in rows:
            generated = answer_question(checkpoint, tokenizer, row["question"],
                                        max_tokens=36, threads=2)
            actual = generated["answer"].strip()
            expected = row["answer"].strip()
            answers.append({"question": row["question"], "expected": expected,
                            "actual": actual, "exact": actual == expected,
                            "eos": generated["stopped_by"] == "eos",
                            "valid_utf8": generated["valid_utf8"],
                            "token_ids": generated["raw_token_ids"]})
        results[name] = {
            "correct": sum(x["exact"] for x in answers),
            "total": len(answers),
            "natural_eos": sum(x["eos"] for x in answers),
            "answers": answers}
    return {
        "completed_updates": payload["completed_steps"],
        "validation_loss": payload["validation_loss"],
        "best_validation_loss": payload["best_validation_loss"],
        "last_training_loss": payload.get("last_training_loss"),
        "curriculum_cycles_completed": payload.get("curriculum_completed", 0),
        "curriculum_cursor": payload.get("curriculum_cursor", 0),
        "curriculum_examples_total": len(payload.get("curriculum_order", [])),
        "curriculum_migration": payload.get("curriculum_lineage", []),
        "probes": results,
    }
