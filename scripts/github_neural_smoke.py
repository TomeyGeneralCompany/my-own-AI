"""CPU-only hosted synthetic training and held-out neural evaluation."""
import json
import os
from pathlib import Path
from time import perf_counter
import torch
from my_own_ai.model import ModelConfig
from my_own_ai.train import train, load_model
from my_own_ai.proposal_training import train_proposals
from my_own_ai.relational_curriculum import (
    load_relational_cases, check_relational_splits, prepare_curriculum, evaluate_relational,
)
from my_own_ai.reasoning import LocalProposer, ProposalRequest, Stage, SUPERVISED_FORMAT

ROOT = Path("examples/relational")
OUTPUT = Path("data/hosted-neural-experiment")


class RecordingProposer:
    def __init__(self, model, tokens=512):
        model.proposal_format = SUPERVISED_FORMAT
        self.proposer = LocalProposer(model, tokens=tokens)
        self.records = []

    def propose(self, request):
        answer = self.proposer.propose(request)
        self.records.append({
            "stage": request.stage.value,
            "question": request.objective,
            "evidence": [p["text"] for p in request.context["evidence"]],
            "generated": answer,
        })
        return answer


def evaluate_checkpoint(path, cases):
    engine = RecordingProposer(load_model(path))
    return evaluate_relational(cases, engine), engine.records


def ask_core(checkpoint, question, evidence):
    """Return actual locally generated words, without supplied reference answers."""
    if not isinstance(question, str) or not 1 <= len(question.strip()) <= 1000:
        raise ValueError("A nonempty question (up to 1000 characters) is required")
    if not isinstance(evidence, str) or len(evidence) > 1200:
        raise ValueError("Evidence is limited to 1200 characters")
    model = load_model(checkpoint)
    model.proposal_format = SUPERVISED_FORMAT
    engine = LocalProposer(model, tokens=320)
    sources = evidence.strip() or "No supporting source was supplied for this question."
    context = {"evidence": [{"text": sources}]}
    formulated = engine.propose(ProposalRequest(
        "research", "grounded_analysis", Stage.FORMULATE,
        "Identify relevant relationships, changed conditions and a testable inference.",
        question, context, "", "", "",
    ))
    response = engine.propose(ProposalRequest(
        "research", "grounded_analysis", Stage.CONSTRUCT,
        "Form a conclusion from cited source evidence without inventing facts.",
        question, context, formulated, "", "",
    ))
    result = {
        "question": question,
        "user_supplied_evidence": evidence.strip() or None,
        "generated_plan": formulated,
        "model_response": response,
        "source": "original-weight CPU neural model; no external inference",
        "evaluated_for_truth": False,
    }
    (OUTPUT / "question-response.json").write_text(json.dumps(result, indent=2))
    summary = [
        "## Your question for the Core",
        "",
        "The neural model is still experimental and may produce malformed or nonsensical text.",
        "The response below is its actual generation, not ChatGPT answering for it.",
        "",
        "**Question:**",
        "",
        chr(96) * 3 + "text",
        question,
        chr(96) * 3,
        "",
        "**Model's response (raw):**",
        "",
        chr(96) * 3 + "text",
        response,
        chr(96) * 3,
        "",
        "**Model's reasoning plan (raw):**",
        "",
        chr(96) * 3 + "text",
        formulated,
        chr(96) * 3,
        "",
        "Answers are not independently verified. See question-response.json in the artifact.",
    ]
    output = "\n".join(summary) + "\n"
    print(output, flush=True)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
            stream.write(output)
    return result


def main():
    started = perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train_cases = load_relational_cases(ROOT / "train.json")
    valid_cases = load_relational_cases(ROOT / "validation.json")
    held_cases = load_relational_cases(ROOT / "held.json")
    check_relational_splits(train_cases, valid_cases, held_cases)

    # Bootstrap ONLY on training-domain premises. Never show the neural
    # model the held-out hunting examples or their answers during training.
    corpus = OUTPUT / "bootstrap-corpus"
    corpus.mkdir(exist_ok=True)
    for row in train_cases:
        (corpus / (row["id"] + ".txt")).write_text(
            (row["evidence"][0] + "\n") * 13, encoding="utf-8"
        )
    curriculum = prepare_curriculum(ROOT / "train.json", ROOT / "validation.json",
                                    ROOT / "held.json", OUTPUT / "curriculum")
    initial = OUTPUT / "untrained-base.pt"
    pretrain = train(
        corpus, initial,
        ModelConfig(context_size=1536, embed_dim=16, heads=4, layers=1),
        steps=3, batch_size=1, checkpoint_every=3,
        schedule_steps=20, warmup_steps=2,
    )
    before, prior_outputs = evaluate_checkpoint(initial, held_cases)

    trained_path = OUTPUT / "relational-brain.pt"
    training = train_proposals(
        initial, curriculum["training"], curriculum["validation"],
        trained_path, steps=240, batch_size=4, learning_rate=.001,
        threads=2, format_name=SUPERVISED_FORMAT,
        pair_batches=True, checkpoint_every=60
    )
    after, trained_outputs = evaluate_checkpoint(trained_path, held_cases)
    result = {
        "experiment": "github-hosted-synthetic-neural-training",
        "seed": 42, "cpu_only": True, "pretrained_external_weights": False,
        "training_domains": sorted({r["domain"] for r in train_cases}),
        "validation_domains": sorted({r["domain"] for r in valid_cases}),
        "heldout_domains": sorted({r["domain"] for r in held_cases}),
        "pretrain": pretrain, "proposal_training": training,
        "before": before, "after": after,
        "delta_exact_consequences":
            after["exact_consequences"] - before["exact_consequences"],
        "delta_counterfactual_families":
            after["counterfactual_passed"] - before["counterfactual_passed"],
        "elapsed_seconds": round(perf_counter() - started, 2),
        "success_condition_demonstrated":
            after["exact_consequences"] > before["exact_consequences"]
            and after["counterfactual_passed"] >= before["counterfactual_passed"],
        "warning": "Synthetic experiment; not proof of cross-disciplinary intelligence.",
    }
    (OUTPUT / "results.json").write_text(json.dumps(result, indent=2))
    (OUTPUT / "raw-generations.json").write_text(json.dumps(
        {"before": prior_outputs, "after": trained_outputs}, indent=2
    ))
    summary = [
        "# Hosted neural training experiment", "",
        "Original random weights, local PyTorch CPU, synthetic examples only.", "",
        "| Measure | Before | After |",
        "|---|---:|---:|",
        f"| Exact held-out consequences | {before['exact_consequences']}/{before['total']} | {after['exact_consequences']}/{after['total']} |",
        f"| Verified quotes | {before['verified_quotes']}/{before['total']} | {after['verified_quotes']}/{after['total']} |",
        f"| Counterfactual families correct | {before['counterfactual_passed']}/{before['counterfactual_families']} | {after['counterfactual_passed']}/{after['counterfactual_families']} |",
        "",
        f"Validation loss: {training['initial_validation_loss']:.4f} to {training['validation_loss']:.4f}",
        f"Success condition demonstrated: **{result['success_condition_demonstrated']}**.",
        "",
        "See results.json and raw-generations.json for measured scores and actual outputs.",
        "See relational-brain.pt for the resulting local checkpoint.",
        "A green workflow means the experiment completed, not that the model passed.",
    ]
    text = "\n".join(summary) + "\n"
    (OUTPUT / "SUMMARY.md").write_text(text)
    question = os.environ.get("CORE_QUESTION", "").strip()
    if question:
        ask_core(trained_path, question, os.environ.get("CORE_EVIDENCE", ""))
    print(text, flush=True)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
            stream.write(text)


if __name__ == "__main__":
    main()
