"""GitHub-hosted original-weight English experiment and browser interrogation.

A completed workflow means that the experiment ran. It does not establish
English proficiency; the frozen gauntlet and raw output expose failures.
"""
import html
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
from time import perf_counter

from my_own_ai.corpus import collect_documents
from my_own_ai.tokenizer import ByteBPETokenizer
from my_own_ai.language_training import train_language, load_dialogues
from my_own_ai.language_inference import answer_question
from my_own_ai.language_evaluation import load_gauntlet, evaluate_english


ROOT = Path("examples/english")
OUT = Path("data/hosted-english")


def run():
    started = perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer_path = OUT / "tokenizer.json"
    sources = collect_documents(ROOT / "corpus/train")
    tokenizer = ByteBPETokenizer.train([doc["text"] for doc in sources],
                                        vocab_size=512, min_frequency=2)
    tokenizer.save(tokenizer_path)
    common = {"tokenizer_path": tokenizer_path, "embed": 64,
              "heads": 4, "layers": 2, "context": 256,
              "batch_size": 2, "threads": 2, "seed": 42}
    pretrain_path = OUT / "pretrain.pt"
    pretrain = train_language(
        **common, phase="pretrain", training=ROOT / "corpus/train",
        validation=ROOT / "corpus/validation", output=pretrain_path,
        steps=60, checkpoint_every=20, lr=.001
    )
    instruction_steps = int(os.environ.get("ENGLISH_INSTRUCTION_STEPS", "120"))
    if not 1 <= instruction_steps <= 5000:
        raise ValueError("English instruction steps must be between 1 and 5000")
    taught_path = OUT / "english-core.pt"
    instruction = train_language(
        **common, phase="instruction", training=ROOT / "dialogue/train.json",
        validation=ROOT / "dialogue/validation.json", output=taught_path,
        init_checkpoint=Path(pretrain["best_checkpoint"]), steps=instruction_steps,
        checkpoint_every=40, lr=.001
    )

    user_question = os.environ.get("CORE_QUESTION", "").strip()
    if not user_question:
        user_question = "Why does a broken communication channel delay a company's decisions?"
    evidence = os.environ.get("CORE_EVIDENCE", "").strip()
    if not 1 <= len(user_question) <= 1000:
        raise ValueError("Question must contain 1–1000 characters")
    if len(evidence) > 1200:
        raise ValueError("Evidence must not exceed 1200 characters")
    # The same prompt is supplied before/after instruction training.
    # Evidence is explicit user text, not a source verified by this demo.
    model_question = (user_question if not evidence else
                      "User-provided evidence (not independently verified):\n"
                      + evidence + "\nQuestion: " + user_question)
    baseline = answer_question(pretrain_path, tokenizer_path, model_question,
                               max_tokens=80, threads=2)
    preferred = Path(instruction["best_checkpoint"])
    response = answer_question(preferred, tokenizer_path, model_question,
                               max_tokens=80, threads=2)
    known = load_dialogues(ROOT / "dialogue/train.json")[0]
    known_response = answer_question(preferred, tokenizer_path, known["question"],
                                     max_tokens=80, threads=2)
    question_record = {
        "displayed_question": user_question,
        "user_supplied_evidence": evidence or None,
        "model_prompt": model_question,
        "checkpoint_sha256": sha256(preferred.read_bytes()).hexdigest(),
        "checkpoint_path": str(preferred),
        "checkpoint_type": "original_subword_english_instruction",
        "training_source": "public_synthetic_demo_only",
        "manuscript_trained": False,
        "before_instruction_training": baseline, "trained_response": response,
        "known_training_question": known["question"],
        "known_training_target": known["answer"],
        "known_training_response": known_response,
    }
    (OUT / "question-response.json").write_text(json.dumps(question_record, indent=2),
                                                  encoding="utf-8")

    cases = load_gauntlet(ROOT / "gauntlet.json")
    limit = int(os.environ.get("ENGLISH_GAUNTLET_LIMIT", "200"))
    report = evaluate_english(preferred, tokenizer_path, cases,
                              limit=limit, max_tokens=40, threads=2)
    (OUT / "gauntlet-results.json").write_text(json.dumps(report, indent=2),
                                                encoding="utf-8")
    comparison = {
        "experiment": "original-weight-english-acquisition",
        "checkpoint_sha256": question_record["checkpoint_sha256"],
        "checkpoint_type": question_record["checkpoint_type"],
        "training_source": question_record["training_source"],
        "manuscript_trained": False,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "tokenizer_sha256": tokenizer.fingerprint(),
        "pretraining": pretrain, "instruction": instruction,
        "instruction_steps": instruction_steps,
        "best_validation_checkpoint": str(preferred),
        "known_training_target_exact": (
            known_response["answer"].strip() == known["answer"].strip()),
        "evaluation_cases": report["total"],
        "auto_exact": report["exact_correct"],
        "auto_scored": report["automatically_scored"],
        "human_review_pending": report["manual_review_pending"],
        "eos_rate": report["natural_eos"] / report["total"],
        "mean_generated_tokens_per_second": report["mean_generated_tokens_per_second"],
        "instruction_target_tokens_per_second": instruction["target_tokens_per_second"],
        "mean_trigram_repetition": report["mean_trigram_repetition"],
        "english_milestone_met": report["english_milestone_met"],
        "elapsed_seconds": round(perf_counter() - started, 2),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "source": "local_original_text_original_random_weights",
    }
    (OUT / "metrics.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    summary = [
        "# Ancestor Core — English acquisition report",
        "",
        "This is the original English model, freshly trained on PUBLIC SYNTHETIC examples. It is NOT the private manuscript-trained checkpoint. No pretrained weights or external inference.",
        "",
        f"**Checkpoint SHA-256:** `{question_record['checkpoint_sha256']}`",
        f"**Training data:** {question_record['training_source']}; manuscript trained: false.",
        "",
        f"**Tokenizer:** {tokenizer.vocab_size} learned/byte-fallback tokens.",
        f"**Pretraining validation loss:** {pretrain['validation_loss']:.4f}",
        f"**Instruction validation loss:** {instruction['validation_loss']:.4f}",
        f"**Best instruction validation loss:** {instruction['best_validation_loss']:.4f} at step {instruction['best_step']}.",
        f"**Instruction updates:** {instruction_steps}.",
        f"**Known training example copied exactly:** {comparison['known_training_target_exact']}.",
        f"**Frozen gauntlet:** {report['total']} items; {report['exact_correct']}/{report['automatically_scored']} strict automatic matches.",
        f"**Manual review pending:** {report['manual_review_pending']} items.",
        f"**Natural EOS:** {report['natural_eos']}/{report['total']} items.",
        f"**CPU training target tokens/s:** {instruction['target_tokens_per_second']}.",
        f"**CPU generation tokens/s:** {report['mean_generated_tokens_per_second']}.",
        f"**Mean repeated 3-gram rate:** {report['mean_trigram_repetition']:.4f}.",
        f"**English milestone demonstrated:** {report['english_milestone_met']}.",
        "",
        "## Your question",
        "",
        "<pre>" + html.escape(user_question) + "</pre>",
        "",
        "**User-supplied evidence (unverified):**",
        "<pre>" + html.escape(evidence or "(none)") + "</pre>",
        "",
        "## Model response — after English training",
        "",
        "<pre>" + html.escape(response["answer"]) + "</pre>",
        "",
        "## Response to a known training question (memorization diagnostic)",
        "",
        "<pre>" + html.escape(known["question"]) + "</pre>",
        "",
        "<pre>" + html.escape(known_response["answer"]) + "</pre>",
        "",
        "## Model response — before instruction training",
        "",
        "<pre>" + html.escape(baseline["answer"]) + "</pre>",
        "",
        "The raw prompts, token IDs, responses, evaluation cases and checkpoints are attached as artifacts.",
        "**A green workflow confirms execution, not language competence.**",
    ]
    text = "\n".join(summary) + "\n"
    (OUT / "SUMMARY.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
            stream.write(text)


if __name__ == "__main__":
    run()
