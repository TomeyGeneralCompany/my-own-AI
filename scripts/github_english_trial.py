"""GitHub-hosted original-weight English experiment and browser interrogation.

A completed workflow means that the experiment ran. It does not establish
English proficiency; the frozen gauntlet and raw output expose failures.
"""
import html
import json
import os
from pathlib import Path
import resource
from time import perf_counter

from my_own_ai.corpus import collect_documents
from my_own_ai.tokenizer import ByteBPETokenizer
from my_own_ai.language_training import train_language
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
    taught_path = OUT / "english-core.pt"
    instruction = train_language(
        **common, phase="instruction", training=ROOT / "dialogue/train.json",
        validation=ROOT / "dialogue/validation.json", output=taught_path,
        init_checkpoint=pretrain_path, steps=120, checkpoint_every=40, lr=.001
    )

    user_question = os.environ.get("CORE_QUESTION", "").strip()
    if not user_question:
        user_question = "Why does a broken communication channel delay a company's decisions?"
    baseline = answer_question(pretrain_path, tokenizer_path, user_question,
                               max_tokens=80, threads=2)
    response = answer_question(taught_path, tokenizer_path, user_question,
                               max_tokens=80, threads=2)
    question_record = {"before_instruction_training": baseline, "trained_response": response}
    (OUT / "question-response.json").write_text(json.dumps(question_record, indent=2),
                                                  encoding="utf-8")

    cases = load_gauntlet(ROOT / "gauntlet.json")
    limit = int(os.environ.get("ENGLISH_GAUNTLET_LIMIT", "200"))
    report = evaluate_english(taught_path, tokenizer_path, cases,
                              limit=limit, max_tokens=40, threads=2)
    (OUT / "gauntlet-results.json").write_text(json.dumps(report, indent=2),
                                                encoding="utf-8")
    comparison = {
        "experiment": "original-weight-english-acquisition",
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "tokenizer_sha256": tokenizer.fingerprint(),
        "pretraining": pretrain, "instruction": instruction,
        "evaluation_cases": report["total"],
        "auto_exact": report["exact_correct"],
        "auto_scored": report["automatically_scored"],
        "human_review_pending": report["manual_review_pending"],
        "eos_rate": report["natural_eos"] / report["total"],
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
        "This is the real locally trained Core. No pretrained weights, external AI or substituted response.",
        "",
        f"**Tokenizer:** {tokenizer.vocab_size} learned/byte-fallback tokens.",
        f"**Pretraining validation loss:** {pretrain['validation_loss']:.4f}",
        f"**Instruction validation loss:** {instruction['validation_loss']:.4f}",
        f"**Frozen gauntlet:** {report['total']} items; {report['exact_correct']}/{report['automatically_scored']} strict automatic matches.",
        f"**Manual review pending:** {report['manual_review_pending']} items.",
        f"**Natural EOS:** {report['natural_eos']}/{report['total']} items.",
        f"**Mean repeated 3-gram rate:** {report['mean_trigram_repetition']:.4f}.",
        f"**English milestone demonstrated:** {report['english_milestone_met']}.",
        "",
        "## Your question",
        "",
        "<pre>" + html.escape(user_question) + "</pre>",
        "",
        "## Model response — after English training",
        "",
        "<pre>" + html.escape(response["answer"]) + "</pre>",
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
