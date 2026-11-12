"""Local-only training and testing on the owner's approved manuscript library.

Never publish private source text or generated outputs to a public Actions run.
"""
import argparse
import json
from pathlib import Path
import torch

from my_own_ai.book_language import prepare_approved_books
from my_own_ai.book_instruction import create_book_continuations
from my_own_ai.corpus import collect_documents
from my_own_ai.tokenizer import ByteBPETokenizer
from my_own_ai.language_training import load_dialogues, train_language
from my_own_ai.progressive_reading import train_progressive
from my_own_ai.language_inference import answer_question


def run(argv=None):
    parser = argparse.ArgumentParser(description="Train original-weight English Core on approved local books")
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New private directory. Must not exist.")
    parser.add_argument("--section-words", type=int, default=512)
    parser.add_argument("--vocab", type=int, default=2048)
    parser.add_argument("--embed", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--pretrain-steps", type=int, default=1000,
                        help="Legacy random-sampled prose training updates when --reading-passes=0")
    parser.add_argument("--reading-passes", type=int, default=5,
                        help="Complete exhaustive passes through all eligible book training tokens; 0 uses legacy sampler")
    parser.add_argument("--reading-overlap", type=int, default=64,
                        help="Context tokens carried into next window, masked from repeated loss")
    parser.add_argument("--reading-patience", type=int, default=0,
                        help="Stop after this many complete passes without validation improvement (0 disables)")
    parser.add_argument("--instruction-steps", type=int, default=600)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--question", default="Why can centralized decisions become slow?")
    parser.add_argument("--max-exercises", type=int, default=200)
    parser.add_argument("--qa-training", type=Path,
                        help="Optional owner-reviewed question/answer examples (private JSON)")
    parser.add_argument("--qa-validation", type=Path,
                        help="Separate held-out owner-reviewed question/answer validation JSON")
    args = parser.parse_args(argv)
    out = args.output.resolve()
    if out.exists():
        parser.error("Output must be a new private directory")
    if args.reading_passes < 0 or args.reading_patience < 0:
        parser.error("Reading passes/patience must not be negative")
    if args.reading_passes and not 0 <= args.reading_overlap < args.context:
        parser.error("Reading overlap must be smaller than model context")
    if args.pretrain_steps < 1 or args.instruction_steps < 1:
        parser.error("Training steps must be positive")
    if args.context < 256 or args.embed < 16 or args.layers < 1 or args.batch < 1:
        parser.error("Invalid model or batch size")
    if (args.qa_training is None) != (args.qa_validation is None):
        parser.error("Both --qa-training and --qa-validation are required together")
    if args.max_exercises < 1:
        parser.error("At least one training exercise required")
    torch.set_num_threads(args.threads)
    out.mkdir(parents=True)
    book_dir = out / "books"
    prepared = prepare_approved_books(
        args.ledger, book_dir, target_words=args.section_words
    )
    curriculum = create_book_continuations(book_dir, out / "exercises",
                                           limit_per_split=args.max_exercises)
    train_books = collect_documents(book_dir / "train")
    tokenizer = ByteBPETokenizer.train(
        [row["text"] for row in train_books], vocab_size=args.vocab,
        min_frequency=2
    )
    tokenizer_path = out / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    config = {
        "tokenizer_path": tokenizer_path, "embed": args.embed,
        "layers": args.layers, "heads": args.heads, "context": args.context,
        "batch_size": args.batch, "threads": args.threads, "seed": args.seed
    }
    if args.reading_passes:
        base = train_progressive(
            tokenizer_path=tokenizer_path,
            training=book_dir / "train", validation=book_dir / "validation",
            output=out / "english-pretrain.pt",
            passes=args.reading_passes, overlap=args.reading_overlap,
            patience=args.reading_patience, embed=args.embed,
            heads=args.heads, layers=args.layers, context=args.context,
            batch_size=args.batch, threads=args.threads, seed=args.seed,
            checkpoint_every=25, lr=.0005
        )
    else:
        base = train_language(
            **config, phase="pretrain", training=book_dir / "train",
            validation=book_dir / "validation", output=out / "english-pretrain.pt",
            steps=args.pretrain_steps, checkpoint_every=max(1, args.pretrain_steps // 10),
            lr=.0005
        )
    dialogues = out / "exercises"
    training_path = dialogues / "train.json"
    validation_path = dialogues / "validation.json"
    reviewed_qa = {"train": 0, "validation": 0}
    if args.qa_training is not None:
        # Never auto-infer a 'correct' answer to the author's arguments.
        # Reviewed Q&A pairs are an explicit separate source of supervision.
        combined = {}
        for split, path, extra in (
            ("train", training_path, args.qa_training),
            ("validation", validation_path, args.qa_validation),
        ):
            existing = load_dialogues(path)
            added = load_dialogues(extra)
            existing_ids = {row["id"] for row in existing}
            if existing_ids & {row["id"] for row in added}:
                raise ValueError("Duplicate identifier between book exercise and reviewed Q&A")
            combined[split] = existing + added
            reviewed_qa[split] = len(added)
        train_questions = {row["question"].casefold().strip() for row in combined["train"]}
        validation_questions = {row["question"].casefold().strip() for row in combined["validation"]}
        if train_questions & validation_questions:
            raise ValueError("Reviewed QA training and validation questions overlap")
        training_path = dialogues / "combined-train.json"
        validation_path = dialogues / "combined-validation.json"
        training_path.write_text(json.dumps(combined["train"], ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        validation_path.write_text(json.dumps(combined["validation"], ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    tuned = train_language(
        **config, phase="instruction", training=training_path,
        validation=validation_path,
        output=out / "english-book-instruct.pt",
        init_checkpoint=base["best_checkpoint"],
        steps=args.instruction_steps,
        checkpoint_every=max(1, args.instruction_steps // 10),
        lr=.0003
    )
    checkpoint = Path(tuned["best_checkpoint"])
    question_result = answer_question(checkpoint, tokenizer_path, args.question,
                                      max_tokens=120, threads=args.threads)
    # Only evaluate untouched test examples after the final candidate is
    # selected. Exact-match is diagnostic, not general English competence.
    test_rows = load_dialogues(dialogues / "test.json")[:12]
    tests = []
    for row in test_rows:
        response = answer_question(checkpoint, tokenizer_path, row["question"],
                                   max_tokens=120, threads=args.threads)
        tests.append({
            "id": row["id"],
            "answer": response["answer"],
            "expected": row["answer"],
            "exact": response["answer"].strip() == row["answer"].strip(),
            "stopped_by": response["stopped_by"]
        })
    report = {
        "prepared_books": prepared, "continuation_exercises": curriculum,
        "owner_reviewed_qa": reviewed_qa,
        "tokenizer_sha256": tokenizer.fingerprint(), "tokens_in_vocabulary": tokenizer.vocab_size,
        "pretrain": base, "instruction": tuned,
        "question_result": question_result,
        "held_out_continuation": {
            "exact": sum(row["exact"] for row in tests),
            "total": len(tests), "cases": tests
        },
        "limitations": [
            "One-book section holdouts are not novel-document generalization.",
            "Next-sentence continuation is not general semantic question answering.",
            "Training on books does not guarantee English fluency.",
            "No private source text is published by this command."
        ]
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    print(json.dumps({
        "approved_book_words": prepared["total_words"],
        "split_policy": prepared["policy"],
        "tokenizer_vocab": tokenizer.vocab_size,
        "best_pretrain_loss": base["best_validation_loss"],
        "best_instruction_loss": tuned["best_validation_loss"],
        "held_out_exact": report["held_out_continuation"]["exact"],
        "held_out_total": len(tests),
        "english_proficiency_proven": False,
        "reading_passes_completed": base.get("completed_passes"),
        "reading_coverage_per_complete_pass": (
            1.0 if base.get("completed_passes", 0) > 0 else None),
        "private_report": str(out / "report.json")
    }, indent=2))


if __name__ == "__main__":
    run()
