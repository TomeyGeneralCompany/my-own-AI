"""Persistent book-trained Core: progressive learning and inference-only questions."""
import argparse
import json
import os
from pathlib import Path
import shutil
import torch

from my_own_ai.book_instruction import create_book_continuations
from my_own_ai.corpus import collect_documents
from my_own_ai.language_training import _example, load_dialogues, load_language, train_language
from my_own_ai.language_inference import answer_question
from my_own_ai.language_evaluation import load_gauntlet, evaluate_english
from my_own_ai.persistent_registry import (
    digest_file, load_active_pointer, package_generation, unpack_generation,
    validate_generation,
)
from my_own_ai.progressive_reading import train_progressive
from my_own_ai.published_corpus import prepare_published_corpus, validate_published_manifest
from my_own_ai.tokenizer import ByteBPETokenizer


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".temporary")
    temporary.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prepare_fitting_dialogues(training, validation, tokenizer, context, output):
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    report, kept = {}, {}
    for split, path in (("train", training), ("validation", validation)):
        rows, used, skipped = load_dialogues(path), [], []
        for row in rows:
            try:
                _example(row, tokenizer, context)
                used.append(row)
            except ValueError:
                skipped.append(row["id"])
        if not used:
            raise ValueError("No instructions fit context in " + split)
        kept[split] = used
        report[split] = {"total": len(rows), "used": len(used), "skipped_ids": skipped}
        write_json(directory / (split + ".json"), used)
    if ({r["question"].casefold().strip() for r in kept["train"]} &
            {r["question"].casefold().strip() for r in kept["validation"]}):
        raise ValueError("Training and validation questions overlap")
    write_json(directory / "fit-report.json", report)
    return report


def train(args):
    generation = validate_generation(args.generation)
    repo, state, scratch = Path(args.repository).resolve(), Path(args.state), Path(args.scratch)
    publications = validate_published_manifest(repo, repo / "published_books/manifest.json")
    scratch.mkdir(parents=True, exist_ok=True)
    books = scratch / "books"
    prepare_published_corpus(repo, repo / "published_books/manifest.json", books,
                             target_words=args.section_words)
    create_book_continuations(books, scratch / "exercises",
                              limit_per_split=args.max_exercises)
    state.mkdir(parents=True, exist_ok=True)
    meta = state / "generation.json"
    saved = read_json(meta) if meta.is_file() else None
    config = {"context": args.context, "embed": args.embed,
              "heads": args.heads, "layers": args.layers,
              "batch_size": args.batch, "seed": args.seed}
    parent = None
    if args.parent_state:
        if saved:
            raise ValueError("Cannot replace a resumed generation's parent")
        older = Path(args.parent_state)
        ancestor = read_json(older / "generation.json")
        if ancestor.get("phase") != "candidate" or ancestor.get("model_config") != config:
            raise ValueError("Incompatible ancestral model")
        parent = older / "assistant.best.pt"
        if not parent.is_file():
            raise ValueError("Missing ancestral model checkpoint")
        shutil.copyfile(older / "tokenizer.json", state / "tokenizer.json")
    if saved:
        if (saved.get("generation") != generation or
                saved.get("published_manifest_sha256") != publications["manifest_sha256"] or
                saved.get("model_config") != config or
                saved.get("target_reading_passes") != args.passes or
                saved.get("target_instruction_steps") != args.instruction_steps):
            raise ValueError("Generation/corpus/model settings changed during resume")
        if saved.get("phase") == "candidate":
            raise ValueError("Finished candidate is immutable")
    tokenizer_path = state / "tokenizer.json"
    if not tokenizer_path.is_file():
        training_docs = collect_documents(books / "train")
        tok = ByteBPETokenizer.train([d["text"] for d in training_docs],
                                     vocab_size=args.vocab)
        tok.save(tokenizer_path)
    tok = ByteBPETokenizer.load(tokenizer_path)
    if saved and saved.get("tokenizer_sha256") != tok.fingerprint():
        raise ValueError("Tokenizer changed after checkpoint")
    if saved is None:
        saved = {
            "format": "ancestor_generation_v1", "generation": generation,
            "phase": "reading",
            "published_manifest_sha256": publications["manifest_sha256"],
            "published_corpus_id": publications["corpus_id"],
            "tokenizer_sha256": tok.fingerprint(),
            "model_config": config, "target_reading_passes": args.passes,
            "target_instruction_steps": args.instruction_steps,
            "completed_reading_passes": 0, "completed_instruction_steps": 0,
            "parent_checkpoint_sha256": digest_file(parent) if parent else None,
        }
    reading = state / "reading.pt"
    if saved["phase"] == "reading":
        result = train_progressive(
            tokenizer_path=tokenizer_path, training=books / "train",
            validation=books / "validation", output=reading,
            passes=args.passes, overlap=args.overlap,
            context=args.context, embed=args.embed, heads=args.heads,
            layers=args.layers, batch_size=args.batch, threads=args.threads,
            seed=args.seed, checkpoint_every=min(25, args.max_reading_updates),
            lr=args.reading_lr, max_updates=args.max_reading_updates,
            resume=reading.exists(),
            init_checkpoint=parent if not reading.exists() else None)
        saved["completed_reading_passes"] = result["completed_passes"]
        saved["reading_progress"] = {
            "next_window": result["next_window"],
            "completed_steps": result["completed_steps"],
            "best_loss": result["best_validation_loss"],
            "stop_reason": result["stop_reason"],
        }
        if result["completed_passes"] < args.passes:
            write_json(meta, saved)
            return saved
        saved["phase"] = "instruction"
    if saved["phase"] == "instruction":
        fitting = prepare_fitting_dialogues(
            scratch / "exercises/train.json", scratch / "exercises/validation.json",
            tok, args.context, scratch / "fitted")
        saved["instruction_fit"] = fitting
        assistant = state / "assistant.pt"
        count = (int(torch.load(assistant, map_location="cpu", weights_only=True)["completed_steps"])
                 if assistant.exists() else 0)
        target = min(args.instruction_steps, count + args.max_instruction_updates)
        if target > count:
            result = train_language(
                tokenizer_path=tokenizer_path,
                training=scratch / "fitted/train.json",
                validation=scratch / "fitted/validation.json",
                output=assistant, phase="instruction",
                steps=target, resume=assistant.exists(),
                init_checkpoint=None if assistant.exists() else state / "reading.best.pt",
                embed=args.embed, heads=args.heads, layers=args.layers,
                context=args.context, batch_size=args.batch, threads=args.threads,
                seed=args.seed, lr=args.instruction_lr,
                checkpoint_every=min(25, args.max_instruction_updates))
            saved["best_instruction_loss"] = result["best_validation_loss"]
        saved["completed_instruction_steps"] = target
        if target == args.instruction_steps:
            saved["phase"] = "candidate"
    write_json(meta, saved)
    return saved


def evaluate(state, cases, output, limit=200, tokens=40, threads=2):
    state = Path(state)
    metadata = read_json(state / "generation.json")
    if metadata.get("phase") != "candidate":
        raise ValueError("Incomplete Core cannot be evaluated")
    checkpoint, tokenizer = state / "assistant.best.pt", state / "tokenizer.json"
    load_language(checkpoint, tokenizer)
    result = evaluate_english(checkpoint, tokenizer, load_gauntlet(cases),
                              limit=limit, max_tokens=tokens, threads=threads)
    report = {
        "format": "ancestor_evaluation_v1",
        "generation": metadata["generation"],
        "gauntlet_sha256": digest_file(cases),
        "checkpoint_sha256": digest_file(checkpoint),
        "tokenizer_sha256": metadata["tokenizer_sha256"],
        "exact_correct": result["exact_correct"],
        "automatically_scored": result["automatically_scored"],
        "human_review_pending": result["manual_review_pending"],
        "evaluated": result["total"],
        "category_scores": result["categories"],
        "promotable": False,
        "reason": "Independent human review required; raw scores alone never promote",
        "raw_results": result,
    }
    write_json(output, report)
    return report


def ask(state, question, evidence, output, active=None, threads=2, tokens=120):
    state = Path(state)
    metadata = read_json(state / "generation.json")
    if metadata.get("phase") != "candidate":
        raise ValueError("Question requires a completed model")
    if active:
        pointer = load_active_pointer(Path(active).read_text(encoding="utf-8"))
        if pointer["generation"] != metadata["generation"]:
            raise ValueError("Downloaded model differs from the active generation")
    if not question.strip() or len(question) > 1000 or len(evidence) > 1200:
        raise ValueError("Invalid question or evidence length")
    prompt = (question if not evidence.strip() else
              "Unverified user evidence:\n" + evidence + "\nQuestion: " + question)
    checkpoint = state / "assistant.best.pt"
    answer = answer_question(checkpoint, state / "tokenizer.json",
                             prompt, max_tokens=tokens, threads=threads)
    report = {
        "format": "ancestor_answer_v1", "generation": metadata["generation"],
        "question": question, "model_prompt": prompt,
        "evidence": evidence or None, "answer": answer,
        "checkpoint_sha256": digest_file(checkpoint),
        "tokenizer_sha256": metadata["tokenizer_sha256"],
        "training_executed": False,
    }
    write_json(output, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    for name in ("state", "scratch", "generation"):
        t.add_argument("--" + name, required=True)
    t.add_argument("--repository", default=".")
    t.add_argument("--parent-state")
    for name, default in (
        ("section-words",512), ("max-exercises",250), ("vocab",2048),
        ("context",512), ("embed",128), ("heads",4), ("layers",4),
        ("batch",2), ("seed",42), ("threads",2), ("passes",5),
        ("overlap",64), ("instruction-steps",600),
        ("max-reading-updates",40), ("max-instruction-updates",40),
    ):
        t.add_argument("--" + name, type=int, default=default)
    t.add_argument("--reading-lr", type=float, default=.0005)
    t.add_argument("--instruction-lr", type=float, default=.0003)
    pack = sub.add_parser("pack")
    pack.add_argument("--state", required=True)
    pack.add_argument("--archive", required=True)
    unpack = sub.add_parser("unpack")
    unpack.add_argument("--state", required=True)
    unpack.add_argument("--archive", required=True)
    unpack.add_argument("--generation")
    unpack.add_argument("--sha256")
    q = sub.add_parser("ask")
    q.add_argument("--state", required=True)
    q.add_argument("--active")
    q.add_argument("--question", default=os.environ.get("CORE_QUESTION",""))
    q.add_argument("--evidence", default=os.environ.get("CORE_EVIDENCE",""))
    q.add_argument("--output", required=True)
    q.add_argument("--tokens", type=int, default=120)
    e = sub.add_parser("evaluate")
    e.add_argument("--state", required=True)
    e.add_argument("--cases", default="examples/english/gauntlet.json")
    e.add_argument("--output", required=True)
    e.add_argument("--limit", type=int, default=200)
    active = sub.add_parser("inspect-active")
    active.add_argument("--file", required=True)
    args = p.parse_args()
    if args.command == "train":
        result = train(args)
    elif args.command == "pack":
        result = package_generation(args.state, args.archive)
    elif args.command == "unpack":
        result = unpack_generation(args.archive, args.state,
                                   expected_generation=args.generation,
                                   expected_archive_sha256=args.sha256)
    elif args.command == "evaluate":
        result = evaluate(args.state, args.cases, args.output, args.limit)
    elif args.command == "ask":
        result = ask(args.state, args.question, args.evidence, args.output,
                     active=args.active, tokens=args.tokens)
    else:
        result = load_active_pointer(Path(args.file).read_text(encoding="utf-8"))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
