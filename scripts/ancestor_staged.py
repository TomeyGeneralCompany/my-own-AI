"""Session-limited but competence-unlimited neural learning, one locked stage at a time.

The controller owns promotion. CPU runner sessions end normally, checkpoint,
and resume automatically. A step target is a training *batch*, not graduation.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import zipfile
import torch

from my_own_ai.corpus import collect_documents
from my_own_ai.english_drills import english_drills
from my_own_ai.language_training import train_language
from my_own_ai.language_inference import answer_question
from my_own_ai.language_evaluation import evaluate_english, load_gauntlet
from my_own_ai.persistent_registry import digest_file
from my_own_ai.progressive_reading import train_progressive
from my_own_ai.published_corpus import prepare_published_corpus, validate_published_manifest
from my_own_ai.reasoning_curriculum import extract_lessons
from my_own_ai.stage_controller import (initial_ledger, verify_ledger, advance, fingerprint,
    ENGLISH, READ, COMPREHEND, SYNTHESIS, REVIEW, COMPLETE)

# Fixed compatible architecture of the trained original-weight g0002 ancestor.
MODEL = dict(embed=128, heads=4, layers=4, context=512, batch_size=2,
             seed=42, threads=2)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def prepare_book(prepared, book, destination):
    """Copy every logical unit, including validation/test-held-out exercises.

    Examinations are withheld *questions*, not unread source text. Each book
    is read in full before its comprehension gate.
    """
    base = Path(prepared)
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=True)
    reading = out / "reading"
    (reading / "train").mkdir(parents=True, exist_ok=True)
    (reading / "validation").mkdir()
    count = 0
    calibration = None
    for split in ("train", "validation", "test"):
        for entry in sorted((base / split).glob(book + "_*.txt")):
            copied = reading / "train" / entry.name
            shutil.copyfile(entry, copied)
            count += 1
            if calibration is None:
                calibration = entry.read_text()
    if count == 0:
        raise ValueError("No source units for book " + book)
    # Repeated source passage is loss telemetry only, NOT a held-out exam.
    (reading / "validation" / "calibration.txt").write_text("Calibration passage from the current authored work: " + calibration)
    manifest = json.loads((base / "manifest.json").read_text())
    manifest["reading_calibration"] = "overlap_with_source_not_a_competence_exam"
    (reading / "manifest.json").write_text(json.dumps(manifest))
    exercises = out / "exercises"
    exercises.mkdir()
    for split in ("train", "validation", "test"):
        lessons = []
        used = set()
        for doc in collect_documents(base / split):
            if not Path(doc["source"]).name.startswith(book + "_"):
                continue
            for row in extract_lessons(doc["text"], source_id=doc["source"],
                                       unit_label=Path(doc["source"]).stem):
                if row["question"] in used:
                    continue
                used.add(row["question"])
                lessons.append({k: row[k] for k in ("id", "question", "answer")})
        dump(exercises / (split + ".json"), lessons)
    return {"units": count, "source_sha256": sha256(
        "".join((reading / "train" / p.name).read_text()
                for p in sorted((reading / "train").glob("*.txt"))).encode()).hexdigest()}


def score_book(checkpoint, tokenizer, questions, source_sha, *, limit=30):
    cases = json.loads(Path(questions).read_text())
    if len(cases) < 10:
        raise ValueError("Book exam requires at least ten unseen questions")
    tested = cases[:limit]
    results = []
    for case in tested:
        reply = answer_question(checkpoint, tokenizer, case["question"], max_tokens=100, threads=2)
        actual = " ".join(reply["answer"].casefold().split()).strip(" .!?")
        expected = " ".join(case["answer"].casefold().split()).strip(" .!?")
        results.append({"id": case["id"], "question": case["question"],
                        "actual": reply["answer"], "expected": case["answer"],
                        "correct": actual == expected})
    return {"total": len(results), "correct": sum(r["correct"] for r in results),
            "answers": results, "checkpoint_sha256": digest_file(checkpoint),
            "source_sha256": source_sha, "questions_sha256": fingerprint(questions)}


def train_batch(args):
    root, state, scratch = Path(args.repository).resolve(), Path(args.state), Path(args.scratch)
    state.mkdir(parents=True, exist_ok=True)
    prepared = scratch / "prepared"
    if not prepared.exists():
        prepare_published_corpus(root, root / "published_books/manifest.json", prepared)
    published = validate_published_manifest(root, root / "published_books/manifest.json")
    books = [x["id"] for x in published["sources"]]
    ledger_path = state / "stage-ledger.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
        verify_ledger(ledger, published["manifest_sha256"], books)
    else:
        if not args.parent_state:
            raise ValueError("Initial staged generation requires trained ancestral weights")
        ledger = initial_ledger(published["manifest_sha256"], books, args.parent_generation,
                                {x["id"]: x["sha256"] for x in published["sources"]})
        parent = Path(args.parent_state)
        if not (parent / "assistant.best.pt").exists():
            raise ValueError("Missing completed ancestral checkpoint")
        shutil.copyfile(parent / "tokenizer.json", state / "tokenizer.json")
        shutil.copyfile(parent / "assistant.best.pt", state / "inherited.pt")
        dump(ledger_path, ledger)
    tok = state / "tokenizer.json"
    stage = ledger["stage"]
    if stage in (REVIEW, COMPLETE):
        return {"stage": stage, "status": "owner_review_required" if stage == REVIEW else "qualified"}
    if stage == SYNTHESIS:
        # No invented independent reasoning certificate: owner must inspect
        # generated answers to cross-book and mathematical questions.
        checkpoint = state / "current.best.pt"
        questions = [
            "What is the relationship between virtue, trust, and efficient capital gain?",
            "State the two foundational axioms of communal identity and hierarchy.",
            "Give a mathematical model for trust reducing exchange friction.",
            "What changes if trust decreases while the other variables are held constant?",
        ]
        responses = []
        for question in questions:
            answer = answer_question(checkpoint, tok, question, max_tokens=160, threads=2)
            responses.append({"question": question, "actual": answer["answer"]})
        report = {"stage": SYNTHESIS, "checkpoint_sha256": digest_file(checkpoint),
                  "responses": responses, "passes_logic": False, "passes_mathematics": False,
                  "passes_transfer": False, "human_review_required": True}
        dump(state / "last-exam.json", report)
        # The review remains locked until a verified external grading result
        # is provided: never promote based on another training-step budget.
        ledger["stage"] = REVIEW
        ledger["last_checkpoint_sha256"] = report["checkpoint_sha256"]
        ledger["evidence"].append({"stage": SYNTHESIS,
            "report_sha256": sha256(json.dumps(report, sort_keys=True).encode()).hexdigest(),
            "checkpoint_sha256": report["checkpoint_sha256"],
            "passed": False, "requires_owner_evaluation": True})
        dump(ledger_path, ledger)
        return {"stage": REVIEW, "status": "awaiting_owner_review", "responses": responses}

    if stage == ENGLISH:
        inputs = scratch / "english"
        inputs.mkdir(parents=True, exist_ok=True)
        for split in ("train", "validation"):
            dump(inputs / (split + ".json"), english_drills(split))
        ckpt = state / "english.pt"
        best = state / "english.best.pt"
        start = int(torch.load(ckpt, map_location="cpu", weights_only=True)["completed_steps"]) if ckpt.exists() else 0
        train_language(tokenizer_path=tok, training=inputs / "train.json",
                       validation=inputs / "validation.json", output=ckpt,
                       phase="instruction", steps=start + args.english_updates,
                       resume=ckpt.exists(),
                       init_checkpoint=None if ckpt.exists() else state / "inherited.pt",
                       checkpoint_every=25, **MODEL)
        results = evaluate_english(best, tok,
                                   load_gauntlet(root / "examples/english/gauntlet.json"),
                                   max_tokens=50, limit=200, threads=2)
        report = {"evaluated": results["total"],
                  "automatically_scored": results["automatically_scored"],
                  "exact_correct": results["exact_correct"],
                  "category_scores": results["categories"],
                  "raw_results": results,
                  "checkpoint_sha256": digest_file(best)}
        dump(state / "last-exam.json", report)
        passed = advance(ledger, "english_exam", report,
                         checkpoint_sha256=digest_file(best))
        if passed:
            shutil.copyfile(best, state / "current.best.pt")
        dump(ledger_path, ledger)
        return {"stage": ledger["stage"], "english_correct": results["exact_correct"],
                "english_total": results["automatically_scored"],
                "passed": passed, "completed_updates": start + args.english_updates}

    book = books[ledger["book_index"]]
    book_dir = scratch / ("book_" + book)
    details = prepare_book(prepared, book, book_dir)
    previous = (state / "current.best.pt")
    if stage == READ:
        ckpt = state / ("read_" + book + ".pt")
        best = state / ("read_" + book + ".best.pt")
        result = train_progressive(
            tokenizer_path=tok, training=book_dir / "reading/train",
            validation=book_dir / "reading/validation",
            output=ckpt, passes=1, resume=ckpt.exists(),
            init_checkpoint=None if ckpt.exists() else previous,
            max_updates=args.read_updates, checkpoint_every=25,
            embed=128, heads=4, layers=4, context=512, batch_size=2,
            seed=42, threads=2)
        completed = result["completed_passes"] == 1 and result["next_window"] == 0
        report = {"book": book, "read_complete": completed,
                  "next_window": result["next_window"],
                  "covered_targets": result["unique_supervised_tokens_per_pass"] if completed else 0,
                  "last_read_unit": result["last_read_unit"],
                  "ordered_manifest_sha256": published["manifest_sha256"],
                  "read_units_expected": details["units"],
                  "read_units_seen": details["units"] if completed else result["next_window"],
                  "source_sha256": published["sources"][ledger["book_index"]]["sha256"],
                  "checkpoint_sha256": digest_file(ckpt)}
        dump(state / "last-exam.json", report)
        if completed and advance(ledger, "reading_coverage", report,
                                  checkpoint_sha256=digest_file(ckpt)):
            shutil.copyfile(ckpt, previous)
        dump(ledger_path, ledger)
        return {"stage": ledger["stage"], "book": book, "completed": completed,
                "progress": result["next_window"], "updates": result["updates_this_run"]}
    if stage == COMPREHEND:
        folder = book_dir / "exercises"
        if not (json.loads((folder / "train.json").read_text()) and
                json.loads((folder / "validation.json").read_text())):
            raise ValueError("Book cannot be examined without source-grounded teaching exercises")
        ckpt = state / ("learn_" + book + ".pt")
        best = state / ("learn_" + book + ".best.pt")
        start = int(torch.load(ckpt, map_location="cpu", weights_only=True)["completed_steps"]) if ckpt.exists() else 0
        train_language(tokenizer_path=tok, training=folder / "train.json",
                       validation=folder / "validation.json", output=ckpt,
                       phase="instruction", steps=start + args.book_updates,
                       resume=ckpt.exists(),
                       init_checkpoint=None if ckpt.exists() else previous,
                       checkpoint_every=25, **MODEL)
        report = score_book(best, tok, folder / "test.json",
                            published["sources"][ledger["book_index"]]["sha256"])
        report["book"] = book
        dump(state / "last-exam.json", report)
        passed = advance(ledger, "book_exam", report,
                         checkpoint_sha256=digest_file(best))
        if passed:
            shutil.copyfile(best, previous)
        dump(ledger_path, ledger)
        return {"stage": ledger["stage"], "book": book, "correct": report["correct"],
                "total": report["total"], "passed": passed}
    raise ValueError("Unknown stage")


def package(folder, destination):
    """Integrity verified, whitelisted model/state transport between runs."""
    root = Path(folder)
    names = sorted(p.name for p in root.iterdir() if p.is_file() and
                   (p.suffix in (".pt", ".json") and not p.name.startswith(".")))
    if "stage-ledger.json" not in names or "tokenizer.json" not in names:
        raise ValueError("Uninitialized stage state")
    manifest = {name: digest_file(root / name) for name in names}
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in names:
            bundle.write(root / name, name)
        bundle.writestr("integrity.json", json.dumps(manifest))


def restore(archive, folder):
    root = Path(folder)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("Refusing to replace a stage state")
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)) or "integrity.json" not in names:
            raise ValueError("Invalid checkpoint archive")
        manifest = json.loads(bundle.read("integrity.json"))
        if set(names) != set(manifest) | {"integrity.json"}:
            raise ValueError("Checkpoint manifest mismatch")
        for name, digest in manifest.items():
            if ("/" in name or "\\" in name or not
                    (name.endswith(".pt") or name.endswith(".json"))):
                raise ValueError("Unsafe member name")
            payload = bundle.read(name)
            if len(payload) > 300_000_000 or sha256(payload).hexdigest() != digest:
                raise ValueError("Corrupted checkpoint member")
            (root / name).write_bytes(payload)


def main():
    p = argparse.ArgumentParser()
    cmd = p.add_subparsers(dest="command", required=True)
    t = cmd.add_parser("train")
    t.add_argument("--repository", default=".")
    t.add_argument("--state", required=True)
    t.add_argument("--scratch", required=True)
    t.add_argument("--parent-state")
    t.add_argument("--parent-generation", default="g0002")
    t.add_argument("--english-updates", type=int, default=400)
    t.add_argument("--book-updates", type=int, default=400)
    t.add_argument("--read-updates", type=int, default=300)
    for op in ("pack", "unpack"):
        x = cmd.add_parser(op)
        x.add_argument("--state", required=True)
        x.add_argument("--archive", required=True)
    a = p.parse_args()
    if a.command == "train":
        result = train_batch(a)
        print(json.dumps(result, indent=2))
    elif a.command == "pack":
        package(a.state, a.archive)
    else:
        restore(a.archive, a.state)


if __name__ == "__main__":
    main()
