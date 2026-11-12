"""Author-text literacy exercises; not a substitute for semantic Q&A.

These examples expose the same language model to English continuation in an
assistant-response format. They never synthesize unsupported interpretations
of the author's philosophy and never introduce held-out book sections into
training. Context is deliberately bounded for CPU model compatibility.
"""
from hashlib import sha256
import json
from pathlib import Path
import re
from .corpus import collect_documents


def _sentences(text):
    # Deterministic conservative segmentation. Avoid claiming full NLP
    # parsing of difficult patristic abbreviations.
    for chunk in re.split(r"(?<=[.!?])\s+(?=[A-Z\u201c\"'])", text):
        clean = " ".join(chunk.split())
        if 35 <= len(clean) <= 260 and clean[-1:] in ".!?":
            yield clean


def create_book_continuations(prepared, destination, *, limit_per_split=250,
                              max_question_chars=360):
    source = Path(prepared).resolve()
    output = Path(destination).resolve()
    if source == output or output in source.parents:
        raise ValueError("Exercise output must not overlap prepared corpus")
    if output.exists():
        raise ValueError("Exercise destination must not already exist")
    if type(limit_per_split) is not int or not 1 <= limit_per_split <= 10000:
        raise ValueError("Invalid exercise limit")
    if type(max_question_chars) is not int or max_question_chars < 120:
        raise ValueError("Question length limit too short")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "approved_books_v1":
        raise ValueError("Expected an approved-book English corpus")
    output.mkdir(parents=True)
    summary = {"format": "book_continuation_exercises_v1", "splits": {},
               "source_manifest_sha256": sha256(
                   (source / "manifest.json").read_bytes()).hexdigest(),
               "task": "next-sentence English continuation, not factual inference"}
    try:
        # Collect before limiting so a repeated *question* across splits
        # can be removed from ALL partitions. Pair-only deduplication is
        # insufficient: the same opening may have different continuations.
        candidates = {}
        for split in ("train", "validation", "test"):
            found = []
            docs = collect_documents(source / split)
            for doc in docs:
                for paragraph in re.split(r"\n\s*\n", doc["text"]):
                    sentences = list(_sentences(paragraph))
                    for first, second in zip(sentences, sentences[1:]):
                        if len(first) > max_question_chars or len(second) > 240:
                            continue
                        prompt = "Continue the following English passage with its next sentence:\n" + first
                        key = sha256((first + "\n" + second).encode("utf-8")).hexdigest()
                        found.append({"id": f"{split}_{key[:24]}",
                                      "question": prompt, "answer": second})
            candidates[split] = found
        memberships = {}
        for split, rows in candidates.items():
            for row in rows:
                normalized = " ".join(row["question"].casefold().split())
                memberships.setdefault(normalized, set()).add(split)
        overlapping = {question for question, splits in memberships.items()
                       if len(splits) > 1}
        summary["cross_split_prompts_excluded"] = len(overlapping)
        for split in ("train", "validation", "test"):
            rows, seen_questions = [], set()
            for row in candidates[split]:
                normalized = " ".join(row["question"].casefold().split())
                if normalized in overlapping or normalized in seen_questions:
                    continue
                seen_questions.add(normalized)
                rows.append(row)
                if len(rows) >= limit_per_split:
                    break
            if not rows:
                raise ValueError(f"No independent consecutive sentences for {split} exercises")
            (output / f"{split}.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            summary["splits"][split] = len(rows)
        (output / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except BaseException:
        import shutil
        shutil.rmtree(output)
        raise
    return {"directory": str(output), **summary}
