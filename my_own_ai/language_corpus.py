"""Local, provenance-preserving, document-disjoint English corpus preparation."""
from hashlib import sha256
import json
from pathlib import Path
import random
import re
import unicodedata

from .corpus import collect_documents


def _words(text):
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold())


def _shingles(text, width=5):
    words = _words(text)
    if len(words) < width:
        return {tuple(words)}
    return {tuple(words[i:i+width]) for i in range(len(words)-width+1)}


def near_duplicate(a, b, threshold=.85):
    x, y = _shingles(a), _shingles(b)
    return len(x & y) / len(x | y) >= threshold if x or y else True


def prepare_english_corpus(source, destination, *, seed=42,
                           validation_ratio=.10, test_ratio=.10):
    """Copy unique source documents into immutable train/validation/test partitions."""
    src, out = Path(source).resolve(), Path(destination).resolve()
    if out.exists():
        raise ValueError("Prepared corpus destination must not exist")
    if out == src or out in src.parents:
        raise ValueError("Output cannot be a source directory")
    if not (0 < validation_ratio < .5 and 0 < test_ratio < .5
            and validation_ratio + test_ratio < .5):
        raise ValueError("Invalid corpus holdout ratios")
    documents = collect_documents(src)
    accepted = []
    rejected = []
    for document in sorted(documents, key=lambda d: d["id"]):
        # Exclude duplicate/near-duplicate works entirely, not just within one partition.
        if any(near_duplicate(document["text"], other["text"]) for other in accepted):
            rejected.append(document["source"])
        else:
            accepted.append(document)
    if len(accepted) < 3:
        raise ValueError("English corpus needs at least three independent documents")
    random.Random(seed).shuffle(accepted)
    n = len(accepted)
    n_test = max(1, round(n * test_ratio))
    n_val = max(1, round(n * validation_ratio))
    partitioned = {
        "test": accepted[:n_test],
        "validation": accepted[n_test:n_test + n_val],
        "train": accepted[n_test + n_val:],
    }
    if any(not rows for rows in partitioned.values()):
        raise ValueError("Corpus split would be empty")
    out.mkdir(parents=True)
    provenance = []
    for split, rows in partitioned.items():
        folder = out / split
        folder.mkdir()
        for doc in rows:
            filename = doc["id"][:32] + ".txt"
            (folder / filename).write_text(doc["text"], encoding="utf-8")
            provenance.append({
                "source": doc["source"], "sha256": doc["id"], "split": split,
                "output": filename, "characters": doc["characters"],
            })
    payload = {"format": "english_document_splits_v1", "seed": seed,
               "source_root": str(src), "documents": provenance,
               "near_duplicates_excluded": rejected}
    (out / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"train": len(partitioned["train"]), "validation": len(partitioned["validation"]),
            "test": len(partitioned["test"]), "near_duplicates_excluded": len(rejected),
            "manifest": str(out / "manifest.json")}
