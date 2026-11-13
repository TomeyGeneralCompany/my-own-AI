"""Unbounded compositional copy lessons, reproducible without keeping samples in memory.

Each complete foundation teaching cycle sees new character combinations rather
than allowing the original model to memorize a finite 72-word lookup table.
The generator has no access to the frozen examination or its answer key.
"""
import random
from hashlib import sha256

VERSION = "character_copy_cycle_v1"
ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def synthetic_copy_rows(cycle, count, seed=42):
    if not isinstance(cycle, int) or cycle < 0 or count <= 0:
        raise ValueError("Invalid dynamic instruction cycle")
    rows = []
    templates = (
        "Repeat exactly the word after the colon: {}",
        "Copy the English word exactly, without extra text: {}",
        "Read the label and answer with only its word: Label = {}",
        "Which word is written here? {}",
    )
    for index in range(count):
        # Hash-based per-item seeds make content independent of Python hash
        # randomization, workers and checkpoint restoration order.
        digest = sha256(f"{VERSION}:{seed}:{cycle}:{index}".encode()).digest()
        rng = random.Random(int.from_bytes(digest, "big"))
        length = 3 + (index % 10)
        word = "".join(rng.choice(ALPHABET) for _ in range(length))
        # A pseudo-word not seen previously still has to be copied directly
        # from its prompt; no answer can be looked up from a fixed lexicon.
        question = templates[index % len(templates)].format(word)
        rows.append({"id":f"synthetic-copy-{cycle}-{index}",
                     "question":question,"answer":word})
    return rows
