"""Deterministic English drills interleaved with the owner's source-grounded lessons.

Language mechanics are skills, not a substitute philosophical authority. The
author's lessons remain the only philosophical reasoning supervision.
"""
from hashlib import sha256
import json
from pathlib import Path


# Partition by lexical families to check transfer to unseen nouns and verbs.
FAMILIES = {
    "train": [
        ("mechanic", "valve", "workshop"), ("gardener", "seed", "garden"),
        ("librarian", "book", "library"), ("engineer", "diagram", "laboratory"),
        ("teacher", "letter", "classroom"), ("builder", "beam", "warehouse"),
        ("farmer", "basket", "field"), ("clerk", "record", "office"),
    ],
    "validation": [
        ("sailor", "compass", "harbor"), ("painter", "brush", "studio"),
        ("baker", "loaf", "bakery"), ("driver", "parcel", "depot"),
    ],
}
TEMPLATES = (
    ("grammar", "Correct the sentence: The {actor} carry the {object}.",
     "The {actor} carries the {object}."),
    ("grammar", "Correct the sentence: A {actor} are in the {place}.",
     "A {actor} is in the {place}."),
    ("grammar", "Correct the sentence: The {object} were in the {place}.",
     "The {object} was in the {place}."),
    ("reading", "Read: The {actor} put the {object} in the {place}. Who put it there?",
     "The {actor}."),
    ("reading", "Read: The {actor} carried the {object} into the {place}. What was carried?",
     "The {object}."),
    ("reading", "Read: The {actor} found the {object} in the {place}. Where was it found?",
     "In the {place}."),
    ("syntax", "Make a sentence: the / {actor} / examined / {object}",
     "The {actor} examined the {object}."),
    ("syntax", "Complete the sentence: The {actor} opened the {place} because",
     "the {place} needed to be opened."),
    ("conversation", "Answer in a full sentence: Where does the {actor} work?",
     "The {actor} works in the {place}."),
    ("conversation", "Answer in a full sentence: What did the {actor} inspect?",
     "The {actor} inspected the {object}."),
)


def english_drills(split):
    if split not in FAMILIES:
        raise ValueError("Unknown English drill split")
    rows = []
    seen = set()
    for actor, _, actor_place in FAMILIES[split]:
        for _, obj, object_place in FAMILIES[split]:
            place = actor_place if actor_place != object_place else object_place
            for category, prompt, answer in TEMPLATES:
                values = {"actor": actor, "object": obj, "place": place}
                question = prompt.format(**values)
                response = answer.format(**values)
                key = sha256((split + "\\0" + category + "\\0" + question).encode()).hexdigest()[:24]
                if question not in seen:
                    seen.add(question)
                    rows.append({"id": "english-" + key, "question": question, "answer": response})
    return rows


def augment_grounded_curriculum(directory):
    """Make a stable bilingual-free *English* curriculum without altering source axioms.

    All philosophical examples are authored-source extracts already in the
    reasoning lessons. English drills teach grammar and reading only.
    """
    folder = Path(directory)
    summary = {}
    for split in ("train", "validation"):
        path = folder / (split + ".json")
        grounded = json.loads(path.read_text(encoding="utf-8"))
        if not grounded:
            raise ValueError("No owner's reasoning lessons to preserve")
        additions = english_drills(split)
        # Do not displace author-grounded reasoning: each English drill is
        # paired with a source-grounded example; retain *all* source lessons.
        merged = grounded + additions
        keys = [r["question"].casefold().strip() for r in merged]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate instruction prompt")
        path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        summary[split] = {"authored_lessons": len(grounded), "english_drills": len(additions)}
    train = json.loads((folder / "train.json").read_text(encoding="utf-8"))
    validation = json.loads((folder / "validation.json").read_text(encoding="utf-8"))
    if {x["question"].casefold() for x in train} & {x["question"].casefold() for x in validation}:
        raise ValueError("Train/validation drill leakage")
    return summary
