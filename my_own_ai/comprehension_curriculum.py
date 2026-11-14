"""Compositional, balanced comprehension instruction; no exam imports.

The full fixed curriculum is retained. Versioned, reproducible per-cycle
lessons balance its skills and rehearse foundations. Only the comprehension
stage opts in. Validation uses a separate lexicon and seed namespace.
"""
from collections import Counter, defaultdict, deque
from hashlib import sha256
import random

from .copy_curriculum import synthetic_copy_rows

VERSION = "comprehension_cycles_v1"
GROUPS = ("syntax", "reading", "qa", "conditional", "paraphrase", "coherence", "followup", "foundation")
CORE = GROUPS[:4]
TRAIN_WORDS = (
    ("porter", "ranger", "clerk", "artisan", "custodian", "courier", "farmer", "seamstress", "editor", "mason", "scout", "brewer"),
    ("lantern", "brush", "saw", "basket", "ribbon", "spool", "hammer", "chisel", "candle", "shovel", "cloth", "bottle"),
    ("attic", "barn", "porch", "shed", "courtyard", "cellar", "tower", "orchard", "hall", "cabin", "garage", "wharf"),
)
VALID_WORDS = (
    ("apothecary", "foreman", "herald", "steward", "locksmith", "beekeeper"),
    ("tongs", "kettle", "carafe", "spindle", "casket", "funnel"),
    ("vestibule", "meadow", "pantry", "conservatory", "granary", "alcove"),
)


def fixed_group(row):
    p = row["question"]
    prefixes = {
        "syntax": ("Correct ", "Make a sentence:", "Complete the sentence:", "Complete the thought:", "Finish the clause:"),
        "reading": ("Read:", "Read and name", "Give only the item:", "Extract the named object"),
        "qa": ("Name the location", "Which place", "Answer in a full sentence:"),
        "conditional": (),
        "paraphrase": ("Rephrase ",),
        "coherence": ("Write three connected ",),
        "followup": ("Person:",),
    }
    if "may proceed exactly when" in p:
        return "conditional"
    for group, starts in prefixes.items():
        if p.startswith(starts):
            return group
    raise ValueError("Unclassified fixed comprehension lesson: " + row["id"])


def _rng(split, cycle, group, index, seed):
    key = f"{VERSION}:{split}:{seed}:{cycle}:{group}:{index}"
    return random.Random(int.from_bytes(sha256(key.encode()).digest(), "big"))


def _row(group, cycle, index, seed, split):
    draw = index // 2 if group == "conditional" else index
    rng = _rng(split, cycle, group, draw, seed)
    actor_pool, object_pool, place_pool = TRAIN_WORDS if split == "train" else VALID_WORDS
    actor, obj, place = (rng.choice(pool) for pool in (actor_pool, object_pool, place_pool))
    other = rng.choice([w for w in object_pool if w != obj])
    elsewhere = rng.choice([w for w in place_pool if w != place])
    # Novel identifiers provide a virtually unbounded compositional signal.
    # They are referenced explicitly, not assumed to carry real-world meaning.
    tag = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5 + draw % 5))
    item = f"{obj} {tag}" if draw % 2 == 0 else obj
    location = f"{place} {tag}" if draw % 2 == 1 else place
    article = "An" if actor[0] in "aeiou" else "A"
    n = index % 4
    if group == "syntax":
        pairs = (
            (f"Complete the clause: The {actor} fetched the {item} since", f"the {item} was required."),
            (f"Repair the verb: {article} {actor} carry the {item} to the {location}.", f"{article} {actor} carries the {item} to the {location}."),
            (f"Correct the agreement: The {item} were beside the {location}.", f"The {item} was beside the {location}."),
            (f"Put these words in order: the / {actor} / moved / the / {item}", f"The {actor} moved the {item}."),
        )
    elif group == "reading":
        pairs = (
            (f"The {actor} set down the {item}, then moved the {other}. Name only the object set down.", item),
            (f"In the {location}, the {actor} examined the {item}. Give the item alone.", item),
            (f"The {other} remained outside. The {actor} carried the {item} indoors. What was carried? Give only its name.", item),
            (f"The {item} belonged to the {actor}. Identify the owner with one word.", actor),
        )
    elif group == "qa":
        pairs = (
            (f"The {actor} was at the {location} with a {item}. Name the place only.", location),
            (f"The {actor} left the {elsewhere} and reached the {location}. Give only the destination.", location),
            (f"The {item} was in the {location}, not the {elsewhere}. In which place was it? Give only the place name.", location),
            (f"The {actor} put the {item} in the {location}. Identify the location, without a sentence.", location),
        )
    elif group == "conditional":
        positive = index % 2 == 0
        fact = f"The {item} is ready." if positive else f"The {item} is not ready."
        if (index // 2) % 2 == 0:
            question = f"The rule permits the {actor} to begin exactly when the {item} is ready. {fact} Is beginning permitted? Give yes or no."
        else:
            question = f"Only a ready {item} permits work at the {location}, and readiness is sufficient. {fact} Under this rule may work begin? Answer yes or no."
        pairs = ((question, "yes" if positive else "no"),) * 4
    elif group == "paraphrase":
        pairs = ((f"Express the same meaning differently: The {actor} returned carrying the {item}.", f"The {actor} came back with the {item}."),) * 4
    elif group == "coherence":
        pairs = ((f"Write three connected sentences: the {actor} seeks the {item}, reaches the {location}, and finds it.", f"The {actor} looked for the {item}. The {actor} reached the {location}. There the {actor} found the {item}."),) * 4
    elif group == "followup":
        pairs = ((f"Person: The {item} is in the {location}. Assistant: I can look there. Person: What will you look for? Assistant:", f"I will look for the {item}."),) * 4
    else:
        raise ValueError("Unknown comprehension competency")
    q, a = pairs[n]
    return {"id": f"comprehension-{split}-{cycle}-{group}-{index}", "question": q, "answer": a}


def quotas(fixed_rows):
    counts = Counter(fixed_group(r) for r in fixed_rows)
    # Equalize the four assessed skills while retaining all fixed examples.
    target = max(counts[g] for g in CORE) + 512
    fresh = {g: target - counts[g] for g in CORE}
    # Pair positive and negative versions even if the fixed counts are odd.
    fresh["conditional"] += fresh["conditional"] % 2
    fresh.update({g: max(128, 768 - counts[g]) for g in GROUPS[4:7]})
    fresh["foundation"] = max(256, target // 3)
    return fresh


def cycle_size(fixed_rows):
    return len(fixed_rows) + sum(quotas(fixed_rows).values())


def lesson_groups(fixed_rows, cycle, seed):
    generated, groups = [], [fixed_group(row) for row in fixed_rows]
    for group, count in quotas(fixed_rows).items():
        rows = (synthetic_copy_rows(cycle, count, seed + 1701) if group == "foundation"
                else [_row(group, cycle, i, seed, "train") for i in range(count)])
        generated.extend(rows)
        groups.extend([group] * len(rows))
    # Interleave shuffled competencies, not contiguous blocks of one skill.
    buckets = defaultdict(list)
    for i, group in enumerate(groups):
        buckets[group].append(i)
    for group, indices in buckets.items():
        _rng("order", cycle, group, 0, seed).shuffle(indices)
    queues = {group: deque(indices) for group, indices in buckets.items()}
    order = []
    while any(queues.values()):
        for group in GROUPS:
            if queues[group]:
                order.append(queues[group].popleft())
    return generated, order


def validation_lessons(seed):
    # Frozen, independent development examples, never part of training.
    rows = [_row(group, 0, i, seed, "validation")
            for group in GROUPS[:-1] for i in range(16)]
    rows.extend(synthetic_copy_rows(0, 16, seed + 9901))
    return rows
