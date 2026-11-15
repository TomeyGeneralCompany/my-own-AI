"""Independent, compositional transfer exercises; no gauntlet dependency."""
import random
import hashlib
import re
from pathlib import Path
from .comprehension_curriculum import GROUPS, _row
from .copy_curriculum import synthetic_copy_rows

VERSION = 2
POOLS = {
    "train": (("child","friend","worker","driver","doctor","cook","neighbor","captain"),
              ("cup","key","shoe","rope","phone","coat","apple","bag"),
              ("house","school","bridge","room","park","street","station","yard")),
    "probe": (("visitor","pupil","nurse","guide","singer","guest","runner","guard"),
              ("coin","bell","hat","fork","ring","towel","plate","pencil"),
              ("museum","church","theater","beach","farm","hotel","bank","clinic")),
}
PHRASES = {
    "reading": ("What did {a} pick up? Name the thing alone.", "Tell me what {a} picked up, just the noun.", "Identify the article lifted by {a}; omit all other words."),
    "qa": ("Where did {a} put it? Give only the place.", "Tell me the destination, and nothing more.", "Identify the place in which the thing was put; use its name alone."),
    "conditional": ("Does the rule allow it? Say yes or no.", "Is the action permitted? Reply with yes or no.", "Given that condition, is permission granted? Answer yes or no."),
    "syntax": ("Fix the verb in this sentence:", "Rewrite with correct subject-verb agreement:", "Repair the grammatical error in the following statement:"),
}
def exercise(group, index, seed, cycle, split="train", variant=0):
    if split not in POOLS:
        raise ValueError("Unknown transfer split")
    pair = index // 2 if group == "conditional" else index
    rng = random.Random(int.from_bytes(hashlib.sha256(
        f"transfer-v2:{split}:{seed}:{cycle}:{group}:{pair}".encode()).digest(),"big"))
    a, item, place = (rng.choice(p) for p in POOLS[split])
    other = rng.choice([x for x in POOLS[split][1] if x != item])
    k = (pair + variant) % 3
    if group == "reading":
        q = f"The {a} left the {other} behind and picked up the {item}. " + PHRASES[group][k].format(a="the "+a)
        answer = item
    elif group == "qa":
        q = f"The {a} put the {item} in the {place}. " + PHRASES[group][k].format(a="the "+a)
        answer = place
    elif group == "conditional":
        yes = index % 2 == 0
        q = f"The {a} is allowed to leave only when the {item} is dry, and dryness is enough. The {item} is " + ("dry. " if yes else "not dry. ") + PHRASES[group][k]
        answer = "yes" if yes else "no"
    elif group == "syntax":
        q = PHRASES[group][k] + f" The {a} hold the {item}."
        answer = f"The {a} holds the {item}."
    elif group == "foundation":
        row = synthetic_copy_rows(cycle, index+1, seed + (71321 if split=="probe" else 39181))[-1]
        q, answer = row["question"], row["answer"]
    else:
        row = _row(group, cycle, index, seed + (71321 if split=="probe" else 39181),
                   "validation" if split=="probe" else "train")
        q, answer = row["question"], row["answer"]
    return {"id":f"transfer-{split}-{cycle}-{group}-{index}-{variant}","question":q,"answer":answer}


def grounded_row(row):
    """Correct ambiguous context-free factual QA without changing source provenance."""
    q, a = row["question"], row["answer"]
    if q.startswith("Answer in a full sentence:"):
        # The original drill asserted an arbitrary answer to an ungrounded question.
        fact = a.rstrip(".") + "."
        return dict(row, question=f"Given the fact: {fact} {q}")
    return row


def author_examples(folder, limit=256):
    """Verbatim short spans from owner-approved TRAIN units only; no worldview labels."""
    if folder is None:
        return []
    rows=[]
    seen=set()
    for path in sorted(Path(folder).glob("*.txt")):
        text=path.read_text(encoding="utf-8")
        for line in text.splitlines():
            for sentence in re.split(r"(?<=[.!?])\s+",line.strip()):
                words=sentence.split()
                if not 5 <= len(words) <= 16 or len(sentence.encode("utf-8"))>100:
                    continue
                if not sentence[0].isalpha() or sentence[-1] not in ".!?":
                    continue
                if sentence in seen:
                    continue
                seen.add(sentence)
                prefix=" ".join(words[:-2])
                tail=" ".join(words[-2:])
                ident=hashlib.sha256((path.name+"\0"+sentence).encode()).hexdigest()[:24]
                rows.append({"id":"owner-language-"+ident,
                             "question":"Continue this exact quotation: "+prefix,
                             "answer":tail})
                if len(rows) >= limit:
                    return rows
    return rows


def probe_rows(group, seed, cycle):
    # Fresh seed namespace, disjoint content vocabulary for core skills.
    return [exercise(group, i, seed, cycle, "probe") for i in range(12)]
