"""Broad, original English instruction drills with strict lexical train/test separation.

English is taught as a communication capability. No external pretrained model,
philosophical teacher or outside worldview enters the authored reasoning system.
Frozen gauntlet examples are never imported or optimized against.
"""
from hashlib import sha256

TRAIN = [
    ("scribe","scroll","archive"), ("mason","stone","quarry"),
    ("sailmaker","canvas","dock"), ("potter","vessel","kiln"),
    ("weaver","thread","loom"), ("carpenter","plank","shed"),
    ("baker","dough","kitchen"), ("surveyor","map","office"),
    ("archivist","ledger","storehouse"), ("cobbler","shoe","shop"),
    ("merchant","crate","market"), ("printer","proof","pressroom"),
    ("botanist","leaf","greenhouse"), ("chemist","sample","laboratory"),
    ("pilot","chart","hangar"), ("musician","score","hall"),
    ("forester","sapling","nursery"), ("seamstress","cloth","workroom"),
    ("metallurgist","ingot","foundry"), ("navigator","compass","ship"),
    ("tailor","coat","studio"), ("farrier","horseshoe","stable"),
    ("watchmaker","spring","bench"), ("editor","manuscript","library"),
]
VALIDATION = [
    ("butcher","cleaver","pantry"), ("florist","bouquet","stall"),
    ("glassblower","flask","furnace"), ("cartographer","atlas","survey"),
    ("jeweler","gem","atelier"), ("beekeeper","hive","meadow"),
]


def drills(split):
    if split not in ("train","validation"):
        raise ValueError("Unknown English acquisition split")
    actors = TRAIN if split=="train" else VALIDATION
    seen = set()
    rows = []
    for actor, _, place in actors:
        for _, obj, _ in actors:
            cases = (
                ("syntax",f"Complete the thought: A {actor} brought the {obj} to the {place} because",
                 f"the {obj} was needed there."),
                ("syntax",f"Finish the clause: The {actor} delivered the {obj} to the {place} since",
                 f"the {obj} was needed there."),
                ("reading",f"Read and name the object, not the person: The {actor} examined the {obj} at the {place}. What object was examined?",
                 obj),
                ("reading",f"Give only the item: A {actor} inspected a {obj} in the {place}. What item did the {actor} inspect?",
                 obj),
                ("qa",f"Name the location and nothing else: The {actor} went to the {place} with a {obj}. Which location was named?",
                 place),
                ("qa",f"Which place appears in this statement? A {actor} found a {obj} near the {place}.",
                 place),
                ("conditional",f"A {actor} may proceed exactly when the {obj} has arrived. It has arrived. May the {actor} proceed? Reply yes or no.",
                 "yes"),
                ("conditional",f"A {actor} may proceed exactly when the {obj} has arrived. It has not arrived. May the {actor} proceed? Reply yes or no.",
                 "no"),
                ("grammar",f"Correct the verb agreement: The {actor} carry the {obj}.",
                 f"The {actor} carries the {obj}."),
                ("grammar",f"Correct this English sentence: The {obj} were near the {place}.",
                 f"The {obj} was near the {place}."),
                ("paraphrase",f"Rephrase without changing the meaning: The {actor} returned to the {place} with the {obj}.",
                 f"The {actor} came back to the {place} carrying the {obj}."),
                ("conversation",f"Person: My {obj} is missing at the {place}. Assistant: Look there first. Person: Why there first? Assistant:",
                 f"Because the {obj} was last seen at the {place}."),
                ("coherence",f"Write three connected English sentences about a {actor} finding a missing {obj} at a {place}.",
                 f"The {actor} noticed the {obj} was missing. The {actor} searched the {place}. Finally, the {actor} found the {obj}."),
                ("reading",f"Extract the named object only. At the {place}, the {actor} placed the {obj} on a table. What was placed there?",
                 obj),
            )
            for category, question, answer in cases:
                if question in seen:
                    continue
                seen.add(question)
                ident=sha256((split+"\0"+question+"\0"+answer).encode()).hexdigest()[:24]
                rows.append({"id":"english-acquisition-"+ident,
                             "question":question,"answer":answer})
    if split=="validation":
        # A stable, independent sub-sample keeps every frequent checkpoint
        # evaluation bounded. The full 200-case frozen exam is still separate.
        rows = rows[::4]
    return rows
