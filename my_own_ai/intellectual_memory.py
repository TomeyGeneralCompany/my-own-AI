"""CPU-bounded, source-grounded memory for English acquisition.

No examination or validation answers enter this store. Fixed lessons and
deterministically reconstructible generated lessons are the source archive;
only a handful of conceptual records and probe statistics need active RAM.
"""
import math
import re
from .comprehension_curriculum import GROUPS, _row, fixed_group
from .copy_curriculum import synthetic_copy_rows

VERSION = 1
RULES = {
    "syntax": "Identify the subject and verb. Match person and number; preserve meaning and word order.",
    "reading": "Identify the requested relation in the passage. Trace the subject, action and object; ignore distractors.",
    "qa": "Identify what the question asks for. Trace the entity to its stated location or relation; obey the requested answer format.",
    "conditional": "Identify the rule and the stated fact. Check whether the necessary and sufficient condition holds, including negation.",
    "paraphrase": "Preserve the original proposition and relationships while changing its wording.",
    "coherence": "Maintain the same entities, temporal order and causal relationships across sentences.",
    "followup": "Resolve references using the preceding conversation and answer the latest question.",
    "foundation": "Read the symbols in sequence and reproduce the requested word without substituting another word.",
}
KEYWORDS = {
    "syntax": "correct repair verb agreement grammar clause sentence order plural",
    "reading": "read passage object carried owner item identify extract",
    "qa": "location place destination where answer question",
    "conditional": "rule exactly when if permitted ready sufficient necessary condition may",
    "paraphrase": "rephrase paraphrase express meaning differently",
    "coherence": "connected sentences story sequence",
    "followup": "person assistant conversation followup",
    "foundation": "copy reproduce spell word",
}


def source_examples(group, seed, cycle=0):
    """Stable training-only anchors, paired for conditionals."""
    if group == "foundation":
        return synthetic_copy_rows(cycle, 4, seed + 1701)
    return [_row(group, cycle, i, seed, "train") for i in range(4)]


def new_memory(rows, seed, cycle):
    return {
        "version": VERSION, "seed": seed, "first_cycle": cycle,
        "covered_through": cycle - 1,
        "source_archive": [dict(r) for r in rows],
        "records": {g: {"rule": RULES[g], "source": source_examples(g, seed),
                        "tier": "acquisition", "streak": 0, "loss": None,
                        "best": None} for g in GROUPS},
    }


def validate_memory(memory, rows, seed, cycle):
    if (memory.get("version") != VERSION or memory.get("seed") != seed
            or memory.get("source_archive") != rows
            or set(memory.get("records", {})) != set(GROUPS)
            or memory["covered_through"] != cycle - 1):
        raise ValueError("Incompatible intellectual memory or coverage")
    for g, record in memory["records"].items():
        if (record["rule"] != RULES[g] or record["source"] != source_examples(g, seed)
                or record["tier"] not in ("acquisition", "consolidation", "established")
                or record["streak"] < 0):
            raise ValueError("Invalid memory record")


def consolidate(memory, losses, cycle):
    """Promote on repeated low loss; demote on regression. No exam feedback."""
    if cycle != memory["covered_through"] + 1 or set(losses) != set(GROUPS):
        raise ValueError("Memory consolidation must follow complete cycle")
    for group, loss in losses.items():
        if not math.isfinite(loss) or loss < 0:
            raise ValueError("Invalid memory probe loss")
        r = memory["records"][group]
        best = r["best"]
        regression = best is not None and loss > best + 0.25
        r["streak"] = r["streak"] + 1 if loss <= 0.5 and not regression else 0
        r["tier"] = ("established" if r["streak"] >= 3 else
                     "consolidation" if r["streak"] or regression else "acquisition")
        r["loss"] = loss
        r["best"] = loss if best is None else min(best, loss)
    memory["covered_through"] = cycle


def replay_group(memory, step):
    """Round-robin coverage with additional priority for weak/regressed skills."""
    if (step // 16) % 2 == 0:
        return GROUPS[(step // 32) % len(GROUPS)]
    def priority(g):
        r = memory["records"][g]
        loss = r["loss"] if r["loss"] is not None else 1.
        return loss + (0.5 if r["tier"] != "established" else 0.)
    ranked = sorted(GROUPS, key=lambda g: (-priority(g), g))
    return ranked[(step // 32) % 3]


def context_for(memory, question, group=None):
    """Retrieve one abstract reasoning rule, never an answer or source exemplar."""
    if group is None:
        words = set(re.findall(r"[a-z]+", question.lower()))
        scores = {g: len(words & set(KEYWORDS[g].split())) for g in GROUPS}
        group = max(GROUPS, key=lambda g: scores[g])
        if not scores[group]:
            return ""
    return memory["records"][group]["rule"]


def contextual_question(memory, question, tokenizer, limit, group=None):
    rule = context_for(memory, question, group)
    candidate = "Reasoning context: " + rule + "\n" + question
    return candidate if rule and len(tokenizer.encode(candidate)) <= limit else question
