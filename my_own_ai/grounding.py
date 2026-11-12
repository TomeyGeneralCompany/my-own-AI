"""Bounded evidence prompts for the span-selection proposal format."""
from dataclasses import dataclass
import re

from .retrieval import STOP, WORDS

SPAN_FORMAT = "grounded_span_v2"
MAX_CHOICES = 3


@dataclass(frozen=True)
class GroundedProposal:
    passage: int
    quote: str
    inference: str
    finished: bool


def _clip(text, budget):
    return text.encode("utf-8")[:budget].decode("utf-8", errors="ignore")


def _excerpt(text, question, budget):
    """Keep an exact word-bounded substring with the best query coverage."""
    terms = {word for word in WORDS.findall(question.casefold()) if word not in STOP}
    words = list(re.finditer(r"\S+", text))
    byte_positions = [0]
    for character in text:
        byte_positions.append(byte_positions[-1] + len(character.encode("utf-8")))
    end = 0
    best = (0, 0, "")
    for start, word in enumerate(words):
        while end < len(words) and byte_positions[words[end].end()] - byte_positions[word.start()] <= budget:
            end += 1
        if end <= start:
            end = start + 1
            continue
        excerpt = text[word.start():words[end - 1].end()]
        hits = terms.intersection(WORDS.findall(excerpt.casefold()))
        score = (len(hits), len(excerpt), excerpt)
        if score[:2] > best[:2]:
            best = score
    return best[2]


def span_prompt(request, capacity):
    """Return the prompt and exactly the evidence choices displayed in it."""
    limit = capacity * 3 // 4
    stage = request.stage.value
    task = {
        "formulate": "Plan the argument.",
        "construct": "Choose evidence, then give a complete inference.",
        "correct": "Choose evidence and revise the inference.",
    }[stage]
    lines = [task, "Question: " + _clip(request.objective.replace("\n", " "), capacity // 5)]
    if stage != "formulate":
        lines.append("Plan: " + _clip(request.plan.replace("\n", " "), capacity // 16))
    if stage == "correct":
        lines.append("Feedback: " + _clip(request.feedback.replace("\n", " "), capacity // 12))
        previous = request.candidate.partition("\nInference:")[2] or request.candidate
        lines.append("Previous: " + _clip(previous.replace("\n", " "), capacity // 16))
    header = "\n".join(lines) + "\n"
    label = "Plan:" if stage == "formulate" else "Choice and inference:"
    remaining = max(0, limit - len((header + label).encode()))
    sources = request.context["evidence"]
    choices = min(MAX_CHOICES, len(sources), max(1, remaining // 76))
    allowance = min(220, remaining // max(1, choices) - 5)
    excerpts = {}
    if allowance >= 8:
        for number, source in enumerate(sources[:choices], 1):
            quote = _excerpt(source["text"], request.objective, allowance)
            if len(quote) >= 8:
                excerpts[number] = quote
    body = "".join(f"{number}: {quote}\n" for number, quote in excerpts.items())
    prefix = (header + body).encode()
    suffix = label.encode()
    # Give the first output byte a stable position across differently sized evidence.
    padding = b" " * max(0, limit - len(prefix) - len(suffix))
    return list(prefix + padding + suffix), excerpts


def evidence_terms(text):
    return {word for word in WORDS.findall(text.casefold()) if len(word) >= 4 and word not in STOP}
