"""Lossless logical reading units for essays, chaptered books and equations.

Unlike the author's numbered treatise, these works have heterogeneous
structure. Each *unit* is a span of the extracted source, not a generated
paraphrase; joining units reconstructs the exact input.
"""
from hashlib import sha256
import re

SEPARATOR = re.compile(r"\n[ \t]*\n+")
HEADING = re.compile(
    r"(?im)^(?:chapter\s+[0-9ivxlcdm]+\b|book\s+(?:one|two|three|four|[0-9ivxlcdm]+)\b|"
    r"(?:(?:[ivxlcdm]+|\d+)\.\s+[A-Z][^\n]{4,90})|"
    r"(?:introduction|conclusion|abstract|notation and formal conventions|bibliography))\s*$"
)
SYMBOLS = re.compile(r"[=∑∂∫∈≠≤≥→←±∞∇√ΦτκβΓ]|(?:\b[A-Z]\([^)]{1,35}\))")


def structured_units(text, *, max_chars=3400):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty source document")
    if not 500 <= max_chars <= 12000:
        raise ValueError("Invalid unit limit")
    # Make boundaries only at observed double-line paragraph separators and
    # independent headings; keep the bytes in each span.
    boundaries = {0, len(text)}
    for match in SEPARATOR.finditer(text):
        boundaries.add(match.end())
    for match in HEADING.finditer(text):
        boundaries.add(match.start())
    markers = sorted(boundaries)
    raw = [text[a:b] for a, b in zip(markers, markers[1:]) if b > a]
    chunks = []
    for item in raw:
        # If a single prose span exceeds context, split conservatively at
        # sentence punctuation nearest the size budget. Never split an isolated
        # equation line out of its surrounding prose because we merge it below.
        while len(item) > max_chars:
            window = item[:max_chars]
            candidates = list(re.finditer(r"(?<=[.!?;])\s+(?=[A-Z(\[])",window))
            if candidates:
                offset = candidates[-1].end()
            else:
                end = window.rfind(" ")
                offset = end + 1 if end > 200 else max_chars
            chunks.append(item[:offset])
            item = item[offset:]
        if item:
            chunks.append(item)
    grouped=[]
    for item in chunks:
        if grouped and (item.lstrip().startswith("[Linearized source equation]") or
                        (len(item.strip()) < 90 and (SYMBOLS.search(item) or
           not re.search(r"[.!?]\s*$", item.strip())))):
            grouped[-1] += item
        else:
            grouped.append(item)
    assert "".join(grouped) == text, "Unit segmentation lost source bytes"
    units=[]
    for i, content in enumerate(grouped):
        if not content.strip():
            if units:
                units[-1]["text"]+=content
                units[-1]["sha256"]=sha256(units[-1]["text"].encode()).hexdigest()
            else:
                # Leading whitespace stays with the next unit.
                if i + 1 < len(grouped):
                    grouped[i+1]=content+grouped[i+1]
            continue
        units.append({"id":f"u{i:06d}","text":content,
                      "sha256":sha256(content.encode()).hexdigest()})
    assert "".join(x["text"] for x in units)==text
    return units
