"""Lossless numbered-paragraph indexing for the owner's chapterless treatises.

The number at the beginning of a line starts a logical paragraph. EVERYTHING
until the next consecutive number, including standalone display equations,
mathematical labels and explanatory prose, belongs to that same paragraph.
We do not normalize whitespace, strip mathematical symbols, or split at pages.
"""
from hashlib import sha256
import re

HEADER = re.compile(r"(?m)^[ \t]*\[(\d{1,6})\][ \t]+")


def numbered_paragraphs(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty treatise")
    source = text.replace("\r\n", "\n").replace("\r", "\n")
    markers = list(HEADER.finditer(source))
    if not markers:
        raise ValueError("Treatise has no numbered paragraphs")
    numbers = [int(m.group(1)) for m in markers]
    if len(numbers) != len(set(numbers)):
        raise ValueError("Repeated numbered paragraph")
    if any(b != a + 1 for a, b in zip(numbers, numbers[1:])):
        raise ValueError("Nonconsecutive paragraph numbering would risk lost content")
    if numbers[0] not in (0, 1):
        raise ValueError("Treatise must begin at numbered paragraph 0 or 1")
    units = []
    preface = source[:markers[0].start()]
    if preface.strip():
        units.append({"id": "frontmatter", "number": None, "text": preface,
                      "sha256": sha256(preface.encode("utf-8")).hexdigest()})
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(source)
        body = source[marker.start():end]
        units.append({"id": "p%06d" % numbers[index], "number": numbers[index],
                      "text": body, "sha256": sha256(body.encode("utf-8")).hexdigest()})
    assert "".join(u["text"] for u in units) == source
    return units


def paragraph_ledger(units):
    numbers = [u["number"] for u in units if u["number"] is not None]
    return {"unit": "numbered_paragraph", "numbered_paragraphs": len(numbers),
            "first_number": min(numbers), "last_number": max(numbers),
            "ordered_sha256": sha256("".join(u["text"] for u in units).encode("utf-8")).hexdigest(),
            "units": [{"id": u["id"], "number": u["number"], "sha256": u["sha256"],
                       "bytes": len(u["text"].encode("utf-8"))} for u in units]}
