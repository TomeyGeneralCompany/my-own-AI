"""Deterministic, lossless byte-level BPE with reserved controller tokens.

Model training never imports third-party tokenizers. The ordinary text API encodes
literal controller-looking strings as bytes, not as privileged delimiters.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

SPECIAL_TOKENS = (
    "<|AXIOM|>", "<|CONTEXT|>", "<|DERIVE|>", "<|END|>", "<|PAD|>",
    "<|DOC|>", "<|PARA|>", "<|EQUATION|>", "<|TERMS|>", "<|CONSEQUENCE|>",
)

@dataclass
class ByteBPE:
    merges: list[tuple[int, int]]
    special_tokens: tuple[str, ...] = SPECIAL_TOKENS

    @property
    def base_vocab_size(self) -> int:
        return 256 + len(self.merges)

    @property
    def vocab_size(self) -> int:
        return self.base_vocab_size + len(self.special_tokens)

    def special_id(self, name: str) -> int:
        return self.base_vocab_size + self.special_tokens.index(name)

    @staticmethod
    def _replace(seq: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        out = []
        i = 0
        while i < len(seq):
            if i + 1 < len(seq) and (seq[i], seq[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        return out

    @classmethod
    def train(cls, texts: list[str], target_base_vocab_size: int = 8192,
              min_pair_frequency: int = 2) -> "ByteBPE":
        if target_base_vocab_size < 256:
            raise ValueError("Target vocabulary must include all 256 bytes")
        if min_pair_frequency < 1:
            raise ValueError("Minimum pair frequency must be positive")
        sequences = [list(text.encode("utf-8")) for text in texts]
        merges: list[tuple[int, int]] = []
        while 256 + len(merges) < target_base_vocab_size:
            counts: Counter[tuple[int, int]] = Counter()
            for seq in sequences:
                counts.update(zip(seq, seq[1:]))
            eligible = [(pair, n) for pair, n in counts.items() if n >= min_pair_frequency]
            if not eligible:
                break
            pair = min(eligible, key=lambda entry: (-entry[1], entry[0]))[0]
            new_id = 256 + len(merges)
            sequences = [cls._replace(seq, pair, new_id) for seq in sequences]
            merges.append(pair)
        return cls(merges)

    def encode(self, text: str) -> list[int]:
        seq = list(text.encode("utf-8"))
        # Apply merges in training order; semantic control markers cannot arise here.
        for new_id, pair in enumerate(self.merges, start=256):
            seq = self._replace(seq, pair, new_id)
        return seq

    def _expansions(self) -> list[bytes]:
        expanded = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            if not (0 <= a < len(expanded) and 0 <= b < len(expanded)):
                raise ValueError("Invalid BPE merge table")
            expanded.append(expanded[a] + expanded[b])
        return expanded

    def decode(self, ids: list[int], *, allow_special: bool = False) -> str:
        expanded = self._expansions()
        parts: list[bytes] = []
        for token in ids:
            if 0 <= token < len(expanded):
                parts.append(expanded[token])
            elif allow_special and len(expanded) <= token < self.vocab_size:
                parts.append(self.special_tokens[token - len(expanded)].encode("utf-8"))
            else:
                raise ValueError(f"Invalid or privileged token in ordinary decode: {token}")
        return b"".join(parts).decode("utf-8")

    def encode_structured(self, sections: list[tuple[str, str]]) -> list[int]:
        """Only controller code can request special tokens via explicit sections."""
        output = []
        for name, text in sections:
            output.append(self.special_id(name))
            output.extend(self.encode(text))
        return output

    def as_dict(self) -> dict:
        return {"format": 1, "merges": [list(x) for x in self.merges],
                "special_tokens": list(self.special_tokens)}

    def digest(self) -> str:
        data = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPE":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw["format"] != 1:
            raise ValueError("Unsupported tokenizer format")
        if tuple(raw["special_tokens"]) != SPECIAL_TOKENS:
            raise ValueError("Special token assignments do not match protocol")
        instance = cls([tuple(pair) for pair in raw["merges"]])
        instance._expansions()
        return instance
