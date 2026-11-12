"""Original deterministic byte-fallback BPE tokenizer.

The first 256 IDs represent raw bytes, providing lossless UTF-8 round trips.
No downloaded vocabulary, pretrained tokenization artifacts or external models.
"""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

PAD = 256
BOS = 257
EOS = 258
USER = 259
ASSISTANT = 260
UNK = 261
FIRST_MERGE = 262
SPECIAL = {"pad": PAD, "bos": BOS, "eos": EOS, "user": USER, "assistant": ASSISTANT, "unk": UNK}


class ByteBPETokenizer:
    def __init__(self, merges=()):
        self.merges = [tuple(map(int, pair)) for pair in merges]
        self.vocab = [bytes([i]) for i in range(256)] + [b""] * (FIRST_MERGE - 256)
        for left, right in self.merges:
            if not (0 <= left < len(self.vocab) and 0 <= right < len(self.vocab)):
                raise ValueError("Merge refers to a missing token")
            self.vocab.append(self.vocab[left] + self.vocab[right])
        self.ranks = {pair: FIRST_MERGE + i for i, pair in enumerate(self.merges)}
        if len(self.ranks) != len(self.merges):
            raise ValueError("Duplicate merge rule")

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text):
        if not isinstance(text, str):
            raise TypeError("Tokenizer input must be text")
        ids = list(text.encode("utf-8"))
        while len(ids) > 1:
            matches = [(self.ranks[(a, b)], i) for i, (a, b) in enumerate(zip(ids, ids[1:]))
                       if (a, b) in self.ranks]
            if not matches:
                break
            token, pos = min(matches)
            ids[pos:pos + 2] = [token]
        return ids

    def decode(self, ids):
        pieces = []
        for token in ids:
            if not isinstance(token, int) or not 0 <= token < len(self.vocab):
                raise ValueError("Invalid token ID")
            if token >= PAD and token < FIRST_MERGE:
                raise ValueError("Special tokens cannot be decoded as ordinary text")
            pieces.append(self.vocab[token])
        return b"".join(pieces).decode("utf-8", errors="strict")

    @classmethod
    def train(cls, documents, vocab_size=8192, min_frequency=2):
        if not FIRST_MERGE <= vocab_size <= 16384 or min_frequency < 2:
            raise ValueError("Invalid BPE configuration")
        sequences = [list(text.encode("utf-8")) for text in documents if text.strip()]
        if not sequences:
            raise ValueError("Cannot train tokenizer on empty corpus")
        merges = []
        while len(merges) + FIRST_MERGE < vocab_size:
            pairs = Counter((a, b) for seq in sequences for a, b in zip(seq, seq[1:]))
            if not pairs:
                break
            best, count = min(pairs.items(), key=lambda pair: (-pair[1], pair[0]))
            if count < min_frequency:
                break
            token = FIRST_MERGE + len(merges)
            merges.append(best)
            updated = []
            for seq in sequences:
                result = []
                i = 0
                while i < len(seq):
                    if i + 1 < len(seq) and (seq[i], seq[i + 1]) == best:
                        result.append(token)
                        i += 2
                    else:
                        result.append(seq[i])
                        i += 1
                updated.append(result)
            sequences = updated
        return cls(merges)

    def state(self):
        return {"format": "original_byte_bpe_v1", "special": SPECIAL,
                "merges": [list(pair) for pair in self.merges]}

    def fingerprint(self):
        return sha256(json.dumps(self.state(), sort_keys=True,
                                 separators=(",", ":")).encode("utf-8")).hexdigest()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state(), ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    @classmethod
    def load(cls, path):
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(state, dict) or set(state) != {"format", "special", "merges"} or (
                state["format"] != "original_byte_bpe_v1" or state["special"] != SPECIAL
                or not isinstance(state["merges"], list)):
            raise ValueError("Unsupported tokenizer format")
        tokenizer = cls(state["merges"])
        if tokenizer.vocab_size > 16384:
            raise ValueError("Tokenizer vocabulary too large")
        return tokenizer
