"""Controller-owned bounded context construction and paragraph segmentation."""
from __future__ import annotations
from dataclasses import dataclass
from .tokenizer import ByteBPE
from .governance import Constitution

@dataclass(frozen=True)
class ContextWindow:
    token_ids: list[int]
    generation_reserve: int
    source_ids: tuple[str, ...]
    axiom_digest: str

class ContextBuilder:
    def __init__(self, tokenizer: ByteBPE, constitution: Constitution, capacity: int):
        if capacity not in (1024,2048) and capacity < 16:
            raise ValueError("Invalid capacity")
        self.tokenizer = tokenizer
        self.constitution = constitution
        self.capacity = capacity

    def build(self, sources: list[tuple[str,str]], task: str, *,
              reserve: int = 128) -> ContextWindow:
        if reserve < 1 or reserve >= self.capacity:
            raise ValueError("Invalid generation reserve")
        axioms = self.constitution.axiom_prefix()
        context = "\n\n".join(f"[{sid}]\n{text}" for sid,text in sources)
        pieces = [("<|AXIOM|>",axioms),("<|CONTEXT|>",context),
                  ("<|DERIVE|>",task)]
        ids = self.tokenizer.encode_structured(pieces)
        if len(ids) + reserve > self.capacity:
            raise ValueError("Mandatory context exceeds available working tier; no silent truncation")
        return ContextWindow(ids,reserve,tuple(sid for sid,_ in sources),
                             self.constitution.digest())

    def paragraph_segments(self, text: str, available_tokens: int) -> list[str]:
        if available_tokens < 1:
            raise ValueError("No token space")
        encoded = self.tokenizer.encode(text)
        if len(encoded) <= available_tokens:
            return [text]
        # Splits at valid UTF-8 character boundaries; never decode partial bytes.
        chunks = []
        current = ""
        for char in text:
            if len(self.tokenizer.encode(current + char)) > available_tokens:
                if not current:
                    raise ValueError("Single UTF-8 character exceeds token budget")
                chunks.append(current)
                current = char
            else:
                current += char
        if current:
            chunks.append(current)
        assert "".join(chunks) == text
        return chunks

    def pack_dependencies(self, records: list[tuple[str,str]], task: str,
                          reserve: int = 128) -> ContextWindow:
        # Mandatory records are never dropped; optional retrieval is a
        # separate policy and must not rewrite required source dependencies.
        return self.build(records, task, reserve=reserve)
