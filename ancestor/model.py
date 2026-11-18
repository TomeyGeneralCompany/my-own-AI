"""Original pre-norm causal decoder-only transformer; no pretrained weights."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F

@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_length: int = 1024
    width: int = 512
    layers: int = 8
    heads: int = 8
    ffn_width: int = 2048
    dropout: float = 0.0

    def validate(self):
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if self.vocab_size < 256 or min(self.context_length, self.width, self.layers, self.heads, self.ffn_width) <= 0:
            raise ValueError("Invalid model dimensions")

class CausalAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.heads = cfg.heads
        self.width = cfg.width
        self.proj_qkv = nn.Linear(cfg.width, 3 * cfg.width)
        self.out = nn.Linear(cfg.width, cfg.width)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.proj_qkv(x).chunk(3, dim=-1)
        def shape(z):
            return z.reshape(b, t, self.heads, d // self.heads).transpose(1, 2)
        q, k, v = shape(q), shape(k), shape(v)
        # is_causal=True prohibits reading future positions.
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                            dropout_p=self.dropout.p if self.training else 0.0)
        return self.out(a.transpose(1, 2).contiguous().reshape(b, t, d))

class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.width)
        self.attn = CausalAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.width)
        self.ffn = nn.Sequential(nn.Linear(cfg.width, cfg.ffn_width),
                                 nn.GELU(), nn.Linear(cfg.ffn_width, cfg.width),
                                 nn.Dropout(cfg.dropout))
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))

class AncestorModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.token = nn.Embedding(cfg.vocab_size, cfg.width)
        self.pos = nn.Embedding(cfg.context_length, cfg.width)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.norm = nn.LayerNorm(cfg.width)
        self.head = nn.Linear(cfg.width, cfg.vocab_size, bias=False)
        self.head.weight = self.token.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.ndim != 2:
            raise ValueError("Expected batch, time token IDs")
        b, t = ids.shape
        if t > self.cfg.context_length or t < 1:
            raise ValueError("Context window overflow or empty sequence")
        x = self.token(ids) + self.pos(torch.arange(t, device=ids.device))[None, :, :]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(ids[:, -self.cfg.context_length:])[:, -1, :]
            prob = torch.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(prob, 1)
            ids = torch.cat([ids, next_id], dim=1)
        return ids

    def manifest(self):
        return asdict(self.cfg)
