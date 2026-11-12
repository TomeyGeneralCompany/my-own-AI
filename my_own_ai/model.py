"""Original randomly initialized causal Transformer, without pretrained weights."""
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.nn import functional as F

@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256
    context_size: int = 128
    embed_dim: int = 64
    heads: int = 4
    layers: int = 2
    dropout: float = 0.0

    def validate(self):
        if (self.vocab_size != 256 and not 262 <= self.vocab_size <= 16384) or min(self.context_size, self.embed_dim, self.heads, self.layers) <= 0:
            raise ValueError("Invalid model dimensions")
        if self.embed_dim % self.heads:
            raise ValueError("embed_dim must be divisible by heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("Invalid dropout")

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.embed_dim)
        self.attn = nn.MultiheadAttention(cfg.embed_dim, cfg.heads, dropout=cfg.dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(cfg.embed_dim)
        self.ff = nn.Sequential(nn.Linear(cfg.embed_dim, 4 * cfg.embed_dim), nn.GELU(),
                                nn.Linear(4 * cfg.embed_dim, cfg.embed_dim), nn.Dropout(cfg.dropout))

    def forward(self, x):
        n = x.size(1)
        mask = torch.ones((n, n), device=x.device, dtype=torch.bool).triu(1)
        z = self.norm1(x)
        attention, _ = self.attn(z, z, z, attn_mask=mask, need_weights=False)
        x = x + attention
        return x + self.ff(self.norm2(x))

class LocalTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        cfg.validate()
        self.config = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.position_embedding = nn.Embedding(cfg.context_size, cfg.embed_dim)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.layers)])
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)

    def forward(self, ids, targets=None):
        if ids.ndim != 2 or not 1 <= ids.size(1) <= self.config.context_size:
            raise ValueError("Expected [batch, tokens] within context size")
        positions = torch.arange(ids.size(1), device=ids.device)
        x = self.token_embedding(ids) + self.position_embedding(positions)
        for layer in self.layers:
            x = layer(x)
        logits = self.head(self.norm(x))
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, prefix, max_new_tokens=60, temperature=1.0):
        if not prefix or max_new_tokens < 0 or temperature <= 0:
            raise ValueError("Invalid generation settings")
        self.eval()
        result = list(prefix)
        device = next(self.parameters()).device
        for _ in range(max_new_tokens):
            ids = torch.tensor([result[-self.config.context_size:]], dtype=torch.long, device=device)
            logits, _ = self(ids)
            probabilities = F.softmax(logits[0, -1] / temperature, dim=-1)
            result.append(int(torch.multinomial(probabilities, 1)))
        return bytes(result)

def encode(text):
    return list(text.encode("utf-8"))

def decode(ids):
    return bytes(ids).decode("utf-8", errors="replace")
