"""Next-token and controller-prefix training with honest independent validation."""
from __future__ import annotations
from dataclasses import dataclass
import math
import random
import torch
import torch.nn.functional as F
from .tokenizer import ByteBPE
from .model import AncestorModel

@dataclass
class TrainConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0
    microbatch: int = 1
    accumulation: int = 1
    block_size: int = 1024

def masked_next_token_loss(logits: torch.Tensor, targets: torch.Tensor,
                           mask: torch.Tensor | None = None) -> torch.Tensor:
    if logits.shape[:2] != targets.shape:
        raise ValueError("Logits and target dimensions mismatch")
    losses = F.cross_entropy(logits.reshape(-1,logits.size(-1)),
                             targets.reshape(-1), reduction="none").reshape(targets.shape)
    if mask is None:
        return losses.mean()
    if mask.shape != targets.shape or not bool((mask >= 0).all()):
        raise ValueError("Invalid loss mask")
    denom = mask.sum()
    if denom.item() <= 0:
        raise ValueError("Loss mask has no active target positions")
    return (losses * mask).sum() / denom

class Trainer:
    def __init__(self, model: AncestorModel, config: TrainConfig):
        if config.block_size > model.cfg.context_length:
            raise ValueError("Training block is longer than model context")
        if min(config.microbatch,config.accumulation) < 1:
            raise ValueError("Invalid batch size")
        self.model = model
        self.config = config
        self.optimizer = torch.optim.AdamW(
            model.parameters(),lr=config.learning_rate,
            betas=(config.beta1,config.beta2), weight_decay=config.weight_decay)
        self.completed_updates = 0
        self.tokens_seen = 0

    def step(self, batches: list[tuple[torch.Tensor,torch.Tensor,torch.Tensor|None]]) -> float:
        if len(batches) != self.config.accumulation:
            raise ValueError("Expected complete gradient accumulation cycle")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total = 0.0
        tokens = 0
        for x,y,mask in batches:
            if x.shape != y.shape or x.ndim != 2 or x.size(0) != self.config.microbatch:
                raise ValueError("Invalid training batch")
            logits = self.model(x)
            loss = masked_next_token_loss(logits,y,mask)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            (loss / len(batches)).backward()
            total += float(loss.detach())
            tokens += int(x.numel())
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.config.max_grad_norm)
        if not torch.isfinite(norm):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("Non-finite gradient norm")
        self.optimizer.step()
        self.completed_updates += 1
        self.tokens_seen += tokens
        return total/len(batches)

    @torch.no_grad()
    def validate(self, batches: list[tuple[torch.Tensor,torch.Tensor,torch.Tensor|None]]) -> float:
        if not batches:
            raise ValueError("Validation requires held-out data")
        self.model.eval()
        scores=[]
        for x,y,mask in batches:
            scores.append(float(masked_next_token_loss(self.model(x),y,mask)))
        return sum(scores)/len(scores)

def flat_batches(ids: list[int], block_size: int, batch_size: int,
                 *, rng: random.Random) -> tuple[torch.Tensor,torch.Tensor, None]:
    if len(ids) < block_size + 1:
        raise ValueError("Insufficient tokens for training window")
    starts=[rng.randrange(0,len(ids)-block_size) for _ in range(batch_size)]
    x=torch.tensor([ids[i:i+block_size] for i in starts],dtype=torch.long)
    y=torch.tensor([ids[i+1:i+block_size+1] for i in starts],dtype=torch.long)
    return x,y,None

def partition_documents(documents: list[tuple[str,str]], train_percent: int = 90,
                        validation_percent: int = 5) -> dict[str,list[tuple[str,str]]]:
    """Split by document identity, not overlapping samples. Lock final test independently."""
    if train_percent <= 0 or validation_percent < 0 or train_percent+validation_percent >= 100:
        raise ValueError("Invalid split")
    if len({name for name,_ in documents}) != len(documents):
        raise ValueError("Duplicate document identity")
    partitions={"train":[],"validation":[],"test":[]}
    # Stable by ID, but enough texts are required for every partition.
    from hashlib import sha256
    for item in documents:
        bucket=int(sha256(item[0].encode()).hexdigest()[:8],16) % 100
        key="train" if bucket < train_percent else "validation" if bucket < train_percent+validation_percent else "test"
        partitions[key].append(item)
    return partitions

def structured_prefix_batch(tokenizer: ByteBPE, sections: list[tuple[str,str]],
                            max_tokens: int) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    """Train next-token generation on the derivation, with earlier prefix as context.

    Because logits[t] predicts token[t+1], a target at position j corresponds
    to a loss mask at position j-1.
    """
    if not sections or sections[-1][0] != "<|DERIVE|>":
        raise ValueError("Final structured section must be DERIVE")
    ids=tokenizer.encode_structured(sections)+[tokenizer.special_id("<|END|>")]
    if len(ids)<2 or len(ids)-1>max_tokens:
        raise ValueError("Structured sample exceeds model context")
    derive_marker=tokenizer.special_id("<|DERIVE|>")
    start=ids.index(derive_marker)+1
    if start>=len(ids)-1:
        raise ValueError("Empty derivation target")
    x=torch.tensor([ids[:-1]],dtype=torch.long)
    y=torch.tensor([ids[1:]],dtype=torch.long)
    mask=torch.tensor([[1.0 if j+1 >= start else 0.0 for j in range(len(ids)-1)]])
    return x,y,mask
