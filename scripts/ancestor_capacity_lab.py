"""Controlled original-weight capacity study, without modifying existing generations.

Reports trainable parameter counts and CPU forward/backward throughput for
three transformer sizes. This benchmark does not claim reasoning ability.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
import torch
from my_own_ai.model import LocalTransformer, ModelConfig


def study(*, context=512, vocab=2048, steps=5, threads=2):
    if steps < 1 or threads < 1:
        raise ValueError("Invalid capacity study settings")
    torch.set_num_threads(threads)
    result=[]
    for embed,layers in ((128,4),(192,6),(256,8)):
        torch.manual_seed(42)
        cfg=ModelConfig(vocab_size=vocab,context_size=context,
                        embed_dim=embed,heads=4,layers=layers)
        net=LocalTransformer(cfg)
        opt=torch.optim.AdamW(net.parameters(),lr=0.0003)
        tokens=torch.randint(0,vocab,(2,min(context,128)))
        labels=tokens.roll(-1,1)
        started=perf_counter()
        for _ in range(steps):
            logits,_=net(tokens)
            loss=torch.nn.functional.cross_entropy(
                logits.reshape(-1,vocab), labels.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        elapsed=perf_counter()-started
        count=sum(p.numel() for p in net.parameters())
        result.append({"embed":embed,"layers":layers,"heads":4,
                       "parameters":count,"steps":steps,
                       "elapsed_seconds":round(elapsed,3),
                       "cpu_steps_per_second":round(steps/max(elapsed,1e-6),3),
                       "loss_is_finite":bool(torch.isfinite(loss).item())})
    if not all(r["loss_is_finite"] for r in result):
        raise ValueError("Capacity candidate has non-finite training loss")
    return {"format":"ancestor_capacity_study_v1",
            "selected_architecture":"unchanged_128x4",
            "capacity_test_only":True,"candidates":result}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",required=True)
    ap.add_argument("--steps",type=int,default=5)
    args=ap.parse_args()
    report=study(steps=args.steps)
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
