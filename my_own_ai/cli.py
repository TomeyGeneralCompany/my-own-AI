"""Local model training, ingestion, benchmarking and module-switch commands."""
import argparse
import json
from pathlib import Path
from .model import ModelConfig
from .train import train, load_model, benchmark
from .corpus import build_manifest
from .modules import CheckpointStore, ModuleController, default_modules

def main(argv=None):
    parser = argparse.ArgumentParser(prog="my-own-ai")
    sub = parser.add_subparsers(dest="command", required=True)
    i = sub.add_parser("ingest")
    i.add_argument("--corpus", type=Path, required=True)
    i.add_argument("--manifest", type=Path, default=Path("data/corpus-manifest.json"))
    t = sub.add_parser("train")
    t.add_argument("--corpus", type=Path, required=True)
    t.add_argument("--output", type=Path, default=Path("data/checkpoints/brain.pt"))
    t.add_argument("--steps", type=int, default=20, help="Total target steps, including resumed steps")
    t.add_argument("--context", type=int, default=64)
    t.add_argument("--embed", type=int, default=64)
    t.add_argument("--heads", type=int, default=4)
    t.add_argument("--layers", type=int, default=2)
    t.add_argument("--batch", type=int, default=4)
    t.add_argument("--lr", type=float, default=.001)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--checkpoint-every", type=int, default=10)
    t.add_argument("--resume", action="store_true")\n    t.add_argument("--schedule-steps", type=int, default=1000)\n    t.add_argument("--warmup-steps", type=int, default=10)
    b = sub.add_parser("benchmark")
    b.add_argument("--checkpoint", type=Path, required=True)
    b.add_argument("--iterations", type=int, default=10)
    b.add_argument("--tokens", type=int, default=16)
    g = sub.add_parser("generate")
    g.add_argument("--checkpoint", type=Path, required=True)
    g.add_argument("--prompt", required=True)
    g.add_argument("--tokens", type=int, default=60)
    s = sub.add_parser("switch")
    s.add_argument("--mode", choices=["conversation", "writing", "coding", "research"], required=True)
    s.add_argument("--memory", type=Path, default=Path("data/module-memory"))
    a = parser.parse_args(argv)
    if a.command == "ingest":
        m = build_manifest(a.corpus, a.manifest)
        print(json.dumps({"documents": len(m["documents"]), "fingerprint": m["fingerprint"]}, indent=2))
    elif a.command == "train":
        cfg = ModelConfig(context_size=a.context, embed_dim=a.embed, heads=a.heads, layers=a.layers)
        print(json.dumps(train(a.corpus, a.output, cfg, a.steps, a.batch, a.lr, a.seed,
                               a.resume, a.checkpoint_every, schedule_steps=a.schedule_steps,\n                               warmup_steps=a.warmup_steps), indent=2))
    elif a.command == "benchmark":
        print(json.dumps(benchmark(a.checkpoint, a.iterations, generation_tokens=a.tokens), indent=2))
    elif a.command == "generate":
        print(load_model(a.checkpoint).generate(a.prompt.encode(), a.tokens).decode("utf-8", errors="replace"))
    else:
        controller = ModuleController(CheckpointStore(a.memory), default_modules())
        controller.switch(a.mode)
        print("Active module:", controller.active.name)
        controller.close()

if __name__ == "__main__":
    main()
