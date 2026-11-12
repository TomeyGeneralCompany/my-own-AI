"""Offline command-line training and module-switch demo."""
import argparse
import json
from pathlib import Path
from .model import ModelConfig
from .train import train, load_model
from .modules import CheckpointStore, ModuleController, default_modules

def main(argv=None):
    parser = argparse.ArgumentParser(prog="my-own-ai")
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--corpus", type=Path, required=True)
    t.add_argument("--output", type=Path, default=Path("data/checkpoints/brain.pt"))
    t.add_argument("--steps", type=int, default=20)
    t.add_argument("--context", type=int, default=64)
    t.add_argument("--embed", type=int, default=64)
    t.add_argument("--heads", type=int, default=4)
    t.add_argument("--layers", type=int, default=2)
    t.add_argument("--batch", type=int, default=4)
    g = sub.add_parser("generate")
    g.add_argument("--checkpoint", type=Path, required=True)
    g.add_argument("--prompt", required=True)
    g.add_argument("--tokens", type=int, default=60)
    s = sub.add_parser("switch")
    s.add_argument("--mode", choices=["conversation", "writing", "coding", "research"], required=True)
    s.add_argument("--memory", type=Path, default=Path("data/module-memory"))
    a = parser.parse_args(argv)
    if a.command == "train":
        cfg = ModelConfig(context_size=a.context, embed_dim=a.embed, heads=a.heads, layers=a.layers)
        print(json.dumps(train(a.corpus, a.output, cfg, a.steps, a.batch), indent=2))
    elif a.command == "generate":
        print(load_model(a.checkpoint).generate(a.prompt.encode(), a.tokens).decode("utf-8", errors="replace"))
    else:
        controller = ModuleController(CheckpointStore(a.memory), default_modules())
        controller.switch(a.mode)
        print("Active module:", controller.active.name)
        controller.close()

if __name__ == "__main__":
    main()
