# my-own-AI

Local-first original neural-network research project with a deterministic exclusive module controller.

The neural model begins from **randomly initialized weights**: no pretrained weights, paid inference API, cloud services, or telemetry. PyTorch is a numerical library, not a hosted neural model. Initial CPU target: Linux, i7-8650U, 16 GB RAM.

## Try locally

Python 3.10+ required. Installing packages may need an internet connection once; the application and training are offline.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
my-own-ai switch --mode writing
my-own-ai train --corpus examples --output data/checkpoints/demo.pt --steps 20
my-own-ai generate --checkpoint data/checkpoints/demo.pt --prompt "This" --tokens 40
```

A few steps on the tiny sample **will not produce a capable conversational AI**. This validates the math, data path, training, and checkpoint reload. Supply your own UTF-8 text files through --corpus; do not put private books into public Git history.

## Architecture

- **Brain**: original byte-level causal Transformer; weights created from scratch.
- **Training**: explicit local corpus, CPU optimizer, atomically stored model checkpoint.
- **Controller**: exactly one active module; snapshot before unload; rollback on activation failure.
- **Modules**: conversation, writing, coding, research — currently lifecycle stubs, not implemented skill engines.
- **Memory**: local atomic JSON checkpoints; optional external archive synchronization is a later feature.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for constraints and upcoming milestones. There is no unattended merge or release system.
