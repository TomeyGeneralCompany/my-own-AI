# my-own-AI

Original-weight, CPU-first modular AI foundation. The neural model starts from random parameters and trains on a user-supplied local corpus. No downloaded neural weights, hosted inference, telemetry, or paid AI service are needed.

## Install and try

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
my-own-ai ingest --corpus examples
my-own-ai train --corpus examples --steps 20 --context 64 --checkpoint-every 10
my-own-ai train --corpus examples --steps 40 --context 64 --resume
my-own-ai benchmark --checkpoint data/checkpoints/brain.pt
my-own-ai switch --mode writing
```

TXT, Markdown, DOCX and text-based PDF are accepted for corpus ingestion. A few iterations on sample text produce meaningless output: the current neural network is a mathematical research foundation, not a useful assistant. Book data should remain in ignored `corpus/private/`, **never committed to this public repo**.

See [training guide](docs/TRAINING.md) and [architecture](docs/ARCHITECTURE.md). The module socket and controller are still lifecycle stubs, not functioning coding/writing assistants. Only the user merges pull requests.
