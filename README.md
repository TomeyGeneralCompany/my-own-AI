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
my-own-ai demo
```

TXT, Markdown, DOCX and text-based PDF are accepted for corpus ingestion. A few iterations on sample text produce meaningless output: the current neural network is a mathematical research foundation, not a useful assistant. Book data should remain in ignored `corpus/private/`, **never committed to this public repo**.

See the [training guide](docs/TRAINING.md) and the [Linux session and module extension guide](docs/ARCHITECTURE.md). A session keeps one CPU model loaded and switches between Conversation, Writing, Coding and Research. The controller saves project progress, restricts tools to the active module and checks proposals against module-specific criteria. Only the user merges pull requests.
