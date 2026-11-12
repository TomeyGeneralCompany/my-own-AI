# my-own-AI

## Original English acquisition (PR #14, experimental)

The Core now has a separate versioned original subword tokenizer, CPU-only
English pretraining and question-answer training, best/latest checkpoint
selection, and a 200-question diagnostic gauntlet. **The demonstration
checkpoint is not a fluent conversational assistant**: the first full
synthetic test scored 0/130 automatically assessed answers. A longer run
memorized a training question but still failed on unseen questions.

[English training and validation guide](docs/ENGLISH_ACQUISITION.md)

To ask it a question without the laptop, use
[Ancestor Core English acquisition](https://github.com/TomeyGeneralCompany/my-own-AI/actions/workflows/english-acquisition.yml)
under GitHub Actions and choose **Run workflow** after PR #14 is merged.
The actual neural answer and diagnostics appear in the run summary. Do not
enter confidential text in public Actions questions or artifacts.


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

## Train on approved private books

For local original-weight English learning from your approved DOCX, PDF, TXT or Markdown manuscripts, see [Private book training](docs/BOOK_LANGUAGE_TRAINING.md). The book preparation splits source material into held-out passages, trains the tokenizer only on training text, then supports English prose pretraining followed by source-faithful continuation and optional owner-reviewed Q&A supervision. Private manuscripts are **not** committed to GitHub or shipped in public Actions artifacts; public CI uses synthetic books only. Book-based training is a learning experiment, not yet a demonstrated solution to unseen-question generalization.
