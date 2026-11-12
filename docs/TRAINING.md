# Offline book training and corpus preparation

The neural weights are randomly initialized; this is a research-scale language model, not a trained commercial assistant. No remote inference, cloud account, or pretrained weights.

## Installation

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Books and privacy

Put author-owned TXT, Markdown, DOCX, or text-based PDF files in an **untracked private directory** (for example `corpus/private/`). Do not upload the manuscripts to this public GitHub repo. Scanned PDFs need an OCR step before ingestion, which is not bundled.

```sh
my-own-ai ingest --corpus corpus/private --manifest data/manifest.json
my-own-ai train --corpus corpus/private --output data/checkpoints/brain.pt --steps 100 --context 64 --checkpoint-every 10
my-own-ai train --corpus corpus/private --output data/checkpoints/brain.pt --steps 200 --context 64 --resume
my-own-ai benchmark --checkpoint data/checkpoints/brain.pt
```

Resume requires identical corpus content, seed, model dimensions, batch size, learning rate, warmup and schedule horizon. An immutable linear-warmup/cosine-decay learning-rate schedule is configured by `--warmup-steps` and `--schedule-steps`; its horizon does not change when a run is resumed with a larger `--steps` target. The target `--steps` is the **total** number of optimizer steps. Checkpoints contain model weights, optimizer state, sampling RNG, corpus fingerprint, and holdout metrics. Writes are atomic at the file level, so an interrupted save preserves the preceding checkpoint.

## Validation

Two or more qualifying documents are separated at the document level, so a book cannot appear in both train and validation sets. A single qualifying document uses a byte split where practical; the reported `split_strategy` indicates this weaker arrangement. Tiny corpora may have no validation segment. `validation_loss` and `validation_perplexity` are sampled held-out indicators, not measures of reasoning ability.

CPU throughput and timing are measured for local comparison; they are not promises of performance on other hardware.

## Limits and future work

No substantive natural-language reasoning, instruction tuning, curriculum learning, model scaling, or cloud sync is implemented yet. Ingestion produces a manifest and in-memory text corpus, not an indexed retrieval database. The learning-rate horizon must be selected independently of total run length to preserve exact resume semantics.
