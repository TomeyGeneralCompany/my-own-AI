# Ancestor

Original, local-first neural intelligence with an owner-governed learning controller.

**Status: foundational implementation under review; NOT a fully trained or certified AI.**

See [the binding architecture](docs/ARCHITECTURE.md) for the owner's six requirements
and the replacement Section IX (Ancestor's intellectual constitution).

## From scratch

- Custom byte-level BPE tokenizer; no third-party tokenizer.
- Fresh decoder-only PyTorch transformer; no pretrained weights.
- Controller-owned axioms/prefixes, context packing, source selection and stage gates.
- Paragraph-indexed source library and a metadata-rejecting dictionary TSV importer.
- AdamW training, masked derivation loss, independently tracked examinations.
- Complete-state checkpoint save and actual fresh-process restoration paths.

## Install

Python 3.10+ and CPU-capable PyTorch are required.

    python -m venv .venv
    . .venv/bin/activate
    pip install -e '.[test]'
    python -m pytest -q

No package install downloads a model or intellectual source documents.

## Owner-controlled governance setup

Prepare an approved JSON file containing the **exact** owner-approved canonical
axiom strings. Do not infer or automatically generate missing axioms from examples.

    ancestor init-governance --axioms /path/axioms.json --output data/governance.json

This command refuses to replace an existing manifest. Governance checks prevent
ordinary training and controller operations from revising the initial axioms.
Owner-authorized revision is a distinct future administrative mechanism.

## Import verified source texts

    ancestor import-manuscript --data data/corpus --source /path/book.txt --id BOOK-001 --title "Canonical edition"
    ancestor import-dictionary --data data/corpus --source /path/verified-webster.tsv

The initial manuscript importer accepts verified UTF-8 text containing numbered
paragraphs; PDF/DOCX extraction should be performed and reviewed separately prior
to import. The dictionary importer requires an explicit headword<TAB>definition
export. It does NOT silently guess StarDict format or treat metadata as words.

## Build original tokenizer

    ancestor build-tokenizer --source /path/english-training.txt --source /path/book.txt --output data/tokenizer.json --vocab 8192

Training and held-out evaluation data must be separated before tokenizer fitting.

## Exercise the next-token learner

    ancestor train --corpus /path/train.txt --validation /path/held-out.txt --tokenizer data/tokenizer.json --governance data/governance.json --out data/checkpoints --updates 10 --block 128

For a checkpoint continuation, provide --resume with the exact checkpoint filename
and reuse all matching model, tokenization and training arguments.

This CLI is a **neural training acceptance tool**. It is not the stage-gated full
education pipeline and cannot grant book- or dictionary-mastery status.
Do not launch long production training until source ingestion, examination banks,
context conditioning and training/recovery integration pass their end-to-end gates.

## Important existing gaps

The current source importer and examination primitives are intentional foundations,
not substitutes for the full authorized education. In particular, the complete
source files have not been supplied and verified in this PR; manuscript equation
block indexing, word-level grammar exams, actual 1913 dictionary validation,
automated curriculum-directed optimizer scheduling, and regression/retention banks
need acceptance against those owner-approved materials. The PR must not be merged
on the assumption those missing features work.
