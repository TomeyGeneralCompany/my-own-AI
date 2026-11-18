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

For Word or PDF originals, use the reviewed extraction workflow:

    ancestor prepare-document --source /path/book.docx --out data/ingest

Inspect the extracted text and record its exact SHA-256 from the manifest, then:

    ancestor approve-document --manifest data/ingest/manifest-<source-sha>.json --approved-hash <extracted-sha> --data data/corpus --id BOOK-001 --title "Canonical edition"

The loader retains the original artifact and refuses to claim the extracted
version is approved until the exact text hash is supplied. Word auto-numbering,
PDF equations, and typography must be reviewed for fidelity. The dictionary importer requires an explicit headword<TAB>definition
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


## Controller-directed course (source-authored, not generic alignment)

After importing and reviewing the sources, create an approved course JSON file
with explicit ordered stages, source IDs, owner-referenced premises, independent
exam IDs, and per-lesson initial update budgets. Every stage has its own exam
bank in a separate JSON file. The stage-gated entrypoint is:

    ancestor run-course --course /path/course.json --exams /path/exams.json --data data/corpus --tokenizer data/tokenizer.json --governance data/governance.json --out data/course-run

The controller trains the paragraph's continuation from the immutable axiom
prefix and approved source prerequisites; grades fresh generated responses;
records coverage/competence separately; and refuses to graduate a failed stage.
This is not an instruction to self-award competence or invent owner exam answers.

For interruption recovery add --resume /path/recovery-checkpoint.pt.
A ledger/checkpoint mismatch aborts rather than resetting or replaying training.
Owner-approved axiom-first derivation examples are also represented by
ancestor/reasoning.py; they require explicit approval and source references.

### Course example (illustrative fixture only)

    {
      "stages": [{
        "id": "ORDERED_MANUSCRIPTS",
        "approved_document_ids": ["BOOK-001"],
        "lessons": [{
          "id": "BOOK1-001",
          "source_id": "BOOK-001:P-000001",
          "task": "Derive the consequence from the cited principles.",
          "prerequisites": ["BOOK-001:P-000002"],
          "exam_id": "BOOK1-001-EXAM",
          "updates": 100
        }],
        "retention_exam_ids": ["EARLIER-REASONING-RETENTION"]
      }]
    }

The owner must replace illustrative IDs, questions, sources, grading thresholds
and prerequisites with a real approved curriculum. The above is a format
illustration, not an approved course plan or valid paragraph relationship.

**Do not run a purported full Ancestor education on demonstration fixtures.**


### Source-coverage gates

For the \`WEBSTER_DICTIONARY\` stage, provide the owner-verified
\`approved_entry_count\` and list **all** indexed \`WEBSTER1913:W-...\`
records as lessons. An incomplete list is rejected before training.

For \`ORDERED_MANUSCRIPTS\`, provide \`approved_document_ids\` and list
**every** numbered paragraph from each of those approved books as lessons.
The stage will reject omitted paragraphs. A source's presence in the library
does not by itself certify that its content has been learned.

The course file is not permission to alter foundational axioms, source
definitions, or neural architecture. Changes to the course/exam files are
included in the checkpoint's source identity and block silent resume.
