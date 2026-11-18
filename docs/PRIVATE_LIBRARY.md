# Importing the owner's eleven manuscripts for Ancestor

This change never publishes a manuscript. Keep the private ZIP on the owner's
local computer; only the importer and synthetic tests belong to GitHub.

## Data artifact

The original archive is named \`ancestor_private_training_library.zip\`.
It contains 11 unmodified original files, their verified extracted text,
source SHA-256 fingerprints, and 3,826 individually addressable units. Existing
DOCX Word paragraphs have stable IDs. PDF units are explicitly page-scoped,
not falsely presented as exact numbered paragraphs.

The Election treatise's 1,139 bracketed paragraph numbers are sequential and
not duplicated. The Latin \`Inane Electromagneticum\` contains two corresponding
numbered sequences. No change to either manuscript is needed.

## Install into the local model's accessible corpus

In the Ancestor repository, after installing the Python package:

    ancestor import-private-library \
        --bundle /path/ancestor_private_training_library.zip \
        --data ./data/owner-library

This imports 11 owner-authored works to a persistent local SQLite-backed
source registry and preserves the original/extracted files privately.

To generate text for the next-token learner and byte-BPE trainer:

    ancestor export-library-text \
        --data ./data/owner-library \
        --output ./data/owner-library-full-text.txt

This output includes **all eleven sources**. It is a learning-coverage corpus,
NOT an independent validation set. Independent held-out English passages and
owner-authored reasoning examination banks remain required to measure competence.
Do not pretend that recycling excerpts from trained books is independent
generalization testing.

The corpus's \`list_paragraphs\` returns Word paragraph units and PDF page units
in order. The controller's context builder splits oversized target text without
discarding its source identity. PDF page-level indexes can later be refined into
verified paragraph units, without discarding or rewriting the original PDF or
extracted text.

## What this action does and does not do

It makes the eleven uploaded manuscripts directly addressable to the installed
Ancestor learning controller, and produces a training text for tokenizer/optimizer
work. It does **not** initialize weights, change the owner's axioms, start
long production training, or declare English mastery. That requires a matching
model/tokenizer and an approved independent evaluation set.

No new lexicographic source is smuggled into this bundle: Webster 1913 is a
separate source and can be imported through the dictionary system.
