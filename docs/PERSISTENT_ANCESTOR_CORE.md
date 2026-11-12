# PR #18 — Persistent published-book Ancestor Core

## Source of truth

The Core is the **original-weight neural model** whose immutable checkpoint is selected by the `ancestor-active` release pointer. There is **no training** in the Ask workflow. The previous `github_neural_smoke.py` and `github_english_trial.py` remain only as *independent synthetic diagnostics*; they do not supply answers from the active Core.

### Published corpus

The author's approved English book text is versioned under `published_books/`, with each edition's exact SHA-256 in `published_books/manifest.json`. The first source is the author-supplied **Revised Second Edition** of *On the Communal Nature of Commerce*. It is not fetched from an invented publisher URL. This repository source is intentionally public because the author explicitly approved training on already published work.

Add each future work as a new distinct, approved text file and manifest entry. File mutation, unapproved books, duplicate editions, path traversal and mismatched hashes fail validation. The corpus preparation creates chronological, disjoint train/validation/test sections for one/two books and whole-book holdouts with three or more independent books. No held-out passages are used for optimization.

### Generation and compatible continuation

A generation identifier is `g0001`, `g0002`, etc. Each generation records a source manifest digest, fixed tokenizer fingerprint, model dimensions, learning targets, progress cursor and checkpoint lineage. The original PR #16 reader uses ordered overlapping token windows and masks overlap from loss: each eligible training target is processed exactly once per complete pass.

Within a generation, each invocation continues from the saved reading checkpoint and optimizer, or from the saved assistant checkpoint and optimizer if the reading stage is finished. A new generation can initialize from the prior **active candidate**. It retains the exact tokenizer/architecture, so new books can be added without corrupting existing token embeddings. An incompatible configuration requires a new, explicitly handled model lineage instead of silent weight transfer.

### Durable GitHub Releases

Actions runners are ephemeral. The training workflow saves a ZIP integrity bundle after each bounded session under an **immutable per-run asset name** in release `ancestor-progress-g0001`. Its manifest contains SHA-256 digests for each serialized member. The next run downloads the latest snapshot, verifies every member, restores the exact optimizer, weights and reading cursor, then continues.

The progress release itself is a container for multiple immutable versioned snapshots. Completed generations get a separate release `ancestor-g0001-candidate` with a single immutable `core-generation.zip`. The candidate release is never overwritten to hide regressions. Two jobs for the same generation cannot run concurrently. An interrupted job may lose only the unpersisted work of that job; the prior durable release snapshot remains available.

Each package contains the tokenizer and both latest and best checkpoints for the completed phases. The full corpus text is **not embedded in checkpoint releases**, because it is already verified in the repository.

## Operating the Core on GitHub

After the owner merges PR #18, use GitHub Actions workflows in this sequence.

1. **Train Ancestor Core — published books**. Input generation `g0001`, cumulative reading passes `5`, instruction updates `600`, and a bounded reading update count (default 40). Repeat this workflow with the **same settings** until the job summary reports `candidate`. Each invocation restores the durable latest checkpoint rather than restarting.
2. **Evaluate Ancestor Core — immutable candidate**. Input `g0001` and the full 200-question English gauntlet. The result, raw tokens, EOS and strict matches are attached to the immutable candidate's release. The evaluation does **not** activate the model automatically.
3. **Activate or roll back Ancestor Core**. Choose `g0001` explicitly. Since the earliest Core has not passed English competence gates, use the deliberate `experimental_bootstrap=true` option to make the actual evolving model available for questions **without implying competence**. Qualified promotion instead requires full automatic thresholds and separately recorded human review (70 open-ended items, score rules in `my_own_ai/promotion_gate.py`). Later generations can be selected or rolled back by re-running this workflow with a prior candidate ID; no weight changes occur.
4. **Ask the Core — persistent published-book neural model**. Enter the question. The workflow reads the active release pointer, validates the immutable model ZIP SHA-256, loads the checkpoint and tokenizer, checks both fingerprints and generates raw tokens. It takes **zero training steps** and reports exactly which checkpoint produced the answer.

The model's active pointer is the JSON body of release `ancestor-active`. Its fields include generation, candidate release tag, ZIP SHA-256, checkpoint SHA-256, tokenizer fingerprint, corpus hash and experimental/evaluated status. Updating this pointer changes the selected checkpoint, not the weights; restoring the previous pointer implements rollback.

### Human review contract

Automatic completion is not English fluency. Qualified activation requires at least 80% strict automatic matches across 130 mechanically scored cases; at least 70% in every automatic category; at least 7 correct conditional pairs; completed review for all 70 open-ended cases with 1–5 grammar, coherence and relevance ratings; mean at least 4/5 and at least 70% passing in every open-ended category. Reviewed scores must be bound to the exact candidate checkpoint and frozen suite through SHA-256 hashes. Previously correct automatic answers must not regress when prior evaluation exists.

Human-reviewed case JSON must be committed as `reviews/g0001.json` via a separate owner-approved change and bind the evaluation file's exact SHA-256. This PR does **not** invent or pre-fill such scores. The bootstrap option explicitly labels an unsuccessful model as `experimental` instead of bypassing the qualified promotion gate under a false success designation.

### Tests and safe limits

Foundation CI checks the book hash, train/validation/test isolation, checkpoint resume fidelity, package integrity and rejected corruption, inference being read-only, and workflow routing. The PR-specific end-to-end smoke test additionally reads the actual approved public book using a small original model, publishes an immutable model candidate as a GitHub release, and downloads it into a **separate runner job** before inference. This tests durable cross-runner transfer, not English fluency. The normal first generation uses the larger configurable CPU model.

GitHub Actions release operations require the repository's workflow token to have Contents write permission; a repository-level restriction can prevent release publication, in which case the job must fail explicitly rather than claim persistence. No external inference service or pretrained weights are used.

### What constitutes progress

Compare validation loss, reading-pass coverage, repeated-token degeneration, natural EOS and actual answers to unfamiliar questions across *immutable successive generations of the same model*. Prior synthetic diagnostic scores are not silently compared as if they came from the same trained checkpoint. The Core can still produce nonsense while the training and release architecture works: that is a measurable learning issue, not an excuse to reset the checkpoint.

## PR #19 — Chapterless paragraph controller and autonomous campaign

**The author's treatise has no chapters.** The authoritative source is the complete numbered sequence `[0]`–`[536]`, 537 numbered paragraphs. The PR #19 parser recognizes only consecutive paragraph headers at the beginnings of lines, records source hashes and paragraph numbers, and preserves **every character from each header to the next header**. An equation, equation number, display-math layout, and accompanying explanation remain part of their numbered paragraph. Frontmatter is indexed separately. No words or equation lines are dropped to enforce a word limit.

The reading corpus is split on **whole paragraphs**, never inside an equation. Long paragraphs are processed by overlapping *token windows within that same paragraph*, with overlap tokens masked from repeated loss. The controller reads paragraphs **in author order** within each pass. It stores the most recent paragraph source, next token window, current pass, optimizer state, and neural weights at each checkpoint.

### Begin or resume one autonomous campaign

On the GitHub Actions page, launch **Train Ancestor Core — autonomous numbered-paragraph reading** once with `generation=g0001`, `reading_passes=5`, `instruction_steps=600`, `session_seconds=2100`, and `auto_continue=true`. There is **no artificial 40-update limit**. The controller trains for the available CPU session budget, saves the verified checkpoint to a new immutable asset in `ancestor-progress-g0001`, then dispatches the next identical workflow on `main` if reading or instruction work remains. Each next job restores the newest durable snapshot and advances from the exact saved training cursor. GitHub workflow concurrency prevents two jobs from advancing the same generation simultaneously.

The next job is queued **only after** the previous checkpoint upload succeeded. If the GitHub token lacks Actions write permission, dispatch fails and the published checkpoint remains available for a manual retry. A failed job does not silently mark any reading as complete. The campaign stops dispatching when the immutable candidate release is created. Stopping a queued campaign or setting `auto_continue=false` prevents further automatic chaining, while the last published checkpoint remains intact.

### Migration from PR #18's existing g0001 snapshots

The old g0001 snapshots used 512-word sections. New paragraph-based training changes the source/validation window identities, so **the old cursor must never be reused as if it pointed to a paragraph**. A strictly limited migration is supported for the known published commerce manifest and its verified predecessor hash. It keeps the current model weights, AdamW moments, tokenizer, RNG state, and completed training-step count; it resets only the *unfinished* reading pass's cursor to the first new paragraph, recomputes validation on the new split, records the previous corpus hashes in the reading history, and preserves the legacy snapshot in the old immutable GitHub Release asset. Re-reading that incomplete pass avoids missing previously unseen paragraph targets.

After migration, ordinary sessions resume the new exact paragraph window index. A changed book edition, unrelated manuscript, tokenizer, model configuration, or training hyperparameter remains an error and is **not** automatically migrated.

### Operational boundary

The controller does not itself interpret mathematical semantics; it supplies numbered prose and equations together as contiguous neural training context. The model still must acquire English and reasoning capability through weight updates and independent held-out evaluation. Auto-resumption solves the manual restart bottleneck; it does not by itself establish fluent English.

## Seven-work authored library (PR #22)

The approved library now contains **seven independently hash-pinned authored works**. The original *On the Communal Nature of Commerce* remains a consecutively numbered 537-paragraph treatise with its existing lossless numbered-paragraph index. Six new sources use chapter, heading, paragraph and equation-aware `structured_units` indexing instead of pretending they have the same paragraph numbering. The 635-page Calvin commentary is represented by three verified source-text transport parts; its three parts are reassembled into one book before indexing. The complete approved source texts are committed, while **the original PDF/DOCX binaries are not**.

All books contribute training, validation, and test units. Repeated PDF running headers and duplicated passages are excluded from the derived partitions to prevent training/evaluation leakage, while the full stored source texts remain untouched. Each independent lesson is traceable through `train-provenance.json`, `validation-provenance.json`, or `test-provenance.json` to a specific source unit.

The existing 200-question frozen English benchmark is **not modified**. A separate seven-work held-out lab adds two source-grounded questions per book (14 total) using a real descendant model. These are evidence of functional training and evaluation wiring until competence scores establish otherwise, and their automatic score does not activate any candidate.

### Import another authored book

Run on a local checkout that has the book available as PDF, DOCX, TXT or Markdown:

```bash
python scripts/import_authored_book.py --input /path/to/work.docx \
  --repository . --id new-work --title "New Work" --dry-run
python scripts/import_authored_book.py --input /path/to/work.docx \
  --repository . --id new-work --title "New Work" --approve
```

The importer refuses to replace a previously approved edition, retains text/OMML math tokens where recoverable, builds source partitions without losing bytes, writes sha256-pinned transport files and updates the author-approved manifest. A scanned PDF with no usable text is rejected. PDF images, some displayed mathematical layouts and OMML matrices still require a visual comparison to their original; reconstructed textual notation is not a mathematical-proof validator.

### Generation ancestry

The expanded library targets **g0002**, inheriting from the immutable `ancestor-g0001-candidate`. It does not rewrite the legacy g0001 corpus, weights or tokenizer. Merge the reasoning-curriculum PR #21 before PR #22. After both are merged, launch `ancestor-train.yml` with `generation=g0002`, `parent_generation=g0001`, `curriculum=reasoning`, `reading_passes=5`, `instruction_steps=600`, `auto_continue=true`. First inspect the seven-work CI results and adjust budgets if needed. The new corpus changes the generation source SHA and must not be used to resume an earlier unrelated g0002 checkpoint.
