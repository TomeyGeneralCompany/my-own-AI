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
