# Progressive Reading and Language Acquisition Controller

## Goal

Teach the *same* original-weight English Core by progressively reading every eligible token in approved full-length books, without feeding entire manuscripts into one neural context window. The controller handles token-aware segmentation, overlap, ordered coverage, multi-pass training, checkpointed progress and validation. The controller does **not** simulate knowledge or provide canned answers.

**Important:** exhaustive reading is only a measurable exposure mechanism. Reading pages is not yet evidence that the model can understand English or generalize to unfamiliar questions. The PR #14 frozen English gauntlet remains the required competence test.

## Book continuity

PR #15 prepares source-approved books in separate train/validation/test folders. The controller reconstructs sections of each original book in section order **inside its own partition**, then tokenizes the continuous text. It never joins training with validation or test text. With only one book, section-based holdouts remain weaker than an unseen-book test.

Each neural window contains at most the configured context length in tokens. Subsequent windows carry the configured overlap from the preceding window for context, but previously learned targets in that overlap are masked from loss. Thus all eligible tokens are supervised **exactly once per complete reading pass**, with no overweighting of overlaps.

The reading order is chronological within a book. Between passes, the order of independent books is shuffled deterministically from the seed and pass number. The model's weights and optimizer persist throughout every pass.

## Train or resume a book-reading session

First prepare the approved books and tokenizer as described in docs/BOOK_LANGUAGE_TRAINING.md, using local private source files. Then:

    my-own-ai read-progressively --tokenizer data/book-tokenizer.json --training data/book-prepared/train --validation data/book-prepared/validation --output data/checkpoints/read-book.pt --passes 5 --overlap 64 --embed 128 --heads 4 --layers 4 --context 512 --batch 1 --threads 4 --checkpoint-every 25

To extend the *same* reading run from five completed passes to ten, **do not recreate the tokenizer**:

    my-own-ai read-progressively --tokenizer data/book-tokenizer.json --training data/book-prepared/train --validation data/book-prepared/validation --output data/checkpoints/read-book.pt --passes 10 --overlap 64 --embed 128 --heads 4 --layers 4 --context 512 --batch 1 --threads 4 --checkpoint-every 25 --resume

This resumes the exact cursor and optimizer state after verifying corpus fingerprints, tokenizer, architecture and training settings. The requested passes are the *cumulative* number of full readings, not additional passes. Checkpoints are saved atomically after the configured update interval and after every complete pass. A separate *.best.pt checkpoint stores the lowest validation loss model. If a run is interrupted mid-pass, it resumes the remaining windows of that pass.

The optional `--patience 3` stops after three **complete passes** without a validation improvement. This is a validation plateau, not a proof of English proficiency. More than 512 context tokens is allowed if the model is created with the matching larger positional embeddings and CPU memory supports it; a checkpoint with a smaller fixed context cannot silently expand its embedding dimensions.

## Integrate with PR #15 one-command book training

The existing private book-training command now uses progressive full-book reading by default:

    python scripts/train_approved_books.py --ledger data/private-books.json --output data/book-generation-v2 --reading-passes 5 --reading-overlap 64 --reading-patience 0 --context 512 --embed 128 --layers 4 --heads 4 --instruction-steps 600

It prepares the private corpus, trains the tokenizer **on train text only**, reads every training token for each pass, and then initializes the existing source-derived instruction-tuning stage from the best book-reading checkpoint. The same model weights flow into the original Core's question-answer engine.

For controlled comparison, `--reading-passes 0` invokes the older random-sampling pretraining pathway and `--pretrain-steps` controls that legacy pathway. The old PR #14 checkpoint format and instructions remain available. Because the whole one-command pipeline creates a fresh run directory, use `my-own-ai read-progressively --resume` for an interrupted standalone reading run rather than re-running the entire book-preparation command.

## What to measure

Read the JSON report and the saved best and latest checkpoints. Each complete pass reports supervised token coverage (1.0), unique supervised tokens, window count, validation loss, and best-pass selection. Resuming rejects changed corpus contents, changed tokenizer, overlap or model configuration.

After reading, the existing instruction trainer practices source-derived English continuations, optionally alongside owner-reviewed Q&A targets. Then ask unfamiliar questions with:

    my-own-ai ask-english --checkpoint data/checkpoints/read-book.best.pt --tokenizer data/book-tokenizer.json --question "Why might central approval delay decisions?"

That example checkpoint is a **prose-only** model, so it may not yet follow user instructions. For conversational testing, use the downstream instruction-trained checkpoint generated by the private book-training script.

Evaluate independently on the frozen 200-question English suite:

    my-own-ai evaluate-english --checkpoint data/book-generation-v2/english-book-instruct.best.pt --tokenizer data/book-generation-v2/tokenizer.json --cases examples/english/gauntlet.json --output data/book-generation-v2/english-evaluation.json

No training or held-out reference answers are supplied in inference prompts. A successful CI result means the controller works, **not** that the language milestones have been met.

## Bounds and privacy

All manuscripts, training sections, tokenizers, checkpoints, and raw responses remain in ignored local paths. Public GitHub CI uses only synthetic test books. Do not upload manuscripts or proprietary checkpoints to public Actions artifacts. The controller retains no automatic long-range semantic summary of the book; it advances through overlapping windows while the model's parameters persist. Research-style retrieval memory is a separate capability and cannot be claimed as a byproduct of overlap.
