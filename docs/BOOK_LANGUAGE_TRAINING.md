# Approved-book English learning (PR #15)

## Purpose and limits

PR #14 reduced language-training loss but scored 0/130 on unfamiliar automatic questions. This implementation gives the original-weight English Core sustained exposure to the author's approved prose and separate author-reviewed answer supervision. Learning to continue an English paragraph is different from knowing how to answer a question. No result is called fluency until unseen-question tests pass.

## Privacy and approval

Books supplied to ChatGPT are not automatically available on the Zorin laptop or GitHub Actions. Copy the original DOCX, readable PDF, TXT or Markdown books to the local ignored directory corpus/private/. Never commit private manuscripts, raw passages, training artifacts, or conversation prompts to the public GitHub repository.

Register one or more books in the existing local owner-approval ledger:

    my-own-ai register-work --ledger data/private-books.json --id commerce_manuscript --kind authored_work --source corpus/private/commerce.docx --origin "Original authored work"
    my-own-ai decide-work --ledger data/private-books.json --id commerce_manuscript --decision approved

Every source is verified against its approved SHA-256 before loading. If a manuscript changes, re-register a fresh version and approve it separately. New books should have distinct IDs and non-duplicate contents.

## One-command local training

    python scripts/train_approved_books.py --ledger data/private-books.json --output data/book-language-v1 --section-words 512 --vocab 2048 --embed 128 --layers 4 --heads 4 --context 512 --batch 2 --threads 4 --pretrain-steps 1000 --instruction-steps 600 --question "Why can excessive central approval slow a company?"

This creates an original tokenizer from training text alone, an original-weight English prose model, a source-based language-continuation curriculum, a further instruction-trained checkpoint, and a local report with the genuine neural response.

Prepared sources are divided into training, validation, and untouched test material. With at least three substantial independent books, entire books are reserved for evaluation (document_disjoint). With one or two books, contiguous sections are held out (section_disjoint_same_book), a weaker test that does not establish cross-domain generalization. Identical sections and repeated continuation prompts are prevented from crossing evaluation boundaries.

No raw book text, tokenizer, private report, or model weights are uploaded automatically. The output must be a new local directory.

## Reviewed question-answer training

Next-sentence continuation helps the model practice English but does not teach semantic explanation. Optionally include independently reviewed training and validation JSON files, each containing objects with exactly id, question and answer. These must use questions and answers reviewed against approved sources, with no overlap across partitions.

    python scripts/train_approved_books.py --ledger data/private-books.json --output data/book-language-v2 --qa-training corpus/private/reviewed-train.json --qa-validation corpus/private/reviewed-validation.json --pretrain-steps 1000 --instruction-steps 600

The script mixes those approved question-answer targets with book continuations but does not silently generate unsupported theological interpretations or pretend its synthetic prompts are general reasoning.

## Inspect before training

    my-own-ai prepare-approved-books --ledger data/private-books.json --output data/book-prepared --section-words 512
    my-own-ai prepare-book-exercises --prepared data/book-prepared --output data/book-exercises --limit 250

Use new output paths on subsequent runs. The preparation manifest records source digests, extraction digests, split policy and section counts without copying source passages into the manifest.

## Evaluation and promotion

The local report includes held-out next-sentence exact matches, validation losses, checkpoint lineage and the actual response to the chosen question. This is a private experiment, not evidence that the 0/130 general-language result has been fixed. Promotion still requires running the independent English gauntlet and manual review on genuinely unfamiliar questions.

GitHub CI uses synthetic books only, verifies split boundaries and real training integration, and intentionally never has access to the private manuscript.
