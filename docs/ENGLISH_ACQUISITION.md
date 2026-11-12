# PR #14 — Original English acquisition

## Scope

The Core's English pathway is a randomly initialized local Transformer; it
does not download trained weights, use hosted inference or substitute scripted
answers. The original byte-model controller and research pipeline remain
available separately. The new model uses a different, versioned vocabulary
and deliberately rejects cross-format loading. The target is a single
language-capable Core that can subsequently learn the author's relational
curriculum through the same neural weights.

## Original tokenizer and approved English corpus

Create document-disjoint, near-deduplicated train/validation/test partitions
from a directory of approved source files. At least three independent
documents are required, and no existing destination is overwritten.

```bash
my-own-ai prepare-english-corpus --source corpus/approved-generation-1 --output data/english-corpus
my-own-ai train-tokenizer --corpus data/english-corpus/train --output data/english/tokenizer.json --vocab 8192
```

The tokenizer learns an original byte-fallback BPE vocabulary from **training
text only**. It can encode and decode arbitrary UTF-8, including unseen words.
The 8,192 setting is a maximum target, not a guarantee that a small corpus has
enough repeated pairs to reach it. All checkpoint weights bind the vocabulary
and merge rules by SHA-256. There are explicit BOS, EOS, user, assistant and
padding IDs. The model rejects an incompatible tokenizer.

Your own manuscripts should remain inside ignored local directories.
The existing `register-work`, `decide-work` and `stage-approved-corpus`
commands establish approval before corpus construction. Public CI uses only
the explicitly original, synthetic demonstration prose.

## Train from new weights

```bash
my-own-ai train-english \
  --phase pretrain --tokenizer data/english/tokenizer.json \
  --training data/english-corpus/train \
  --validation data/english-corpus/validation \
  --output data/checkpoints/english-pretrain.pt \
  --embed 128 --heads 4 --layers 4 --context 512 \
  --batch 2 --steps 1000 --checkpoint-every 50 --threads 4
```

This example is an **engineering configuration**, not a promise of fluent
English or a prescribed optimal training size. CPU use and source provenance
must be measured. Data are processed under causal next-token prediction;
validation text is excluded from weight updates.

The checkpoint stores tokenizer configuration and digest, weights,
optimizer, PyTorch and sampler RNG states, phase and corpus digests,
completed steps and validation loss. Every run keeps a latest checkpoint
for exact resumption and a separate `.best.pt` checkpoint selected by the
lowest measured validation loss. Use the best checkpoint for inference or
new-generation initialization; do not assume the latest update is strongest. An interrupted run may use `--resume`
with unchanged configuration and source files. `--steps` means cumulative
total steps. A new approved work begins a new generation, passing
`--init-checkpoint` to initialize from previous weights with a fresh optimizer
and a **different output**; the original model stays intact.

## Teach answering instead of merely continuing prose

```bash
my-own-ai train-english \
  --phase instruction --tokenizer data/english/tokenizer.json \
  --training examples/english/dialogue/train.json \
  --validation examples/english/dialogue/validation.json \
  --init-checkpoint data/checkpoints/english-pretrain.pt \
  --output data/checkpoints/english-assistant.pt \
  --embed 128 --heads 4 --layers 4 --context 512 \
  --batch 2 --steps 1000 --checkpoint-every 50 --threads 4
```

Dialogue examples have exactly `id`, `question` and `answer`. The shared
format consists of BOS, USER, question tokens, EOS, ASSISTANT, answer tokens,
EOS. Only response tokens and the final EOS contribute to loss. The inference
format is identical and does not contain reference answers.

```bash
my-own-ai ask-english --checkpoint data/checkpoints/english-assistant.pt \
  --tokenizer data/english/tokenizer.json \
  --question "Why might a central approval step slow down a company?" \
  --tokens 120 --temperature 0
```

The output includes raw generated and prompt token IDs, UTF-8 validity,
termination reason, generation settings, model and tokenizer identity.
Deterministic greedy decoding and seeded top-p sampling are available.

## Browser questions without the laptop

Open GitHub Actions, select **Ancestor Core English acquisition**, click
**Run workflow**, enter a question and choose the full gauntlet. The workflow
trains a **fresh synthetic demonstration model** and displays its actual raw
response in the run summary, also publishing the tokenizer, two checkpoints,
raw responses and metrics as artifacts. This does not constitute a persistent
conversation or draw from your private manuscript corpus.

## Frozen gauntlet and regression protection

`examples/english/gauntlet.json` contains 200 distinct tasks: 40 syntax,
40 reading, 30 paraphrase, 30 direct question answering, 20 multi-sentence
coherence, 20 matched yes/no premise switches, and 20 conversational
follow-ups. This is a synthetic, partially templated **software test corpus**,
not a naturalistic human benchmark. Its answers never enter inference
prompts. Exact scoring is intentionally conservative; valid alternative
phrasings require human judgment. The 70 open-ended tasks remain explicitly
unscored until human review. Results also measure natural EOS, UTF-8,
3-gram repetition and counterfactual-pair accuracy.

```bash
my-own-ai evaluate-english --checkpoint data/checkpoints/english-assistant.pt \
  --tokenizer data/english/tokenizer.json \
  --cases examples/english/gauntlet.json \
  --output data/english/report.json --threads 4
```

A release must score at least 80% on mechanically verifiable language tasks,
at least 70% in every such category, and pass independent human review for
grammar, coherence and relevance. Previously correct items must not regress.
It must be reviewed on genuinely held-out questions beyond synthetic
templates before broad English proficiency can be claimed.

`gate-english` compares previous and candidate language checkpoints on the
identical frozen cases and refuses promotion without independent review.

First produce an initial comparison report. Without manual review, this command
intentionally exits nonzero but still writes the measured answers.

```bash
my-own-ai gate-english --previous data/checkpoints/english-previous.pt \
  --candidate data/checkpoints/english-candidate.pt \
  --tokenizer data/english/tokenizer.json \
  --cases examples/english/gauntlet.json \
  --report data/english/comparison.json

my-own-ai prepare-english-review \
  --comparison data/english/comparison.json \
  --template data/english/review.json \
  --worksheet data/english/review.md
```

The worksheet shows the previous and candidate responses side by side for
all 70 open-ended cases. The JSON template starts with all scores at **zero**
and approval **false**. An independent reviewer must enter integer 1–5
grammar, coherence and relevance grades for both versions of every case
and explicitly approve after completing the assessment. The gate requires
average 4/5, 70% successful responses in each open-ended category, and
preservation of previously strong manual responses.

```bash
my-own-ai gate-english --previous data/checkpoints/english-previous.pt \
  --candidate data/checkpoints/english-candidate.pt \
  --tokenizer data/english/tokenizer.json \
  --cases examples/english/gauntlet.json \
  --review data/english/review.json \
  --report data/english/final-comparison.json \
  --promote-to data/checkpoints/english-approved.pt
```

Promotion additionally requires at least 80% exact automatic performance
overall and 70% in each objectively scored category, including at least
seven of ten correctly answered yes/no premise-switch pairs.
 The
review must be bound to both candidate and evaluation-set SHA-256 hashes.

## Original relational reasoning survives

`prepare-english-relational` converts already approved PR #10 worked cases
into instruction supervision for the same language-capable Transformer.
The input comprises question and premises; the target comprises identity,
hierarchy, channels, states, disturbance, feedback and conclusion. No worked
conclusion is supplied in evaluation prompts.

```bash
my-own-ai prepare-english-relational \
  --training examples/relational/train.json \
  --validation examples/relational/validation.json \
  --held-out examples/relational/held.json \
  --output data/english-relational
```

This curriculum may need 2,048 or more tokens of model context, which must
be selected **at the beginning** of the checkpoint lineage. A model created
with 512 positional embeddings cannot silently inherit 2,048-position
weights. Use the same config throughout or train a new compatible model.

The original controller, evidence checks, Research module and other modules
are not removed. The relational model's transfer to an unseen discipline
must be tested with `evaluate-english-relational` before promotion.

## First actual hosted result and status

Initial smoke: tokenizer built successfully; pretraining validation loss
5.4137, dialogue-tuning validation loss 4.6052; 0/14 automatic questions
correct, and the model responded "A ce." after instruction training.
A later 1,200-step run exactly reproduced one *training* answer (about a
sensor), but scored 0/130 on unseen automatically scored cases. Its
validation loss increased to 6.1580, demonstrating overfitting and
motivating best-checkpoint preservation. That is evidence of memorization,
not generalization.

**That is an infrastructure success, not English fluency.** A tiny model
trained on approximately 20 paragraphs and 30 short dialogue examples
cannot establish broad English. Meaningful English acquisition still
requires a sufficiently large approved language corpus, longer training,
capacity benchmarks, and an independently reviewed gauntlet result.

**Privacy:** GitHub Actions workflow inputs, summaries, logs and artifacts in
a public repository may be publicly accessible. Do not enter private company
information or unpublished manuscript text into the browser question field.
