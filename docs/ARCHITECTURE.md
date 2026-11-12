# Local modular runtime

A session loads one original-weight CPU Transformer and retains it for the life of the session. The controller owns the active module, tools, memory and reasoning transitions. Modules implement the procedures and acceptance checks for their operations. The core uses the Python standard library, the existing local PyTorch model and the repository's existing optional book parsers.

## Run on Linux

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
python -m compileall -q my_own_ai tests
CUDA_VISIBLE_DEVICES="" python -m pytest -q
CUDA_VISIBLE_DEVICES="" my-own-ai demo
```

Install dependencies once, or install them from a local wheelhouse using pip's `--no-index --find-links` options. Training, sessions, memory and the demo then run offline. The demo uses the original Transformer for formulation and scripted source proposals to induce and repair a compiler failure reproducibly. It saves the pending correction, switches through the other three modules, restores the coding operation, verifies the correction and reopens the saved result from a fresh controller.

Train or reuse an existing local checkpoint. This example keeps the same model loaded across switches. The command stream is also suitable for an interactive terminal, with one JSON object per line.

```sh
my-own-ai train --corpus examples --steps 20 --context 64
my-own-ai session --checkpoint data/checkpoints/brain.pt --threads 4 --workspace . <<'JSONL'
{"command":"switch","mode":"conversation"}
{"command":"remember","key":"hometown","value":"Manistee"}
{"command":"start","operation":"reply","project":"greetings","objective":"Say hello and mention Manistee.","inputs":{"required_terms":["Manistee"]}}
{"command":"step"}
{"command":"checkpoint"}
{"command":"switch","mode":"writing"}
{"command":"switch","mode":"conversation"}
{"command":"progress","project":"greetings"}
{"command":"metrics"}
{"command":"quit"}
JSONL
```

Continue with `step` until the result is terminal. Each response reports `ok` and either a result or an error. `start` accepts an optional project identifier, input object and `max_corrections` (default 2). `progress` reads the latest or named project in the active module. `export` takes a new destination and optional project. Existing files are protected. `--memory` selects the state directory and `--tokens` sets the generation budget. The standalone `switch` command still checks activation and saves state before exiting.

## Switching and persistence

The controller serializes owner commands and holds a Linux file lease for the session's memory directory. Before releasing the current module it suspends it, snapshots its state, validates and syncs the checkpoint, unloads the module, and closes its resource scope. It then loads the replacement's resources, restores its checkpoint, resumes the module and publishes it as active. A bad target checkpoint is caught in preflight. A failure before unloading resumes the current module. A failed replacement is cleaned up before the controller reloads the previous module. If resource cleanup cannot be confirmed, the controller disables further activation and retains the lease for that process.

A checkpoint holds the objective, accepted project work, pending operation, stage history, evaluation evidence and contextual inputs. The store validates checksummed JSON before atomic replacement and retains one validated prior generation. Interrupted stages resume at the last committed stage. A damaged primary falls back to the previous generation. If both generations are invalid, activation fails with the records preserved for inspection. Legacy flat module JSON is read and upgraded on the next save. Declared module versions must match their saved version. Exported artifacts and checkpoint data are ordinary local files.

Project names are private to a module. Each module has one pending operation at a time and can retain multiple named projects. Shared memory is a separate key-value store. The owner writes entries explicitly with `remember`. A module can copy those entries into its context through the scoped `shared()` interface. A switch does not copy private project data between modules.

## Reasoning and tool scope

The controller starts at Formulate, moves to Construct, then runs Evaluate using the active module's checks. A failed evaluation moves to Correct if the correction budget permits. The revised candidate always returns to Evaluate. Success requires a passing evaluation. Exhaustion or an invalid proposal yields an unsuccessful result. An exception during a stage leaves its previous committed state available to retry.

The model receives a data-only proposal request and returns text. It receives no controller, tool handle or session command stream. Only trusted procedures can use the controller-issued tool capability. It restricts tool names and stages to the active manifest, rejects access from another thread and is revoked before the stage commits. The owner interface alone can switch modules, write shared memory and export artifacts.

| Module and operation | Local capabilities | Acceptance checks |
| --- | --- | --- |
| Conversation `reply` | Previous accepted exchange, explicit personal context, reply proposals | Printable response and optional `required_terms` |
| Writing `manuscript` | Outline, Markdown draft, optional workspace `reference`, revision and export | Markdown headings and optional required `headings` |
| Coding `python_source` | Plan, optional workspace Python `source`, generate or revise source, compile, inspect and export | Compilation plus requested `declarations` or scalar `literals` |
| Research `evidence_notes` | Acquire workspace `sources`, retain extracted text and SHA-256 provenance, organize evidence and export | At least one verified exact quote per source on a line like `- [1] quote` |

Workspace inputs are operator-selected relative paths resolved within the workspace. Text and Markdown are built in. PDF and DOCX extraction uses the existing optional book parsers. Imported files are limited to 1 MiB and 64,000 extracted characters each. Each module checkpoint is limited to 8 MiB.

## Install a module

Install trusted local Python code in the same environment and expose a factory taking the workspace path and returning a fresh `BasicModule` subclass. Register it using `my-own-ai session ... --module my_package:build_module`, or pass the instance alongside `default_modules()` when constructing a `ModuleController` in Python. Neither route requires changes to the model.

1. Supply a `Manifest` with an identifier, version, installed distribution dependencies, local assets, RAM estimate in MiB, stage-scoped `Tool` entries and supported operation names. Dependencies are Python packages. Other functional modules cannot be loaded as dependencies.
2. Supply a contextual description for each `Stage` in `procedures`. Implement `validate_inputs`, `prepare`, and `evaluate`. Evaluate should return `Evaluation(passed, evidence)` from observed checks. Use `validate_context` and `validate_state` to reject incomplete extension checkpoints. Set `exportable = True` for text exports of evaluated projects.
3. Map declared tool names to callables in `tool_handlers()`. Procedures invoke them through `tools.call(tool_name, **arguments)`. Tool inputs and outputs are JSON values. Stage tools should be read-only or replay-safe because an interruption before the checkpoint can replay a stage.
4. Acquire resources in `load(scope)`. Declared files are opened under the configured asset root and exposed as `scope.assets`. Use `scope.cache()`, `scope.enter(context_manager)`, or `scope.own(cleanup, *args)` for resources such as caches, memory maps, optional adapters or handles. Use `scope.worker(argv)` for a managed Linux worker group. Managed workers are single processes. The scope reaps them on unload and Linux signals them if the owner dies. Register cleanup immediately after acquisition.
5. Use `activate(saved_state)` to restore JSON working state, `resume()` and `suspend()` to manage activity, and `deactivate()` to clear extension references. Deactivation must tolerate partial initialization. The resource scope closes even when a lifecycle callback fails. Controller-owned operation state uses the reserved `_reasoning` and `_accepted` fields.

Extensions run as trusted code in the owner process. The capability boundary isolates model proposals from tools. Installed Python code and local model files should come from sources the operator trusts. Background workers should quiesce in `suspend()` and must avoid changing controller-owned state directly.

## Measurement and current limits

`metrics` reports CPU affinity, Linux available RAM, resident and peak RSS, and accumulated process CPU time. `switch` reports wall time, process CPU time and resident memory before unload, after unload and after the replacement loads. `--memory-budget-mib` can cap manifest admission below the reported host RAM. Admission uses declared estimates. Resource cleanup frees ownership and closes handles, while Python and PyTorch allocators may retain process RSS. The demo also reports measured formulation time and supplies a repeatable starting point for measuring the Latitude.

The byte-level model still needs training for useful proposals. Small context windows sharply limit how much project context the proposer can use. The checks verify the stated local criteria. They do not assess writing quality, execute generated code or establish the truth of research claims beyond matching the quoted text. The runtime has no web acquisition, OCR, semantic retrieval, manuscript layout engine or generated-code execution. Bundled modules use the shared model directly. Optional neural adapters are resources extensions can manage, and the repository ships no adapters.

## Search and evaluate a local corpus

The Research module can use an optional SQLite FTS5 passage index. Install the existing book parsers with `python -m pip install -e '.[books]'`. Linux's Python build needs SQLite with FTS5. The index stores extracted text and belongs alongside the private books. Keep indexes, checkpoints, personal evaluation files, and session memory under ignored `data/` or outside the checkout.

```sh
my-own-ai index-corpus --corpus corpus/private --index data/corpus.sqlite
my-own-ai search-corpus --index data/corpus.sqlite --query 'your search terms'
my-own-ai session --checkpoint data/checkpoints/brain.pt --corpus-index data/corpus.sqlite --tokens 256
```

In the session, switch to Research and start `grounded_analysis` with an objective that states the question. The optional input `limit` selects up to 12 passages. Formulate searches the index and checkpoints the ranked passages with relative filenames, locators, and text digests. The controller closes the index on unload. After formulation the operation can resume from the checkpointed passages, including in a session without the index configured. Construct and Correct receive numbered excerpts within the model's context budget. Evaluate verifies exact quotes in a `Premises:` section and checks that references in `Inference:` point to verified premises. Quotes use normalized extracted text. PDF locators are one-based file pages, and DOCX locators are one-based body paragraphs.

For a reproducible evaluation with the public sample cases:

```sh
my-own-ai index-corpus --corpus examples/grounded_corpus --index data/demo-corpus.sqlite
my-own-ai evaluate-corpus --index data/demo-corpus.sqlite \
  --cases examples/grounded_cases.json --checkpoint data/checkpoints/brain.pt
```

Each held-out case has an ID, a question, expected source paths relative to the indexed corpus, and required inference terms. The CLI uses the real local checkpoint for proposals and exits with status 1 when any case misses its rubric. The report distinguishes expected-source retrieval, verified citation coverage, inference term coverage, and operation outcome. The expected sources and terms stay outside the model prompt. Keep case files outside the training corpus. Assess substantive reasoning separately because exact quotes and required terms can be verified mechanically, while the soundness of an inference requires reviewing the argument. A tiny continuation-trained checkpoint may retrieve the right passages through the index while still producing unusable proposals.

## Train on worked proposals

`train-proposals` starts from an original-weight checkpoint created with `train` and writes a separate CPU checkpoint. The training objective predicts the worked answer bytes and an end marker. The prompt bytes are masked out of the loss. The same prompt formatter supplies the numbered passages during training and inference. The controller still owns evaluation, so the training file can contain targets for Formulate, Construct and Correct.

A row has an `id`, a `group` for variants of one exercise, a `stage`, a `question`, numbered `passages` as a list of text strings, and `plan`, `candidate`, `feedback` and `answer` strings. Formulate uses empty plan, candidate and feedback fields. Construct includes the prior plan. Correct also supplies the rejected candidate and evaluation feedback. Construct and Correct targets must cite an exact excerpt visible in the model prompt and state an inference. Keep all variants of an exercise together. The loader checks group separation between training and validation and detects repeated normalized questions across those files and the held-out cases. Review paraphrases and near-duplicates when authoring the splits.

```sh
my-own-ai train --corpus corpus/private --steps 20 --context 384 \
  --output data/example-base.pt
my-own-ai train-proposals --checkpoint data/example-base.pt \
  --examples examples/proposals_train.json --validation examples/proposals_validation.json \
  --held-out examples/grounded_cases.json --output data/example-proposals.pt
```

The starting checkpoint must have a context of at least 256 bytes and must have training documents longer than its context. Use the private manuscript corpus or another local text source with sufficiently long documents. Review validation loss and use the same held-out questions and decoding budget for both checkpoints. Validation loss measures byte prediction against worked targets. Report held-out citation and inference checks independently.

### Evidence selection and inference

Choose `--format grounded_span_v2` when training from a checkpoint with at least 512 bytes of context. Construct and Correct examples in this format use one verified premise. Their learning targets are the displayed passage number followed by the inference text. The model chooses among up to three bounded, word-aligned excerpts and generates the inference. The controller inserts the exact excerpt as the premise, checks that it exists in the saved corpus evidence, and runs the usual citation checks. An unfinished inference, an unavailable selection, or a response without a specific term from the selected evidence fails evaluation. Older checkpoints keep their existing generation behavior.

```sh
my-own-ai train --corpus corpus/private --steps 1800 --context 512 --embed 96 \
  --heads 4 --layers 2 --batch 4 --schedule-steps 1800 --warmup-steps 40 \
  --output data/base-512.pt
my-own-ai train-proposals --format grounded_span_v2 --checkpoint data/base-512.pt \
  --examples examples/proposals_train.json --validation examples/proposals_validation.json \
  --held-out examples/grounded_cases.json --output data/span-proposals.pt
my-own-ai evaluate-corpus --index data/demo-corpus.sqlite --cases examples/grounded_cases.json \
  --checkpoint data/span-proposals.pt --tokens 160 --include-answers
```

`--include-answers` prints answers and the source locations of verified quotes for human review. Treat this output as private when the index contains manuscript text. Review whether the inference addresses the question and follows from the cited premise. The structural and keyword checks alone cannot establish that.

## Verified rulebook reasoning

Research can use a locally reviewed rulebook for bounded questions. Use
`--rulebook` to select this path explicitly. A session can also load a
checkpoint for its other operations. Rulebook Research runs offline and does
not require model weights.

```sh
my-own-ai index-corpus --corpus examples/rulebook_corpus --index data/rulebook-demo.sqlite
my-own-ai evaluate-corpus --index data/rulebook-demo.sqlite \
  --rulebook examples/inference_rules.json --cases examples/rulebook_cases.json --include-answers
my-own-ai session --corpus-index data/rulebook-demo.sqlite --rulebook examples/inference_rules.json
```

A rulebook is a versioned JSON file with extractors and rules. An extractor
declares whether it reads the question or indexed evidence, a regular
expression, and the fact it recognizes. Evidence extractors may identify an
exact corpus-relative source path. Facts are arrays beginning with a
predicate name. Variables begin with `$` and refer to named captures in
extractors. In a rule, the same variable must bind to the same value in each
required fact. A rule's `produces` fact can be used by another rule. A
terminal rule supplies an `answer` paragraph and can refer to bound
variables with braces.

The controller retrieves text for matching source-linked premises, derives
the result, and checks the exact citations and derivation before completion.
A terminal result needs both a question match and indexed evidence. When the
available evidence supplies no answer or competing answers, the operation
returns unresolved. Evaluation feedback records the applied rule IDs.
Checkpoints retain the rulebook digest and the exact evidence. A pending
operation requires the original rulebook when resumed.

Patterns and logical rules must be written or reviewed by the operator.
Their coverage and the evidence they retrieve bound the conclusions.
A tested rulebook gives usable results on covered questions. General neural
reasoning remains a separate capability. The optional answer audit may
contain private manuscript excerpts.

## Experimental neural evidence ranking

The neural judge learns to rank candidate conclusions against a question and
evidence. It uses original CPU weights, local lexical features and supervised
contrast examples. It receives no answer key or Research rulebook at inference.
When the top score is weak, the margin is narrow, or retrieved passages favor
different candidates, it can abstain. These checks do not establish semantic
entailment.

Provide training, validation and test files with separate case groups and
questions. Each row has this format:

```json
{
  "id": "delay_example_a",
  "group": "delay_example",
  "question": "Will a late notice delay the response?",
  "evidence": ["The keeper can act only after reading the notice."],
  "candidates": [
    "The late notice delays the keeper's response.",
    "An independent signal triggers the response without the notice."
  ],
  "accepted": 0
}
```

Use a zero-based candidate index for an accepted conclusion and `null` when
the material supplies no supported answer. To test evidence dependence, keep
the question, candidates and their order fixed while changing the premise and
the expected choice in two cases with the same group. The evaluator reports
fully correct groups separately from individual cases. It exits with status 1
if any case is wrong.

```sh
my-own-ai train-neural-judge --examples data/train.json --validation data/validation.json \
  --held-out data/test.json --output data/neural-judge.pt
my-own-ai evaluate-neural-judge --checkpoint data/neural-judge.pt --cases data/test.json
my-own-ai draft-neural-outline --checkpoint data/neural-judge.pt --index data/corpus.sqlite \
  --brief data/brief.json
```

A brief contains a question and a list of candidate theses. The outline
command retrieves corpus passages and returns the selected thesis with its
exact source and locator, or an unresolved result. It does not generate the
candidate prose. The original local proposal engine and verified Research
remain available through their respective commands.


## Experimental autonomous neural inference

`my-own-ai infer-neural` uses a **locally trained grounded-supervised proposal
Transformer** and a separately trained evidence judge. The user supplies only a
question, not candidate conclusions. The proposal model formulates a plan,
samples multiple possible source-grounded conclusions, and the judge ranks the
model's own valid alternatives. The existing Research controller checks source
quotes, handles correction feedback, and records completion.

```sh
my-own-ai infer-neural --checkpoint data/checkpoints/proposals.pt \
  --judge data/checkpoints/judge.pt --index data/books.sqlite \
  --question "How does the change in condition affect an institution?" \
  --attempts 4 --tokens 160
```

This path is **experimental**: when the model fails to generate valid proposals,
the controller reports failure rather than inventing a proof. The judge compares
alternatives; it does not establish logical entailment. A completed controller
run verifies quotation/format requirements, **not** the substantive truth of
the inference. The output explicitly reports `semantic_entailment_verified: false`.
Private manuscripts and neural checkpoints are not added to GitHub.

Train the proposal model with `train-proposals --format grounded_supervised_v1`
and the ranker with `train-neural-judge`. Measure generalization on untouched
cases, particularly paired premise changes. Increasing sample count does not
by itself establish the model learned the user's ontology.

Run the same held-out manuscript evaluation used for earlier grounded Research
trials, without giving the generator the expected sources or inference terms:

```sh
my-own-ai evaluate-neural-inference \
  --checkpoint data/checkpoints/proposals.pt \
  --judge data/checkpoints/judge.pt \
  --index data/books.sqlite --cases private/held-out.json
```

The independent evaluation rubric remains outside the model's input.
A full-pass requires retrieved and cited expected sources, substantive target
terms and controller completion; the rubric is still not an entailment proof.


## Relational reasoning curriculum (PR #10)

The two foundational claims of the author's ontology are **identity as a unity
of self and other**, where relationships are constitutive of identity, and
**hierarchically ordered communion**, where roles, direction and asymmetry
matter. The training procedure operationalizes these principles without
treating them as generic graph slogans: examples require analysis of
constitutive identity, ordered roles, communication channels, internal states,
disturbances, feedback and the **derived consequence**. Training targets supply
worked derivations; the runtime model receives only the question, source
evidence and its own generated plan.

A relational input JSON file is a list of case objects:

```json
{
  "id": "case_open",
  "group": "counterfactual_family",
  "domain": "engineering",
  "question": "How does the signal affect adjustment?",
  "evidence": ["A controller receives pressure readings and adjusts its valve."],
  "analysis": {
    "identity": "The effective system depends on this sensor-controller relation.",
    "hierarchy": "The controller commands the valve while the sensor informs it.",
    "channels": "Pressure measurements flow from sensor to controller.",
    "states": "The controller has an updated measurement.",
    "disturbance": "Changing pressure disturbs the target state.",
    "feedback": "A subsequent valve correction counteracts the disturbance."
  },
  "conclusion": "The controller can adjust the valve in response to pressure changes."
}
```

Every counterfactual family must contain at least two cases with the same
question and domain, **changed evidence and changed conclusions**. Families
are disjoint across training, validation and held-out files. Evaluation
domains must be entirely absent from training and validation. The supplied
`examples/relational` corpus is a small, **synthetic software fixture**, not
the user's manuscripts; private authored examples belong outside Git.

Generate worked examples:

```sh
my-own-ai prepare-relational \
  --training examples/relational/train.json \
  --validation examples/relational/validation.json \
  --held-out examples/relational/held.json \
  --output data/relational
```

The command creates two validated worked-example files for the **existing
original-weight Transformer**. It does not create a new neural brain, publish
private source documents, train the model automatically or expose held-out
answer keys to the training input. For substantive research, replace these
demonstration examples with author-approved cases based on privately indexed
writings, with distinct held-out disciplines.

Train with an existing proposal-capable local checkpoint and sufficient
context for the six worked operations; use a context size of **at least 2048
bytes**, preferably 4096 or more for longer worked analysis:

```sh
my-own-ai train-proposals \
  --checkpoint data/checkpoints/brain.pt \
  --examples data/relational/relational_train.json \
  --validation data/relational/relational_validation.json \
  --output data/checkpoints/relational-brain.pt \
  --format grounded_supervised_v1 --pair-batches --steps 400
```

Then test **real neural outputs**, not scripted examples:

```sh
my-own-ai evaluate-relational \
  --checkpoint data/checkpoints/relational-brain.pt \
  --training examples/relational/train.json \
  --validation examples/relational/validation.json \
  --cases examples/relational/held.json
```

This reports exact conclusions, correctly cited evidence, complete
counterfactual families, and per-domain held-out results. With `--pair-batches`, supervised updates sample whole counterfactual families: both source states and both Formulate/Construct stages remain together. This is paired exposure under the existing token-prediction objective, **not a separately validated causal-learning loss**. Use validation and held-out changes to decide whether it improves premise sensitivity.\n\nThe strict exact
conclusion metric is conservative; paraphrases may fail it, and even an exact
match is not a proof of semantic entailment. More importantly, **a green
software test suite is not evidence that the neural model has learned the
ontology**. Novel-discipline success requires measured held-out outputs.


## Cumulative intellectual development: owner-approved inheritance (PR #11)

This system deliberately separates *preserving an intellectual record* from
*training on it as accepted truth*. Every record has a local immutable file
digest, a declared origin (`authored_work`, `derived_work` or `reasoning`),
optional evidence locators, and an explicit `pending / approved / rejected`
decision. A Research operation's format validation does **not** authorize its
claim as training data. The owner decides; neither model nor controller
automatically approves its own inventions.

```sh
my-own-ai register-work --ledger data/intellectual-ledger.json \
  --id book_2030 --kind authored_work --source corpus/private/book.txt \
  --origin "author: original manuscript"
my-own-ai decide-work --ledger data/intellectual-ledger.json \
  --id book_2030 --decision approved
my-own-ai approved-works --ledger data/intellectual-ledger.json
```

Approved manuscripts can be staged in a fresh **private** local training directory
and fed into the existing corpus trainer. Registration alone never silently
changes the model's weights:

```sh
my-own-ai stage-approved-corpus --ledger data/intellectual-ledger.json \
  --output corpus/approved-generation-1
my-own-ai train --corpus corpus/approved-generation-1 \
  --output data/checkpoints/new-corpus-brain.pt --context 512 --steps 200
```

Every staged source is copied only after its SHA-256 matches the approved record;
pending or rejected works never enter that corpus. Its manifest preserves
the origin and approval lineage. A new generation uses a fresh output directory;
existing corpora are not overwritten automatically. To train on an
approved *worked reasoning file*, register it as `reasoning`, manually approve
it, and build a rehearsal dataset containing both the previously learned
examples and new approved material:

```sh
my-own-ai prepare-replay \
  --ledger data/intellectual-ledger.json \
  --retained data/previous-worked.json \
  --validation data/stable-validation.json \
  --work-id approved_reasoning_1 \
  --output data/rehearsal.json
```

No previous accepted training example is silently dropped. The old examples
and approved additions are checked for duplicated IDs, groups and validation
questions. The output stays local and private.

`train-proposals` can resume interrupted runs from the same output checkpoint,
preserving **optimizer state, PyTorch RNG, data sampler RNG, cumulative step count,
data fingerprints and training settings**:

```sh
my-own-ai train-proposals --checkpoint data/checkpoints/brain.pt \
  --examples data/rehearsal.json --validation data/stable-validation.json \
  --output data/checkpoints/candidate.pt --format grounded_supervised_v1 \
  --steps 100 --checkpoint-every 25
my-own-ai train-proposals --checkpoint data/checkpoints/candidate.pt \
  --examples data/rehearsal.json --validation data/stable-validation.json \
  --output data/checkpoints/candidate.pt --format grounded_supervised_v1 \
  --steps 200 --checkpoint-every 25 --resume
```

`--steps` on resume means the **total target step count**, not steps to add.
Changing the training/validation examples requires a *new training run from the
previous checkpoint weights*, with a new output file and optimizer; it is not
permitted to pretend the prior optimizer trajectory is still identical.

Before adopting a candidate, compare it with the previous checkpoint on
identical old held-out cases and new held-out cases from genuinely unseen
disciplines. Neither set's answer labels enter the neural model prompt.
Promotion requires: no loss of **any individually correct old case**, retention
of previously correct counterfactual families, and strictly improved new-case
accuracy without reducing new counterfactual-pair results.

```sh
my-own-ai gate-learning --baseline data/checkpoints/previous.pt \
  --candidate data/checkpoints/candidate.pt \
  --old-cases corpus/private/old-heldout.json \
  --new-cases corpus/private/new-heldout.json \
  --training-examples data/rehearsal.json \
  --validation-examples data/stable-validation.json \
  --report data/learning-comparison.json \
  --promote-to data/checkpoints/active-approved.pt
```

The report records immutable input SHA-256 identities and per-suite scores.
On failure, the previous released checkpoint is left untouched. On pass, the
candidate is copied into the release destination via a same-directory atomic
replacement. Success is based on conservative exact-consequence and paired
metrics, not proof of philosophical correctness.

**Important:** passing software CI demonstrates that the mechanism works.
The requested success criterion—measured new learning *and* non-regression on
real old tasks—requires local private training and a passing independent
`gate-learning` run, not merely a green PR.

The gate also rejects a held-out question or counterfactual family found in the supplied training or validation examples. The evaluated training and validation file digests are recorded in the release report.


Completed neural Research investigations can be archived automatically **pending**
author review without teaching the model its own unverified assertions:

```sh
my-own-ai infer-neural --checkpoint data/checkpoints/candidate.pt \
  --judge data/checkpoints/judge.pt --index data/manuscripts.sqlite \
  --question "How does this new condition alter the relation?" \
  --ledger data/intellectual-ledger.json \
  --archive-dir data/reasoning-archive
```

This stores the model's candidate, exact citation locators and controller status,
registers the file under a digest, and leaves approval pending. To reuse an
approved record as training, an author must convert the approved reasoning into
an independently reviewed worked-example dataset; raw archived JSON isn't
implicitly a valid supervised example. No archived argument auto-edits weights.
