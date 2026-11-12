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
