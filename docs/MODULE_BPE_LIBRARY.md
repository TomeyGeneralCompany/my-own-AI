# Development module runtime and Finance library

This PR adds an explicit BPE session route, preserving the existing byte-model
runtime and leaving training untouched. The original checkpoint is loaded by
the existing validated tokenizer/checkpoint loader once. BPEProposer passes
the module request (objective, procedure, evidence, prior output and feedback)
to the native chat framing. No external model, rulebook or answer substitution
is used. This is not qualification or production activation. A request that
exceeds the model context fails visibly; long-document retrieval and iterative
context packing need separate implementation. Raw invalid UTF-8 also fails
rather than being silently repaired. Use development checkpoints and separate
memory, never the production training state.

## Finance reference library

Acquire owner-approved editions listed in finance_library, including the
owner's existing economics manuscript. Keep the actual documents in a private
local folder. Build the SQLite index using the existing corpus indexer:

```sh
my-own-ai index-corpus --corpus data/finance-books --index data/finance.sqlite
my-own-ai session --checkpoint data/dev-core.pt --tokenizer data/dev-tokenizer.json \
  --memory data/dev-module-memory --workspace . --tokens 128 \
  --module my_own_ai.finance_module:build_module
```

```json
{"command":"switch","mode":"finance"}
{"command":"start","operation":"library_research","objective":"Investigate price signals and dispersed knowledge","inputs":{"index":"data/finance.sqlite","limit":1}}
{"command":"step"}
```

Repeat step. The objective supplies the retrieval query; the original passage,
source locator and digest are saved in the operation's state. The existing
citation verifier checks exact quoted passages; substantive reasoning still
requires independent review. This is one retrieval operation, not yet
autonomous iterative library search, curriculum training or a strategy
evaluator. The existing scenario replay and retrospective oracle remain
separate from reference research. Full-text acquisition, chronological
market experiments and owner-bound qualification remain outstanding.

Tests include real randomly initialized BPE forward execution (not a
competency exam), unchanged weights, oversized-request rejection, and a
labeled fixture that demonstrates library search, source preservation and
citation evaluation after restart.
