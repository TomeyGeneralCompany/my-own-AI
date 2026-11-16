# Operational module implementation: initial audit and Finance instruments

Authority: [owner-governed module specification](MODULE_DEVELOPMENT_SPEC.md), merged in PR #37.

## Existing runtime audit (initial, not activation certification)

The repository already has a single-active-module controller, scoped stage tools, owner commands, resource cleanup, and persistent operation snapshots. Retain them. Do not rename the existing module IDs.

| Module | Existing implementation | Next work before operational qualification |
| --- | --- | --- |
| Conversation | Local reply, shared context, prior exchange | Real qualified BPE checkpoint adapter; multi-turn factual and reference tests; owner style review |
| Authorship (writing) | Markdown draft and heading checks | Durable long-document composition, corpus provenance, math checking, DOCX output, owner evaluation |
| Engineering (coding) | Python source proposal and compile/static checks | Isolated execution and independent functional testing, workspace change control, reproducible build evidence |
| Research | Local retrieval, quote and inference-format checks; optional rulebook | Real neural proposal provenance; substantive owner-reviewed derivation and counterfactual tests; exclude rulebook substitution in qualified runtime |
| Finance | New opt-in offline scenario replay | Sequential observation/replay without future-data leakage, timestamped test data, instrument schema, independent competency and execution authorization |
| Business | Not implemented | Financial modeling and sourced due diligence instruments, adaptive plans, measurable task outcomes |

Cross-cutting gaps: the existing session's byte-oriented LocalProposer and generic checkpoint loading are not yet an approved adapter for the current competency-locked BPE Core. Existing scripted fixtures, generic extension registration and optional RulebookProposer are development mechanisms, not proof of compliant owner-authorized activation. The existing controller also limits correction count and proposal/context length. These require an explicit compatibility design and tests before production module qualification. Do not alter the English trainer to solve them.

## Finance development slice

The opt-in `my_own_ai.finance_module:build_module` factory exposes only `scenario_replay`. It is not a default module and has no brokerage interface, network dependency, credentials, or monetary side effects.

The owner supplies the local scenario filename, initial cents, target cents and fee cents. The Core/proposal engine supplies an order array. The deterministic ledger performs arithmetic and measures the submitted orders; it does not select trades or generate proposals. Failed candidates get measurable feedback. The controller persists the original scenario (with source digest) and owner parameters across switches and cold restores.

Example scenario fixture: `{"prices_cents":[20,50]}`. All observations are shown to the proposer; this deliberately makes the operation a retrospective arithmetic/interface exercise, **not** a prospective strategy test or realistic performance backtest. No timestamps, market liquidity, fractional lots or live fill assumptions are modeled. Final inventory settles at the fixture's last price and incurs the specified fee. No simulation result counts as passing the owner's actual $1-to-$100 examination.

Registration is available only through the existing explicit development extension path, using an isolated development memory directory and a compatible development checkpoint. Do not use the active training checkpoint for this example:

```sh
my-own-ai session --checkpoint data/dev-byte-core.pt --memory data/dev-finance-memory \
  --workspace data/dev-fixtures --module my_own_ai.finance_module:build_module
```

Example owner commands:

```json
{"command":"switch","mode":"finance"}
{"command":"start","operation":"scenario_replay","objective":"Fixture only: finish with 180 simulated cents","inputs":{"scenario":"scenario.json","initial_cents":100,"target_cents":180,"fee_cents":2}}
{"command":"step"}
```

Repeat steps until the operation is terminal. Tests exercise cash/fee/inventory reconciliation, rejection of invalid orders, owner parameters, explicit registration, correction, switch and restart. The test proposer is a labeled fixture, not proof of neural intelligence. No Core weights, curriculum, production registry, existing activation gates, or default modules are changed.

Next: design the qualified Core runtime adapter and owner-bound activation evidence, then extend Finance evaluation with chronological, unseen observations and independently measured outcomes. Live external authority is not part of this PR. The other five modules remain in their stated implementation stages.
