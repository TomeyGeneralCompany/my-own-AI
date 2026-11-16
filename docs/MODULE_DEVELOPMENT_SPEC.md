# Ancestor operational module development specification

Status: proposed for explicit owner approval. This document governs future module implementation; it does not activate a module, certify the Core, or amend the existing training curriculum.

## 1. Authority and intellectual provenance

One original-weight local Core supplies language, intellectual identity, premises, inference, and strategic decisions. The owner's approved corpus and explicitly approved subsequent intellectual records supply its philosophical foundation. The owner defines objectives, permissions, available resources, and risk parameters. Module developers supply instruments and measurable execution procedures, not an independent personality or philosophy.

The owner's two foundational axioms are identity as a unity of self and other and hierarchically ordered communion. Their meanings and consequences are learned from the author's works; this specification does not substitute a developer's interpretation or a hard-coded inference rule for learned reasoning.

External books, market data, observations, software documentation and experimental results are evidence, not automatically authoritative axioms. No GPT, Claude, Gemini or other external pretrained model may provide hidden reasoning, training targets, evaluations of philosophical correctness, or substitute decisions for the Core. Existing experimental rulebook or judge mechanisms must not silently become the operational decision-maker. Any owner-approved deterministic computation is an instrument, not an alternate intellectual Core.

## 2. Non-negotiable implementation invariants

- Preserve one loaded Core and one explicitly selected active operational module. No automatic mode switching, background parallel module, second language model or replacement of the learned weights.
- Preserve the existing staged education, checkpoint lineage, verified memories and owner approval gates. Module work must not change the currently running trainer or unlock operational execution prematurely.
- No hidden system prompts, undisclosed normative rubrics, compulsory conversational style, imposed risk appetite, canned counterarguments, or silent changes to the owner's objective. Operational instructions may describe formats, factual constraints, tools, and execution preconditions; they must be auditable.
- Distinguish the owner's objective from the Core's proposed strategy, observations, deterministic tool computations, and measured outcomes. A controller may enforce permissions, validate arguments, reject invalid operations and record actual errors; it must never silently invent a strategy or pretend a failed operation succeeded.
- Claims and objections use the same evidentiary standard. An identified risk is an input to reasoning. A demonstrated defect in a strategy identifies the relevant condition, supporting evidence and why the strategy's existing response does not handle it. Unknowns remain unknown; neither obligatory hedging nor fabricated certainty is a grading criterion.
- The owner selects and approves module boundaries and activation. No module may silently expand its tools, financial permissions, credentials, filesystem scope or other external authority. Legal and provider requirements are explicit preconditions, not fabricated legal conclusions. No unauthorized external side effects.
- Default target: four CPU cores, eight hardware threads, 16 GiB RAM, no dedicated GPU. Measure peak RSS, CPU time, wall time, memory use across switches, recovery and tool resources; select resource budgets explicitly. No promise of performance without measurements.

## 3. Common operational contract

Use the existing controller and scoped tool lifecycle where practical. Map proposed module names to existing IDs: conversation -> conversation; authorship -> writing; engineering -> coding; research -> research. Finance and business are new proposed IDs, not replacements. Audit existing procedures and experimental fixtures against this contract before production activation.

Each implementation must provide a versioned manifest, supported operations, tool permissions and input/output schemas, checkpoint and recovery contract, resource budget, explicit owner objective, acceptance criteria and independent tests. Record actual Core proposals with checkpoint identity, tool requests and observations, objective changes authorized by the owner, feedback, next decisions, and final evidence. Redact credentials from traces. Record only auditable operational decisions, not invented private reasoning.

The Core chooses among available lawful and authorized methods; deterministic tools execute and report. Use an observation, formulation, execution, evaluation and correction loop. Failures return concrete observations; alternative approaches remain available. Completion requires independently measured evidence. Time and resource bounds are owner parameters; runner slices may checkpoint/resume without silently terminating an objective. An unmet goal is reported as unmet.

## 4. Module design briefs

| Module | Instruments | Independent acceptance evidence |
| --- | --- | --- |
| Conversation | Local dialogue, owner-provided context, memory retrieval and approved memory writes | Held-out multi-turn tasks with factual consistency, correct reference resolution and substantive responses; owner reviews style. |
| Authorship (writing) | Approved corpus retrieval, long-document outline and section state, citations, mathematical notation, editorial checks, DOCX export | Owner-defined original manuscript (eventually 600 pages), argument consistency, source fidelity, independently checked mathematics and owner-approved philosophical derivations; no padding counted as success. |
| Engineering (coding) | Workspace inspection, source editing, local compiler and isolated tests, dependency documentation, reproducible builds | Previously unseen project specification, functioning implementation, independent tests and reproducible results within approved workspace; no scripted solution substituted. |
| Research | Source retrieval, provenance, calculations, experiments and counterfactual comparisons | Held-out questions with verified evidence, sound derivation judged independently, premise-change behavior and reproducible experiment; citation presence alone is insufficient. |
| Finance | Read-only market data, historical data, deterministic calculations, replay/simulation, ledger and explicitly authorized execution interfaces | Owner's initial target: USD 1 to USD 100 in 12 hours from actual starting capital, measured by settled realizable balance after costs. Simulation is labeled and cannot satisfy a real-money examination. Subsequent target tiers require separately approved capital and measurement rules. No deposits or fabricated fills counted as trading returns. |
| Business | Business research, financial modeling, acquisition analysis, workflow planning, due diligence and owner-approved operational interfaces | Unseen commercial objective, sourced premises, measured results, adaptive alternatives and owner review. External commitments require appropriate authority. |

The finance examination target is a required outcome, not a prediction or assurance of returns. Initial development uses deterministic fixtures and simulated adapters; live financial authority, applicable legal/provider requirements, execution permissions, and owner-defined limits are separate implementation decisions. No broker credentials, actual trades or capital transfers are authorized by this specification.

## 5. Required evidence before module activation

1. Source audit: enumerate prompts, templates, tool interfaces, dependency graph, deterministic rules, evaluation criteria, and all external call sites. Identify the origin and purpose of each. Owner reviews the exact commit and manifest.
2. Provenance test: real local Core supplies recorded proposals; fixed fixtures are explicitly labeled and cannot certify competency. No external reasoning endpoint or hidden canned proposal.
3. Objective test: owner objective survives switches, errors and restart without silent replacement. Permission and tool-schema tests cover unauthorized operations and side effects.
4. Reasoning transfer: owner-approved held-out cases from the authored corpus and changed-premise pairs; expected conclusions kept outside model input. Evaluate substantive inference, not merely required phrases. No developer-authored philosophical answer key treated as owner authority.
5. Operational examination: real outputs and independent outcome measurements on previously unseen tasks. Failure does not unlock the module. Simulation, unit-test success and declining loss are not evidence of live task completion.
6. Resource and isolation test: CPU-only target, single active module, cleanup, checkpoint/restart, memory budget and non-regression of previously qualified abilities.
7. Explicit owner approval binds the reviewed code identity, Core checkpoint, module manifest, permissions and examination evidence. Changes to authority or material behavior require new review.

## 6. Implementation order and scope

First approve this specification. Next audit the existing four modules and controller for alignment and enumerate gaps; prepare individual module specifications, starting with Finance as requested. Implement each in a separate scoped PR with tests and documented owner-review decisions. Do not replace working architecture merely to rename modules. Keep all operational modules locked from the unqualified production Core. Existing development fixtures remain fixtures.

This specification is a reviewable development contract, not a technical guarantee that arbitrary trusted Python extensions cannot violate it. Enforcement requires implementation, independent tests, exact-commit review and explicit activation. No runtime code or training state is changed by this document.
