# Architecture contract — foundation v0.1

## Design goals

One local brain, one active capability module, deterministic controller, persistent project memory, and user-owned corpus. Permanent weights are original/randomly initialized, not downloaded pretrained models. No paid inference, remote runtime, mandatory accounts or telemetry. Google Drive can later be an optional archive, never required for execution.

## Implemented

- `model.py`: randomly initialized, byte-tokenized, decoder-only causal Transformer with a masked multi-head attention block. This establishes a **trainable experiment**, not a generally intelligent assistant.
- `train.py`: CPU-only local plaintext training, deterministic seed, model checkpoints, basic loss metrics.
- `modules.py`: exclusive lifecycle, JSON snapshots, atomic writes, rollback following activation failure.
- `cli.py`: reproducible local training and demo module switching.
- CI unit tests cover training/reload, future masking, exclusive switching, checkpoint failure and rollback.

## Shared reasoning objective

**Formulate → construct → evaluate → correct** is the universal scheme. Contextual mappings: writing (outline/draft/critique/revise), coding (plan/implement/test/debug). This is a planned controller feature and is not yet wired into model inference.

## Required next work

1. Corpus ingestion of user-authored books, structured metadata, deduplication, and document-disjoint train/validation tests. Keep books out of Git.
2. Offline perplexity/loss evaluation and CPU throughput/RAM profiling on the actual target laptop.
3. Secure tool permissions and fully implemented module lifecycle with durable work-product snapshots.
4. Local retrieval/indexing and opt-in external archive adapters.
5. Specialized behavior datasets and individually loadable adapters or tools.
6. Linux packaging and realistic integrated application tests.

The future one-core Altoids wearable is a later optimization goal. Existing CPU limitations must be measured, not assumed away. Machine intelligence depends on data, optimization and extensive evaluation, not only modularity.
