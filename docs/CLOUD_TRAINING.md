# Ancestor — public-source, off-laptop training

The eleven manuscripts are published works. The owner has expressly authorized
them to be made public. They do **not** require a second private GitHub repository.
Ancestor trains on GitHub-hosted Linux CPUs even while the owner's Zorin laptop
is powered off.

## The one published training-input release

Place the exact prebuilt eleven-manuscript archive in the PUBLIC code repository,
\`TomeyGeneralCompany/my-own-AI\`, as an asset on a release tagged:

    ancestor-training-inputs

The release must contain:

    ancestor_private_training_library.zip
    governance.json
    tokenizer.json
    heldout.txt

The first filename is historical: despite the word "private", it may be publicly
hosted with the owner's authorization. Changing the filename is not necessary.
The archive includes the original eleven manuscripts, extracted text, source IDs,
and a SHA-256-verified index. The GitHub workflow checks every original,
extraction and indexed record before using any of them.

The canonical governance file must be created from the EXACT owner-approved
axioms, using \`ancestor init-governance\`. Do not create generic substitute
axioms or use an example list. The tokenizer must be the frozen custom byte-BPE
vocabulary used by the model. The held-out sample must not be training content.

All four inputs can be public, but the GitHub source code does not manufacture
missing items. If the published input release does not exist, the scheduled
job exits cleanly without training. If an input is missing/corrupt, it fails
instead of training a model on a fabricated source.

## What runs, unattended

The workflow \`.github/workflows/ancestor-cloud-training.yml\` triggers every
six hours and can also be run through GitHub Actions → Run workflow.

1. GitHub checks out the original unpretrained Ancestor source code.
2. Downloads approved published input assets from this SAME repository.
3. Validates and imports all eleven works into the ephemeral SQLite corpus.
4. Retrieves the newest SHA-256-verified \`ancestor-neural-checkpoint\`
   GitHub Actions artifact from a previous workflow run.
5. Continues the same model with optimizer and RNG state.
6. Executes owner-axiom-prefixed, paragraph-conditioned updates for a bounded
   period within the hosted runner time limit.
7. Measures token-prediction loss on independent held-out text.
8. Uploads complete model and optimizer checkpoint as a new Actions artifact.

There is no personal access token, separate storage account, remote shell,
self-hosted runner, or requirement to power on the laptop. The standard
\`GITHUB_TOKEN\` is restricted to contents:read and actions:read.

## Recovery and limits

GitHub Actions artifacts are retained for up to the repository's configured
retention period (typically 90 days), and **not indefinitely**. Each scheduled
job saves a fresh artifact, providing continuity as long as regular runs succeed.
Interrupted training loses only work not yet checkpointed; verified earlier
artifacts remain available until retention expiry. If artifacts exist but
none passes SHA-256 verification, the workflow refuses to silently start again
from fresh weights.

Actions artifacts on a PUBLIC repository are accessible to people with repository
read access. This is a consequence of using public GitHub-hosted storage;
the owner should take this into account for checkpoint publication as distinct
from manuscript publication.

Training does not, by itself, certify learned philosophy, theology, mathematics
or owner-method derivation. The existing independent stage examinations are still
required to claim that Ancestor has actually learned the intellectual constitution.

## Exact integration status

The merged previous cloud workflow depended on a private data repository and
owner-created PAT. This PR removes that requirement entirely.

The **public** manuscript archive release has not been created remotely by
the GitHub connector in this conversation: the connector can edit GitHub files
and pull requests but cannot upload an existing 15 MB binary release asset.
The owner-uploaded archive exists in the conversation and can be downloaded;
it is not a publicly reachable GitHub release URL. The owner's exact canonical
axiom manifest, tokenizer, and independent held-out test file likewise have not
been uploaded as release assets. Therefore this PR fixes the remote storage
architecture but does NOT claim an active cloud training process.

Do not merge changes to the owner's corpus content or replace his governing
axioms in an attempt to avoid these input-integrity requirements.
