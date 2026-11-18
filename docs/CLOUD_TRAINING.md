# Ancestor: off-laptop training (private remote data, public source code)

## Architecture

GitHub-hosted Ubuntu runners execute the original Ancestor transformer. The
owner's laptop does **not** need to be turned on. Scheduled jobs resume a
complete checkpoint and carry out small bounded amounts of training.

The public code repository contains only software. Its Actions logs must never
include manuscript text, owner axioms, private tokens, or private checkpoints.
**Never upload the training archive or trained weights as a public GitHub Actions
artifact**. The workflow uses a separate PRIVATE GitHub repository's release
assets for both private inputs and private checkpoints.

GitHub standard hosted jobs have a six-hour per-job limit. This workflow uses
shorter sessions and checks in all model weights and optimizer/RNG state between
runs rather than relying on an uninterrupted process. This is compute run by
GitHub, not on the owner's computer. Public-repository standard-runner minutes
are currently generally free, subject to GitHub policy and service limits.

## Required private storage

A separate GitHub repository **with visibility PRIVATE** must exist and be
accessible to a narrowly scoped GitHub token. Do not assume a public repository
is appropriate just because it allows more Actions minutes.

Create a release with tag:

    ancestor-training-inputs

Upload exactly these assets under that release:

    ancestor_private_training_library.zip
    governance.json
    tokenizer.json
    heldout.txt

The manuscript ZIP is the prepared local archive containing all eleven originals,
the extracted text and source-traceable records; it is **not** code and is not
committed to public GitHub.

- governance.json is generated from the **exact owner-approved canonical axioms**
  using the existing governance code. Do not invent or substitute a list.
- tokenizer.json is the frozen versioned original byte-level tokenizer.
- heldout.txt is genuinely independent English validation text, not a passage
  already included in one of the owner's eleven training manuscripts.

Store the private-data repository full name as a variable on the public code
repository:

    ANCESTOR_PRIVATE_DATA_REPO = OWNER/PRIVATE-REPOSITORY

Store a fine-grained token with read/write contents permission limited to
that PRIVATE data repository as the code repository Actions secret:

    ANCESTOR_PRIVATE_REPO_TOKEN

The ordinary public-code GITHUB_TOKEN does not grant access to the owner's
separate private repository; do not put the PAT in a file or log.

**Enable the scheduled worker only after verifying all four input assets and
the private repository access**:

    ANCESTOR_CLOUD_ENABLED = true

Without this variable equal to true, GitHub Actions will skip every scheduled
and manual cloud training job. The workflow never runs on untrusted PRs.

## Every hosted cycle

1. Check that the data repository is PRIVATE and token access works.
2. Download the four immutable input artifacts from the private release.
3. Validate the hash-checked source archive and load the local SQLite index.
4. Restore the most recent model + optimizer + RNG checkpoint if it exists.
5. Execute bounded paragraph-conditioned, owner-axiom-prefixed neural updates.
6. Evaluate on an independently supplied held-out English sample.
7. Save an integrity-checked complete checkpoint.
8. Upload latest.pt, latest.json and summary.json to the *private*
   \`ancestor-cloud-checkpoints\` release.

The workflow runs every six hours once activated. There is a single concurrency
group to avoid simultaneous training jobs overwriting the checkpoint.

This is **foundational training exposure**, not certification that the model
understands the owner's philosophy or theology. Full independent curriculum
and stage-gate examinations remain necessary. Generic assistant instruction
weights are not loaded. The owner's foundational axioms are never altered.

## Limitations / prerequisites

1. The current GitHub integration can edit the public source repository but
   cannot create the separate private repository or write its release assets.
2. The eleven-manuscript archive is prepared in the current conversation but
   has not been transferred to a remote private GitHub release.
3. The definitive governing axiom manifest, frozen tokenizer and independent
   validation data are not yet remotely available.
4. Consequently, no hosted training has yet started. The workflow is disabled
   until the necessary private storage is provisioned.
5. Hosted runner limits and usage policies may change. Before using full-time
   training, review GitHub's current Actions billing and quotas.
6. Abrupt runner failure before uploading its final checkpoint loses the
   unuploaded work of that session. The alternate private slot retains the
   previous complete checkpoint if an upload fails midway. A later version
   can add intermediate remote uploads if sessions become long.

This workflow does not require user access to the Zorin laptop once all private
cloud inputs are available.
