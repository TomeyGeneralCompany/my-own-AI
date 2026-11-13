# Competency-locked original neural education

The intended outcome is **actual language and reasoning ability**, not a
fixed number of gradient updates. The owner's corpus supplies philosophical
premises and mathematical meanings. English lessons teach communication
mechanics and conditional reasoning; they do not replace those premises.

## Hard stages

1. **Word-reference foundation.** The g0002 original-weight checkpoint is inherited. The Core must first correctly reproduce at least 80% of unseen reference words from a separate word-copy and identification examination. Failure retains this stage and preserves its neural checkpoint. No grammar or reading stage unlocks before it succeeds.\n2. **English acquisition.** After word-reference competence has been demonstrated, the inherited trained weights proceed to full English instruction.
   New, independently generated English exercises teach grammar, object and
   location extraction, sentence construction, paraphrase, conversation,
   coherent prose and yes/no conditional reasoning. The frozen 200-question
   English examination remains outside optimization. To pass, the exact
   checkpoint must score at least **104/130 correct automatic answers**,
   at least **70% in each automatic category**, and **7 of 10 conditional
   premise-change pairs**. The 70 open-ended answers remain available for
   additional human review. Failure preserves weights and repeats learning;
   it does not unlock a book.
3. **Book reading.** Approved works are processed in the manifest's original
   order. The complete work is split into lossless, ordered text units,
   including the original mathematical expressions. The neural reader
   processes **every unit**, including repeated units, and checkpoints its
   exact next-token window, source SHA, optimizer, RNG and covered-token count.
   A partial pass cannot unlock comprehension. No entire authored work is
   discarded as a test partition.
4. **Book comprehension.** Independently selected, source-grounded questions
   from each work's untouched instruction test partition examine the model.
   A minimum of ten unseen prompts and **80% correct** are required.
   Failure trains further on source-grounded examples for the same book,
   without changing the held-out exam; success permits the next book.
5. **Cross-work reasoning and mathematics.** The final model receives actual
   questions about virtue, trust and capital gain; the user's two axioms;
   a mathematical trust/friction relationship; and counterfactual changes.
   Every raw neural response is preserved. An owner review must explicitly
   establish valid inference, correct mathematics and transfer before
   qualification.
6. **Candidate publication and activation.** The owner-controlled review
   workflow binds the verdict to the exact checkpoint and exam evidence
   before publishing an immutable native candidate. Activation remains
   separate. No unit count, elapsed time, token-validity check or declining
   loss can substitute for the gates.

## Autonomous operation

The main-branch auto-scheduler launches the staged workflow after a merge or
approved curriculum/corpus update and checks for interrupted campaigns daily.
Every successful session packages its durable learner state and immediately
dispatches another session while an examination remains locked. This keeps
resource consumption bounded *per GitHub Actions runner* without declaring
the education finished on a timer or step limit.

Checkpoint releases are keyed to the exact approved corpus manifest digest.
Changed books cannot silently resume stale state. The old train workflow
rejects training generations after g0002, so its step-based candidate
publishing cannot bypass the new stage controller.

## See the model yourself

The **Preview the actual neural Core response** workflow lets the owner ask
the current checkpoint any English question. It chooses the checkpoint
corresponding to the currently locked stage and returns its actual generated
response with a SHA-256 fingerprint. A preview does not train, certify,
graduate or activate the model.

The **Owner review and qualify original Core reasoning** workflow is only
available to the repository owner after the final synthesis responses exist.
The reviewer must explicitly mark all three requirements met and supply
a substantive rationale tied to the exact saved exam and weights. The
resulting g0003 candidate remains inactive until deliberately activated.

## Important measurement boundaries

Infrastructure/test CI confirms these mechanisms execute, including a
real g0002-inherited neural gradient update and 200 new model-generated
answers. **A green infrastructure CI run does not certify English fluency.**
The controller's examination ledger is the competency result; if the
English score is insufficient, its stage remains English. The immutable
g0002 baseline is preserved and is not retroactively called competent.
