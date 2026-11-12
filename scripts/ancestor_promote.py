"""Build explicit, auditable active checkpoint pointer; never modify weights."""
import argparse
import json
from pathlib import Path

from my_own_ai.language_training import load_language
from my_own_ai.persistent_registry import digest_file, load_active_pointer
from my_own_ai.promotion_gate import assess_promotion


def make_pointer(state, archive, *, experimental=False,
                 evaluation=None, review=None, previous_evaluation=None,
                 gauntlet="examples/english/gauntlet.json"):
    state = Path(state)
    meta = json.loads((state / "generation.json").read_text(encoding="utf-8"))
    if meta.get("format") != "ancestor_generation_v1" or meta.get("phase") != "candidate":
        raise ValueError("Only a completed original model candidate can be activated")
    checkpoint = state / "assistant.best.pt"
    load_language(checkpoint, state / "tokenizer.json")
    if experimental:
        if evaluation or review:
            raise ValueError("Experimental activation must not be mistaken for qualified promotion")
        status = "experimental"
    else:
        if not evaluation or not review:
            raise ValueError("Qualified promotion requires an actual evaluation and owner-reviewed scores")
        result = assess_promotion(evaluation, review, gauntlet,
                                  previous_evaluation=previous_evaluation)
        if result["checkpoint_sha256"] != digest_file(checkpoint):
            raise ValueError("Evaluation checkpoint differs from immutable release")
        status = "evaluated"
    pointer = {
        "format": "ancestor_active_v1", "generation": meta["generation"],
        "candidate_tag": "ancestor-" + meta["generation"] + "-candidate",
        "bundle_sha256": digest_file(archive),
        "checkpoint_sha256": digest_file(checkpoint),
        "tokenizer_sha256": meta["tokenizer_sha256"],
        "corpus_manifest_sha256": meta["published_manifest_sha256"],
        "status": status,
    }
    return load_active_pointer(json.dumps(pointer))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--experimental", action="store_true")
    parser.add_argument("--evaluation")
    parser.add_argument("--review")
    parser.add_argument("--previous-evaluation")
    args = parser.parse_args()
    pointer = make_pointer(args.state, args.archive,
                           experimental=args.experimental,
                           evaluation=args.evaluation,
                           review=args.review,
                           previous_evaluation=args.previous_evaluation)
    Path(args.output).write_text(json.dumps(pointer, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"generation": pointer["generation"],
                      "status": pointer["status"],
                      "checkpoint_sha256": pointer["checkpoint_sha256"],
                      "bundle_sha256": pointer["bundle_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
