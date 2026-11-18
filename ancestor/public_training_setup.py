"""Deterministic initial training inputs from the owner's explicit three axioms.

The full intellectual constitution is the existing verbatim owner-approved
Constitution (loaded and verified by that class). The axioms below are exactly
the author's three stated imperatives; this setup is NOT a license to infer,
invent, or rewrite further theological or mathematical premises.

Held-out text is an original neutral language-mechanics diagnostic, not an
owner-authored premise or a philosophical authority.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
from .corpus import Corpus
from .governance import Constitution
from .tokenizer import ByteBPE

OWNER_THREE_AXIOMS = ("honor God", "love your kin", "defeat your enemies")

HELDOUT_ENGLISH = """A visitor entered the workshop on a rainy afternoon. The
workbench stood near the eastern wall, and several tools lay beside a folded
sheet of paper. One craftsman examined a wooden frame while another recorded
measurements in a notebook. The work continued until the last of the daylight
faded. Each participant could explain the purpose of the instruments, the order
of the operations, and the reasons for checking a measurement a second time.
Nothing in the workshop required the visitor to share the opinions of its
occupants. The purpose of the account is to measure ordinary English
continuation on a passage not present in the owner's manuscript library.
It supplies no substitute premises for the owner's philosophical system.
"""

def prepare(data_root: str | Path, output: str | Path) -> dict:
    data_root, output = Path(data_root), Path(output)
    corpus = Corpus(data_root)
    documents = corpus.db.execute(
        "SELECT id,exact_text_sha FROM documents ORDER BY id").fetchall()
    if [x[0] for x in documents] != [f"WORK-{i:03d}" for i in range(1,12)]:
        raise ValueError("Expected all eleven owner-authored works before preparation")
    output.mkdir(parents=True,exist_ok=True)
    constitution_path = output/"governance.json"
    if constitution_path.exists():
        existing = Constitution.load(constitution_path)
        if existing.axioms != OWNER_THREE_AXIOMS:
            raise ValueError("Existing canonical axioms differ; no silent replacement")
    else:
        Constitution(OWNER_THREE_AXIOMS).save_initial(constitution_path)
    examples = []
    for document_id,sha in documents:
        data = (data_root/"extracted"/(document_id+".txt")).read_text(encoding="utf-8")
        if not data:
            raise ValueError("Empty owner teaching source: "+document_id)
        # Deterministic character-bounded sample for original BPE vocabulary.
        # Training itself uses the FULL documents, not these BPE samples.
        examples.append(data[:20000])
    tokenizer_path=output/"tokenizer.json"
    trained=ByteBPE.train(examples, target_base_vocab_size=320)
    if tokenizer_path.exists():
        prior=ByteBPE.load(tokenizer_path)
        if prior.digest()!=trained.digest():
            raise ValueError("Tokenizer changed between remote training runs")
    else:
        trained.save(tokenizer_path)
    heldout_path=output/"heldout.txt"
    if heldout_path.exists() and heldout_path.read_text(encoding="utf-8")!=HELDOUT_ENGLISH:
        raise ValueError("Held-out diagnostic changed between cloud runs")
    heldout_path.write_text(HELDOUT_ENGLISH,encoding="utf-8")
    return {"documents":len(documents),
            "vocab_size":trained.vocab_size,
            "governance_sha256":Constitution(OWNER_THREE_AXIOMS).digest(),
            "tokenizer_sha256":trained.digest(),
            "diagnostic_kind":"independent language mechanics, not intellectual mastery"}

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",required=True)
    parser.add_argument("--out",required=True)
    args=parser.parse_args(argv)
    print(json.dumps(prepare(args.data,args.out),indent=2))

if __name__ == "__main__":
    main()
