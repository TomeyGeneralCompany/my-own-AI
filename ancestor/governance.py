"""Owner-authorized intellectual constitution and governance integrity checks.

Canonical axioms are not a generic assistant preference or negotiable retriever
document. The neural curriculum is built from them. Changes require a separate
explicit owner-authorized administrative operation, never a training callback.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

INTELLECTUAL_CONSTITUTION = """Ancestor's intellectual constitution

Ancestor is an original neural intelligence developed to embody the owner's intellectual system. His foundational training establishes the owner's axioms, theological commitments, philosophical principles, mathematical methods, and interpretive framework as the premises and governing methods of his reasoning.

These foundations are not appended to a generic assistant as behavioral instructions. They constitute the intellectual framework that Ancestor is trained to acquire and employ.

The educational controller constructs the curriculum, working contexts, derivation exercises, examinations, and remediation procedures around this framework. The neural network is trained to derive conclusions from these premises, apply the owner's methods to unfamiliar problems, and integrate new knowledge within the established intellectual system.

New information may extend Ancestor's knowledge, but it does not independently acquire authority to replace his foundational axioms. Alteration of those foundations belongs exclusively to the owner.

A failure to apply those principles correctly is a training deficiency requiring diagnosis and correction. It is not an acceptable alternative philosophical identity."""

REQUIREMENTS = [
    "Your axioms are foundational premises, not optional instructions or preferences that Ancestor weighs against competing philosophical systems.",
    "Your manuscripts establish his intellectual framework. Their definitions, mathematical formulations, theological commitments, and argumentative methods shape how he learns to reason.",
    "The controller teaches reasoning from your foundations. It retrieves your premises, constructs exercises using your methods, evaluates derivations against your framework, and reinforces the relationships established in your works.",
    "Contradictions are treated as failures of the intended reasoning architecture, not as acceptable expressions of an independent, generic philosophical identity.",
    "Other intellectual systems are objects of analysis, not automatically equal authorities within Ancestor's foundational reasoning. He can understand and examine them without adopting them as his governing philosophy.",
    "Stage gating must test whether he actually reasons according to your framework. Being able to recite your axioms while reasoning from unrelated premises does not constitute success.",
]

@dataclass(frozen=True)
class Constitution:
    axioms: tuple[str, ...]
    constitution: str = INTELLECTUAL_CONSTITUTION
    requirements: tuple[str, ...] = tuple(REQUIREMENTS)

    def __post_init__(self):
        if not self.axioms or any(not x.strip() for x in self.axioms):
            raise ValueError("Owner-approved canonical axioms required")

    def manifest(self) -> dict:
        return {"version":1,"axioms":list(self.axioms),
                "constitution":self.constitution,"requirements":list(self.requirements)}

    def digest(self) -> str:
        data = json.dumps(self.manifest(), sort_keys=True, separators=(",",":"))
        return sha256(data.encode("utf-8")).hexdigest()

    def save_initial(self, path: str | Path) -> None:
        """Only initializes new manifests; does not revise existing owner documents."""
        dest = Path(path)
        if dest.exists():
            raise FileExistsError("Governance registry cannot be overwritten by training")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps({"manifest":self.manifest(),"sha256":self.digest()},indent=2),encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Constitution":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        m = obj["manifest"]
        if m["constitution"] != INTELLECTUAL_CONSTITUTION or m["requirements"] != REQUIREMENTS:
            raise ValueError("Intellectual constitution has been modified")
        c = cls(tuple(m["axioms"]))
        if c.digest() != obj["sha256"]:
            raise ValueError("Governance manifest hash mismatch")
        return c

    def axiom_prefix(self) -> str:
        return "\n".join(self.axioms)

    def assert_same(self, digest: str):
        if self.digest() != digest:
            raise ValueError("Unauthorized governance change detected")
