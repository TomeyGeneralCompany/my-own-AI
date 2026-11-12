"""Built-in procedures and deterministic checks for the four local modules."""
import ast
import hashlib
from pathlib import Path
import re

from .corpus import SUPPORTED, extract_text
from .modules import BasicModule, Manifest, Tool
from .reasoning import Evaluation, Stage
from .grounding import evidence_terms
from .retrieval import CorpusIndex, PASSAGE_LIMIT, digest


def _procedures(formulate, construct, evaluate, correct):
    return dict(zip(Stage, (formulate, construct, evaluate, correct)))


def _options(inputs, allowed):
    if inputs.keys() - allowed:
        raise ValueError("Unknown operation options: " + ", ".join(sorted(inputs.keys() - allowed)))


def _text_options(inputs, scalar=(), lists=()):
    for key in scalar:
        if key in inputs and (not isinstance(inputs[key], str) or not inputs[key].strip()):
            raise ValueError(f"{key} must be nonempty text")
    for key in lists:
        values = inputs.get(key, [])
        if (not isinstance(values, list) or len(values) > 16
                or any(not isinstance(value, str) or not value.strip() for value in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"{key} must contain up to 16 distinct nonempty strings")


def _check_text(candidate, required=(), min_chars=12):
    if len(candidate.strip()) < min_chars or any(not c.isprintable() and c not in "\t\n\r" for c in candidate):
        return Evaluation(False, f"Expected printable text with at least {min_chars} characters")
    missing = [term for term in required if term.casefold() not in candidate.casefold()]
    if missing:
        return Evaluation(False, "Missing required text: " + ", ".join(missing))
    return Evaluation(True, "Output meets the configured text requirements")


class LocalFiles(BasicModule):
    exportable = True

    def __init__(self, name, workspace):
        super().__init__(name)
        self.workspace = Path(workspace).resolve()

    def _file(self, name, suffixes):
        relative = Path(name)
        path = (self.workspace / relative).resolve()
        if (relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(self.workspace)
                or path.suffix.lower() not in suffixes or not path.is_file()):
            raise ValueError(f"Unsupported or unavailable workspace file: {name}")
        if path.stat().st_size > 1024 * 1024:
            raise ValueError(f"Workspace file exceeds 1 MiB: {name}")
        return path

    def read_document(self, name):
        text = extract_text(self._file(name, SUPPORTED))
        if not text.strip() or len(text) > 64_000:
            raise ValueError(f"Expected 1–64,000 characters of extracted text: {name}")
        return {"source": name, "text": text, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


class Conversation(BasicModule):
    manifest = Manifest("conversation", memory_mib=1, operations=("reply",))
    procedures = _procedures("Plan a reply using relevant dialogue and personal context",
                             "Write the conversational response", "Check the response requirements",
                             "Revise the response using evaluation feedback")

    def __init__(self):
        super().__init__("conversation")

    def validate_inputs(self, operation, inputs):
        _options(inputs, {"required_terms"})
        _text_options(inputs, lists=("required_terms",))

    def prepare(self, run, tools, prior):
        return {"personal_context": tools.shared(), "last_exchange": prior}

    def evaluate(self, run, tools):
        return _check_text(run["candidate"], run["inputs"].get("required_terms", ()))


class Writing(LocalFiles):
    manifest = Manifest("writing", memory_mib=4, tools=(Tool("read_document", (Stage.FORMULATE,)),),
                        operations=("manuscript",))
    procedures = _procedures("Outline the manuscript against the objective and references",
                             "Draft the manuscript as Markdown", "Check manuscript structure and required sections",
                             "Revise the manuscript against the review feedback")

    def __init__(self, workspace):
        super().__init__("writing", workspace)

    def tool_handlers(self):
        return {"read_document": self.read_document}

    def validate_inputs(self, operation, inputs):
        _options(inputs, {"reference", "headings"})
        _text_options(inputs, scalar=("reference",), lists=("headings",))

    def prepare(self, run, tools, prior):
        context = super().prepare(run, tools, prior)
        if reference := run["inputs"].get("reference"):
            context["reference"] = tools.call("read_document", name=reference)
        return context

    def evaluate(self, run, tools):
        candidate = run["candidate"]
        result = _check_text(candidate, min_chars=24)
        if not result.passed:
            return result
        headings = re.findall(r"^#{1,6}[ \t]+(.+?)\s*$", candidate, re.MULTILINE)
        required = run["inputs"].get("headings", [])
        missing = [heading for heading in required if heading not in headings]
        if missing or not headings:
            return Evaluation(False, "Expected Markdown headings: " + (", ".join(missing) or "at least one"))
        return Evaluation(True, f"Found {len(headings)} Markdown headings and all requested sections")


class Coding(LocalFiles):
    manifest = Manifest("coding", memory_mib=4,
                        tools=(Tool("read_source", (Stage.FORMULATE,)), Tool("check_python", (Stage.EVALUATE,))),
                        operations=("python_source",))
    procedures = _procedures("Plan a Python source change and its acceptance checks",
                             "Propose complete Python source without Markdown fences",
                             "Compile Python and check the requested declarations and literal assignments",
                             "Debug the source using compiler or acceptance-check feedback")

    def __init__(self, workspace):
        super().__init__("coding", workspace)

    def read_source(self, name):
        with self._file(name, {".py"}).open(encoding="utf-8-sig") as handle:
            text = handle.read(64_001)
        if len(text) > 64_000:
            raise ValueError("Source exceeds 64,000 characters")
        return text

    def check_python(self, source, declarations, literals):
        try:
            compile(source, "<proposal>", "exec")
            tree = ast.parse(source)
        except (SyntaxError, ValueError, RecursionError, OverflowError) as exc:
            return {"passed": False, "evidence": f"Python compile check failed: {exc}"}
        names = {node.name for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        assignments = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assignments[target.id] = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                assignments[node.target.id] = node.value
        missing = sorted(set(declarations) - names)
        mismatched = [name for name, value in literals.items()
                      if name not in assignments or not isinstance(assignments[name], ast.Constant)
                      or type(assignments[name].value) is not type(value) or assignments[name].value != value]
        if missing or mismatched:
            return {"passed": False, "evidence": f"Missing declarations {missing}, literal mismatches {mismatched}"}
        return {"passed": True, "evidence": "Python compiled and all configured static checks passed"}

    def tool_handlers(self):
        return {"read_source": self.read_source, "check_python": self.check_python}

    def validate_inputs(self, operation, inputs):
        _options(inputs, {"source", "declarations", "literals"})
        _text_options(inputs, scalar=("source",), lists=("declarations",))
        declarations, literals = inputs.get("declarations", []), inputs.get("literals", {})
        if (not isinstance(literals, dict) or len(literals) > 32
                or any(not key.isidentifier() or type(value) not in (str, int, float, bool, type(None))
                       for key, value in literals.items()) or any(not name.isidentifier() for name in declarations)):
            raise ValueError("Expected Python declaration names and scalar literal checks")
        if not declarations and not literals:
            raise ValueError("Coding requires at least one declaration or literal acceptance check")

    def prepare(self, run, tools, prior):
        context = super().prepare(run, tools, prior)
        context["checks"] = run["inputs"]
        if source := run["inputs"].get("source"):
            context["source"] = tools.call("read_source", name=source)
        return context

    def evaluate(self, run, tools):
        return Evaluation(**tools.call("check_python", source=run["candidate"],
                                       declarations=run["inputs"].get("declarations", []),
                                       literals=run["inputs"].get("literals", {})))


class Research(LocalFiles):
    manifest = Manifest("research", memory_mib=16,
                        tools=(Tool("acquire_source", (Stage.FORMULATE,)),
                               Tool("search_corpus", (Stage.FORMULATE,))),
                        operations=("evidence_notes", "grounded_analysis"))
    procedures = _procedures("Plan a source analysis using the acquired document excerpts",
                             "Write evidence notes including an exact quote on a line '- [n] quote' for each source",
                             "Verify each evidence quote against the acquired source text",
                             "Correct missing or inaccurate quotes using the evaluation feedback")

    grounded_procedures = _procedures("Plan an inference using the numbered passages",
                                     "Use Premises: then - [n] exact quote, then Inference: and your reasoning",
                                     "Verify quoted premises and check inference structure",
                                     "Fix citation or inference defects. Use Premises: and Inference:")

    def __init__(self, workspace, corpus_index=None, rulebook=None):
        super().__init__("research", workspace)
        self.index_path = corpus_index
        self.index = None
        self.rulebook = rulebook

    def procedure(self, operation, stage):
        if operation == "grounded_analysis":
            return self.grounded_procedures[stage]
        return super().procedure(operation, stage)

    def load(self, resources):
        if self.index_path is not None:
            self.index = resources.enter(CorpusIndex(self.index_path))

    def deactivate(self):
        self.index = None
        super().deactivate()

    def search(self, query, limit):
        if self.index is None:
            raise ValueError("Configure a local corpus index with --corpus-index")
        hits = self.index.search(query, limit)
        if self.rulebook is None:
            return hits
        support = self.rulebook.search_support(query, self.index, limit)
        distinct = {}
        for hit in support + hits:
            distinct.setdefault((hit["source"], hit["locator"]), hit)
        return list(distinct.values())[:limit]

    def tool_handlers(self):
        return {"acquire_source": self.read_document, "search_corpus": self.search}

    def validate_inputs(self, operation, inputs):
        if operation == "grounded_analysis":
            _options(inputs, {"limit"})
            limit = inputs.get("limit", 5)
            if type(limit) is not int or not 1 <= limit <= 12:
                raise ValueError("Search limit must be between 1 and 12")
            return
        _options(inputs, {"sources"})
        _text_options(inputs, lists=("sources",))
        if not inputs.get("sources"):
            raise ValueError("Research requires one or more local sources")

    def prepare(self, run, tools, prior):
        context = super().prepare(run, tools, prior)
        if run["operation"] == "grounded_analysis":
            if self.rulebook is not None:
                context["rulebook"] = self.rulebook.digest
            context["evidence"] = tools.call("search_corpus", query=run["objective"],
                                             limit=run["inputs"].get("limit", 5))
        else:
            context["sources"] = [tools.call("acquire_source", name=name) for name in run["inputs"]["sources"]]
        return context

    def validate_context(self, operation, inputs, context):
        if operation == "grounded_analysis":
            identity = context.get("rulebook")
            if identity is not None and (not isinstance(identity, str) or not re.fullmatch("[0-9a-f]{64}", identity)):
                raise ValueError("Invalid saved rulebook identity")
            evidence = context.get("evidence")
            if not isinstance(evidence, list) or len(evidence) > inputs.get("limit", 5):
                raise ValueError("Invalid retrieved evidence")
            for source in evidence:
                if (not isinstance(source, dict) or set(source) != {"source", "locator", "text", "sha256"}
                        or not all(isinstance(value, str) and value for value in source.values())
                        or len(source["text"]) > PASSAGE_LIMIT or digest(source["text"]) != source["sha256"]):
                    raise ValueError("Invalid corpus passage checkpoint")
            return
        sources = context.get("sources")
        if not isinstance(sources, list) or len(sources) != len(inputs["sources"]):
            raise ValueError("Research checkpoint is missing acquired sources")
        for expected, source in zip(inputs["sources"], sources):
            if (not isinstance(source, dict) or set(source) != {"source", "text", "sha256"}
                    or source["source"] != expected or not isinstance(source["text"], str)
                    or not source["text"].strip() or len(source["text"]) > 64_000
                    or hashlib.sha256(source["text"].encode("utf-8")).hexdigest() != source["sha256"]):
                raise ValueError("Invalid acquired source in research checkpoint")

    def validate_execution(self, operation, context):
        identity = context.get("rulebook")
        if identity is not None and (self.rulebook is None or self.rulebook.digest != identity):
            raise ValueError("The original rulebook is required to resume this grounded operation")

    def render_grounded(self, proposal, context):
        evidence = context.get("evidence", [])
        index = proposal.passage
        if not (1 <= index <= len(evidence) and len(proposal.quote) >= 8
                and proposal.quote in evidence[index - 1]["text"]):
            return "No verified evidence span was selected."
        inference = proposal.inference.strip()
        if (not proposal.finished or not inference.endswith((".", "?", "!"))
                or "\n" in inference or any(not character.isprintable() for character in inference)):
            return f"Unfinished inference: {inference}"
        if not evidence_terms(inference) & evidence_terms(proposal.quote):
            return f"Inference lacks a term from the selected evidence: {inference}"
        return f"Premises:\n- [{index}] {proposal.quote}\nInference: {inference}"

    def evaluate(self, run, tools):
        if run["operation"] == "grounded_analysis":
            if run["context"].get("rulebook") is not None:
                result, steps = self.rulebook.resolve(run["objective"], run["context"]["evidence"])
                if not steps:
                    return Evaluation(False, result)
                if run["candidate"] != result:
                    return Evaluation(False, "The proposal differs from the verified rule derivation")
                verified, _, error = grounded_evidence(result, run["context"]["evidence"])
                return Evaluation(not error, error or "Verified rules: " + ", ".join(steps))
            if run["candidate"].startswith("Inference lacks a term from the selected evidence:"):
                return Evaluation(False, "Relate the inference to a specific term in the selected evidence")
            if run["candidate"].startswith("Unfinished inference:"):
                return Evaluation(False, "Finish the inference with a complete sentence")
            verified, _, error = grounded_evidence(run["candidate"], run["context"]["evidence"])
            return Evaluation(not error, error or f"Verified {len(verified)} quoted premises and an inference section")
        sources = run["context"]["sources"]
        verified = set()
        for line in run["candidate"].splitlines():
            if match := re.fullmatch(r"- \[([1-9][0-9]{0,2})\] (.+)", line):
                number = int(match[1])
                quote = match[2].strip()
                if not (number <= len(sources) and len(quote) >= 8 and quote in sources[number - 1]["text"]):
                    return Evaluation(False, f"Unverified evidence quote for source {number}")
                verified.add(number)
            elif line.lstrip().startswith("- ["):
                return Evaluation(False, "Evidence bullets must use '- [n] exact quote'")
        missing = sorted(set(range(1, len(sources) + 1)) - verified)
        if missing:
            return Evaluation(False, f"Missing exact quotes for sources {missing}")
        return Evaluation(True, f"Verified exact evidence quotes for {len(verified)} local sources")


def grounded_evidence(candidate, sources):
    """Return verified citation numbers, inference text and a deterministic format error."""
    if not sources:
        return set(), "", "No matching passages in the local corpus"
    heading = re.fullmatch(r"\s*Premises:[ \t]*\n(.+?)\nInference:[ \t]*(.+)\s*", candidate, re.DOTALL)
    if not heading or len(heading[2].strip()) < 16:
        return set(), "", "Use Premises: with exact quote bullets followed by a substantive Inference: section"
    verified = set()
    for line in heading[1].splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"- \[([1-9][0-9]?)\] (.+)", line)
        if not match:
            return set(), "", "Premise bullets must use '- [n] exact quote'"
        number, quote = int(match[1]), match[2].strip()
        if number > len(sources) or len(quote) < 8 or quote not in sources[number - 1]["text"]:
            return set(), "", f"Unverified premise quote for passage {number}"
        verified.add(number)
    if not verified:
        return set(), "", "Include at least one verified premise"
    inference = heading[2].strip()
    references = set(re.findall(r"\[([0-9]+)\]", inference))
    if references - {str(number) for number in verified}:
        return set(), "", "Inference refers to a passage without a verified premise quote"
    return verified, inference, ""
