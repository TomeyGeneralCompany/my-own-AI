"""Offline inference over owner-reviewed, source-linked rules."""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
from string import Formatter

from .module_memory import IDENTIFIER
from .reasoning import Stage

LIMIT = 768
NO_PROOF = "No unique source-backed derivation covers this question."


@dataclass(frozen=True)
class Rule:
    identifier: str
    requires: tuple
    produces: tuple
    answer: str | None


@dataclass(frozen=True)
class Derivation:
    citations: tuple = ()
    steps: tuple = ()
    question_matched: bool = False


def _atom(value):
    if (not isinstance(value, list) or not value or len(value) > 6
            or any(not isinstance(item, str) or not item or len(item) > 180 for item in value)
            or not IDENTIFIER.fullmatch(value[0])):
        raise ValueError("Invalid rulebook atom")
    return tuple(value)


def _variables(atom):
    return {item[1:] for item in atom[1:] if item.startswith("$")}


def _text(value):
    return " ".join(value.casefold().split())


def _bind(pattern, fact, environment):
    bound = environment.copy()
    for key, value in zip(pattern[1:], fact[1:]):
        if key.startswith("$"):
            key = key[1:]
            if key in bound and bound[key] != value:
                return None
            bound[key] = value
        elif _text(key) != value:
            return None
    return bound


class Rulebook:
    """Rule bodies unify over observed facts and derive finite, auditable conclusions."""

    @classmethod
    def load(cls, path):
        with Path(path).open(encoding="utf-8") as handle:
            return cls(json.load(handle))

    def __init__(self, spec):
        if (not isinstance(spec, dict) or set(spec) != {"version", "extractors", "rules"}
                or type(spec["version"]) is not int or spec["version"] != 1):
            raise ValueError("Unsupported rulebook schema")
        self.digest = hashlib.sha256(json.dumps(spec, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if (not isinstance(spec["extractors"], list) or not 1 <= len(spec["extractors"]) <= 192
                or not isinstance(spec["rules"], list) or not 1 <= len(spec["rules"]) <= 96):
            raise ValueError("Invalid rulebook size")
        self.extractors = []
        self.rules = []
        names = set()
        arity = {}
        origins = {}

        def register(value):
            atom = _atom(value)
            if arity.setdefault(atom[0], len(atom)) != len(atom):
                raise ValueError(f"Inconsistent arity for {atom[0]}")
            return atom

        for item in spec["extractors"]:
            if (not isinstance(item, dict) or set(item) - {"input", "pattern", "fact", "source"}
                    or item.get("input") not in ("question", "evidence")
                    or not isinstance(item.get("pattern"), str) or not 1 <= len(item["pattern"]) <= 1024
                    or "fact" not in item):
                raise ValueError("Invalid rulebook extractor")
            try:
                pattern = re.compile(item["pattern"], re.IGNORECASE)
            except re.error as exc:
                raise ValueError(f"Invalid rulebook pattern: {exc}") from exc
            atom = register(item["fact"])
            source = item.get("source")
            if source is not None and (item["input"] != "evidence" or not isinstance(source, str) or not source):
                raise ValueError("Source selectors apply only to evidence extractors")
            if not _variables(atom) <= set(pattern.groupindex):
                raise ValueError("Fact variables must name extractor captures")
            if origins.setdefault(atom[0], item["input"]) != item["input"]:
                raise ValueError("Use distinct predicates for question and evidence observations")
            self.extractors.append((item["input"], pattern, atom, source))
        for item in spec["rules"]:
            if (not isinstance(item, dict) or set(item) - {"id", "requires", "produces", "answer"}
                    or not isinstance(item.get("id"), str) or not IDENTIFIER.fullmatch(item["id"])
                    or item["id"] in names or not isinstance(item.get("requires"), list)
                    or not 1 <= len(item["requires"]) <= 8 or "produces" not in item):
                raise ValueError("Invalid or duplicate inference rule")
            requires = tuple(map(register, item["requires"]))
            output = register(item["produces"])
            variables = set().union(*map(_variables, requires))
            answer = item.get("answer")
            if not _variables(output) <= variables:
                raise ValueError("Rule output variables must occur in its premises")
            if answer is not None:
                if not isinstance(answer, str) or not 16 <= len(answer) <= 2000 or "\n" in answer:
                    raise ValueError("Rule answers must be a bounded paragraph")
                fields = list(Formatter().parse(answer))
                if (not answer.strip() or any(spec or conversion or field not in variables
                       for _, field, spec, conversion in fields if field is not None)):
                    raise ValueError("Answer templates may only reference bound variables")
            names.add(item["id"])
            self.rules.append(Rule(item["id"], requires, output, answer))

        self.extractors = tuple(self.extractors)
        self.rules = tuple(self.rules)
        self._producers = {}
        for rule in self.rules:
            if rule.produces[0] in origins:
                raise ValueError("Derived predicates must differ from observed predicates")
            self._producers.setdefault(rule.produces[0], []).append(rule.requires)

        visiting, checked = set(), set()

        def check(predicate):
            if predicate in visiting:
                raise ValueError("Rules may not have circular dependencies")
            if predicate not in checked:
                visiting.add(predicate)
                for body in self._producers.get(predicate, ()):
                    for item in body:
                        check(item[0])
                visiting.remove(predicate)
                checked.add(predicate)

        for predicate in self._producers:
            check(predicate)

    def search_support(self, question, index, limit):
        """Follow the premises of question-scoped rules to source-bound index entries."""
        question_facts = {fact[0] for fact in self._observations(question, [])}
        evidence = {}
        for origin, pattern, atom, source in self.extractors:
            if origin == "evidence":
                evidence.setdefault(atom[0], []).append((source, pattern))

        @lru_cache(maxsize=None)
        def dependencies(predicate):
            if predicate in question_facts:
                return set()
            supported = list(evidence.get(predicate, ()))
            feasible = predicate in evidence
            for body in self._producers.get(predicate, ()):
                premises = [dependencies(atom[0]) for atom in body]
                if all(premise is not None for premise in premises):
                    feasible = True
                    supported.extend(set().union(*premises))
            return set(supported) if feasible else None

        selectors = set()
        for rule in self.rules:
            if rule.answer is None:
                continue
            premises = [dependencies(atom[0]) for atom in rule.requires]
            if all(premise is not None for premise in premises):
                selectors.update(set().union(*premises))
        results = {}
        for origin, pattern, _, source in self.extractors:
            if source is not None and (source, pattern) in selectors:
                hit = index.find_support(source, pattern)
                if hit:
                    results.setdefault((hit["source"], hit["locator"]), hit)
                    if len(results) == limit:
                        break
        return list(results.values())

    def _observations(self, question, passages):
        facts = {}
        for origin, pattern, atom, source in self.extractors:
            texts = [(0, question)] if origin == "question" else [
                (number, passage["text"]) for number, passage in enumerate(passages, 1)
                if source is None or source == passage["source"]
            ]
            variables = _variables(atom)
            for number, text in texts:
                matches = pattern.finditer(text) if variables else (pattern.search(text),)
                for match in matches:
                    if match is None:
                        continue
                    binding = match.groupdict()
                    if any(not binding.get(variable) for variable in variables):
                        continue
                    terms = tuple(_text(binding[item[1:]] if item.startswith("$") else item)
                                  for item in atom[1:])
                    citation = ()
                    if number:
                        # Keep the indexed passage intact for verification and audit.
                        citation = ((number, text),)
                    facts.setdefault((atom[0], *terms), Derivation(citation, question_matched=not number))
                    if len(facts) >= LIMIT:
                        raise ValueError("Too many rulebook observations")
        return facts

    def _joins(self, requires, facts, budget, bindings=None, derivation=Derivation()):
        if not requires:
            yield bindings, derivation
            return
        premise, *rest = requires
        for fact, previous in facts.get(premise[0], ()):
            budget[0] -= 1
            if budget[0] < 0:
                raise ValueError("Rulebook derivation exceeded its join limit")
            bound = _bind(premise, fact, bindings or {})
            if bound is not None:
                yield from self._joins(rest, facts, budget, bound, Derivation(
                    tuple(dict.fromkeys(derivation.citations + previous.citations)),
                    tuple(dict.fromkeys(derivation.steps + previous.steps)),
                    derivation.question_matched or previous.question_matched))

    def resolve(self, question, passages):
        observations = self._observations(question, passages)
        answers = {}
        facts = {(fact, proof.question_matched, bool(proof.citations)): proof
                 for fact, proof in observations.items()}
        budget = [40_000]
        while True:
            by_predicate = {}
            for (fact, _, _), proof in facts.items():
                by_predicate.setdefault(fact[0], []).append((fact, proof))
            updated = {}
            for rule in self.rules:
                for bound, proof in self._joins(rule.requires, by_predicate, budget):
                    fact = (rule.produces[0], *(_text(bound[item[1:]] if item.startswith("$") else item)
                                                for item in rule.produces[1:]))
                    trace = Derivation(proof.citations, (*proof.steps, rule.identifier), proof.question_matched)
                    provenance = (fact, trace.question_matched, bool(trace.citations))
                    if provenance not in facts:
                        updated.setdefault(provenance, trace)
                    if rule.answer is not None and proof.citations and proof.question_matched:
                        answer = rule.answer.format(**bound)
                        answers.setdefault(answer, trace)
                    if len(facts) + len(updated) >= LIMIT or len(answers) >= LIMIT:
                        raise ValueError("Rulebook derivation exceeded its fact limit")
            if not updated:
                break
            facts.update(updated)
        if len(answers) != 1:
            return NO_PROOF, ()
        answer, proof = next(iter(answers.items()))
        quotes = "\n".join(f"- [{number}] {text}" for number, text in proof.citations)
        return f"Premises:\n{quotes}\nInference: {answer}", proof.steps


class RulebookProposer:
    """Use an explicitly configured rulebook for Research and the existing model elsewhere."""

    def __init__(self, book, fallback=None):
        self.book, self.fallback = book, fallback

    def propose(self, request):
        if request.operation != "grounded_analysis" or request.context.get("rulebook") != self.book.digest:
            if self.fallback is None:
                raise ValueError("A neural checkpoint is required for this operation")
            return self.fallback.propose(request)
        answer, steps = self.book.resolve(request.objective, request.context["evidence"])
        if request.stage == Stage.FORMULATE:
            return "Apply and verify source-linked rules: " + ", ".join(steps) if steps else NO_PROOF
        return answer
