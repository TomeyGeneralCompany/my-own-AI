"""Local learned ranking of candidate conclusions against source text."""
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import blake2s
import json
import math
from pathlib import Path
import re
from time import perf_counter
import unicodedata

import torch
from torch import nn
from torch.nn import functional as F

from .module_memory import IDENTIFIER, encode_json
from .modules import hardware_metrics
from .retrieval import STOP, digest
from .train import _atomic_save

FORMAT = "evidence_judge_v1"
WIDTH = 4096
TOKENS = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)
RETAIN = {"before", "after", "only", "while", "when", "without", "instead", "never", "no", "not"}
GENERIC = {"state", "text", "record", "document", "person", "rule", "thing", "fact", "give", "allow"}
# Aliases normalize ordinary language variants before feature hashing.
ALIASES = {
    alias: canonical
    for canonical, alternatives in {
        "approve": "approved approval authorisation authorization clearance signoff",
        "grant": "grants granting granted issue issued issues issuing",
        "authority": "mandate authorised authorized permitted permission delegate delegates delegated delegation",
        "delay": "delays delayed delaying postpones postpone postponed stalls stalled",
        "late": "later overdue belated",
        "before": "prior beforehand earlier",
        "immediately": "instant promptly directly",
        "report": "reports bulletin bulletins notice notices message messages update updates",
        "record": "records recording recorded logs logging logged register registered filing filed",
        "write": "writes wrote written author authored authors compose composed composition drafted",
        "distribute": "distributed distributes distribution republishes circulated delivered delivers",
        "verify": "verification verifies verified checking checked confirm confirms confirmation",
        "independent": "independently separately separate autonomous",
        "replacement": "replacements substitute alternate alternative backup",
        "withdraw": "withdrawal depart departure leave leaving exit",
        "fair": "fairness justice just ethical",
        "cost": "costs expense expenses expenditure outlays",
        "criterion": "criteria threshold thresholds cutoff classification classify",
        "copy": "copies copied duplicate duplicates duplicated",
    }.items()
    for alias in alternatives.split()
}



@dataclass(frozen=True)
class Case:
    identifier: str
    group: str
    question: str
    evidence: tuple[str, ...]
    candidates: tuple[str, ...]
    accepted: int | None


def _tokens(text):
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    return TOKENS.findall(text)


def _key(text):
    return " ".join(_tokens(text))


def _stem(word):
    if word in ALIASES:
        return ALIASES[word]
    for suffix in ("ments", "ment", "ingly", "ing", "edly", "ed", "ies", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[:-len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def _features(text):
    words = _tokens(text)[:150]
    unigrams = [_stem(word) for word in words if word not in STOP or word in RETAIN]
    bigrams = [_stem(first) + " " + _stem(second) for first, second in zip(words, words[1:])
               if first in RETAIN or second in RETAIN or first not in STOP and second not in STOP]
    return unigrams, bigrams


def _slot(key, width):
    hashed = int.from_bytes(blake2s(key.encode(), digest_size=4).digest(), "little")
    return hashed % width, 1 if hashed >> 31 else -1


def _focus(evidence, important, topical):
    words = list(TOKENS.finditer(evidence))
    if len(words) <= 46:
        return evidence
    best = (0.0, 0)
    start = 0
    for offset in range(0, len(words), 12):
        region = words[offset:offset + 46]
        shared = {_stem(match.group().casefold()) for match in region}
        score = 2 * len(shared & important) + .6 * len(shared & topical)
        if score > best[0]:
            best = (score, offset)
            start = offset
    return evidence[words[start].start():words[min(start + 45, len(words) - 1)].end()]


def _rows(question, evidence, candidate, rivals):
    q, qb = _features(question)
    c, cb = _features(candidate)
    unique = set(c) - set().union(*(set(_features(other)[0]) for other in rivals))
    focused = _focus(evidence, unique, set(c + q))
    e, eb = _features(focused)
    feature = torch.zeros(WIDTH, dtype=torch.float32)

    def put(values, offset, width, scale):
        entries = Counter(values)
        factor = scale / math.sqrt(max(1, len(entries)))
        bucket = Counter()
        for value, count in entries.items():
            index, sign = _slot(value, width)
            bucket[offset + index] += sign * factor * min(count, 3)
        if bucket:
            feature[list(bucket)] = torch.tensor(list(bucket.values()))

    put(q + qb, 0, 256, 0.45)
    put(e + eb, 256, 512, 0.55)
    put(c + cb, 768, 512, 0.45)
    put(sorted(set(e) & set(c)), 1280, 512, 1.35)
    put(sorted(set(q) & set(c)), 1792, 256, 0.55)
    left = tuple(dict.fromkeys(e))[:32]
    right = tuple(dict.fromkeys(c))[:24]
    put((x + "|" + y for x in left for y in right), 2048, 1220, 1.20)
    put(sorted(set(e) & unique), 3268, 260, 1.1)
    flags = RETAIN | {"can", "must", "need", "authority", "immediately", "delay", "only", "requires"}
    cue_left = [word for word in e if word in flags]
    cue_right = [word for word in c if word in flags]
    put((x + "|" + y for x in cue_left for y in right), 3528, 252, .70)
    put((x + "|" + y for x in left for y in cue_right), 3780, 252, .70)
    for offset, (first, second) in enumerate(((e, c), (q, c), (e, q), (eb, cb))):
        a, b = set(first), set(second)
        overlap = len(a & b)
        feature[4032 + 3 * offset:4035 + 3 * offset] = torch.tensor([
            overlap / max(1, len(a)), overlap / max(1, len(b)), overlap / max(1, len(a | b))
        ])
    shared = set(e) & unique
    feature[4044:4048] = torch.tensor([
        len(shared) / max(1, len(unique)),
        min(len(shared), 8) / 8,
        bool(set(e) & RETAIN),
        bool(set(c) & RETAIN),
    ]) * 2
    return feature


def load_cases(path):
    with Path(path).open(encoding="utf-8") as stream:
        return parse_cases(json.load(stream))


def parse_cases(items):
    if not isinstance(items, list) or not 1 <= len(items) <= 20_000:
        raise ValueError("Expected a nonempty list of neural cases")
    rows, identifiers = [], set()
    for item in items:
        if not isinstance(item, dict) or set(item) != {"id", "group", "question", "evidence", "candidates", "accepted"}:
            raise ValueError("Invalid neural case fields")
        for field in ("id", "group"):
            if not isinstance(item[field], str) or not IDENTIFIER.fullmatch(item[field]):
                raise ValueError("Invalid neural case identifier")
        if item["id"] in identifiers:
            raise ValueError("Duplicate neural case ID")
        identifiers.add(item["id"])
        question, evidence, candidates, accepted = (item[k] for k in ("question", "evidence", "candidates", "accepted"))
        if not isinstance(question, str) or not 1 <= len(question.strip()) <= 1500:
            raise ValueError("Invalid neural case question")
        for values, max_count, size, minimum in ((evidence, 6, 1500, 1), (candidates, 8, 600, 2)):
            if (not isinstance(values, list) or not minimum <= len(values) <= max_count
                    or any(not isinstance(value, str) or not _key(value) or len(value) > size
                           or "\0" in value for value in values)
                    or len({_key(value) for value in values}) != len(values)):
                raise ValueError("Invalid or duplicate neural case evidence or candidates")
        if accepted is not None and (type(accepted) is not int or not 0 <= accepted < len(candidates)):
            raise ValueError("Accepted candidate index is out of range")
        rows.append(Case(item["id"], item["group"], question, tuple(evidence), tuple(candidates), accepted))
    return rows


def check_split(*partitions):
    questions, groups, identifiers = set(), set(), set()
    for partition in partitions:
        observed = {_key(row.question) for row in partition}
        families = {row.group for row in partition}
        names = {row.identifier for row in partition}
        if observed & questions or families & groups or names & identifiers:
            raise ValueError("Neural case IDs, groups and normalized questions must be disjoint between splits")
        questions.update(observed)
        groups.update(families)
        identifiers.update(names)


class ClaimNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(WIDTH, 72), nn.GELU(), nn.Dropout(.12), nn.Linear(72, 1))

    def forward(self, features):
        return self.network(features).squeeze(-1)


def _encode(case):
    return torch.stack([_rows(case.question, premise, candidate, case.candidates[:i] + case.candidates[i+1:])
                        for i, candidate in enumerate(case.candidates) for premise in case.evidence])


def _availability(case):
    def terms(text):
        return {word for word in _features(text)[0] if len(word) >= 4 and word not in GENERIC}
    candidates = [terms(value) for value in case.candidates]
    evidence = [terms(value) for value in case.evidence]
    topic = terms(case.question)
    return torch.tensor([[bool(claim & passage) and len((claim | topic) & passage) >= 2
                          for passage in evidence] for claim in candidates])


def _labels(case):
    return torch.tensor([float(number == case.accepted) for number in range(len(case.candidates))],
                        dtype=torch.float32)


@dataclass(frozen=True)
class Calibration:
    threshold: float
    margin: float


def _decision(scores, settings):
    values, indices = torch.topk(scores, 2)
    if not torch.isfinite(values[0]) or float(values[0]) < settings.threshold or float(values[0] - values[1]) < settings.margin:
        return None
    return int(indices[0])


def _selection(support, settings):
    scores, sources = support.max(dim=1)
    selected = _decision(scores, settings)
    individual = {_decision(passage, settings) for passage in support.T} - {None}
    if len(individual) > 1:
        selected = None
    return selected, scores, sources


class NeuralJudge:
    def __init__(self, model, settings):
        self.model, self.settings = model.eval(), settings

    @classmethod
    def load(cls, path, threads=4):
        _cpu_threads(threads)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format") != FORMAT:
            raise ValueError("Unsupported neural evidence checkpoint")
        settings = Calibration(**payload["calibration"])
        if not all(math.isfinite(value) for value in asdict(settings).values()) or settings.margin < 0:
            raise ValueError("Invalid calibration")
        model = ClaimNetwork()
        model.load_state_dict(payload["state_dict"])
        return cls(model, settings)

    @torch.inference_mode()
    def rank(self, question, evidence, candidates):
        # The production model receives no rubric, labels or rulebook.
        if not isinstance(evidence, (list, tuple)) or not isinstance(candidates, (list, tuple)):
            raise ValueError("Evidence and candidates must be ordered collections of text")
        request = parse_cases([{"id": "inference", "group": "inference", "question": question,
                                "evidence": list(evidence), "candidates": list(candidates), "accepted": None}])[0]
        support = self.model(_encode(request)).reshape(len(candidates), len(evidence))
        support = support.masked_fill(~_availability(request), float("-inf"))
        selected, scores, sources = _selection(support, self.settings)
        return {"selected": selected, "scores": [float(score) if torch.isfinite(score) else None
                                                   for score in scores],
                "evidence_positions": sources.tolist()}


def _cpu_threads(threads):
    if type(threads) is not int or threads < 1:
        raise ValueError("Thread count must be positive")
    torch.set_num_threads(min(threads, hardware_metrics()["available_cpus"]))


@torch.inference_mode()
def _validate(model, validation, encoded):
    model.eval()
    support = [model(features).reshape(len(row.candidates), len(row.evidence))
               for row, features in zip(validation, encoded)]
    loss = sum(float(F.binary_cross_entropy_with_logits(score.max(dim=1).values, _labels(row)))
               for score, row in zip(support, validation)) / len(validation)
    available = [score.masked_fill(~_availability(row), float("-inf"))
                 for score, row in zip(support, validation)]
    choices = [Calibration(threshold, margin) for threshold in (-3, -2, -1, 0, .5, 1, 2, 3, 4)
               for margin in (.15, .4, .8, 1.4, 2.2)]
    def quality(settings):
        results = [_selection(score, settings)[0] for score in available]
        false_accepts = sum(result is not None and result != row.accepted
                            for result, row in zip(results, validation))
        correct = sum(result == row.accepted for result, row in zip(results, validation))
        return correct - 2 * false_accepts, -false_accepts, correct
    settings = max(choices, key=quality)
    return loss, settings, quality(settings)[2]


def train_judge(training, validation, output, *, held_out=None, steps=300, batch_size=24, seed=42, threads=4):
    inputs = [training, validation] + ([held_out] if held_out else [])
    if Path(output).resolve() in {Path(path).resolve() for path in inputs}:
        raise ValueError("Choose a separate output checkpoint")
    if any(type(value) is not int or value < 1 for value in (steps, batch_size)):
        raise ValueError("Training settings must be positive integers")
    train, valid = load_cases(training), load_cases(validation)
    if sum(len(row.evidence) * len(row.candidates) for row in train + valid) > 14_000:
        raise ValueError("Training and validation exceed the 14,000 passage-claim feature budget")
    check_split(train, valid, load_cases(held_out) if held_out else [])
    for partition in (train, valid):
        if not any(row.accepted is None for row in partition) or not any(row.accepted is not None for row in partition):
            raise ValueError("Include both supported and unresolved examples in each training split")
    _cpu_threads(threads)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    start = perf_counter()
    encoded = [_encode(row) for row in train]
    validation_features = [_encode(row) for row in valid]
    targets = [_labels(row) for row in train]
    model = ClaimNetwork()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.004)
    initial, _, _ = _validate(model, valid, validation_features)
    for _ in range(steps):
        model.train()
        indices = torch.randint(len(train), (batch_size,), generator=generator).tolist()
        sizes = [len(encoded[i]) for i in indices]
        flat = model(torch.cat([encoded[i] for i in indices]))
        scores = [group.reshape(len(train[i].candidates), len(train[i].evidence)).max(dim=1).values
                  for i, group in zip(indices, flat.split(sizes))]
        expected = torch.cat([targets[i] for i in indices])
        loss = F.binary_cross_entropy_with_logits(torch.cat(scores), expected)
        ranking_losses = [-score.log_softmax(dim=0)[train[i].accepted]
                          for i, score in zip(indices, scores) if train[i].accepted is not None]
        if ranking_losses:
            loss = loss + .35 * torch.stack(ranking_losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    final, settings, correct = _validate(model, valid, validation_features)
    report = {"steps": steps, "training_cases": len(train), "validation_cases": len(valid),
              "initial_validation_loss": initial, "validation_loss": final, "validation_correct": correct,
              "calibration": asdict(settings), "seconds": perf_counter() - start, "hardware": hardware_metrics()}
    _atomic_save({"format": FORMAT, "state_dict": model.state_dict(), "calibration": asdict(settings),
                  "data_sha256": blake2s(encode_json([asdict(row) for row in train])).hexdigest(),
                  "training_report": report}, output)
    return report


def evaluate_judge(cases, judge, *, include_scores=False):
    results = []
    for case in cases:
        result = judge.rank(case.question, case.evidence, case.candidates)
        selected = result["selected"]
        row = {"id": case.identifier, "group": case.group, "passed": selected == case.accepted,
               "accepted": case.accepted, "selected": selected}
        if include_scores:
            row["scores"] = result["scores"]
        results.append(row)
    paired = {}
    for row in results:
        paired.setdefault(row["group"], []).append(row)
    # A valid premise switch holds the question and candidate ordering constant.
    # Merely changing the answer label within a group is not a counterfactual test.
    source_cases = {case.identifier: case for case in cases}
    counters = []
    for rows in paired.values():
        by_task = {}
        for row in rows:
            case = source_cases[row["id"]]
            key = (_key(case.question), tuple(_key(value) for value in case.candidates))
            by_task.setdefault(key, []).append((row, case))
        for task_rows in by_task.values():
            if (len({row["accepted"] for row, _ in task_rows}) > 1
                    and len({tuple(_key(value) for value in case.evidence) for _, case in task_rows}) > 1):
                counters.append([row for row, _ in task_rows])
    unresolved = [row for row in results if row["accepted"] is None]
    return {"passed": sum(row["passed"] for row in results), "total": len(results),
            "unresolved_correct": sum(row["passed"] for row in unresolved), "unresolved_total": len(unresolved),
            "wrong_selections": sum(row["selected"] is not None and not row["passed"] for row in results),
            "counterfactual_passed": sum(all(row["passed"] for row in rows) for rows in counters),
            "counterfactual_total": len(counters), "cases": results}


def draft_outline(index, judge, question, candidates, *, limit=5):
    """Prepare a source-backed outline around the selected candidate thesis."""
    if type(limit) is not int or not 1 <= limit <= 6:
        raise ValueError("Neural drafting uses between one and six retrieved passages")
    if not isinstance(candidates, list):
        raise ValueError("A draft brief needs a list of candidate theses")
    parse_cases([{"id": "outline", "group": "outline", "question": question,
                  "evidence": ["Evidence will be retrieved."], "candidates": candidates, "accepted": None}])
    passages = index.search(question, limit)
    if not passages:
        return {"status": "unresolved", "reason": "No matching local passages"}
    result = judge.rank(question, [p["text"] for p in passages], candidates)
    if result["selected"] is None:
        return {"status": "unresolved", "reason": "Neural support was insufficient or conflicting"}
    selected = result["selected"]
    passage = passages[result["evidence_positions"][selected]]
    if digest(passage["text"]) != passage["sha256"]:
        raise ValueError("The selected source failed its integrity check")
    return {"status": "draft_outline", "question": question, "selected": selected,
            "thesis": candidates[selected], "evidence": passage}
