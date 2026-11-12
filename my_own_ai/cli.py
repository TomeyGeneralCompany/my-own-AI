"""Local model training, ingestion, benchmarking and module-switch commands."""
import argparse
import json
import sys
from pathlib import Path
from .model import ModelConfig
from .train import train, load_model, benchmark
from .corpus import build_manifest
from .modules import CheckpointStore, ModuleController, default_modules
from .session import configure_session, demo, load_proposer, session
from .corpus_eval import evaluate_cases, load_cases
from .retrieval import CorpusIndex, build_index
from .proposal_training import train_proposals
from .grounding import SPAN_FORMAT
from . import neural_judge
from .rulebook import Rulebook, RulebookProposer
from .reasoning import SUPERVISED_FORMAT

def main(argv=None):
    parser = argparse.ArgumentParser(prog="my-own-ai")
    sub = parser.add_subparsers(dest="command", required=True)
    i = sub.add_parser("ingest")
    i.add_argument("--corpus", type=Path, required=True)
    i.add_argument("--manifest", type=Path, default=Path("data/corpus-manifest.json"))
    t = sub.add_parser("train")
    t.add_argument("--corpus", type=Path, required=True)
    t.add_argument("--output", type=Path, default=Path("data/checkpoints/brain.pt"))
    t.add_argument("--steps", type=int, default=20, help="Total target steps, including resumed steps")
    t.add_argument("--context", type=int, default=64)
    t.add_argument("--embed", type=int, default=64)
    t.add_argument("--heads", type=int, default=4)
    t.add_argument("--layers", type=int, default=2)
    t.add_argument("--batch", type=int, default=4)
    t.add_argument("--lr", type=float, default=.001)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--checkpoint-every", type=int, default=10)
    t.add_argument("--resume", action="store_true")
    t.add_argument("--schedule-steps", type=int, default=1000)
    t.add_argument("--warmup-steps", type=int, default=10)
    b = sub.add_parser("benchmark")
    b.add_argument("--checkpoint", type=Path, required=True)
    b.add_argument("--iterations", type=int, default=10)
    b.add_argument("--tokens", type=int, default=16)
    g = sub.add_parser("generate")
    g.add_argument("--checkpoint", type=Path, required=True)
    g.add_argument("--prompt", required=True)
    g.add_argument("--tokens", type=int, default=60)
    s = sub.add_parser("switch")
    s.add_argument("--mode", choices=["conversation", "writing", "coding", "research"], required=True)
    s.add_argument("--memory", type=Path, default=Path("data/module-memory"))
    configure_session(sub.add_parser("session", help="Run a local JSON-lines controller session"))
    sub.add_parser("demo", help="Demonstrate local reasoning, switching and checkpoint restoration")
    index = sub.add_parser("index-corpus", help="Build an offline passage index")
    index.add_argument("--corpus", type=Path, required=True)
    index.add_argument("--index", type=Path, required=True)
    search = sub.add_parser("search-corpus", help="Search the local passage index")
    search.add_argument("--index", type=Path, required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--limit", type=int, default=5)
    evaluate = sub.add_parser("evaluate-corpus", help="Measure grounded reasoning on held-out cases")
    evaluate.add_argument("--index", type=Path, required=True)
    evaluate.add_argument("--cases", type=Path, required=True)
    route = evaluate.add_mutually_exclusive_group(required=True)
    route.add_argument("--checkpoint", type=Path)
    route.add_argument("--rulebook", type=Path)
    evaluate.add_argument("--threads", type=int, default=4)
    evaluate.add_argument("--tokens", type=int, default=256)
    evaluate.add_argument("--limit", type=int, default=5)
    evaluate.add_argument("--max-corrections", type=int, default=2)
    evaluate.add_argument("--include-answers", action="store_true",
                          help="Include private answer text and source locators for review")
    proposals = sub.add_parser("train-proposals", help="Train on worked reasoning proposals")
    proposals.add_argument("--checkpoint", type=Path, required=True)
    proposals.add_argument("--examples", type=Path, required=True)
    proposals.add_argument("--validation", type=Path, required=True)
    proposals.add_argument("--held-out", type=Path)
    proposals.add_argument("--output", type=Path, required=True)
    proposals.add_argument("--steps", type=int, default=200)
    proposals.add_argument("--batch", type=int, default=4)
    proposals.add_argument("--lr", type=float, default=.001)
    proposals.add_argument("--seed", type=int, default=42)
    proposals.add_argument("--threads", type=int, default=4)
    proposals.add_argument("--format", choices=(SUPERVISED_FORMAT, SPAN_FORMAT), default=SUPERVISED_FORMAT)
    judge_training = sub.add_parser("train-neural-judge", help="Train a local evidence-conditioned claim ranker")
    judge_training.add_argument("--examples", type=Path, required=True)
    judge_training.add_argument("--validation", type=Path, required=True)
    judge_training.add_argument("--held-out", type=Path)
    judge_training.add_argument("--output", type=Path, required=True)
    judge_training.add_argument("--steps", type=int, default=300)
    judge_training.add_argument("--threads", type=int, default=4)
    judge_evaluation = sub.add_parser("evaluate-neural-judge", help="Test evidence-sensitive claim ranking")
    judge_evaluation.add_argument("--checkpoint", type=Path, required=True)
    judge_evaluation.add_argument("--cases", type=Path, required=True)
    judge_evaluation.add_argument("--threads", type=int, default=4)
    judge_evaluation.add_argument("--include-scores", action="store_true")
    draft = sub.add_parser("draft-neural-outline", help="Prepare an offline outline around a supported thesis")
    draft.add_argument("--checkpoint", type=Path, required=True)
    draft.add_argument("--index", type=Path, required=True)
    draft.add_argument("--brief", type=Path, required=True)
    draft.add_argument("--limit", type=int, default=5)
    draft.add_argument("--threads", type=int, default=4)
    a = parser.parse_args(argv)
    if a.command == "ingest":
        m = build_manifest(a.corpus, a.manifest)
        print(json.dumps({"documents": len(m["documents"]), "fingerprint": m["fingerprint"]}, indent=2))
    elif a.command == "train":
        cfg = ModelConfig(context_size=a.context, embed_dim=a.embed, heads=a.heads, layers=a.layers)
        print(json.dumps(train(a.corpus, a.output, cfg, a.steps, a.batch, a.lr, a.seed,
                               a.resume, a.checkpoint_every, schedule_steps=a.schedule_steps,
                               warmup_steps=a.warmup_steps), indent=2))
    elif a.command == "benchmark":
        print(json.dumps(benchmark(a.checkpoint, a.iterations, generation_tokens=a.tokens), indent=2))
    elif a.command == "generate":
        print(load_model(a.checkpoint).generate(a.prompt.encode(), a.tokens).decode("utf-8", errors="replace"))
    elif a.command == "session":
        session(a, sys.stdin, sys.stdout)
    elif a.command == "demo":
        print(json.dumps(demo(), indent=2))
    elif a.command == "index-corpus":
        print(json.dumps(build_index(a.corpus, a.index), indent=2))
    elif a.command == "search-corpus":
        with CorpusIndex(a.index) as corpus:
            print(json.dumps(corpus.search(a.query, a.limit), indent=2))
    elif a.command == "train-proposals":
        report = train_proposals(a.checkpoint, a.examples, a.validation, a.output, held_out=a.held_out,
                                 steps=a.steps, batch_size=a.batch, learning_rate=a.lr, seed=a.seed,
                                 threads=a.threads, format_name=a.format)
        print(json.dumps(report, indent=2))
    elif a.command == "train-neural-judge":
        report = neural_judge.train_judge(a.examples, a.validation, a.output, held_out=a.held_out,
                                          steps=a.steps, threads=a.threads)
        print(json.dumps(report, indent=2))
    elif a.command == "evaluate-neural-judge":
        judge = neural_judge.NeuralJudge.load(a.checkpoint, a.threads)
        report = neural_judge.evaluate_judge(neural_judge.load_cases(a.cases), judge,
                                            include_scores=a.include_scores)
        print(json.dumps(report, indent=2))
        return int(report["passed"] != report["total"])
    elif a.command == "draft-neural-outline":
        with a.brief.open(encoding="utf-8") as stream:
            brief = json.load(stream)
        if not isinstance(brief, dict) or set(brief) != {"question", "candidates"}:
            raise ValueError("Expected a question and candidate theses in the brief")
        judge = neural_judge.NeuralJudge.load(a.checkpoint, a.threads)
        with CorpusIndex(a.index) as index:
            result = neural_judge.draft_outline(index, judge, **brief, limit=a.limit)
        print(json.dumps(result, indent=2))
        return int(result["status"] != "draft_outline")
    elif a.command == "evaluate-corpus":
        cases = load_cases(a.cases)
        book = Rulebook.load(a.rulebook) if a.rulebook else None
        engine = RulebookProposer(book) if book is not None else load_proposer(a.checkpoint, a.threads, a.tokens)
        report = evaluate_cases(a.index, cases, engine, limit=a.limit,
                                max_corrections=a.max_corrections, include_answers=a.include_answers, rulebook=book)
        print(json.dumps(report, indent=2))
        return int(report["passed"] != report["total"])
    else:
        with ModuleController(CheckpointStore(a.memory), default_modules()) as controller:
            controller.switch(a.mode)
            print("Active module:", controller.active.name)

if __name__ == "__main__":
    raise SystemExit(main())
