"""Local model training, ingestion, benchmarking and module-switch commands."""
import argparse
import json
import sys
from pathlib import Path
from .model import ModelConfig
from .tokenizer import ByteBPETokenizer
from .language_training import train_language
from .language_inference import answer_question
from .language_evaluation import load_gauntlet, evaluate_english
from .language_corpus import prepare_english_corpus
from .language_regression import promote_english
from .train import train, load_model, benchmark
from .corpus import build_manifest
from .modules import CheckpointStore, ModuleController, default_modules
from .session import configure_session, demo, load_proposer, session
from .corpus_eval import evaluate_cases, load_cases
from .retrieval import CorpusIndex, build_index
from .proposal_training import train_proposals
from .intellectual_ledger import register, decide, approved_sources, stage_approved_corpus
from .cumulative_learning import prepare_replay, gate_and_promote, capture_successful_reasoning
from .relational_curriculum import prepare_curriculum, evaluate_relational, load_relational_cases, check_relational_splits
from .grounding import SPAN_FORMAT
from . import neural_judge
from .autonomous_inference import GeneratingProposer, infer
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
    t.add_argument("--init-checkpoint", type=Path, help="Start new corpus generation from previous model weights")
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
    proposals.add_argument("--pair-batches", action="store_true", help="Keep both counterfactual variants together in updates")
    proposals.add_argument("--resume", action="store_true", help="Resume identical data/settings from saved output optimizer state")
    proposals.add_argument("--checkpoint-every", type=int, default=25)
    works = sub.add_parser("register-work", help="Register a source in the local intellectual ledger")
    works.add_argument("--ledger", type=Path, required=True)
    works.add_argument("--id", required=True)
    works.add_argument("--kind", choices=["authored_work", "derived_work", "reasoning"], required=True)
    works.add_argument("--source", type=Path, required=True)
    works.add_argument("--origin", required=True)
    works.add_argument("--evidence", action="append", default=[])
    approval = sub.add_parser("decide-work", help="Owner approves or rejects one pending work")
    approval.add_argument("--ledger", type=Path, required=True)
    approval.add_argument("--id", required=True)
    approval.add_argument("--decision", choices=["approved", "rejected"], required=True)
    inventory = sub.add_parser("approved-works", help="Verify approved source digests")
    inventory.add_argument("--ledger", type=Path, required=True)
    approved_corpus = sub.add_parser("stage-approved-corpus", help="Stage digest-verified approved manuscripts for training")
    approved_corpus.add_argument("--ledger", type=Path, required=True)
    approved_corpus.add_argument("--output", type=Path, required=True)
    replay = sub.add_parser("prepare-replay", help="Keep old worked examples with approved new training")
    replay.add_argument("--ledger", type=Path, required=True)
    replay.add_argument("--retained", type=Path, required=True)
    replay.add_argument("--validation", type=Path, required=True)
    replay.add_argument("--work-id", action="append", required=True)
    replay.add_argument("--output", type=Path, required=True)
    promotion = sub.add_parser("gate-learning", help="Regress both old and unseen domains before promoting")
    promotion.add_argument("--baseline", type=Path, required=True)
    promotion.add_argument("--candidate", type=Path, required=True)
    promotion.add_argument("--old-cases", type=Path, required=True)
    promotion.add_argument("--new-cases", type=Path, required=True)
    promotion.add_argument("--training-examples", type=Path, required=True)
    promotion.add_argument("--validation-examples", type=Path, required=True)
    promotion.add_argument("--report", type=Path, required=True)
    promotion.add_argument("--promote-to", type=Path)
    promotion.add_argument("--threads", type=int, default=4)
    promotion.add_argument("--tokens", type=int, default=512)
    relational = sub.add_parser("prepare-relational", help="Prepare author-supervised relational training from private cases")
    relational.add_argument("--training", type=Path, required=True)
    relational.add_argument("--validation", type=Path, required=True)
    relational.add_argument("--held-out", type=Path, required=True)
    relational.add_argument("--output", type=Path, required=True)
    relational_eval = sub.add_parser("evaluate-relational", help="Measure held-out relational neural generation")
    relational_eval.add_argument("--checkpoint", type=Path, required=True)
    relational_eval.add_argument("--cases", type=Path, required=True)
    relational_eval.add_argument("--training", type=Path, required=True)
    relational_eval.add_argument("--validation", type=Path, required=True)
    relational_eval.add_argument("--threads", type=int, default=4)
    relational_eval.add_argument("--tokens", type=int, default=512)
    relational_eval.add_argument("--include-answers", action="store_true")
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
    inference = sub.add_parser("infer-neural", help="Generate neural hypotheses without supplied conclusions")
    inference.add_argument("--checkpoint", type=Path, required=True, help="Trained proposal Transformer")
    inference.add_argument("--judge", type=Path, required=True, help="Trained neural evidence ranker")
    inference.add_argument("--index", type=Path, required=True)
    inference.add_argument("--question", required=True)
    inference.add_argument("--limit", type=int, default=5)
    inference.add_argument("--attempts", type=int, default=4)
    inference.add_argument("--tokens", type=int, default=160)
    inference.add_argument("--temperature", type=float, default=.85)
    inference.add_argument("--seed", type=int, default=42)
    inference.add_argument("--threads", type=int, default=4)
    inference.add_argument("--max-corrections", type=int, default=2)
    inference.add_argument("--ledger", type=Path)
    inference.add_argument("--archive-dir", type=Path)
    neural_eval = sub.add_parser("evaluate-neural-inference", help="Evaluate generated conclusions on held-out questions")
    neural_eval.add_argument("--checkpoint", type=Path, required=True)
    neural_eval.add_argument("--judge", type=Path, required=True)
    neural_eval.add_argument("--index", type=Path, required=True)
    neural_eval.add_argument("--cases", type=Path, required=True)
    neural_eval.add_argument("--attempts", type=int, default=4)
    neural_eval.add_argument("--tokens", type=int, default=160)
    neural_eval.add_argument("--temperature", type=float, default=.85)
    neural_eval.add_argument("--seed", type=int, default=42)
    neural_eval.add_argument("--threads", type=int, default=4)
    neural_eval.add_argument("--limit", type=int, default=5)
    neural_eval.add_argument("--max-corrections", type=int, default=2)
    neural_eval.add_argument("--include-answers", action="store_true")
    draft = sub.add_parser("draft-neural-outline", help="Prepare an offline outline around a supported thesis")
    draft.add_argument("--checkpoint", type=Path, required=True)
    draft.add_argument("--index", type=Path, required=True)
    draft.add_argument("--brief", type=Path, required=True)
    draft.add_argument("--limit", type=int, default=5)
    draft.add_argument("--threads", type=int, default=4)
    prepare_language = sub.add_parser("prepare-english-corpus", help="Create disjoint, deduplicated English corpus partitions")
    prepare_language.add_argument("--source", type=Path, required=True)
    prepare_language.add_argument("--output", type=Path, required=True)
    prepare_language.add_argument("--seed", type=int, default=42)
    english_gate = sub.add_parser("gate-english", help="Reject regressions and require manual evaluation before promotion")
    english_gate.add_argument("--previous", type=Path, required=True)
    english_gate.add_argument("--candidate", type=Path, required=True)
    english_gate.add_argument("--tokenizer", type=Path, required=True)
    english_gate.add_argument("--cases", type=Path, required=True)
    english_gate.add_argument("--report", type=Path, required=True)
    english_gate.add_argument("--review", type=Path)
    english_gate.add_argument("--promote-to", type=Path)
    english_gate.add_argument("--tokens", type=int, default=80)
    english_gate.add_argument("--threads", type=int, default=2)
    token = sub.add_parser("train-tokenizer", help="Train original subword vocabulary on an approved English corpus")
    token.add_argument("--corpus", type=Path, required=True)
    token.add_argument("--output", type=Path, required=True)
    token.add_argument("--vocab", type=int, default=8192)
    language = sub.add_parser("train-english", help="CPU English pretraining or supervised dialogue tuning")
    language.add_argument("--tokenizer", type=Path, required=True)
    language.add_argument("--training", type=Path, required=True)
    language.add_argument("--validation", type=Path, required=True)
    language.add_argument("--output", type=Path, required=True)
    language.add_argument("--phase", choices=["pretrain", "instruction"], required=True)
    language.add_argument("--steps", type=int, required=True)
    language.add_argument("--init-checkpoint", type=Path)
    language.add_argument("--resume", action="store_true")
    language.add_argument("--embed", type=int, default=64)
    language.add_argument("--heads", type=int, default=4)
    language.add_argument("--layers", type=int, default=2)
    language.add_argument("--context", type=int, default=256)
    language.add_argument("--batch", type=int, default=2)
    language.add_argument("--lr", type=float, default=.001)
    language.add_argument("--seed", type=int, default=42)
    language.add_argument("--threads", type=int, default=2)
    language.add_argument("--checkpoint-every", type=int, default=20)
    ask_english = sub.add_parser("ask-english", help="Generate with the original English-trained Core")
    ask_english.add_argument("--checkpoint", type=Path, required=True)
    ask_english.add_argument("--tokenizer", type=Path, required=True)
    ask_english.add_argument("--question", required=True)
    ask_english.add_argument("--tokens", type=int, default=120)
    ask_english.add_argument("--temperature", type=float, default=0)
    ask_english.add_argument("--top-p", type=float, default=1.)
    ask_english.add_argument("--seed", type=int, default=42)
    ask_english.add_argument("--threads", type=int, default=2)
    gauntlet = sub.add_parser("evaluate-english", help="Evaluate original-model answers on frozen English gauntlet")
    gauntlet.add_argument("--checkpoint", type=Path, required=True)
    gauntlet.add_argument("--tokenizer", type=Path, required=True)
    gauntlet.add_argument("--cases", type=Path, required=True)
    gauntlet.add_argument("--output", type=Path, required=True)
    gauntlet.add_argument("--limit", type=int)
    gauntlet.add_argument("--tokens", type=int, default=120)
    gauntlet.add_argument("--threads", type=int, default=2)
    a = parser.parse_args(argv)
    if a.command == "prepare-english-corpus":
        print(json.dumps(prepare_english_corpus(a.source, a.output, seed=a.seed), indent=2))
    elif a.command == "gate-english":
        result = promote_english(a.previous, a.candidate, a.tokenizer, a.cases, a.report,
                                 destination=a.promote_to, review_path=a.review,
                                 tokens=a.tokens, threads=a.threads)
        print(json.dumps({k: v for k, v in result.items() if k not in ("baseline", "candidate")}, indent=2))
        return int(not result["promotable"])
    elif a.command == "train-tokenizer":
        from .corpus import collect_documents
        rows = collect_documents(a.corpus)
        vocab = ByteBPETokenizer.train([row["text"] for row in rows], vocab_size=a.vocab)
        vocab.save(a.output)
        print(json.dumps({"vocab_size": vocab.vocab_size,
                          "tokenizer_sha256": vocab.fingerprint()}, indent=2))
    elif a.command == "train-english":
        report = train_language(tokenizer_path=a.tokenizer, training=a.training,
                                validation=a.validation, output=a.output,
                                phase=a.phase, steps=a.steps, init_checkpoint=a.init_checkpoint,
                                resume=a.resume, embed=a.embed, layers=a.layers,
                                heads=a.heads, context=a.context, batch_size=a.batch,
                                lr=a.lr, seed=a.seed, threads=a.threads,
                                checkpoint_every=a.checkpoint_every)
        print(json.dumps(report, indent=2))
    elif a.command == "ask-english":
        print(json.dumps(answer_question(a.checkpoint, a.tokenizer, a.question,
                                         max_tokens=a.tokens, temperature=a.temperature,
                                         top_p=a.top_p, seed=a.seed, threads=a.threads), indent=2))
    elif a.command == "evaluate-english":
        report = evaluate_english(a.checkpoint, a.tokenizer, load_gauntlet(a.cases),
                                  limit=a.limit, max_tokens=a.tokens, threads=a.threads)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))
        return int(not report["english_milestone_met"])
    elif a.command == "ingest":
        m = build_manifest(a.corpus, a.manifest)
        print(json.dumps({"documents": len(m["documents"]), "fingerprint": m["fingerprint"]}, indent=2))
    elif a.command == "train":
        cfg = ModelConfig(context_size=a.context, embed_dim=a.embed, heads=a.heads, layers=a.layers)
        print(json.dumps(train(a.corpus, a.output, cfg, a.steps, a.batch, a.lr, a.seed,
                               a.resume, a.checkpoint_every, schedule_steps=a.schedule_steps,
                               warmup_steps=a.warmup_steps, init_checkpoint=a.init_checkpoint), indent=2))
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
                                 threads=a.threads, format_name=a.format, pair_batches=a.pair_batches,
                                 resume=a.resume, checkpoint_every=a.checkpoint_every)
        print(json.dumps(report, indent=2))
    elif a.command == "register-work":
        print(json.dumps(register(a.ledger, a.id, a.kind, a.source, origin=a.origin,
                                  evidence=a.evidence), indent=2))
    elif a.command == "decide-work":
        print(json.dumps(decide(a.ledger, a.id, a.decision), indent=2))
    elif a.command == "approved-works":
        print(json.dumps(approved_sources(a.ledger), indent=2))
    elif a.command == "stage-approved-corpus":
        print(json.dumps(stage_approved_corpus(a.ledger, a.output), indent=2))
    elif a.command == "prepare-replay":
        print(json.dumps(prepare_replay(a.ledger, a.retained, a.validation, a.output,
                                        accepted_ids=a.work_id), indent=2))
    elif a.command == "gate-learning":
        result = gate_and_promote(a.baseline, a.candidate, a.old_cases, a.new_cases,
                                  a.report, training_examples=a.training_examples,
                                  validation_examples=a.validation_examples,
                                  destination=a.promote_to, threads=a.threads, tokens=a.tokens)
        print(json.dumps(result, indent=2))
        return int(not result["promotable"])
    elif a.command == "prepare-relational":
        result = prepare_curriculum(a.training, a.validation, a.held_out, a.output)
        print(json.dumps(result, indent=2))
    elif a.command == "evaluate-relational":
        train_cases = load_relational_cases(a.training)
        validation_cases = load_relational_cases(a.validation)
        held_cases = load_relational_cases(a.cases)
        check_relational_splits(train_cases, validation_cases, held_cases)
        engine = load_proposer(a.checkpoint, a.threads, a.tokens)
        report = evaluate_relational(held_cases, engine)
        if not a.include_answers:
            for case in report["cases"]:
                case.pop("generated_consequence", None)
        print(json.dumps(report, indent=2))
        return int(report["exact_consequences"] != report["total"]
                   or report["counterfactual_passed"] != report["counterfactual_families"])
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
    elif a.command == "infer-neural":
        model = load_proposer(a.checkpoint, a.threads, a.tokens).model
        judge = neural_judge.NeuralJudge.load(a.judge, a.threads)
        if (a.ledger is None) != (a.archive_dir is None):
            raise ValueError("Provide both --ledger and --archive-dir to archive reasoning")
        result = infer(a.index, model, judge, a.question, limit=a.limit, attempts=a.attempts,
                       tokens=a.tokens, temperature=a.temperature, seed=a.seed,
                       max_corrections=a.max_corrections)
        if a.ledger is not None and result["status"] == "completed":
            result["intellectual_record"] = capture_successful_reasoning(a.ledger, a.archive_dir, result)
        print(json.dumps(result, indent=2))
        return int(result["status"] != "completed")
    elif a.command == "evaluate-neural-inference":
        model = load_proposer(a.checkpoint, a.threads, a.tokens).model
        judge = neural_judge.NeuralJudge.load(a.judge, a.threads)
        engine = GeneratingProposer(model, judge, attempts=a.attempts, tokens=a.tokens,
                                    temperature=a.temperature, seed=a.seed)
        report = evaluate_cases(a.index, load_cases(a.cases), engine, limit=a.limit,
                                max_corrections=a.max_corrections, include_answers=a.include_answers)
        report["semantic_entailment_verified"] = False
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
