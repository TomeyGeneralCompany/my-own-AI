"""Explicit local commands. No command downloads pretrained weights or self-modifies code."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import random
import torch
from .tokenizer import ByteBPE
from .model import AncestorModel, ModelConfig
from .training import TrainConfig, Trainer, flat_batches
from .checkpoints import save_checkpoint, restore_checkpoint
from .governance import Constitution
from .corpus import Corpus

def args_parser():
    parser=argparse.ArgumentParser(prog="ancestor")
    sub=parser.add_subparsers(dest="command",required=True)
    g=sub.add_parser("init-governance",help="Create owner-approved manifest (refuses overwrite)")
    g.add_argument("--axioms",required=True,help="JSON file containing an array of exact owner-approved axiom strings")
    g.add_argument("--output",required=True)
    t=sub.add_parser("build-tokenizer")
    t.add_argument("--source",action="append",required=True)
    t.add_argument("--output",required=True)
    t.add_argument("--vocab",type=int,default=8192)
    c=sub.add_parser("import-manuscript")
    c.add_argument("--data",required=True)
    c.add_argument("--source",required=True)
    c.add_argument("--id",required=True)
    c.add_argument("--title",required=True)
    prep=sub.add_parser("prepare-document", help="Preserve original DOCX/PDF/TXT and extract text for owner review")
    prep.add_argument("--source",required=True)
    prep.add_argument("--out",required=True)
    approve=sub.add_parser("approve-document", help="Index reviewed text by exact SHA-256")
    approve.add_argument("--manifest",required=True)
    approve.add_argument("--approved-hash",required=True)
    approve.add_argument("--data",required=True)
    approve.add_argument("--id",required=True)
    approve.add_argument("--title",required=True)
    d=sub.add_parser("import-dictionary")
    d.add_argument("--data",required=True)
    d.add_argument("--source",required=True,help="Verified headword<TAB>definition text")
    p=sub.add_parser("train")
    p.add_argument("--corpus",required=True,help="Approved UTF-8 training text")
    p.add_argument("--validation",required=True,help="Independent held-out UTF-8 text")
    p.add_argument("--tokenizer",required=True)
    p.add_argument("--governance",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--updates",required=True,type=int)
    p.add_argument("--block",type=int,default=128)
    p.add_argument("--width",type=int,default=384)
    p.add_argument("--layers",type=int,default=6)
    p.add_argument("--heads",type=int,default=6)
    p.add_argument("--ffn",type=int,default=1536)
    p.add_argument("--accumulation",type=int,default=1)
    p.add_argument("--save-every",type=int,default=250)
    p.add_argument("--resume",default=None)
    p.add_argument("--seed",type=int,default=1729)
    return parser

def main(argv=None):
    args=args_parser().parse_args(argv)
    if args.command=="init-governance":
        axioms=json.loads(Path(args.axioms).read_text(encoding="utf-8"))
        if not isinstance(axioms,list) or not all(isinstance(x,str) for x in axioms):
            raise ValueError("Axioms file must be a JSON list of exact strings")
        c=Constitution(tuple(axioms))
        c.save_initial(args.output)
        print("Governance manifest:",c.digest())
    elif args.command=="build-tokenizer":
        texts=[Path(p).read_text(encoding="utf-8") for p in args.source]
        t=ByteBPE.train(texts,args.vocab)
        t.save(args.output)
        print("Tokenizer:",t.digest(),"base vocab:",t.base_vocab_size)
    elif args.command=="prepare-document":
        from .importers import prepare_document
        print(json.dumps(prepare_document(args.source,args.out),indent=2))
    elif args.command=="approve-document":
        from .importers import commit_reviewed_text
        corpus=Corpus(args.data)
        count=commit_reviewed_text(corpus,args.manifest,document_id=args.id,
                                   title=args.title,approved_hash=args.approved_hash)
        print("Approved paragraphs indexed:",count)
    elif args.command=="import-manuscript":
        corpus=Corpus(args.data)
        print("Paragraphs:",corpus.ingest_text(args.source,args.id,args.title))
    elif args.command=="import-dictionary":
        corpus=Corpus(args.data)
        print("Entries:",corpus.import_dictionary_tsv(args.source))
    elif args.command=="train":
        if args.updates<1 or args.save_every<1:
            raise ValueError("Updates and save interval must be positive")
        t=ByteBPE.load(args.tokenizer)
        c=Constitution.load(args.governance)
        train_text=Path(args.corpus).read_text(encoding="utf-8")
        valid_text=Path(args.validation).read_text(encoding="utf-8")
        if not train_text or not valid_text or train_text==valid_text:
            raise ValueError("Training and validation must be nonempty and distinct")
        random.seed(args.seed);torch.manual_seed(args.seed)
        model=AncestorModel(ModelConfig(vocab_size=t.vocab_size,
            context_length=args.block,width=args.width,layers=args.layers,
            heads=args.heads,ffn_width=args.ffn))
        conf=TrainConfig(block_size=args.block,accumulation=args.accumulation)
        trainer=Trainer(model,conf)
        # Bind checkpoint to exact approved source bytes, rather than changing silently.
        corpus_hash=sha256((sha256(train_text.encode()).hexdigest()+":"+sha256(valid_text.encode()).hexdigest()).encode()).hexdigest()
        out=Path(args.out)
        out.mkdir(parents=True,exist_ok=True)
        if args.resume:
            restored=restore_checkpoint(args.resume,trainer,tokenizer_hash=t.digest(),
                                        constitution=c,corpus_manifest_hash=corpus_hash)
            print("Resumed:",restored)
        train_ids=t.encode(train_text)
        valid_ids=t.encode(valid_text)
        rng=random.Random(args.seed)
        for _ in range(trainer.completed_updates):
            # Preserve batch sampling position when resuming. This is a simple
            # deterministic sampler; full lineage checkpointing is recommended
            # for production multi-document curricula.
            for _ in range(args.accumulation):
                rng.randrange(0,len(train_ids)-args.block)
        for _ in range(args.updates):
            batches=[flat_batches(train_ids,args.block,1,rng=rng) for _ in range(args.accumulation)]
            loss=trainer.step(batches)
            if trainer.completed_updates % args.save_every == 0 or _==args.updates-1:
                validation=trainer.validate([flat_batches(valid_ids,args.block,1,rng=random.Random(91))])
                checkpoint=save_checkpoint(out,trainer,tokenizer_hash=t.digest(),constitution=c,
                    corpus_manifest_hash=corpus_hash,stage="ENGLISH_PRETRAINING",
                    objective_id="NEXT_TOKEN_ENGLISH")
                print(json.dumps({"update":trainer.completed_updates,"loss":loss,
                    "validation_loss":validation,"checkpoint":str(checkpoint)}))
if __name__=="__main__":
    main()
