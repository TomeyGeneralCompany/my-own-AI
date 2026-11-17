"""Checkpointed, CPU-bounded English mastery; no teacher model or answer injection."""
from dataclasses import asdict
from hashlib import sha256
from difflib import SequenceMatcher
import json
import math
from pathlib import Path
from time import monotonic
import torch

from .model import LocalTransformer, ModelConfig
from .tokenizer import ByteBPETokenizer, BOS, EOS, USER, ASSISTANT, PAD
from .language_training import _example, _batch, _loss
from .train import _atomic_save
from .english_mastery_curriculum import Curriculum, STAGES, clean, CURRICULUM_VERSION
from .webster1913 import BLOB
from .verified_memory import VerifiedMemory

VERSION=1
ASSESSMENT_VERSION=2
# Re-examine frequently and change supervision after failed independent tests.
# A failure must not burn hundreds of identical teacher-forced updates.
REPETITIONS=(8,16,32)

def digest(value):
    return sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def new_progress(curriculum, step, identity):
    return dict(version=VERSION,source=identity,totals=curriculum.totals,
                curriculum_version=CURRICULUM_VERSION,stage_index=0,cursor=[STAGES[0],0,0],phase="teach",remaining=1,
                stage_passes={},passed_items={s:0 for s in STAGES},
                remediation=[],remediation_round=0,root_failures=0,
                outstanding_missing=[],missing_count=0,
                cumulative_passes=0,cumulative_started=None,cumulative_items={s:0 for s in STAGES},
                audit_cursor=0,audit_return=None,retention_tests=0,retention_failures=0,
                attempts=0,correct=0,remedial_updates=0,training_updates=0,initial_step=step,
                last_answer=None,qualified=False)

def migrate_curriculum(p,curriculum,identity,step):
    if p.get("curriculum_version")==CURRICULUM_VERSION:
        return p
    if (p.get("version")!=VERSION or p.get("source")!=identity or
            p.get("totals")!=curriculum.totals or
            p["initial_step"]+p["training_updates"]!=step):
        raise ValueError("Incompatible legacy curriculum")
    # The old cursor, credits and pending remediation refer to a different
    # order and unit layout. Archive them, never reinterpret as new credit.
    old=dict(p)
    fresh=new_progress(curriculum,step,identity)
    fresh["legacy_curriculum"]=old
    fresh["partial_answers"]=dict(old.get("partial_answers",{}))
    fresh["initial_step"]=old["initial_step"]
    fresh["training_updates"]=old["training_updates"]
    fresh["remedial_updates"]=old.get("remedial_updates",0)
    return fresh


def verify_progress(p,curriculum,identity,completed):
    if (p.get("version")!=VERSION or p.get("source")!=identity or
        p.get("totals")!=curriculum.totals or
        p["initial_step"]+p["training_updates"]!=completed or
        not 0<=p["stage_index"]<=len(STAGES) or
        p["phase"] not in ("teach","exam","remediate","remedial_exam","complete") or
        not 0<=p["remaining"]<=300):
        raise ValueError("Incompatible saved English mastery progress")
    for stage,total in curriculum.totals.items():
        if not 0<=p["passed_items"][stage]<=total or not 0<=p["cumulative_items"][stage]<=total:
            raise ValueError("Invalid mastery count")
    for key in ([p["cursor"]] if p["cursor"] else [])+p["remediation"]:
        curriculum.unit(key)
    if p["qualified"] and (p["phase"]!="complete" or p["cursor"] is not None or
            p["remediation"] or p["cumulative_items"]!=curriculum.totals or
            p["cumulative_started"]!=completed):
        raise ValueError("Unverified English graduation")
    return p

def training_row(unit):
    return {k:unit[k] for k in ("id","question","answer")}

@torch.inference_mode()
def generate(model,tokenizer,question,expected):
    """Actual unaided autoregressive output; never supply reference tokens."""
    history=[BOS,USER]+tokenizer.encode(question)+[EOS,ASSISTANT]
    limit=min(model.config.context_size-len(history),len(tokenizer.encode(expected))*2+16)
    if limit<1:
        raise ValueError("Exam prompt exceeds context")
    output=[]
    model.eval()
    ended=False
    for _ in range(limit):
        logits,_=model(torch.tensor([history],dtype=torch.long))
        scores=logits[0,-1].clone()
        scores[[PAD,BOS,USER,ASSISTANT,261]]=float("-inf")
        token=int(scores.argmax())
        if token==EOS:
            ended=True
            break
        history.append(token)
        output.append(token)
    try:
        answer=tokenizer.decode(output)
    except UnicodeDecodeError:
        answer=""
    model.train()
    return answer,ended

def correct(unit,answer,ended):
    # Case and punctuation are substantive in grammatical tasks; whitespace
    # differences alone are not a language error.
    return ended and clean(answer)==clean(unit["answer"])

def record_partial(p, unit, answer, ended, tokenizer):
    """Persist confirmed reference-token positions; never use them as exam output."""
    reference = tokenizer.encode(clean(unit["answer"])) + [EOS]
    actual = tokenizer.encode(clean(answer)) + ([EOS] if ended else [])
    identity = digest([unit["exam"], reference])
    table = p.setdefault("partial_answers", {})
    previous = table.get(identity, {})
    retained = set(previous.get("retained", []))
    # Exact answers are unambiguous; otherwise align in reference order.
    alignment = SequenceMatcher(None, reference, actual, autojunk=False)
    matched = set()
    for block in alignment.get_matching_blocks():
        matched.update(range(block.a, block.a + block.size))
    new = matched - retained
    regressed = retained - matched
    retained.update(matched)
    # Count insertions as errors too; a fully matched subsequence is not an exact answer.
    distance = sum(max(b-a, d-c) for tag,a,b,c,d in alignment.get_opcodes() if tag!="equal")
    best = previous.get("best_distance", distance)
    stagnant = 0 if distance < best else previous.get("stagnant", 0)+1
    first = next((i for i,(x,y) in enumerate(zip(reference,actual)) if x!=y),
                 min(len(reference),len(actual)))
    table[identity] = dict(retained=sorted(retained), latest=sorted(matched),
                           missing=sorted(set(range(len(reference)))-matched),
                           regressions=sorted(regressed), attempts=previous.get("attempts",0)+1,
                           distance=distance,best_distance=min(best,distance),stagnant=stagnant,
                           first_divergence=first,actual=answer,ended=ended)
    p["partial_new_tokens"] = p.get("partial_new_tokens",0)+len(new)
    p["partial_regressions"] = p.get("partial_regressions",0)+len(regressed)
    return table[identity]

def correction_sample(p, unit, tokenizer, context):
    """Extra supervision for current errors; ordinary full-answer loss remains."""
    reference = tokenizer.encode(clean(unit["answer"])) + [EOS]
    record = p.get("partial_answers", {}).get(digest([unit["exam"], reference]))
    if not record or not record["missing"]:
        return None
    inputs, labels = _example(dict(training_row(unit),question=unit["exam"]), tokenizer, context)
    # Only target positions if training and assessment tokenizations agree.
    if [v for v in labels if v != -100] != reference:
        return None
    start = len(labels)-len(reference)
    missing = set(record["missing"])
    return inputs, [v if i-start in missing else -100 for i,v in enumerate(labels)]

def remediation_samples(p, unit, tokenizer, context):
    """Teach the failed prompt itself and repair the first autoregressive error.

    Recovery is an explicit restart exercise: a causal decoder cannot erase
    an already emitted wrong token. No failed prefix is inserted into exams.
    """
    reference=tokenizer.encode(clean(unit["answer"]))+[EOS]
    record=p.get("partial_answers",{}).get(digest([unit["exam"],reference]),{})
    # Include the full answer under the exam wording even when no partial
    # record exists (e.g. a checkpoint from before this trainer version).
    row=dict(training_row(unit),question=unit["exam"])
    inputs,labels=_example(row,tokenizer,context)
    samples=[(inputs,labels)]
    targeted=correction_sample(p,unit,tokenizer,context)
    if targeted is not None:
        samples.append(targeted)
    # The first wrong/extra token includes insertion errors missed by
    # matching reference subsequences. Train the correct next token there.
    first=record.get("first_divergence")
    if first is not None and 0<=first<len(reference) and [v for v in labels if v!=-100]==reference:
        pos=len(labels)-len(reference)+first
        samples.append((inputs,[v if i==pos else -100 for i,v in enumerate(labels)]))
    # Change the supervision on a persistent plateau, instead of merely
    # repeating the same near-zero-loss teacher-forced row.
    if record.get("stagnant",0)>=3 or record.get("attempts",0)>=3:
        actual=record.get("actual","")
        if actual:
            actual_tokens=tokenizer.encode(clean(actual))
            try:
                prefix=tokenizer.decode(actual_tokens[:(first if first is not None else 0)+1])
            except UnicodeDecodeError:
                prefix=actual
            recovery=dict(row,question=(unit["exam"]+"\nYour attempt began: "+prefix+
                                        "\nRestart and give the complete corrected answer."))
            if len(tokenizer.encode(recovery["question"]))+len(reference)+4<=context:
                samples.append(_example(recovery,tokenizer,context))
        # Distinct teaching paraphrase, not the held-out transfer wording.
        paraphrase=dict(row,question="Follow this instruction exactly: "+unit["question"])
        if len(tokenizer.encode(paraphrase["question"]))+len(reference)+4<=context:
            samples.append(_example(paraphrase,tokenizer,context))
        # Teach equivalent commands with the same bounded target, without
        # teaching the held-out transfer question or supplying exam answers.
        for instruction in (
            "Write the exact passage requested by "+unit["id"]+".",
            "Give only the complete source text for "+unit["id"]+".",
            "Return the requested dictionary text exactly, for "+unit["id"]+".",
        ):
            alternative=dict(row,question=instruction)
            if len(tokenizer.encode(instruction))+len(reference)+4<=context:
                samples.append(_example(alternative,tokenizer,context))
    return samples


def transfer_question(unit):
    """Held-out wording; mandatory independent examination."""
    if unit["id"].startswith("dictionary-"):
        # Preserve the exact source entry and section identity.
        return unit["exam"].replace("From memory, reproduce ","Recall verbatim ").replace(", section ",", passage ",1)
    return "Give the requested response without changing the task: "+unit["question"]


def migrate_assessment(p, step):
    """Keep all learned state; old mastery is provisional until current-weight re-examination."""
    if p.get("assessment_version")==ASSESSMENT_VERSION:
        return
    p["assessment_version"]=ASSESSMENT_VERSION
    p["independent_cumulative_units"]=0
    p["legacy_mastery_credit"]=dict(p["passed_items"])
    p["legacy_unaided_tests"]=p.get("unaided_tests",0)
    p["legacy_unaided_correct"]=p.get("unaided_correct",0)
    p["legacy_transfer_tests"]=p.get("transfer_tests",0)
    p["legacy_transfer_correct"]=p.get("transfer_correct",0)
    for key in ("unaided_tests","unaided_correct","transfer_tests","transfer_correct"):
        p[key]=0
    if p["stage_index"]==len(STAGES):
        p["qualified"]=False
        if not p["remediation"]:
            reset_cumulative(p,step)


def assess_independently(model, tokenizer, unit, progress):
    """No retrieval, supplied answer, teacher forcing, or advancement on failed transfer."""
    checks=[("recall",unit["exam"],unit)]
    checks.append(("transfer",transfer_question(unit),unit))
    if "composition_transfer" in unit:
        extra=unit["composition_transfer"]
        checks.append(("composition",extra["exam"],extra))
    results=[]
    for kind,question,expected in checks:
        if len(tokenizer.encode(question))+4+len(tokenizer.encode(expected["answer"]))+1>model.config.context_size:
            raise ValueError("Independent examination cannot fit checkpoint context")
        answer,ended=generate(model,tokenizer,question,expected["answer"])
        ok=correct(expected,answer,ended)
        row=dict(id=unit["id"],kind=kind,question=question,actual=answer,
                 expected=expected["answer"],correct=ok,eos=ended,origin="neural")
        results.append(row)
        progress["unaided_tests"]=progress.get("unaided_tests",0)+1
        progress["unaided_correct"]=progress.get("unaided_correct",0)+int(ok)
        if kind=="transfer":
            progress["transfer_tests"]=progress.get("transfer_tests",0)+1
            progress["transfer_correct"]=progress.get("transfer_correct",0)+int(ok)
            progress["last_transfer"]=row
        if unit.get("usage_kind") and kind=="recall":
            progress["word_sentence_tests"]=progress.get("word_sentence_tests",0)+1
            progress["word_sentence_correct"]=progress.get("word_sentence_correct",0)+int(ok)
        if kind=="composition":
            progress["composition_tests"]=progress.get("composition_tests",0)+1
            progress["composition_correct"]=progress.get("composition_correct",0)+int(ok)
        if not ok:
            break
    progress["last_independent_exam"]=results
    return all(row["correct"] for row in results),results


def reset_cumulative(p,step):
    p["cumulative_items"]={s:0 for s in STAGES}
    p["cumulative_started"]=step
    p["independent_cumulative_units"]=0
    p["cursor"]=[STAGES[0],0,0]
    p["phase"]="exam"
    p["remaining"]=0

def advance_cursor(p,curriculum,step):
    key=p["cursor"]
    if p["audit_return"] is not None:
        p["cursor"]=p["audit_return"]
        p["audit_return"]=None
        p["root_failures"]=0
        p["phase"]="teach"
        p["remaining"]=1
        return
    stage,index,part=key
    next_key=curriculum.next(key)
    completed_item=next_key is None or next_key[1]!=index
    cumulative=p["stage_index"]==len(STAGES)
    if completed_item:
        table=p["cumulative_items"] if cumulative else p["passed_items"]
        if table[stage]!=index:
            raise ValueError("Mastery coverage must be contiguous")
        table[stage]+=1
    p["root_failures"]=0
    if next_key is None:
        if not cumulative:
            if p["passed_items"][stage]!=curriculum.totals[stage]:
                raise ValueError("Incomplete stage")
            p["stage_passes"][stage]=step
            p["stage_index"]+=1
            if p["stage_index"]==len(STAGES):
                reset_cumulative(p,step)
                return
            next_key=[STAGES[p["stage_index"]],0,0]
        else:
            if stage==STAGES[-1]:
                if (p["cumulative_items"]!=curriculum.totals or p["cumulative_started"]!=step or
                        p.get("independent_cumulative_units",0)<sum(curriculum.totals.values())):
                    raise ValueError("Cumulative verification did not use fixed weights")
                p["qualified"]=True
                p["cursor"]=None
                p["phase"]="complete"
                p["remaining"]=0
                return
            next_key=[STAGES[STAGES.index(stage)+1],0,0]
    p["cursor"]=next_key
    p["phase"]="exam" if cumulative else "teach"
    p["remaining"]=0 if cumulative else 1
    # Periodically retest previously mastered records; the learner cannot
    # continue with a failed retention probe left unresolved.
    if not cumulative and completed_item and sum(p["passed_items"].values())%16==0:
        eligible=[s for s in STAGES if p["passed_items"][s]]
        chosen=eligible[p["audit_cursor"]%len(eligible)]
        idx=(p["audit_cursor"]//len(eligible))%p["passed_items"][chosen]
        parts=curriculum.units(chosen,idx)
        audit=[chosen,idx,(p["audit_cursor"]//max(1,p["passed_items"][chosen]))%len(parts)]
        p["audit_cursor"]+=1
        p["audit_return"]=next_key
        p["cursor"]=audit
        p["phase"]="exam"
        p["remaining"]=0
        p["retention_tests"]+=1

def fail_root(p,curriculum,unit,actual):
    p["root_failures"]+=1
    if p["audit_return"] is not None:
        p["retention_failures"]+=1
    indices,missing=curriculum.indices(unit["lexical_text"])
    p["missing_count"]+=len(missing)
    p["outstanding_missing"]=sorted(set(p["outstanding_missing"]+missing))[:2048]
    pending=[]
    # Never jump ahead of alphabetical vocabulary mastery or follow
    # unrelated words hallucinated in an answer.
    for i in (indices if p["cursor"][0] not in ("dictionary","usage") else []):
        for part in range(len(curriculum.units("dictionary",i))):
            key=["dictionary",i,part]
            if key!=p["cursor"]:
                pending.append(key)
    # The original failed unit is always retaught after every relevant
    # dictionary definition. No recursive expansion of a remedial definition.
    pending.append(list(p["cursor"]))
    p["remediation"]=pending
    p["remediation_round"]=0
    p["phase"]="remediate"
    p["remaining"]=REPETITIONS[min(p["root_failures"]-1,2)]

def mastered_replay(p,curriculum,step):
    eligible=[s for s in STAGES if p["passed_items"][s]]
    if not eligible:
        return None
    s=eligible[step%len(eligible)]
    count=p["passed_items"][s]
    index=(step*7919)%count
    units=curriculum.units(s,index)
    return training_row(units[(step//max(1,count))%len(units)])

def train_session(checkpoint,tokenizer_path,records,*,parent=None,budget=400,
                  threads=2,max_tests=64,time_budget=1500,identity=BLOB, prior_exam=None):
    if budget<1 or max_tests<1 or time_budget<=0:
        raise ValueError("Invalid session budget")
    checkpoint=Path(checkpoint)
    source=checkpoint if checkpoint.exists() else Path(parent)
    payload=torch.load(source,map_location="cpu",weights_only=True)
    tokenizer=ByteBPETokenizer.load(tokenizer_path)
    cfg=ModelConfig(**payload["model_config"])
    if payload["tokenizer_sha256"]!=tokenizer.fingerprint():
        raise ValueError("Tokenizer changed")
    torch.set_num_threads(threads)
    model=LocalTransformer(cfg)
    model.load_state_dict(payload["state_dict"])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.000125)
    continuing=checkpoint.exists()
    if continuing and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    for group in optimizer.param_groups:
        group["lr"]=.000125
    if continuing and "torch_rng" in payload:
        torch.set_rng_state(payload["torch_rng"])
    curriculum=Curriculum(records,tokenizer,cfg.context_size)
    step=int(payload["completed_steps"]) if continuing else 0
    if not continuing:
        # A fresh stage inheriting a different checkpoint is not a resume.
        payload=dict(payload)
        payload.pop("english_mastery",None)
    p=payload.get("english_mastery") or new_progress(curriculum,step,identity)
    p=migrate_curriculum(p,curriculum,identity,step)
    verify_progress(p,curriculum,identity,step)
    memory=VerifiedMemory(payload.get("verified_memory",{}))
    if prior_exam is not None and "english_mastery" not in payload:
        memory.import_exam(prior_exam,step)
    migrate_assessment(p,step)
    p.setdefault("retrieval_hits",0)
    p.setdefault("new_verified",0)
    p.setdefault("unaided_tests",0)
    p.setdefault("unaided_correct",0)
    start=step
    tests=0
    started=monotonic()
    last_loss=None
    def save():
        verify_progress(p,curriculum,identity,step)
        payload.update(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),
                       completed_steps=step,torch_rng=torch.get_rng_state(),
                       english_mastery=p,verified_memory=memory.records,sampling_policy="english_mastery_v1",
                       phase="instruction")
        _atomic_save(payload,checkpoint)
    while not p["qualified"] and step-start<budget and tests<max_tests and monotonic()-started<time_budget:
        key=p["remediation"][0] if p["phase"] in ("remediate","remedial_exam") else p["cursor"]
        unit=curriculum.unit(key)
        if p["phase"] in ("teach","remediate"):
            rows=[training_row(unit)]
            if "composition_practice" in unit:
                rows.append(unit["composition_practice"])
            replay=mastered_replay(p,curriculum,step) if step%4==0 else None
            if replay is not None:
                rows.append(replay)
            samples=[_example(row,tokenizer,cfg.context_size) for row in rows]
            if p["phase"]=="remediate":
                # The old record lacks first-divergence/raw-answer metadata.
                # Hydrate it from the saved failure, without resetting state.
                last=p.get("last_answer") or {}
                reference=tokenizer.encode(clean(unit["answer"]))+[EOS]
                prior=p.get("partial_answers",{}).get(digest([unit["exam"],reference]),{})
                if last.get("id")==unit["id"] and not last.get("correct") and "first_divergence" not in prior:
                    record_partial(p,unit,last["actual"],last.get("eos",False),tokenizer)
                samples.extend(remediation_samples(p,unit,tokenizer,cfg.context_size))
                last_checks=p.get("last_independent_exam",[])
                if last_checks and last_checks[-1]["id"]==unit["id"] and last_checks[-1]["kind"]=="transfer" and not last_checks[-1]["correct"]:
                    alternate_teaching=dict(training_row(unit),question="Follow this instruction exactly: "+unit["question"])
                    if len(tokenizer.encode(alternate_teaching["question"]))+len(reference)+4<=cfg.context_size:
                        samples.append(_example(alternate_teaching,tokenizer,cfg.context_size))
            else:
                targeted=correction_sample(p,unit,tokenizer,cfg.context_size)
                if targeted is not None:
                    samples.append(targeted)
            model.train()
            loss=_loss(model,_batch(samples,list(range(len(samples)))))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite English mastery loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            step+=1
            p["training_updates"]+=1
            last_loss=float(loss.detach())
            if p["phase"]=="remediate":
                p["remedial_updates"]+=1
            p["remaining"]-=1
            if p["remaining"]==0:
                p["phase"]="remedial_exam" if p["phase"]=="remediate" else "exam"
        else:
            ok,results=assess_independently(model,tokenizer,unit,p)
            primary=results[0]
            answer,ended=primary["actual"],primary["eos"]
            partial=record_partial(p,unit,answer,ended,tokenizer)
            if primary["correct"]:
                p["new_verified"]+=int(memory.remember(unit["exam"],answer,expected=unit["answer"],step=step,provenance="neural"))
            tests+=1
            p["attempts"]+=1
            p["correct"]+=int(ok)
            p["last_answer"]=dict(id=unit["id"],question=unit["exam"],
                                  actual=answer,expected=unit["answer"],correct=ok,eos=ended,origin="neural",
                                  partial=partial,independent_exam=results)
            if ok and p["stage_index"]==len(STAGES) and p["phase"]=="exam":
                p["independent_cumulative_units"]+=1
            if p["phase"]=="remedial_exam":
                if not ok:
                    p["remediation_round"]+=1
                    p["remaining"]=REPETITIONS[min(p["root_failures"]+p["remediation_round"]-1,2)]
                    p["phase"]="remediate"
                else:
                    p["remediation"].pop(0)
                    p["remediation_round"]=0
                    if p["remediation"]:
                        p["remaining"]=REPETITIONS[min(p["root_failures"]-1,2)]
                        p["phase"]="remediate"
                    elif p["stage_index"]==len(STAGES):
                        # No pre-remediation cumulative result can certify
                        # the changed weights. Restart the entire sweep.
                        p["root_failures"]=0
                        reset_cumulative(p,step)
                    else:
                        advance_cursor(p,curriculum,step)
            elif ok:
                advance_cursor(p,curriculum,step)
            else:
                fail_root(p,curriculum,unit,results[-1]["actual"])
        if step!=start and step%100==0 and p["phase"] in ("exam","remedial_exam"):
            save()
    save()
    p["verified_memory_count"]=len(memory.records)
    return dict(substage=STAGES[p["stage_index"]] if p["stage_index"]<len(STAGES) else "cumulative",
                completed_updates=step,session_updates=step-start,session_tests=tests,
                mastery=p,passed=p["qualified"],last_training_loss=last_loss,
                checkpoint_sha256=sha256(checkpoint.read_bytes()).hexdigest())

def graduation_report(result):
    p=result["mastery"]
    return dict(format="english_mastery_v1",checkpoint_sha256=result["checkpoint_sha256"],
                source=p["source"],totals=p["totals"],passed_items=p["passed_items"],
                stage_passes=p["stage_passes"],cumulative_items=p["cumulative_items"],
                cumulative_started=p["cumulative_started"],completed_updates=result["completed_updates"],
                remediation_pending=len(p["remediation"]),qualified=p["qualified"],
                phase=p["phase"],cursor=p["cursor"],
                assessment_version=p.get("assessment_version"),
                curriculum_version=p.get("curriculum_version"),
                word_sentence_tests=p.get("word_sentence_tests",0),
                word_sentence_correct=p.get("word_sentence_correct",0),
                independent_cumulative_units=p.get("independent_cumulative_units",0))
