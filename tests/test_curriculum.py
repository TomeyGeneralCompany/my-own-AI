import json
import pytest
import torch
from ancestor.curriculum import CourseStage,Lesson,load_course,CurriculumRunner
from ancestor.governance import Constitution
from ancestor.context import ContextBuilder
from ancestor.tokenizer import ByteBPE
from ancestor.corpus import Corpus
from ancestor.ledger import LearningLedger
from ancestor.controller import Controller
from ancestor.model import AncestorModel,ModelConfig
from ancestor.training import Trainer,TrainConfig

def setup(tmp_path):
    source=tmp_path/"source.txt"
    source.write_text("1. Reason from the governing principle.",encoding="utf-8")
    corpus=Corpus(tmp_path/"corpus")
    corpus.ingest_text(source,"BOOK","One")
    c=Constitution(("Identity is a unity of self and other.",))
    t=ByteBPE([])
    context=ContextBuilder(t,c,1024)
    controller=Controller(corpus,context,LearningLedger(tmp_path/"ledger.jsonl"),c)
    model=AncestorModel(ModelConfig(t.vocab_size,128,24,1,4,48))
    trainer=Trainer(model,TrainConfig(block_size=128))
    return controller,trainer,t

def test_ordered_curriculum_config(tmp_path):
    path=tmp_path/"course.json"
    path.write_text(json.dumps({"stages":[
      {"id":"ELEMENTARY_ENGLISH","lessons":[{"id":"L1","source_id":"BOOK:P-000001",
      "task":"Continue the author's reasoning","exam_id":"E1"}]}]}))
    c=load_course(path)
    assert c[0].lessons[0].exam_id=="E1"

def test_real_lesson_gradient_update(tmp_path):
    controller,trainer,t=setup(tmp_path)
    runner=CurriculumRunner(controller,trainer,t.digest(),"corpus-sha",tmp_path/"checkpoints")
    lesson=Lesson("LESSON","BOOK:P-000001","Continue the derivation",exam_id="E1")
    loss=runner.lesson_step(lesson,reserve=8)
    assert loss>=0
    assert trainer.completed_updates==1
    assert not controller.ledger.covered()
    # Only successful completion of the full declared lesson gives coverage.

def test_no_unearned_graduation(tmp_path):
    controller,trainer,t=setup(tmp_path)
    runner=CurriculumRunner(controller,trainer,t.digest(),"corpus-sha",tmp_path/"checkpoints")
    stage=CourseStage("ELEMENTARY_ENGLISH",(Lesson("LESSON","BOOK:P-000001",
                                      "Continue the derivation",exam_id="MISSING"),))
    with pytest.raises(ValueError):
        runner.run_stage(stage,{})
    assert not any(x["type"]=="STAGE_COMPLETED" for x in controller.ledger.events())


def test_dictionary_and_book_stages_cannot_skip_sources(tmp_path):
    controller,trainer,t=setup(tmp_path)
    runner=CurriculumRunner(controller,trainer,t.digest(),"source",tmp_path/"checkpoints")
    d=tmp_path/"dictionary.tsv"
    d.write_text("Able\tcapable\nAbout\tconcerning\n",encoding="utf-8")
    controller.corpus.import_dictionary_tsv(d)
    incomplete=CourseStage("WEBSTER_DICTIONARY",
        (Lesson("L1","WEBSTER1913:W-000000","Use word",exam_id="E1"),),
        approved_entry_count=2)
    with pytest.raises(ValueError):
        runner.run_stage(incomplete,{})
    incomplete_books=CourseStage("ORDERED_MANUSCRIPTS",
        (Lesson("L2","BOOK:P-000001","Reason",exam_id="E2"),),
        approved_document_ids=("BOOK",))
    # Approved inventory exists, so its one actual numbered paragraph is complete.
    # The exam is still mandatory; missing exam blocks the next operation.
    with pytest.raises(ValueError):
        runner.run_stage(incomplete_books,{})
