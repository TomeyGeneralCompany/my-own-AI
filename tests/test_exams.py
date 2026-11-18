import pytest
from ancestor.exams import ExamItem,run_exam,rubric_grade

def test_owner_axiomatic_derivation_requires_ordered_reasoning():
    item=ExamItem("E1","Derive consequence","", "philosophy",grading="ordered_derivation",
        required_reasoning_steps=("Premise A","Therefore B","Consequence C"),
        required_axioms=("Identity is a unity of self and other.",))
    right="Identity is a unity of self and other.\nPremise A\nTherefore B\nConsequence C"
    wrong="Identity is a unity of self and other.\nConsequence C\nPremise A\nTherefore B"
    assert rubric_grade(right,item)
    assert not rubric_grade(wrong,item)
    assert run_exam("EXAM1",[item],lambda p:right).passed
    assert not run_exam("EXAM1",[item],lambda p:wrong).passed

def test_unknown_exam_grader_fails_closed():
    item=ExamItem("E2","Question","Answer","grammar",grading="unknown")
    with pytest.raises(ValueError):
        run_exam("EXAM2",[item],lambda q:"Answer")
