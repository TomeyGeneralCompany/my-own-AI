import json
from ancestor.curriculum import load_course

def test_remediation_only_from_approved_source_references(tmp_path):
    p=tmp_path/"course.json"
    p.write_text(json.dumps({"stages":[{"id":"ELEMENTARY_ENGLISH",
      "lessons":[{"id":"L","source_id":"B:P-000001","task":"Study",
        "exam_id":"E","remediation_lessons":[{"source_id":"B:P-000002",
        "task":"Revisit the recorded prerequisite"}]}]}]}))
    lesson=load_course(p)[0].lessons[0]
    assert lesson.remediation_lessons==(("B:P-000002","Revisit the recorded prerequisite"),)
