"""Regression coverage for browser question routing.

A successful run must never silently use the old byte-level model when a
user asks the English Core a question.
"""
from pathlib import Path
import ast


ROOT = Path(__file__).resolve().parents[1]


def test_ask_core_workflow_routes_to_original_english_pipeline():
    workflow = (ROOT / ".github/workflows/core-neural-smoke.yml").read_text()
    assert "python -u scripts/github_english_trial.py" in workflow
    assert "python -u scripts/github_neural_smoke.py" not in workflow
    assert "public demo" in workflow.lower()
    assert "CORE_QUESTION:" in workflow
    assert "CORE_EVIDENCE:" in workflow
    assert "data/hosted-english/question-response.json" in workflow


def test_english_question_run_records_model_provenance():
    source = (ROOT / "scripts/github_english_trial.py").read_text()
    parsed = ast.parse(source)
    assert isinstance(parsed, ast.Module)
    assert '"checkpoint_sha256"' in source
    assert '"checkpoint_type"' in source
    assert '"training_source"' in source
    assert '"manuscript_trained": False' in source
    assert 'sha256(preferred.read_bytes()).hexdigest()' in source
    assert 'model_question' in source
    assert '"user_supplied_evidence"' in source
    assert 'answer_question(preferred, tokenizer_path, model_question' in source
