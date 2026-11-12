"""Prevent a regression to the synthetic Core or ephemeral model transport."""
from pathlib import Path
import ast


ROOT = Path(__file__).resolve().parents[1]


def test_question_workflow_loads_exact_active_generation_and_never_trains():
    workflow = (ROOT / ".github/workflows/core-neural-smoke.yml").read_text()
    assert "gh release view ancestor-active" in workflow
    assert "gh release download" in workflow
    assert "--sha256" in workflow
    assert "scripts/ancestor_core.py ask" in workflow
    assert "scripts/github_neural_smoke.py" not in workflow
    assert "scripts/github_english_trial.py" not in workflow
    assert "scripts/ancestor_core.py train" not in workflow
    assert "core-generation.zip" in workflow
    assert "python -m pip install" in workflow


def test_train_workflow_uses_durable_versioned_release_snapshots():
    workflow = (ROOT / ".github/workflows/ancestor-train.yml").read_text()
    assert "ancestor-progress-" in workflow
    assert "ancestor-state-" in workflow
    assert "scripts/ancestor_core.py unpack" in workflow
    assert "scripts/ancestor_core.py train" in workflow
    assert "scripts/ancestor_core.py pack" in workflow
    assert "gh release upload" in workflow
    assert "ancestor-" in workflow and "-candidate" in workflow
    assert "concurrency:" in workflow
    assert "cancel-in-progress: false" in workflow


def test_promotion_is_explicit_and_reversible_without_changing_checkpoint():
    workflow = (ROOT / ".github/workflows/ancestor-promote.yml").read_text()
    assert "workflow_dispatch:" in workflow
    assert "experimental_bootstrap" in workflow
    assert "ancestor-active" in workflow
    assert "scripts/ancestor_promote.py" in workflow
    assert "gh release edit ancestor-active --notes-file data/active.json" in workflow
    assert "scripts/ancestor_core.py train" not in workflow
    assert "gh release download" in workflow


def test_real_inference_does_not_implicitly_create_weights():
    source = (ROOT / "scripts/ancestor_core.py").read_text()
    parsed = ast.parse(source)
    functions = {node.name: node for node in parsed.body if isinstance(node, ast.FunctionDef)}
    answer = ast.get_source_segment(source, functions["ask"])
    assert "answer_question(" in answer
    assert "train_language(" not in answer
    assert "train_progressive(" not in answer
    assert "ByteBPETokenizer.train(" not in answer
    assert "checkpoint_sha256" in answer
    assert '"training_executed": False' in answer
