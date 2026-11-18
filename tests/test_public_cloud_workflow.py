"""No private-PAT dependency may return to owner's off-laptop training path."""
from pathlib import Path
import yaml


def test_cloud_training_uses_public_owner_release_and_actions_artifacts():
    path=Path(__file__).resolve().parents[1]/".github"/"workflows"/"ancestor-cloud-training.yml"
    text=path.read_text(encoding="utf-8")
    workflow=yaml.safe_load(text)
    assert workflow["name"]=="Ancestor off-laptop training"
    job=workflow["jobs"]["train"]
    assert job["timeout-minutes"]<=360
    assert "schedule" in str(workflow) or "17 */6 * * *" in text
    assert "ANCESTOR_PRIVATE_REPO_TOKEN" not in text
    assert "ANCESTOR_PRIVATE_DATA_REPO" not in text
    assert "PRIVATE_REPO" not in text
    assert "ancestor-training-inputs" in text
    assert "github.token" in text
    assert "actions/upload-artifact@v4" in text
    assert "ancestor-neural-checkpoint" in text
    assert "checkpoint_metadata" in text
    assert "refusing silent model reset" in text
    assert "pull_request_target" not in text
    assert "originals/WORK-001.docx" not in text


def test_cloud_training_documented_input_readiness():
    doc=(Path(__file__).resolve().parents[1]/"docs"/"CLOUD_TRAINING.md").read_text(encoding="utf-8")
    for value in ("governance.json","tokenizer.json","heldout.txt",
                  "ancestor_private_training_library.zip"):
        assert value in doc
    assert "does NOT claim an active cloud training process" in doc
