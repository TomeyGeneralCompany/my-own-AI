"""Cloud learning uses the owner's published texts; no missing release gate."""
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]

def test_cloud_training_reads_published_book_texts_and_retains_checkpoints():
    workflow=(ROOT/".github"/"workflows"/"ancestor-cloud-training.yml").read_text(encoding="utf-8")
    data=yaml.safe_load(workflow)
    assert data["name"]=="Ancestor off-laptop training"
    assert data["jobs"]["train"]["timeout-minutes"]<=360
    assert "17 */6 * * *" in workflow
    assert "published_library" in workflow
    assert "python -m ancestor.public_library" in workflow
    assert "python -m ancestor.public_training_setup" in workflow
    assert "python -m ancestor.cloud_training" in workflow
    assert "ancestor-training-inputs" not in workflow
    assert "PRIVATE_REPO" not in workflow
    assert "ANCESTOR_PRIVATE_REPO_TOKEN" not in workflow
    assert "github.token" in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "ancestor-neural-checkpoint" in workflow
    assert "checkpoint_metadata" in workflow
    assert "Prior checkpoints found but none passed integrity verification." in workflow
    assert "pull_request_target" not in workflow

def test_public_library_metadata_has_all_eleven_sources():
    from ancestor.public_library import PUBLISHED_SOURCES
    assert len(PUBLISHED_SOURCES)==11
    assert [x[0] for x in PUBLISHED_SOURCES]==[
        f"WORK-{i:03d}" for i in range(1,12)]
    assert sum(len(x[2]) for x in PUBLISHED_SOURCES)==15
    for document_id,title,paths,checksums in PUBLISHED_SOURCES:
        assert title.strip()
        assert len(paths)==len(checksums)
        for name,checksum in zip(paths,checksums):
            assert (ROOT/"published_library"/name).is_file()
            assert len(checksum)==40
