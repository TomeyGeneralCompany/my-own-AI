"""Checkpoint storage only: simulate a full GitHub release without real deletion."""
import pytest
from scripts.stage_checkpoint_storage import latest, rotate


class FakeRelease:
    def __init__(self, count=1000):
        self.assets = [
            {"id": i + 1, "name": f"stage-state-{100000+i}-1.zip",
             "state": "uploaded", "size": 42,
             "created_at": f"{i:06d}"}
            for i in range(count)
        ]
        self.deleted = []
        self.read_pages = []

    def __call__(self, path, *, delete=False):
        if delete:
            asset_id = int(path.rsplit("/", 1)[-1])
            self.deleted.append(asset_id)
            self.assets = [a for a in self.assets if a["id"] != asset_id]
            return None
        if "/releases/tags/" in path:
            return {"id": 12}
        page = int(path.rsplit("=", 1)[-1])
        self.read_pages.append(page)
        return self.assets[(page-1)*100:page*100]


def test_full_release_rotates_only_oldest_and_restores_latest():
    release = FakeRelease()
    before = latest("owner/repo", "stage", release)
    assert before == "stage-state-100999-1.zip"
    assert 11 in release.read_pages
    assert rotate("owner/repo", "stage", release) == 20
    assert release.deleted == list(range(1, 21))
    assert len(release.assets) == 980
    assert latest("owner/repo", "stage", release) == before
    release.assets.append({"id": 1001, "name": "stage-state-101000-1.zip",
                           "state": "uploaded", "size": 42, "created_at": "001000"})
    assert latest("owner/repo", "stage", release) == "stage-state-101000-1.zip"
    assert rotate("owner/repo", "stage", release) == 0


def test_rotation_protects_unrelated_and_incomplete_assets():
    release = FakeRelease()
    release.assets[0]["name"] = "unrelated.zip"
    release.assets[1]["state"] = "starter"
    release.assets[-1]["state"] = "starter"
    assert latest("owner/repo", "stage", release) == "stage-state-100998-1.zip"
    assert rotate("owner/repo", "stage", release) == 20
    assert 1 not in release.deleted and 2 not in release.deleted
    assert 999 not in release.deleted and 1000 not in release.deleted


def test_failed_inventory_or_insufficient_recovery_state_fails_closed():
    release = FakeRelease()
    for asset in release.assets[:-2]:
        asset["name"] = "unrelated-" + str(asset["id"])
    with pytest.raises(ValueError, match="Insufficient"):
        rotate("owner/repo", "stage", release)
    assert not release.deleted
    release.assets[-1]["state"] = "starter"
    release.assets[-2]["state"] = "starter"
    with pytest.raises(ValueError, match="no usable"):
        latest("owner/repo", "stage", release)


def test_small_release_is_untouched():
    release = FakeRelease(100)
    assert rotate("owner/repo", "stage", release) == 0
    assert not release.deleted
