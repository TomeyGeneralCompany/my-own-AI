from pathlib import Path
import pytest
from ancestor.importers import prepare_document,commit_reviewed_text
from ancestor.corpus import Corpus

def test_original_and_extracted_are_separate_and_reviewed(tmp_path):
    source=tmp_path/"work.txt"
    source.write_text("1. Definition.\n2. Terms. x is a variable.",encoding="utf-8")
    manifest=prepare_document(source,tmp_path/"prepared")
    assert manifest["approval_status"]=="PENDING_OWNER_REVIEW"
    assert Path(manifest["original_path"]).read_bytes()==source.read_bytes()
    assert Path(manifest["extracted_path"]).read_text()==source.read_text()
    corpus=Corpus(tmp_path/"db")
    manifests=list((tmp_path/"prepared").glob("manifest-*.json"))
    with pytest.raises(PermissionError):
        commit_reviewed_text(corpus,manifests[0],document_id="B1",title="Work",approved_hash="wrong")
    assert commit_reviewed_text(corpus,manifests[0],document_id="B1",
                                title="Work",approved_hash=manifest["extracted_sha256"])==2
    blocks=corpus.subblocks("B1:P-000002")
    assert any(item.kind=="terms" for item in blocks)
