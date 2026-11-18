"""Local library import requires actual exact bytes; cannot certify corrupted documents."""
import hashlib
import json
import zipfile
import pytest
from ancestor.private_library import import_private_library,export_training_text
from ancestor.corpus import Corpus

def sha(data):return hashlib.sha256(data).hexdigest()

def make_bundle(path,corrupt=False):
    sources=[]
    with zipfile.ZipFile(path,"w") as z:
        for name,ext,text,kind in [
            ("WORK-001","docx","[0] First.\n[1] Second.\n","word_paragraph"),
            ("WORK-002","pdf","Page one\fPage two","pdf_page")]:
            original=(name+" source").encode()
            data=text.encode()
            if kind=="word_paragraph":
                parts=[("[0] First.\n",0,11,1),("[1] Second.\n",11,len(text),2)]
                # Derive exact offsets so all bytes are verified.
                parts=[];offset=0
                for i,line in enumerate(text.splitlines(keepends=True),1):
                    parts.append((line,offset,offset+len(line),i))
                    offset+=len(line)
            else:
                parts=[("Page one",0,8,1),("Page two",9,len(text),2)]
            index=[]
            for line,start,end,i in parts:
                index.append({"id":f"{name}:W-{i:06d}" if kind=="word_paragraph"
                             else f"{name}:PAGE-{i:04d}",
                    "document_id":name,"kind":kind,"ordinal":i,
                    "original_number":i-1 if kind=="word_paragraph" else None,
                    "number_occurrence":1 if kind=="word_paragraph" else None,
                    "start_char":start,"end_char":end,"text":line,
                    "sha256":sha(line.encode())})
            rows="".join(json.dumps(x)+"\n" for x in index).encode()
            z.writestr(f"originals/{name}.{ext}",original)
            z.writestr(f"extracted/{name}.txt",data)
            z.writestr(f"index/{name}.jsonl",rows)
            sources.append({"id":name,"filename":name+"."+ext,"format":ext,
                "original_sha256":sha(original),
                "extracted_sha256":"invalid" if corrupt and name=="WORK-001" else sha(data),
                "records_sha256":sha(rows),"record_count":len(index)})
        z.writestr("manifest.json",json.dumps({"version":2,"source_count":2,
            "sources":sources}))
def test_exact_private_index_and_local_training_text(tmp_path):
    bundle=tmp_path/"library.zip"; make_bundle(bundle)
    result=import_private_library(bundle,tmp_path/"data")
    assert result["imported_source_count"]==2
    assert result["indexed_learning_units"]==4
    db=Corpus(tmp_path/"data")
    assert len(db.list_paragraphs("WORK-001"))==2
    assert len(db.list_paragraphs("WORK-002"))==2
    text=export_training_text(tmp_path/"data",tmp_path/"train.txt")
    assert text["document_count"]==2
    assert "First." in (tmp_path/"train.txt").read_text()
    # Reimport is idempotent; original documents do not change.
    assert import_private_library(bundle,tmp_path/"data")["indexed_learning_units"]==4

def test_corrupted_bundle_rejected_before_local_import(tmp_path):
    bundle=tmp_path/"bad.zip";make_bundle(bundle,corrupt=True)
    with pytest.raises(ValueError,match="Extracted text mismatch"):
        import_private_library(bundle,tmp_path/"data")
    assert not (tmp_path/"data"/"index.sqlite3").exists()
