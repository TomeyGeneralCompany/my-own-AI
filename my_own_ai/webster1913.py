"""Immutable 1913 Webster source: complete StarDict index, all record text retained."""
import gzip
import hashlib
import io
from pathlib import Path
import struct
import tarfile
from urllib.request import urlopen

REV = "7f9c6d6725eb1b0d8545a6f74904610fa2a2c71e"
BLOB = "a0cf8f69bdd5168e22265e367fbbcbfdc462e7eb"
URL = ("https://raw.githubusercontent.com/ahacop/websters-dict-1913-stardict/"
       + REV + "/stardict-dictd-web1913-2.4.2.tgz")
SIZE = 28584542

def git_blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()

def load_archive(path):
    raw = Path(path).read_bytes()
    if len(raw) != SIZE or git_blob(raw) != BLOB:
        raise ValueError("1913 dictionary archive integrity mismatch")
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        def member(suffix):
            matches = [m for m in archive.getmembers() if m.isfile() and m.name.endswith(suffix)]
            if len(matches) != 1 or matches[0].size > 200_000_000:
                raise ValueError("Invalid dictionary archive member " + suffix)
            return archive.extractfile(matches[0]).read()
        info = dict(line.split("=",1) for line in member(".ifo").decode("utf-8-sig").splitlines() if "=" in line)
        index = member(".idx")
        data = gzip.decompress(member(".dict.dz"))
    if info.get("idxoffsetbits","32") != "32" or info.get("sametypesequence") != "m":
        raise ValueError("Unsupported StarDict encoding: " + str(info))
    records = parse_index(index, data)
    if len(index) != int(info["idxfilesize"]) or len(records) != int(info["wordcount"]):
        raise ValueError("Dictionary index coverage mismatch")
    if len(records) < 100000:
        raise ValueError("Incomplete dictionary")
    return records

def parse_index(index, data):
    records=[]
    cursor=0
    while cursor < len(index):
        end=index.index(b"\0",cursor)
        word=index[cursor:end].decode("utf-8")
        offset,size=struct.unpack_from(">II",index,end+1)
        if not word or not size or offset+size > len(data):
            raise ValueError("Invalid dictionary record")
        definition=data[offset:offset+size].decode("utf-8").strip()
        if not definition:
            raise ValueError("Empty dictionary record")
        records.append((word,definition))
        cursor=end+9
    return records

def acquire(folder):
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/"webster1913.tgz"
    if not path.exists():
        with urlopen(URL,timeout=90) as response:
            raw=response.read(SIZE+1)
        if len(raw)!=SIZE or git_blob(raw)!=BLOB:
            raise ValueError("Invalid download")
        tmp=path.with_suffix(".tmp")
        tmp.write_bytes(raw)
        tmp.replace(path)
    return load_archive(path)
