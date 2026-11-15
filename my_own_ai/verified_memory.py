"""Persistent verified answer index. Only independently verified outputs enter it."""
import json
import re
from hashlib import sha256
from pathlib import Path
from collections import defaultdict

def normalize(s):
    return " ".join(s.split()).casefold()

def key(s):
    return sha256(normalize(s).encode()).hexdigest()

class VerifiedMemory:
    def __init__(self, records=None):
        self.records={} if records is None else records
        self.index=defaultdict(set)
        for k,r in self.records.items():
            if key(r["question"])!=k or not r["answer"].strip():
                raise ValueError("Invalid verified memory record")
            self._index(k,r["question"])
    def _index(self,k,q):
        for word in set(re.findall(r"\w+",normalize(q))):
            self.index[word].add(k)
    def remember(self,question,actual,*,expected,step,provenance):
        if not actual.strip() or normalize(actual)!=normalize(expected):
            raise ValueError("Only verified answers may be indexed")
        k=key(question)
        prior=self.records.get(k)
        if prior:
            if normalize(prior["answer"])!=normalize(actual):
                raise ValueError("Conflicting verified answer")
            return False
        self.records[k]={"question":question,"answer":actual,"step":step,"provenance":provenance}
        self._index(k,question)
        return True
    def exact(self,question):
        return self.records.get(key(question))
    def related(self,question,limit=3):
        words=set(re.findall(r"\w+",normalize(question)))
        scores=defaultdict(float)
        for word in words:
            keys=self.index.get(word,())
            if not keys or len(keys)>4096:
                continue
            for k in keys:
                scores[k]+=1/len(keys)
        return [self.records[k] for k in sorted(scores,key=lambda k:(-scores[k],k))[:limit]]
    def contextual(self,question,tokenizer,budget):
        prefix=""
        for row in self.related(question):
            candidate=prefix+"Previously verified: "+row["question"]+"\nAnswer: "+row["answer"]+"\n"
            if len(tokenizer.encode(candidate+question))<=budget:
                prefix=candidate
        return prefix+question
    def import_exam(self,path,step):
        if not Path(path).is_file():
            return 0
        report=json.loads(Path(path).read_text())
        rows=report.get("raw_results",{}).get("results",[])
        added=0
        for r in rows:
            actual=r.get("answer",r.get("actual",r.get("response","")))
            expected=r.get("reference_answer",r.get("reference",""))
            if r.get("exact_match") is True and isinstance(actual,str) and isinstance(expected,str) and actual.strip() and normalize(actual)==normalize(expected):
                added+=self.remember(r["question"],actual,expected=expected,step=step,provenance="prior_exact_exam")
        return added
