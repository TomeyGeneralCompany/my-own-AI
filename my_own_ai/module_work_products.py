"""Owner-facing local instruments. No generated strategies or replacement neural model."""
import json
from pathlib import Path
from hashlib import sha256
from xml.sax.saxutils import escape
from zipfile import ZipFile, ZIP_DEFLATED
from .finance_module import strict_json
from .functional_modules import grounded_evidence


def append_exchange(history, question, answer):
    """Preserve the actual exchange without rewriting it."""
    if not isinstance(history,list) or any(
        not isinstance(row,dict) or set(row)!={"question","answer"} or
        any(not isinstance(v,str) for v in row.values()) for row in history):
        raise ValueError("Invalid dialogue history")
    if any(not isinstance(v,str) or not v.strip() for v in (question,answer)):
        raise ValueError("Expected actual question and answer")
    return [*history,{"question":question,"answer":answer}]


def assemble_manuscript(sections, order, destination):
    """Export owner-selected section artifacts in the specified order; do not generate prose."""
    if not isinstance(sections,dict) or not isinstance(order,list) or not order:
        raise ValueError("Expected manuscript sections and explicit order")
    if any(not isinstance(key,str) for key in order) or len(set(order))!=len(order) or set(order)!=set(sections):
        raise ValueError("Section order must identify every section exactly once")
    if any(not isinstance(v,str) or not v.strip() for v in sections.values()):
        raise ValueError("Empty or invalid section")
    destination=Path(destination)
    if destination.suffix.lower()!=".docx":
        raise ValueError("Expected DOCX destination")
    # Minimal OOXML with Times New Roman, 12 pt, double spacing.
    paras=[]
    for key in order:
        for line in sections[key].splitlines():
            if any(ord(c)<32 and c not in "\t\n\r" for c in line):
                raise ValueError("Invalid document character")
            paras.append('<w:p><w:pPr><w:spacing w:line="480" w:lineRule="auto"/></w:pPr><w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="24"/></w:rPr><w:t xml:space="preserve">'+escape(line)+'</w:t></w:r></w:p>')
    document='<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'+''.join(paras)+'<w:sectPr/></w:body></w:document>'
    with ZipFile(destination,"x",compression=ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml",'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
        archive.writestr("_rels/.rels",'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
        archive.writestr("word/document.xml",document)
    return {"sections":len(order),"sha256":sha256(destination.read_bytes()).hexdigest(),
            "semantic_review_completed":False}


def compare_observations(candidate, observations):
    """Check cited empirical observations, without deciding philosophical conclusions."""
    outcome, inference, error = grounded_evidence(candidate, observations)
    return {"citation_numbers":sorted(outcome),"inference":inference,"error":error,
            "semantic_entailment_verified":False}


def check_engineering_artifact(source, expected_sha256=None):
    """Independently fingerprint and compile proposed Python, without executing it."""
    if not isinstance(source,str):
        raise ValueError("Expected source text")
    identity=sha256(source.encode("utf-8")).hexdigest()
    if expected_sha256 is not None and expected_sha256!=identity:
        raise ValueError("Artifact differs from owner-reviewed identity")
    try:
        compile(source,"<proposal>","exec")
    except (ValueError,SyntaxError,OverflowError,RecursionError) as exc:
        return {"sha256":identity,"compiled":False,"error":str(exc),"functional_tests_passed":False}
    return {"sha256":identity,"compiled":True,"functional_tests_passed":False}
