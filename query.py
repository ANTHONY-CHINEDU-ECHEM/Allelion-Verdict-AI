"""Query understanding: entities and intent from a plain language question."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

from allelion.config import KB
from allelion.kb.hgvs import parse_c, parse_p

CODE_RE = re.compile(r"\b(PVS1|PS[1-4]|PM[1-6]|PP[1-5]|BA1|BS[1-4]|BP[1-7])\b", re.I)
HGVS_C_RE = re.compile(r"(?:[A-Z]{2}_\d+(?:\.\d+)?:)?c\.[-*]?\d+(?:[+-]\d+)?(?:_[-*]?\d+(?:[+-]\d+)?)?(?:[ACGT]+>[ACGT]+|del[ACGT]*ins[ACGT]+|del[ACGT]*|dup[ACGT]*|ins[ACGT]+)")
HGVS_P_RE = re.compile(r"(?:p\.)?\(?(?:Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val|Ter)\d+(?:Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val|Ter|=|fs\w*|del|dup)\)?")
PMID_RE = re.compile(r"\bPMID[:\s]*(\d{6,9})\b", re.I)
HPO_ID_RE = re.compile(r"\bHP:\d{7}\b")
DISEASE_ID_RE = re.compile(r"\b(?:OMIM|ORPHA):\d+\b")

INTENT_WORDS = {
    "phenotype": ["define", "definition", "what does", "mean", "what is a", "what is an", "term"],
    "disease": ["features", "phenotypes", "clinical", "characterise", "characterize", "symptoms", "syndrome", "disease", "inheritance of"],
    "literature": ["study", "report", "paper", "abstract", "published", "described", "reported", "found", "patients", "family", "families"],
    "guidance": ["criterion", "criteria", "guideline", "points", "strength", "should", "apply", "applied", "svi", "clingen", "acmg",
                 "downgrade", "combine", "threshold", "rule", "recommend"],
    "gene": ["gene", "constraint", "pli", "loftool", "mechanism", "haploinsufficiency", "spectrum", "exon", "evidence pool"],
}


@dataclass
class QueryInfo:
    text: str
    genes: list[str] = field(default_factory=list)
    codes: list[str] = field(default_factory=list)
    hgvs_c: list[str] = field(default_factory=list)
    hgvs_p: list[str] = field(default_factory=list)
    pmids: list[str] = field(default_factory=list)
    hpo: list[str] = field(default_factory=list)
    diseases: list[str] = field(default_factory=list)
    intent: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@lru_cache(maxsize=1)
def _lexicons():
    genes = set(pd.read_parquet(KB / "genes.parquet", columns=["gene"])["gene"])
    terms = pd.read_parquet(KB / "hpo_terms.parquet", columns=["hpo_id", "name", "synonyms"])
    hpo_names = {}
    for r in terms.itertuples():
        for n in [r.name] + list(r.synonyms):
            n = n.lower().strip()
            if len(n) >= 6:
                hpo_names.setdefault(n, r.hpo_id)
    dis = pd.read_parquet(KB / "diseases.parquet", columns=["disease_id", "name"])
    dnames = {}
    for r in dis.itertuples():
        n = re.sub(r"\s+", " ", r.name.lower()).strip()
        if len(n) >= 6 and " " in n:
            dnames.setdefault(n, []).append(r.disease_id)
    return genes, hpo_names, dnames


def understand(q: str) -> QueryInfo:
    genes_all, hpo_names, dnames = _lexicons()
    info = QueryInfo(text=q)
    bare = re.sub(r"\[[A-Z_0-9]+\]", " ", q)  # redaction placeholders such as [NHS_NUMBER] are not genes
    for w in re.findall(r"\b[A-Z][A-Z0-9]{1,9}\b", bare):
        if w in genes_all and w not in info.genes and not CODE_RE.fullmatch(w) and w not in ("DNA", "RNA", "SVI", "ACMG", "AMP", "PMID", "HPO", "OMIM", "UTR"):
            info.genes.append(w)
    info.codes = sorted({c.upper() for c in CODE_RE.findall(q)})
    info.hgvs_c = [m for m in HGVS_C_RE.findall(q) if parse_c(m)]
    info.hgvs_p = [m for m in HGVS_P_RE.findall(q) if parse_p(m if m.startswith("p.") else "p." + m.strip("()"))]
    info.pmids = PMID_RE.findall(q)
    info.hpo = HPO_ID_RE.findall(q)
    info.diseases = DISEASE_ID_RE.findall(q)
    ql = " " + re.sub(r"[^a-z0-9 ]+", " ", q.lower()) + " "
    ql = re.sub(r"\s+", " ", ql)
    # longest phenotype and disease name matches
    for name, hid in sorted(hpo_names.items(), key=lambda x: -len(x[0])):
        if f" {name} " in ql and hid not in info.hpo:
            info.hpo.append(hid)
            if len(info.hpo) >= 3:
                break
    for name, ids in sorted(dnames.items(), key=lambda x: -len(x[0])):
        key = re.sub(r"[^a-z0-9 ]+", " ", name)
        key = re.sub(r"\s+", " ", key).strip()
        if len(key) >= 8 and f" {key} " in ql:
            for d in ids:
                if d not in info.diseases:
                    info.diseases.append(d)
            if len(info.diseases) >= 4:
                break
    low = q.lower()
    for k, words in INTENT_WORDS.items():
        info.intent[k] = float(sum(1 for w in words if w in low))
    if info.codes:
        info.intent["guidance"] += 2
    if info.genes:
        info.intent["gene"] += 1
    return info
