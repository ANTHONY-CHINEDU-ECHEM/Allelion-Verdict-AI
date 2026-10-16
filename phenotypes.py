"""Human Phenotype Ontology: terms, gene to disease links, disease profiles and information content.

Used by the phenotype specificity criterion (PP4) and by the knowledge graph.
Phenotype similarity follows the usual best match average over the
information content of the most informative common ancestor (Resnik).
"""
from __future__ import annotations

import json
import math
import pickle
from collections import defaultdict

import pandas as pd

from allelion.config import KB, RAW

INHERITANCE_TERMS = {
    "HP:0000006": "AD", "HP:0000007": "AR", "HP:0001417": "XL", "HP:0001419": "XLR", "HP:0001423": "XLD",
    "HP:0001427": "MT", "HP:0001428": "SOM", "HP:0010982": "POLY", "HP:0001450": "Y",
}
ONSET_ROOT = "HP:0003674"


def parse_obo(path=RAW / "hpo" / "hp.obo") -> dict[str, dict]:
    terms, cur = {}, None
    for line in open(path, encoding="utf8"):
        line = line.rstrip("\n")
        if line == "[Term]":
            cur = {"parents": [], "synonyms": []}
            continue
        if line.startswith("[") and line.endswith("]"):
            cur = None
            continue
        if cur is None or ": " not in line:
            continue
        k, v = line.split(": ", 1)
        if k == "id":
            cur["id"] = v
            terms[v] = cur
        elif k == "name":
            cur["name"] = v
        elif k == "def":
            cur["def"] = v.split('" [')[0].strip('"')
        elif k == "synonym":
            cur["synonyms"].append(v.split('"')[1])
        elif k == "is_a":
            cur["parents"].append(v.split(" ! ")[0])
        elif k == "is_obsolete" and v == "true":
            cur["obsolete"] = True
    return {k: v for k, v in terms.items() if not v.get("obsolete")}


class Ontology:
    def __init__(self, terms: dict[str, dict]):
        self.terms = terms
        self._anc: dict[str, frozenset] = {}
        self.ic: dict[str, float] = {}

    def ancestors(self, t: str) -> frozenset:
        if t in self._anc:
            return self._anc[t]
        out = {t}
        for p in self.terms.get(t, {}).get("parents", []):
            out |= self.ancestors(p)
        self._anc[t] = frozenset(out)
        return self._anc[t]

    def fit_ic(self, disease_terms: dict[str, set[str]]) -> None:
        counts = defaultdict(int)
        for ts in disease_terms.values():
            closed = set()
            for t in ts:
                closed |= self.ancestors(t)
            for t in closed:
                counts[t] += 1
        n = max(len(disease_terms), 1)
        self.ic = {t: -math.log(c / n) for t, c in counts.items()}

    def resnik(self, a: str, b: str) -> float:
        common = self.ancestors(a) & self.ancestors(b)
        return max((self.ic.get(t, 0.0) for t in common), default=0.0)

    def bma(self, query: list[str], target: list[str]) -> float:
        """Best match average similarity of a query profile against a target profile (query to target direction)."""
        if not query or not target:
            return 0.0
        return sum(max(self.resnik(q, t) for t in target) for q in query) / len(query)


def build() -> dict:
    terms = parse_obo()
    hpoa = pd.read_csv(RAW / "hpo" / "phenotype.hpoa", sep="\t", comment="#", dtype=str)
    if "database_id" not in hpoa.columns:
        hpoa = pd.read_csv(RAW / "hpo" / "phenotype.hpoa", sep="\t", skiprows=4, dtype=str)
    g2p = pd.read_csv(RAW / "hpo" / "genes_to_phenotype.txt", sep="\t", dtype=str)

    disease_name = hpoa.groupby("database_id")["disease_name"].first().to_dict()
    pheno = hpoa[(hpoa["aspect"] == "P") & (hpoa["qualifier"].fillna("") != "NOT")]
    disease_terms = pheno.groupby("database_id")["hpo_id"].apply(lambda s: sorted(set(s))).to_dict()
    freq_terms = {"HP:0040280": 1.0, "HP:0040281": 0.9, "HP:0040282": 0.55, "HP:0040283": 0.17, "HP:0040284": 0.02, "HP:0040285": 0.0}

    def _freq(x) -> float:
        if not isinstance(x, str) or not x:
            return 0.5
        if x in freq_terms:
            return freq_terms[x]
        if "/" in x:
            a, b = x.split("/", 1)
            return float(a) / float(b) if b.strip("0") else 0.5
        if x.endswith("%"):
            return float(x[:-1]) / 100
        return 0.5
    pheno = pheno.assign(f=pheno["frequency"].map(_freq))
    disease_freq: dict = defaultdict(dict)
    for (d, t), f in pheno.groupby(["database_id", "hpo_id"])["f"].max().items():
        disease_freq[d][t] = round(float(f), 3)
    inh = hpoa[hpoa["hpo_id"].isin(INHERITANCE_TERMS)]
    disease_inh = inh.groupby("database_id")["hpo_id"].apply(lambda s: sorted({INHERITANCE_TERMS[x] for x in s})).to_dict()
    onset = hpoa[hpoa["aspect"] == "C"]
    disease_onset = onset.groupby("database_id")["hpo_id"].apply(lambda s: sorted(set(s))).to_dict()

    gene_diseases = g2p.groupby("gene_symbol")["disease_id"].apply(lambda s: sorted(set(s))).to_dict()
    ont = Ontology(terms)
    ont.fit_ic({d: set(ts) for d, ts in disease_terms.items()})

    diseases = []
    for d, name in disease_name.items():
        diseases.append({"disease_id": d, "name": name, "terms": disease_terms.get(d, []),
                         "term_freq": json.dumps(disease_freq.get(d, {})),
                         "inheritance": disease_inh.get(d, []), "onset": disease_onset.get(d, [])})
    pd.DataFrame(diseases).to_parquet(KB / "diseases.parquet", index=False)
    term_rows = [{"hpo_id": k, "name": v.get("name", ""), "definition": v.get("def", ""), "synonyms": v.get("synonyms", []),
                  "parents": v.get("parents", []), "ic": round(ont.ic.get(k, 0.0), 4)} for k, v in terms.items()]
    pd.DataFrame(term_rows).to_parquet(KB / "hpo_terms.parquet", index=False)
    (KB / "gene_diseases.json").write_text(json.dumps(gene_diseases))
    with open(KB / "ontology.pkl", "wb") as fh:
        pickle.dump({"terms": {k: {"parents": v["parents"], "name": v.get("name", "")} for k, v in terms.items()}, "ic": ont.ic}, fh)
    stats = {"terms": len(terms), "diseases": len(diseases), "diseases_with_profiles": len(disease_terms),
             "genes_with_diseases": len(gene_diseases), "terms_with_definition": sum(1 for t in term_rows if t["definition"])}
    print(stats)
    return stats


def load_ontology() -> Ontology:
    with open(KB / "ontology.pkl", "rb") as fh:
        d = pickle.load(fh)
    ont = Ontology(d["terms"])
    ont.ic = d["ic"]
    return ont


if __name__ == "__main__":
    build()
