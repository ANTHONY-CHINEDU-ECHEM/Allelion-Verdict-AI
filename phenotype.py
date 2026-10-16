"""Phenotype specificity for PP4: how well a patient's HPO terms match a gene's disease profiles.

Score: the best, over the gene's diseases, of the best match average Resnik
similarity from patient terms to disease terms. Empirical p value: the share
of a fixed background of 300 genes (every k-th gene with a disease profile, in
alphabetical order) whose best score is at least as high, with the usual plus
one correction. "Specific" means the best matching disease has at most two
associated genes, i.e. a disease with essentially a single genetic cause.
The TypeScript engine (web/src/engine/phenotype.ts) uses the same background.
"""
from __future__ import annotations

from functools import cached_property

import pandas as pd

from allelion.config import KB
from allelion.kb.phenotypes import load_ontology

BACKGROUND = 300


class PhenotypeMatcher:
    @cached_property
    def ont(self):
        return load_ontology()

    @cached_property
    def gene_dis(self) -> dict:
        g = pd.read_parquet(KB / "genes.parquet", columns=["gene", "diseases"])
        return {r.gene: list(r.diseases) for r in g.itertuples()}

    @cached_property
    def dis_terms(self) -> dict:
        d = pd.read_parquet(KB / "diseases.parquet", columns=["disease_id", "terms"])
        return {r.disease_id: list(r.terms) for r in d.itertuples() if len(r.terms)}

    @cached_property
    def dis_gene_count(self) -> dict:
        c: dict = {}
        for g, ds in self.gene_dis.items():
            for d in ds:
                c[d] = c.get(d, 0) + 1
        return c

    @cached_property
    def background(self) -> list[str]:
        genes = sorted(g for g, ds in self.gene_dis.items() if any(d in self.dis_terms for d in ds))
        step = max(1, len(genes) // BACKGROUND)
        return genes[::step][:BACKGROUND]

    def gene_score(self, gene: str, terms: list[str]) -> tuple[float, str | None]:
        best, best_d = 0.0, None
        for d in self.gene_dis.get(gene, []):
            dt = self.dis_terms.get(d)
            if not dt:
                continue
            s = self.ont.bma(terms, dt)
            if s > best:
                best, best_d = s, d
        return best, best_d

    def match(self, gene: str, terms: list[str]) -> dict:
        terms = [t for t in terms if t in self.ont.terms]
        score, d = self.gene_score(gene, terms)
        bg = [self.gene_score(g, terms)[0] for g in self.background]
        p = (1 + sum(1 for s in bg if s >= score)) / (1 + len(bg))
        return {"score": round(score, 4), "p_value": round(p, 4), "best_disease": d,
                "specific": bool(d) and self.dis_gene_count.get(d, 0) <= 2, "n_terms": len(terms)}
