"""Hybrid retrieval: candidate generators, fusion, a learned reranker and diversity.

Generators
    bm25        BM25 over all passages
    sif         SIF word vector cosine
    dual        dual encoder cosine
    entity      BM25 restricted to passages about the genes, codes, diseases,
                phenotype terms or PubMed identifiers named in the question
    kind        BM25 restricted to the passage kind the question's intent points to
Fusion: reciprocal rank fusion of every generator (k = 60).
Reranking: LightGBM LambdaMART over per candidate features (see FEATURES).
Diversity: at most two passages from the same source in the final list.
"""
from __future__ import annotations

import json
import time
from functools import cached_property

import lightgbm as lgb
import numpy as np
import pandas as pd

from allelion.config import MODELS
from allelion.retrieval import dense
from allelion.retrieval.bm25 import BM25
from allelion.retrieval.query import QueryInfo, understand
from allelion.retrieval.text import tokenize

KINDS = ["guidance", "gene", "disease", "phenotype", "literature"]
RRF_K = 60
FEATURES = (
    ["bm25", "bm25_rank", "bm25_norm", "sif", "sif_rank", "dual", "dual_rank", "entity_rank", "kind_rank", "rrf", "rrf_rank",
     "n_generators", "q_coverage", "title_coverage", "len_log", "gene_in_source", "gene_in_genes", "code_match", "disease_match",
     "hpo_match", "pmid_match", "n_query_genes", "n_query_codes"]
    + [f"kind_{k}" for k in KINDS] + [f"intent_{k}" for k in KINDS] + [f"intent_x_kind_{k}" for k in KINDS]
)


class Retriever:
    def __init__(self):
        self.passages = pd.read_parquet(MODELS / "passages.parquet")
        self.bm25 = BM25.load()
        self.sif, self.de, self.P_sif, self.P_dual = dense.load()
        self.kind = self.passages["kind"].values
        self.source = self.passages["source"].values
        self.genes = self.passages["genes"].values
        self.title_toks = [set(tokenize(t)) for t in self.passages["title"]]
        self.text_toks = None
        self._by_source: dict[str, list[int]] = {}
        for i, s in enumerate(self.source):
            self._by_source.setdefault(s, []).append(i)
        self._codes = {}
        for i, cs in enumerate(self.passages["codes"]):
            for c in cs:
                self._codes.setdefault(c, []).append(i)
        self._gene_idx = {}
        for i, gs in enumerate(self.genes):
            for g in gs:
                self._gene_idx.setdefault(g, []).append(i)

    @cached_property
    def reranker(self):
        p = MODELS / "reranker.txt"
        return lgb.Booster(model_file=str(p)) if p.exists() else None

    @cached_property
    def lengths(self) -> np.ndarray:
        return np.log1p(self.passages["text"].str.split().map(len).values.astype(float))

    # ------------------------------------------------------------------ generators
    def entity_mask(self, qi: QueryInfo) -> np.ndarray | None:
        idx = set()
        for g in qi.genes:
            idx.update(self._by_source.get(f"gene:{g}", []))
            idx.update(self._gene_idx.get(g, []))
        for c in qi.codes:
            idx.update(self._codes.get(c, []))
        for d in qi.diseases:
            idx.update(self._by_source.get(f"disease:{d}", []))
        for h in qi.hpo:
            idx.update(self._by_source.get(f"hpo:{h}", []))
        for p in qi.pmids:
            for k in ("tmvar", "seth", "mf", "osiris"):
                idx.update(self._by_source.get(f"lit:{k}:{p}", []))
        if not idx:
            return None
        m = np.zeros(len(self.passages), dtype=bool)
        m[list(idx)] = True
        return m

    def kind_mask(self, qi: QueryInfo) -> np.ndarray | None:
        best = max(qi.intent.items(), key=lambda x: x[1])
        if best[1] <= 0:
            return None
        return self.kind == best[0]

    def candidates(self, q: str, qi: QueryInfo, depth: int = 100) -> tuple[dict, dict]:
        lists = {}
        bm = self.bm25.top(q, depth)
        lists["bm25"] = bm
        qs = self.sif.embed([q])
        s_sif = self.P_sif @ qs[0]
        top = np.argpartition(-s_sif, depth)[:depth]
        lists["sif"] = [(int(i), float(s_sif[i])) for i in top[np.argsort(-s_sif[top])]]
        qd = self.de.q(qs)[0]
        s_dual = self.P_dual @ qd
        top = np.argpartition(-s_dual, depth)[:depth]
        lists["dual"] = [(int(i), float(s_dual[i])) for i in top[np.argsort(-s_dual[top])]]
        em = self.entity_mask(qi)
        lists["entity"] = self.bm25.top(q, 50, em) if em is not None else []
        if em is not None and not lists["entity"]:
            lists["entity"] = [(int(i), 0.0) for i in np.where(em)[0][:50]]
        km = self.kind_mask(qi)
        lists["kind"] = self.bm25.top(q, 50, km) if km is not None else []
        raw = {"sif": s_sif, "dual": s_dual}
        return lists, raw

    @staticmethod
    def rrf(lists: dict, names: list[str]) -> list[tuple[int, float]]:
        sc = {}
        for n in names:
            for r, (i, _) in enumerate(lists.get(n, [])):
                sc[i] = sc.get(i, 0.0) + 1.0 / (RRF_K + r + 1)
        return sorted(sc.items(), key=lambda x: -x[1])

    # ------------------------------------------------------------------ features
    def features(self, q: str, qi: QueryInfo, lists: dict, raw: dict, cand: list[int]) -> pd.DataFrame:
        ranks = {n: {i: r for r, (i, _) in enumerate(l)} for n, l in lists.items()}
        bm_scores = dict(lists["bm25"])
        bm_all = self.bm25.scores(q)
        mx = max(bm_all.max(), 1e-6)
        fused = dict(self.rrf(lists, ["bm25", "sif", "dual", "entity", "kind"]))
        fused_rank = {i: r for r, (i, _) in enumerate(sorted(fused.items(), key=lambda x: -x[1]))}
        qt = set(tokenize(q))
        rows = []
        for i in cand:
            src = self.source[i]
            kind = self.kind[i]
            text = self.passages.at[i, "text"]
            tt = set(tokenize(self.passages.at[i, "title"] + " " + text))
            f = {
                "bm25": float(bm_all[i]), "bm25_rank": ranks["bm25"].get(i, 200), "bm25_norm": float(bm_all[i] / mx),
                "sif": float(raw["sif"][i]), "sif_rank": ranks["sif"].get(i, 200), "dual": float(raw["dual"][i]), "dual_rank": ranks["dual"].get(i, 200),
                "entity_rank": ranks["entity"].get(i, 200), "kind_rank": ranks["kind"].get(i, 200), "rrf": fused.get(i, 0.0),
                "rrf_rank": fused_rank.get(i, 500), "n_generators": sum(1 for n in ranks if i in ranks[n]),
                "q_coverage": len(qt & tt) / max(len(qt), 1), "title_coverage": len(qt & self.title_toks[i]) / max(len(qt), 1),
                "len_log": float(self.lengths[i]),
                "gene_in_source": float(any(src == f"gene:{g}" for g in qi.genes)),
                "gene_in_genes": float(any(g in self.genes[i] for g in qi.genes)),
                "code_match": float(any(c in self.passages.at[i, "codes"] for c in qi.codes)),
                "disease_match": float(any(src == f"disease:{d}" for d in qi.diseases)),
                "hpo_match": float(any(src == f"hpo:{h}" for h in qi.hpo)),
                "pmid_match": float(any(src.endswith(f":{p}") for p in qi.pmids)),
                "n_query_genes": len(qi.genes), "n_query_codes": len(qi.codes),
            }
            for k in KINDS:
                f[f"kind_{k}"] = float(kind == k)
                f[f"intent_{k}"] = qi.intent.get(k, 0.0)
                f[f"intent_x_kind_{k}"] = qi.intent.get(k, 0.0) * float(kind == k)
            rows.append(f)
        return pd.DataFrame(rows, index=cand)[FEATURES]

    # ------------------------------------------------------------------ search
    def search(self, q: str, k: int = 8, method: str = "full", max_per_source: int = 2) -> dict:
        t0 = time.perf_counter()
        qi = understand(q)
        lists, raw = self.candidates(q, qi)
        if method in ("bm25", "sif", "dual", "entity"):
            ranked = [i for i, _ in lists[method]]
            if method == "entity" and not ranked:
                ranked = [i for i, _ in lists["bm25"]]
        elif method == "bm25_dual":
            ranked = [i for i, _ in self.rrf(lists, ["bm25", "dual"])]
        elif method == "fused":
            ranked = [i for i, _ in self.rrf(lists, ["bm25", "sif", "dual", "entity", "kind"])]
        else:
            fused = self.rrf(lists, ["bm25", "sif", "dual", "entity", "kind"])
            cand = [i for i, _ in fused[:150]]
            X = self.features(q, qi, lists, raw, cand)
            s = self.reranker.predict(X) if self.reranker is not None else X["rrf"].values
            ranked = [cand[j] for j in np.argsort(-s)]
        out, seen = [], {}
        for i in ranked:
            src = self.source[i]
            if seen.get(src, 0) >= max_per_source:
                continue
            seen[src] = seen.get(src, 0) + 1
            out.append(int(i))
            if len(out) >= k:
                break
        return {"query": qi.to_dict(), "ids": out, "ms": round((time.perf_counter() - t0) * 1000, 1)}

    def passage(self, i: int) -> dict:
        r = self.passages.iloc[i]
        return {"pid": int(r["pid"]), "source": r["source"], "kind": r["kind"], "title": r["title"], "text": r["text"], "level": r["level"],
                "refs": list(r["refs"]), "genes": list(r["genes"])}
