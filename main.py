"""FastAPI service. The React interface talks to it when VITE_API_URL is set, otherwise it runs the same contract in the browser.

Run:  uvicorn allelion.api.main:app --port 8000
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from allelion.acmg.engine import Engine, prepare
from allelion.acmg.phenotype import PhenotypeMatcher
from allelion.config import CLASSES, KB, MODELS, REPORTS, ROOT
from allelion.copilot.copilot import Copilot, read_audit
from allelion.models import meta

app = FastAPI(title="Allelion Verdict AI", version="1.0.0",
              description="ACMG/AMP variant interpretation with calibrated evidence, a cited copilot and a blind evaluation.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@lru_cache(maxsize=1)
def engine() -> Engine:
    return Engine()


@lru_cache(maxsize=1)
def copilot() -> Copilot:
    return Copilot(engine=engine())


@lru_cache(maxsize=1)
def matcher() -> PhenotypeMatcher:
    return PhenotypeMatcher()


@lru_cache(maxsize=1)
def variants() -> pd.DataFrame:
    v = pd.read_parquet(KB / "variants.parquet")
    v["c_short"] = v["hgvs_c"].fillna("").str.split(":").str[-1]
    return v


@lru_cache(maxsize=1)
def meta_model():
    return meta.load()


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_clean(v) for v in o.tolist()]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


class Population(BaseModel):
    ac: float | None = None
    an: float | None = None
    grpmax_af: float | None = None
    faf95: float | None = None
    hom: float | None = None
    hemi: float | None = None


class ClassifyRequest(BaseModel):
    variation_id: str | None = None
    gene: str | None = None
    hgvs_c: str | None = None
    hgvs_p: str | None = None
    population: Population | None = None
    case: dict = Field(default_factory=dict)


class CopilotRequest(BaseModel):
    question: str


class PhenotypeRequest(BaseModel):
    gene: str | None = None
    terms: list[str]


class BatchRequest(BaseModel):
    lines: list[str]


def _lookup(req: ClassifyRequest) -> dict:
    v = variants()
    if req.variation_id:
        hit = v[v["variation_id"] == req.variation_id]
        if hit.empty:
            raise HTTPException(404, "variant not found")
        return engine().score_batch(hit.head(1)).iloc[0].to_dict()
    if not req.gene or not req.hgvs_c:
        raise HTTPException(422, "give a variation_id, or a gene and a coding HGVS description")
    cs = req.hgvs_c.split(":")[-1]
    hit = v[(v["gene"] == req.gene) & (v["c_short"] == cs)]
    if not hit.empty and req.population is None:
        return engine().score_batch(hit.head(1)).iloc[0].to_dict()
    try:
        pop = req.population.model_dump(exclude_none=True) if req.population else None
        return prepare(req.gene, cs, req.hgvs_p, pop)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/classify")
def classify(req: ClassifyRequest):
    v = _lookup(req)
    case = dict(req.case or {})
    if case.get("phenotype_terms"):
        case["phenotype_match"] = matcher().match(v["gene"], case.pop("phenotype_terms"))
    res = engine().evaluate(v, case)
    X = meta.frame(engine(), pd.DataFrame([v]))
    probs = meta_model().predict(X)[0]
    res["learned_verdict"] = {"classification": CLASSES[int(np.argmax(probs))], "probabilities": dict(zip(CLASSES, [round(float(p), 4) for p in probs]))}
    res["reference"] = {"clinvar_class": v.get("clinvar_class"), "stars": v.get("stars"), "note": "Shown for comparison only; never used as evidence."}
    res["population"] = {k: v.get(k) for k in ("gn_AC", "gn_AN", "gn_faf95", "gn_grpmax_2000", "gn_Hom", "gn_Hemi")}
    return _clean(res)


@app.get("/api/variants/search")
def search_variants(q: str = "", gene: str | None = None, limit: int = Query(25, le=200)):
    v = variants()
    sub = v[v["gene"] == gene] if gene else v
    if q:
        ql = q.lower()
        sub = sub[sub["c_short"].str.lower().str.contains(ql, regex=False) | sub["protein_change"].str.lower().str.contains(ql, regex=False)
                  | sub["hgvs_p"].fillna("").str.lower().str.contains(ql, regex=False)]
    cols = ["variation_id", "gene", "hgvs_c", "hgvs_p", "consequence", "clinvar_class", "stars", "gn_faf95"]
    return _clean(sub.head(limit)[cols].to_dict("records"))


@app.get("/api/genes/{symbol}")
def gene(symbol: str):
    g = engine().r.gene(symbol)
    if not g:
        raise HTTPException(404, "gene not found")
    stats = engine().r.tables.stats(symbol)
    tx = engine().r.transcripts.get(symbol)
    T = engine().r.tables
    mis = [[pos, r[0], r[2]] for (gg, pos), lst in T.by_residue.items() if gg == symbol for r in lst]
    return _clean({"gene": symbol, **g, "pool_stats": stats, "missense": mis, "truncating": T.plp_truncating.get(symbol, np.array([])).tolist(),
                   "transcript": tx.to_dict() if tx is not None and tx.anchored else None})


@app.post("/api/copilot")
def ask(req: CopilotRequest):
    return _clean(copilot().answer(req.question))


@app.post("/api/phenotype/match")
def phenotype_match(req: PhenotypeRequest):
    if not req.gene:
        raise HTTPException(422, "gene required")
    return matcher().match(req.gene, req.terms)


@app.post("/api/phenotype/rank")
def phenotype_rank(req: PhenotypeRequest, limit: int = 25):
    m = matcher()
    scored = [(g, *m.gene_score(g, req.terms)) for g in m.gene_dis if m.gene_dis[g]]
    scored = sorted([x for x in scored if x[1] > 0], key=lambda x: -x[1])[:limit]
    return [{"gene": g, "score": round(s, 4), "disease": d} for g, s, d in scored]


@app.post("/api/batch")
def batch(req: BatchRequest):
    out = []
    for line in req.lines[:500]:
        parts = line.replace(",", " ").replace("\t", " ").split()
        if len(parts) < 2:
            out.append({"input": line, "error": "expected: GENE c.HGVS [p.HGVS]"})
            continue
        try:
            r = classify(ClassifyRequest(gene=parts[0], hgvs_c=parts[1], hgvs_p=parts[2] if len(parts) > 2 else None))
            out.append({"input": line, "classification": r["classification"], "points": r["points"], "met": r["met"],
                        "learned": r["learned_verdict"]["classification"], "reference": r["reference"]["clinvar_class"]})
        except HTTPException as e:
            out.append({"input": line, "error": e.detail})
    return out


@app.get("/api/conflicts")
def conflicts(gene: str | None = None, limit: int = 100):
    v = variants()
    sub = v[v["conflicted"] == 1]
    if gene:
        sub = sub[sub["gene"] == gene]
    rows = []
    for r in sub.head(limit).to_dict("records"):
        res = engine().evaluate(engine().score_batch(pd.DataFrame([r])).iloc[0].to_dict())
        rows.append({"variation_id": r["variation_id"], "gene": r["gene"], "hgvs_c": r["hgvs_c"], "submissions": r["submissions"],
                     "engine": res["classification"], "points": res["points"], "met": res["met"]})
    return _clean(rows)


@app.get("/api/calibration")
def calibration():
    return {k: json.loads((MODELS / f"{k}_calibration.json").read_text()) for k in ("missense", "splice")}


@app.get("/api/evaluation")
def evaluation():
    p = REPORTS / "evaluation_summary.json"
    return json.loads(p.read_text()) if p.exists() else {}


@app.get("/api/audit")
def audit(limit: int = 200):
    return read_audit(limit)


dist = ROOT / "web" / "dist"
if dist.exists():
    app.mount("/", StaticFiles(directory=dist, html=True), name="web")
