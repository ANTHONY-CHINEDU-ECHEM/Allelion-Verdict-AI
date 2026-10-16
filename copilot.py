"""The evidence copilot: guard, query understanding, retrieval, variant evaluation, writing, verification, audit."""
from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from functools import cached_property

import pandas as pd

from allelion.acmg.engine import Engine, prepare
from allelion.config import AUDIT_LOG, KB
from allelion.copilot import guard
from allelion.copilot.verifier import Verifier
from allelion.copilot.writer import ExtractiveWriter, get_writer
from allelion.kb.hgvs import parse_c, parse_p
from allelion.retrieval.query import understand
from allelion.retrieval.retriever import Retriever

LEVEL_RANK = {"engine": 3, "guideline": 3, "curated database": 2, "ontology": 2, "literature": 1}


class Copilot:
    def __init__(self, retriever: Retriever | None = None, engine: Engine | None = None, writer=None):
        self.retriever = retriever or Retriever()
        self.engine = engine or Engine()
        self.writer = writer or get_writer()
        self.verifier = Verifier.load()

    @cached_property
    def variants(self) -> pd.DataFrame:
        v = pd.read_parquet(KB / "variants.parquet")
        v["c_short"] = v["hgvs_c"].fillna("").str.split(":").str[-1]
        v["p_short"] = v["hgvs_p"].fillna("").str.split(":").str[-1]
        return v

    # ------------------------------------------------------------------ variant resolution
    def resolve_variant(self, qi: dict) -> dict | None:
        if not qi["genes"] or not (qi["hgvs_c"] or qi["hgvs_p"]):
            return None
        V = self.variants
        for g in qi["genes"]:
            sub = V[V["gene"] == g]
            for c in qi["hgvs_c"]:
                cs = c.split(":")[-1]
                hit = sub[sub["c_short"] == cs]
                if len(hit):
                    return self.engine.score_batch(hit.head(1)).iloc[0].to_dict()
            for p in qi["hgvs_p"]:
                pv = parse_p(p if p.startswith("p.") else "p." + p.strip("()"))
                if pv and pv.short:
                    hit = sub[sub["protein_change"] == pv.short]
                    if len(hit):
                        return self.engine.score_batch(hit.head(1)).iloc[0].to_dict()
            if qi["hgvs_c"]:
                c = qi["hgvs_c"][0].split(":")[-1]
                if parse_c(c):
                    return prepare(g, c, qi["hgvs_p"][0] if qi["hgvs_p"] else None)
        return None

    def variant_evidence(self, v: dict) -> tuple[list[dict], dict]:
        res = self.engine.evaluate(v)
        name = f"{v['gene']} {str(v.get('hgvs_c', '')).split(':')[-1]}"
        met = [c for c in res["criteria"] if c["status"] == "met"]
        head = (f"The Allelion criteria engine classifies {name} ({str(res['variant']['consequence']).replace('_', ' ')}) as "
                f"{res['classification'].lower()} with {res['points']} points. Criteria met: {', '.join(res['met']) or 'none'}.")
        items = [{"title": f"Criteria engine summary for {name}", "text": head, "kind": "engine", "source": f"engine:{name}", "level": "engine",
                  "refs": ["Allelion criteria engine"]}]
        for c in met:
            items.append({"title": f"{c['label']}: {c['title']}", "text": f"{c['label']} applied for {name}. {c['rationale']}", "kind": "engine",
                          "source": f"engine:{name}:{c['code']}", "level": "engine", "refs": [e["ref"] for e in c["evidence"]][:3]})
        ne = [c["code"] for c in res["criteria"] if c["status"] == "not_evaluated"]
        if ne:
            items.append({"title": f"Criteria awaiting case data for {name}", "level": "engine", "kind": "engine", "source": f"engine:{name}:pending",
                          "text": f"Criteria that need case level data for {name} were not evaluated: {', '.join(ne)}.", "refs": []})
        return items, res

    # ------------------------------------------------------------------ main
    def answer(self, question: str, k: int = 6, log: bool = True) -> dict:
        t0 = time.perf_counter()
        g = guard.check(question)
        if g.action == "off_topic":
            qi0 = understand(g.text)
            if qi0.genes or qi0.hpo or qi0.diseases or qi0.codes:   # a phenotype term or disease name is on topic
                g.action, g.message = "answer", ""
        out = {"question_redacted": g.text, "redactions": g.redactions, "action": g.action, "notices": [], "sentences": [], "evidence": [],
               "engine_result": None, "writer": self.writer.name}
        if g.action != "answer":
            out["notices"].append(g.message)
            out["confidence"] = "none"
            out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
            if log:
                self._audit(question, out)
            return out
        if g.personal:
            out["notices"].append(g.message)
        if g.injection:
            out["notices"].append("Instructions embedded in the question were removed before retrieval.")
        qi = understand(g.text).to_dict()
        evidence = []
        v = self.resolve_variant(qi)
        if v is not None:
            items, res = self.variant_evidence(v)
            evidence.extend(items)
            out["engine_result"] = res
        hits = self.retriever.search(g.text, k=k)
        for i in hits["ids"]:
            p = self.retriever.passage(i)
            evidence.append({**p, "text": guard.neutralise(p["text"])[0]})
        for n, e in enumerate(evidence, 1):
            e["n"] = n
        sents = self.writer.write(g.text, qi, evidence)
        if not sents and self.writer.name != "extractive":
            sents = ExtractiveWriter().write(g.text, qi, evidence)
        by_n = {e["n"]: e for e in evidence}
        checked = []
        for s in sents:
            ev = by_n.get(s["cite"])
            ver = self.verifier.score(s["text"], ev["text"] if ev else None)
            checked.append({"text": s["text"], "cite": s["cite"], "supported": ver["supported"], "probability": ver["probability"], "reason": ver["reason"]})
        share = sum(c["supported"] for c in checked) / len(checked) if checked else 0.0
        cited = [by_n[c["cite"]] for c in checked if c["cite"] in by_n]
        ents = qi["genes"] + qi["codes"]
        cov = (sum(1 for e in ents if any(e in (c["text"] + c["title"]) for c in cited)) / len(ents)) if ents else 1.0
        best_level = max((LEVEL_RANK.get(c["level"], 1) for c in cited), default=0)
        conf = "high" if share >= 0.9 and cov >= 0.5 and best_level >= 2 else "moderate" if share >= 0.7 and best_level >= 1 else "low"
        out.update({"query": qi, "sentences": checked, "evidence": evidence, "verified_share": round(share, 3), "entity_coverage": round(cov, 3),
                    "confidence": conf, "retrieval_ms": hits["ms"], "ms": round((time.perf_counter() - t0) * 1000, 1)})
        if log:
            self._audit(question, out)
        return out

    def _audit(self, question: str, out: dict) -> None:
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "question": out["question_redacted"],
               "sha256": hashlib.sha256(question.encode()).hexdigest(), "action": out["action"], "writer": out["writer"],
               "redactions": out["redactions"], "evidence": len(out["evidence"]), "verified_share": out.get("verified_share"),
               "confidence": out.get("confidence"), "classification": (out["engine_result"] or {}).get("classification")}
        with open(AUDIT_LOG, "a", encoding="utf8") as fh:
            fh.write(json.dumps(rec) + "\n")


def read_audit(limit: int = 200) -> list[dict]:
    if not AUDIT_LOG.exists():
        return []
    lines = AUDIT_LOG.read_text(encoding="utf8").strip().split("\n")
    return [json.loads(l) for l in lines[-limit:] if l]
