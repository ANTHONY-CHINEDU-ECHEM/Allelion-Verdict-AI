"""Write the shared parity fixture (web/test/parity_fixture.json).

The Python engine and the TypeScript engine must agree on: HGVS parsing,
tokenisation, every criterion's status and strength, points and class for a
stratified sample of bundled variants and for variants entered by HGVS
without population data, predictor scores, the learned verdict, verifier
probabilities, phenotype matches and the top retrieval results.
"""
from __future__ import annotations

import gzip
import json
import random

import pandas as pd

from allelion.acmg.engine import Engine, prepare
from allelion.acmg.phenotype import PhenotypeMatcher
from allelion.export.bundle import hydrate
from allelion.config import BUNDLE, CLASSES, ROOT, SEED
from allelion.copilot.verifier import Verifier
from allelion.kb.hgvs import parse_c, parse_p
from allelion.models import meta
from allelion.retrieval.retriever import Retriever
from allelion.retrieval.text import tokenize

OUT = ROOT / "web" / "test" / "parity_fixture.json"

CUSTOM = [("TP53", "c.743G>A", "p.Arg248Gln"), ("BRCA1", "c.5266dupC", "p.Gln1756ProfsTer74"), ("CFTR", "c.1521_1523delCTT", "p.Phe508del"),
          ("MLH1", "c.1A>G", "p.Met1?"), ("MSH2", "c.942+3A>T", None), ("PTEN", "c.388C>T", "p.Arg130Ter"), ("SCN5A", "c.4846+1G>A", None),
          ("FBN1", "c.3037G>A", "p.Gly1013Arg"), ("APC", "c.3927_3931delAAAGA", "p.Glu1309AspfsTer4"), ("LDLR", "c.-12C>T", None),
          ("MYBPC3", "c.1504C>T", "p.Arg502Trp"), ("KCNQ1", "c.1032+50G>A", None), ("GLA", "c.644A>G", "p.Asn215Ser"),
          ("BRCA2", "c.9976A>T", "p.Lys3326Ter"), ("NOTAGENE", "c.100A>G", "p.Lys34Glu"), ("DMD", "c.10171C>T", "p.Arg3391Ter")]
HGVS = ["NM_007294.3:c.68_69delAG", "c.8754+5G>A", "c.-26G>A", "c.*105A>G", "c.1521_1523delCTT", "c.6997_7012del16", "c.330C=", "c.123_124insA",
        "NM_000059.3:c.9976A>T", "c.4846+1G>A", "c.1032-21_1032-20del", "c.100_101delinsGT"]
PROT = ["p.Phe508del", "p.Arg1443Ter", "p.Met1?", "p.Ter394GlnextTer?", "p.Glu23ValfsTer17", "p.Pro367=", "p.Lys1_Gly3dup", "NP_1.1:p.Ala55Ala=",
        "p.Leu420_Leu421=", "p.Gly333_Lys350del+", "p.R248Q", "p.(Arg248Gln)"]
QUERIES = ["At what strength should PM2 be applied?", "What are the clinical features of Marfan syndrome?", "Is loss of function an established mechanism for MYH7?",
           "Which report describes the F11 Asp506Gly mutation?", "How is the maximum credible allele frequency calculated?", "Define ectopia lentis.",
           "What diseases are associated with SCN1A?", "When does a premature stop escape nonsense mediated decay?",
           "Which genes cause Noonan syndrome?", "What does BP7 require for intronic variants?"]


def build(n: int = 600) -> dict:
    eng = Engine()
    rng = random.Random(SEED)
    v1, v2 = (json.loads(gzip.open(BUNDLE / f"variants_{i}.json.gz").read()) for i in (1, 2))
    recs = [hydrate(dict(zip(v1["cols"], r))) for r in v1["rows"] + v2["rows"]]
    by_cons: dict = {}
    for r in recs:
        by_cons.setdefault(r["consequence"], []).append(r)
    sample = []
    per = max(1, n // len(by_cons))
    for c, lst in sorted(by_cons.items()):
        sample += rng.sample(lst, min(per, len(lst)))
    sample = sample[:n]
    df = eng.score_batch(pd.DataFrame(sample))
    mm = meta.load()
    cases = []
    for v in df.to_dict("records"):
        v = {k: (None if isinstance(x, float) and x != x else x) for k, x in v.items()}
        res = eng.evaluate(v)
        X = meta.frame(eng, pd.DataFrame([v]))
        learned = CLASSES[int(mm.predict(X)[0].argmax())]
        cases.append({"variation_id": v["variation_id"], "expected": _expect(res) | {"learned": learned}})
    custom = []
    for g, c, p in CUSTOM:
        v = prepare(g, c, p)
        res = eng.evaluate(v)
        custom.append({"gene": g, "c": c, "p": p, "expected": _expect(res)})
    case_level = []
    v0 = prepare("FBN1", "c.3037G>A", "p.Gly1013Arg")
    for kase in ({"de_novo": [{"confirmed": True, "phenotype": "specific"}]}, {"segregation_meioses": 5}, {"probands": 3, "functional": {"direction": "damaging", "strength": "Moderate"}},
                 {"in_trans": [{"partner_class": "Pathogenic", "phase": "trans"}, {"partner_class": "Likely pathogenic", "phase": "unknown"}]},
                 {"nonsegregation": True, "alternate_cause": True}):
        case_level.append({"case": kase, "expected": _expect(eng.evaluate(v0, kase))})
    pm = PhenotypeMatcher()
    pheno = [{"gene": g, "terms": t, "expected": pm.match(g, t)} for g, t in (("FBN1", ["HP:0001166", "HP:0001083", "HP:0002616"]), ("BRCA1", ["HP:0003002"]),
                                                                               ("SCN1A", ["HP:0001250", "HP:0001263"]), ("CFTR", ["HP:0006528", "HP:0001508"]))]
    ret = Retriever()
    retrieval = [{"q": q, "top5": [ret.source[i] for i in ret.search(q, k=5)["ids"]], "bm25_top5": [ret.source[i] for i in ret.search(q, k=5, method="bm25")["ids"]]} for q in QUERIES]
    ver = Verifier.load()
    P = ret.passages
    vs = []
    for i in [0, 5, 100, 4300, 10000, 20000, 36000]:
        text = P.at[i, "text"]
        s = text.split(". ")[0]
        vs.append({"sentence": s, "evidence": text, "expected": ver.score(s, text)["probability"]})
        vs.append({"sentence": s.replace("1", "7"), "evidence": text, "expected": ver.score(s.replace("1", "7"), text)["probability"]})
    out = {
        "variants": cases, "custom": custom, "case_level": case_level, "phenotype": pheno, "retrieval": retrieval, "verifier": vs,
        "tokenize": [{"text": t, "tokens": tokenize(t)} for t in QUERIES + ["Haemochromatosis and tumours in BRCA1 c.68_69delAG carriers; rs1801133 p.Arg248Gln"]],
        "hgvs_c": [{"text": h, "parsed": (lambda c: None if c is None else {"kind": c.kind, "region": c.start.region, "base": c.start.base, "offset": c.start.offset,
                                                                          "length_change": c.length_change, "splice_distance": c.splice_distance})(parse_c(h))} for h in HGVS],
        "hgvs_p": [{"text": h, "parsed": (lambda p: None if p is None else {"kind": p.kind, "short": p.short})(parse_p(h))} for h in PROT],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, default=str))
    print(f"parity fixture: {len(cases)} variants, {len(custom)} custom, {len(retrieval)} queries")
    return out


def _expect(res: dict) -> dict:
    return {"classification": res["classification"], "classification_2015": res["classification_2015"], "points": res["points"], "met": res["met"],
            "status": {c["code"]: c["status"] for c in res["criteria"]}, "scores": res["scores"], "posterior": res["posterior"]}


if __name__ == "__main__":
    build()
