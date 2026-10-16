"""End to end evaluation. Writes reports/evaluation_summary.json.

    python -m allelion.eval.evaluate            development sets only
    python -m allelion.eval.evaluate --blind    adds the blind variant and question sets (checks both hashes first)

Every blind run is counted in data/eval/blind_runs.json so the README can state
how many times the blind sets were scored.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from allelion.acmg.engine import Engine
from allelion.acmg.tables import consequence_group
from allelion.config import CLASSES, EVAL, KB, MODELS, RAW, REPORTS
from allelion.eval import retrieval_eval, splits
from allelion.eval.concordance import metrics
from allelion.models import meta
from allelion.models.calibrate import strength_for

ABLATIONS = {
    "all criteria": None,
    "without computational (PP3, BP4, BP7)": {"PP3", "BP4", "BP7"},
    "without allelic (PS1, PM5)": {"PS1", "PM5"},
    "without hot spot (PM1)": {"PM1"},
    "without gene level (PP2, BP1)": {"PP2", "BP1"},
    "without PVS1": {"PVS1"},
    "population criteria only": "population",
}
POP = {"BA1", "BS1", "BS2", "PM2"}


def _rows(ids: pd.Series) -> pd.DataFrame:
    v = pd.read_parquet(KB / "variants.parquet")
    return v[v["variation_id"].isin(set(ids))]


def classify(eng: Engine, rows: pd.DataFrame, truth: pd.Series) -> tuple[pd.DataFrame, dict]:
    rows = eng.score_batch(rows)
    recs = rows.to_dict("records")
    from allelion.acmg.engine import ORDER
    out = []
    for v in recs:
        r = eng.evaluate(v)
        out.append({"variation_id": v["variation_id"], "gene": v["gene"], "consequence": v["consequence"], "pred": r["classification"],
                    "pred_2015": r["classification_2015"], "points": r["points"], "posterior": r["posterior"], "met": r["met"],
                    "missense_score": r["scores"]["missense"], "splice_score": r["scores"]["splice"]})
    df = pd.DataFrame(out)
    df["truth"] = df["variation_id"].map(truth)
    abl = {}
    for name, drop in ABLATIONS.items():
        if drop is None:
            preds = df["pred"]
        else:
            use = POP if drop == "population" else set(ORDER) - drop
            preds = pd.Series([eng.evaluate(v, use=use)["classification"] for v in recs])
        m = metrics(df["truth"], preds.reset_index(drop=True))
        abl[name] = {k: m[k] for k in ("exact_5tier", "accuracy_3tier", "balanced_accuracy_3tier", "severe_discordance_rate", "per_class_3tier")}
    # meta classifier
    X = meta.frame(eng, rows)
    probs = meta.load().predict(X)
    df["pred_meta"] = [CLASSES[i] for i in probs.argmax(1)]
    return df, abl


def summarise(df: pd.DataFrame, by_panel: bool) -> dict:
    res = {"rules_points": metrics(df["truth"], df["pred"]), "rules_2015": metrics(df["truth"], df["pred_2015"]),
           "learned_meta": metrics(df["truth"], df["pred_meta"])}
    res["definitive_call_rate"] = round(float((df["pred"] != "Uncertain significance").mean()), 4)
    grp = df["consequence"].map(consequence_group)
    res["by_consequence"] = {}
    for g, sub in df.groupby(grp):
        m = metrics(sub["truth"], sub["pred"])
        res["by_consequence"][g] = {"n": m["n"], "exact_5tier": m["exact_5tier"], "accuracy_3tier": m["accuracy_3tier"],
                                    "severe_discordance": m["severe_discordance"]}
    # posterior reliability: share of reference P/LP by posterior band
    bands = [(0, 0.001), (0.001, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, 0.99), (0.99, 1.01)]
    rel = []
    for lo, hi in bands:
        sub = df[(df["posterior"] >= lo) & (df["posterior"] < hi)]
        if len(sub):
            rel.append({"band": f"{lo:g} to {min(hi, 1):g}", "n": int(len(sub)),
                        "reference_pathogenic_share": round(float(sub["truth"].isin(["Pathogenic", "Likely pathogenic"]).mean()), 4)})
    res["posterior_reliability"] = rel
    # most frequent criteria by reference class
    crit = {}
    for cls, sub in df.groupby("truth"):
        c = pd.Series([x for m in sub["met"] for x in m]).value_counts()
        crit[cls] = {k: round(int(v) / len(sub), 3) for k, v in c.head(8).items()}
    res["criteria_frequency_by_reference_class"] = crit
    sev = df[((df["truth"].isin(["Pathogenic", "Likely pathogenic"])) & df["pred"].isin(["Benign", "Likely benign"])) |
             ((df["truth"].isin(["Benign", "Likely benign"])) & df["pred"].isin(["Pathogenic", "Likely pathogenic"]))]
    res["severe_examples"] = sev.head(25)[["variation_id", "gene", "consequence", "truth", "pred", "points", "met"]].assign(
        met=lambda d: d["met"].map(lambda x: ", ".join(x))).to_dict("records")
    return res


def missense_on(df: pd.DataFrame) -> dict:
    m = df[df["missense_score"].notna() & df["truth"].isin(["Pathogenic", "Likely pathogenic", "Benign", "Likely benign"])]
    if m["truth"].nunique() < 2:
        return {}
    y = m["truth"].isin(["Pathogenic", "Likely pathogenic"]).astype(int).values
    s = m["missense_score"].astype(float).values
    cal = json.loads((MODELS / "missense_calibration.json").read_text())
    strengths = pd.Series([("{}_{}".format(*strength_for(x, cal)) if strength_for(x, cal)[0] else "indeterminate") for x in s])
    return {"n": int(len(y)), "n_pathogenic": int(y.sum()), "auroc": round(float(roc_auc_score(y, s)), 4),
            "average_precision": round(float(average_precision_score(y, s)), 4),
            "strength_share_pathogenic": strengths[y == 1].value_counts(normalize=True).round(3).to_dict(),
            "strength_share_benign": strengths[y == 0].value_counts(normalize=True).round(3).to_dict()}


def run(blind: bool) -> dict:
    eng = Engine()
    summary = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    dev = pd.read_parquet(EVAL / "dev_variants.parquet")
    ddf, dabl = classify(eng, _rows(dev["variation_id"]), dev.set_index("variation_id")["truth5"])
    summary["classification_dev"] = summarise(ddf, False) | {"ablations": dabl, "missense_model": missense_on(ddf)}
    if blind:
        assert splits.verify_blind(), "blind variant file does not match its recorded hash"
        bl = pd.read_parquet(EVAL / "blind_variants.parquet")
        bdf, babl = classify(eng, _rows(bl["variation_id"]), bl.set_index("variation_id")["truth5"])
        bdf["panel"] = bdf["variation_id"].map(bl.set_index("variation_id")["panel"])
        bres = summarise(bdf, True) | {"ablations": babl, "missense_model": missense_on(bdf)}
        bres["by_panel"] = {}
        for p, sub in bdf.groupby("panel"):
            if len(sub) < 20:
                continue
            m = metrics(sub["truth"], sub["pred"])
            mm = metrics(sub["truth"], sub["pred_meta"])
            bres["by_panel"][p] = {"n": m["n"], "genes": sorted(sub["gene"].dropna().unique().tolist()), "exact_5tier": m["exact_5tier"],
                                   "accuracy_3tier": m["accuracy_3tier"], "balanced_accuracy_3tier": m["balanced_accuracy_3tier"],
                                   "severe_discordance": m["severe_discordance"], "meta_accuracy_3tier": mm["accuracy_3tier"]}
        summary["classification_blind"] = bres
        summary["blind_hash_verified"] = True
        bdf.to_parquet(REPORTS / "blind_predictions.parquet", index=False)
        runs_p = EVAL / "blind_runs.json"
        runs = json.loads(runs_p.read_text()) if runs_p.exists() else []
        runs.append(summary["generated_utc"])
        runs_p.write_text(json.dumps(runs))
        summary["blind_runs"] = len(runs)
    ddf.to_parquet(REPORTS / "dev_predictions.parquet", index=False)
    summary["retrieval"] = retrieval_eval.run(blind)
    for name in ("insilico_report", "meta_report", "extractor_report", "verifier_report", "reranker_report", "index_report", "passages_stats"):
        p = MODELS / f"{name}.json"
        if p.exists():
            summary[name] = json.loads(p.read_text())
    rules = json.loads((MODELS / "rules.json").read_text())
    summary["rules"] = {"selected": rules["selected"], "default": rules["default_config_result"], "searched": rules["searched"]}
    summary["splits"] = json.loads((EVAL / "splits_summary.json").read_text())
    summary["knowledge_base"] = kb_stats()
    (REPORTS / "evaluation_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: summary[k] for k in summary if k.startswith("classification")}, default=str)[:3000])
    return summary


def kb_stats() -> dict:
    v = pd.read_parquet(KB / "variants.parquet", columns=["gene", "gn_seen"])
    g = pd.read_parquet(KB / "genes.parquet", columns=["gene", "exon_model", "diseases"])
    man = json.loads((RAW / "MANIFEST.json").read_text())
    lit = pd.read_parquet(KB / "literature_docs.parquet", columns=["corpus"])
    links = pd.read_parquet(KB / "literature_links.parquet")
    return {"variants": len(v), "variants_seen_in_gnomad": int(v["gn_seen"].sum()), "genes": len(g), "genes_with_exon_model": int(g["exon_model"].sum()),
            "genes_with_diseases": int((g["diseases"].map(len) > 0).sum()), "diseases": len(pd.read_parquet(KB / "diseases.parquet", columns=["disease_id"])),
            "hpo_terms": len(pd.read_parquet(KB / "hpo_terms.parquet", columns=["hpo_id"])), "literature_documents": len(lit),
            "literature_by_corpus": lit["corpus"].value_counts().to_dict(),
            "literature_variant_links": int(links["variation_id"].notna().sum()),
            "graph": json.loads((KB / "graph_stats.json").read_text()), "raw_files": man["total_files"], "raw_mb": round(man["total_bytes"] / 1e6, 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--blind", action="store_true")
    run(ap.parse_args().blind)
