"""A learned classifier over the engine's evidence (comparator to the rule based verdict).

Inputs: the signed points of every criterion as the engine applied it, the two
predictor scores, population frequency, consequence and gene level context.
Target: the five tier ClinVar class. Training uses a class balanced sample of
the evidence pool; each training variant's criteria are computed with that
variant removed from the evidence tables, exactly as at inference time.

This model answers a different question from the rule engine: not "what do
the ACMG/AMP rules conclude from this evidence" but "how do laboratories
submitting to ClinVar tend to classify variants with evidence like this". It
can learn submitter habits that the rules do not encode, which is why it is
reported next to the rules and never replaces them.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from allelion.acmg.engine import ORDER, Engine
from allelion.config import CLASSES, EVAL, KB, MODELS, POINTS, SEED
from allelion.eval.concordance import metrics

CONSEQ = ["missense", "synonymous", "intron", "frameshift", "nonsense", "splice_donor", "splice_acceptor", "inframe_indel",
          "utr5", "utr3", "start_loss", "stop_loss", "other"]
FEATURES = [f"pts_{c}" for c in ORDER] + ["ba1", "points", "missense_score", "splice_score", "log_af", "seen", "hom"] + \
           [f"cons_{c}" for c in CONSEQ] + ["pLI", "loftool", "dominant", "recessive", "haploinsufficient"]


def featurise(res: dict, v: dict, g: dict) -> dict:
    f = {f"pts_{c}": 0.0 for c in ORDER}
    for c in res["criteria"]:
        if c["status"] == "met" and c["strength"] != "StandAlone":
            p = POINTS[c["strength"]]
            f[f"pts_{c['code']}"] = float(p if c["direction"] == "pathogenic" else -p)
    f["ba1"] = float("BA1" in res["met"])
    f["points"] = float(res["points"])
    f["missense_score"] = res["scores"]["missense"] if res["scores"]["missense"] is not None else np.nan
    f["splice_score"] = res["scores"]["splice"] if res["scores"]["splice"] is not None else np.nan
    af = v.get("gn_faf95")
    f["log_af"] = float(np.log10(af)) if af is not None and not (isinstance(af, float) and np.isnan(af)) and af > 0 else -7.0
    ac = v.get("gn_AC")
    f["seen"] = float(ac is not None and not (isinstance(ac, float) and np.isnan(ac)) and ac > 0)
    hom = v.get("gn_Hom")
    f["hom"] = float(hom) if hom is not None and not (isinstance(hom, float) and np.isnan(hom)) else 0.0
    cons = v.get("consequence") if v.get("consequence") in CONSEQ else "other"
    for c in CONSEQ:
        f[f"cons_{c}"] = float(cons == c)
    f["pLI"] = g.get("pLI", np.nan)
    f["loftool"] = g.get("loftool", np.nan)
    for k in ("dominant", "recessive", "haploinsufficient"):
        f[k] = float(bool(g.get(k)))
    return f


def frame(engine: Engine, rows: pd.DataFrame) -> pd.DataFrame:
    rows = engine.score_batch(rows)
    feats = []
    for v in rows.to_dict("records"):
        res = engine.evaluate(v)
        feats.append(featurise(res, v, engine.r.gene(v.get("gene") or "")))
    return pd.DataFrame(feats, index=rows.index)[FEATURES]


PARAMS = dict(objective="multiclass", num_class=5, learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=SEED, num_threads=2)


def train(per_class: int = 12000) -> dict:
    eng = Engine()
    v = pd.read_parquet(KB / "variants.parquet")
    pool_ids = set(pd.read_parquet(EVAL / "pool_ids.parquet")["variation_id"])
    pool = v[v["variation_id"].isin(pool_ids) & v["gene"].notna()]
    parts = [pool[pool["clinvar_class"] == c].sample(min(per_class, (pool["clinvar_class"] == c).sum()), random_state=SEED) for c in CLASSES]
    tr = pd.concat(parts)
    X = frame(eng, tr)
    y = tr["clinvar_class"].map(CLASSES.index).values
    groups = tr["gene"].values
    oof = np.zeros((len(y), 5))
    best = []
    for trn, val in GroupKFold(n_splits=5).split(X, y, groups):
        m = lgb.train(PARAMS, lgb.Dataset(X.iloc[trn], y[trn]), 600, valid_sets=[lgb.Dataset(X.iloc[val], y[val])],
                      callbacks=[lgb.early_stopping(40, verbose=False)])
        oof[val] = m.predict(X.iloc[val], num_iteration=m.best_iteration)
        best.append(m.best_iteration)
    final = lgb.train(PARAMS, lgb.Dataset(X, y), int(np.median(best)))
    final.save_model(str(MODELS / "meta_model.txt"))
    oof_pred = pd.Series([CLASSES[i] for i in oof.argmax(1)])
    report = {"n_train": int(len(y)), "per_class": per_class, "best_iterations": best,
              "grouped_cv_pool": {k: v for k, v in metrics(pd.Series([CLASSES[i] for i in y]), oof_pred).items() if k != "confusion_5tier"}}
    dev = pd.read_parquet(EVAL / "dev_variants.parquet")
    drows = v[v["variation_id"].isin(dev["variation_id"])]
    Xd = frame(eng, drows)
    pd_ = pd.Series([CLASSES[i] for i in final.predict(Xd).argmax(1)])
    truth = dev.set_index("variation_id")["truth5"].reindex(drows["variation_id"]).reset_index(drop=True)
    report["dev"] = {k: vv for k, vv in metrics(truth, pd_).items()}
    imp = pd.Series(final.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    report["importance_gain_top"] = {k: round(float(x), 1) for k, x in imp.head(15).items()}
    (MODELS / "meta_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({"cv": report["grouped_cv_pool"]["balanced_accuracy_3tier"], "dev_bal3": report["dev"]["balanced_accuracy_3tier"],
                      "dev_exact5": report["dev"]["exact_5tier"], "dev_severe": report["dev"]["severe_discordance_rate"],
                      "top": report["importance_gain_top"]}, indent=1))
    return report


def load():
    return lgb.Booster(model_file=str(MODELS / "meta_model.txt"))


if __name__ == "__main__":
    train()
