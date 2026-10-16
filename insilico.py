"""Train and calibrate the in silico predictors (missense and splice region).

Training data: evidence pool only (pathogenic or likely pathogenic = 1, benign
or likely benign = 0). Development and blind variants are never seen.

Validation: five fold cross validation grouped by gene, so a gene's variants
are never split between training and validation. The out of fold scores give
the honest performance estimate and feed the evidence strength calibration;
the final model is then refit on the whole pool.

Outputs (data/models):
    missense_model.txt, splice_model.txt          LightGBM boosters
    missense_calibration.json, splice_calibration.json
    insilico_report.json                          AUROC, average precision, baselines, feature importance
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from allelion.config import EVAL, KB, MODELS, SEED
from allelion.models.calibrate import calibrate
from allelion.models.features import MISSENSE_FEATURES, SPLICE_FEATURES, gene_frame, missense_features, splice_features

PLP = {"Pathogenic", "Likely pathogenic"}
BLB = {"Benign", "Likely benign"}

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=40, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=SEED, num_threads=2)


def _pool() -> pd.DataFrame:
    v = pd.read_parquet(KB / "variants.parquet")
    ids = set(pd.read_parquet(EVAL / "pool_ids.parquet")["variation_id"])
    p = v[v["variation_id"].isin(ids) & v["clinvar_class"].isin(PLP | BLB) & v["gene"].notna()].copy()
    p["y"] = p["clinvar_class"].isin(PLP).astype(int)
    return p


def _cv(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, rounds: int = 400) -> tuple[np.ndarray, list[int]]:
    oof = np.zeros(len(y))
    best = []
    for k, (tr, va) in enumerate(GroupKFold(n_splits=5).split(X, y, groups)):
        dtr = lgb.Dataset(X.iloc[tr], y[tr])
        dva = lgb.Dataset(X.iloc[va], y[va])
        m = lgb.train(PARAMS, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(40, verbose=False)])
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        best.append(m.best_iteration)
    return oof, best


def _metrics(y, s) -> dict:
    ok = ~np.isnan(s)
    return {"auroc": round(float(roc_auc_score(y[ok], s[ok])), 4), "average_precision": round(float(average_precision_score(y[ok], s[ok])), 4),
            "n": int(ok.sum())}


def train() -> dict:
    pool = _pool()
    genes = pd.read_parquet(KB / "genes.parquet")
    gf = gene_frame(genes)
    report = {}

    # ---------------- missense ----------------
    mis = pool[(pool["consequence"] == "missense") & pool["p_pos"].notna() & pool["p_ref"].isin(list("ACDEFGHIKLMNPQRSTVWY"))
               & pool["p_alt"].isin(list("ACDEFGHIKLMNPQRSTVWY"))].reset_index(drop=True)
    Xm = missense_features(mis, gf)
    ym = mis["y"].values
    oof, best = _cv(Xm, ym, mis["gene"].values)
    report["missense"] = {
        "n_train": int(len(ym)), "n_pathogenic": int(ym.sum()), "n_genes": int(mis["gene"].nunique()),
        "grouped_cv": _metrics(ym, oof),
        "baselines": {
            "BLOSUM62 (negated)": _metrics(ym, -Xm["blosum62"].values.astype(float)),
            "Grantham distance": _metrics(ym, Xm["grantham"].values.astype(float)),
            "gene pLI": _metrics(ym, Xm["pLI"].values.astype(float)),
            "LoFtool (negated)": _metrics(ym, -Xm["loftool"].values.astype(float)),
        },
        "best_iterations": best,
    }
    n_rounds = int(np.median(best))
    final = lgb.train(PARAMS, lgb.Dataset(Xm, ym), n_rounds)
    final.save_model(str(MODELS / "missense_model.txt"))
    imp = pd.Series(final.feature_importance("gain"), index=MISSENSE_FEATURES).sort_values(ascending=False)
    report["missense"]["importance_gain_top"] = {k: round(float(v), 1) for k, v in imp.head(15).items()}
    cal = calibrate(oof, ym)
    (MODELS / "missense_calibration.json").write_text(json.dumps(cal))
    report["missense"]["thresholds"] = {"pathogenic": cal["pathogenic_thresholds"], "benign": cal["benign_thresholds"]}
    # share of pool missense reaching each strength out of fold
    report["missense"]["oof_strength_share"] = _strength_share(oof, ym, cal)
    pd.DataFrame({"variation_id": mis["variation_id"], "gene": mis["gene"], "y": ym, "oof": oof}).to_parquet(MODELS / "missense_oof.parquet", index=False)

    # ---------------- splice region (intronic offsets 3 to 50, outside the canonical dinucleotide) ----------------
    sp = pool[(pool["c_offset"].abs() >= 3) & (pool["c_offset"].abs() <= 50) & (pool["c_region"] == "cds")].reset_index(drop=True)
    Xs = splice_features(sp, gf)
    ys = sp["y"].values
    oofs, bests = _cv(Xs, ys, sp["gene"].values)
    report["splice"] = {"n_train": int(len(ys)), "n_pathogenic": int(ys.sum()), "n_genes": int(sp["gene"].nunique()),
                        "grouped_cv": _metrics(ys, oofs),
                        "baselines": {"closeness to exon (negated offset)": _metrics(ys, -Xs["abs_offset"].values.astype(float))},
                        "best_iterations": bests}
    final_s = lgb.train(PARAMS, lgb.Dataset(Xs, ys), int(np.median(bests)))
    final_s.save_model(str(MODELS / "splice_model.txt"))
    cal_s = calibrate(oofs, ys)
    (MODELS / "splice_calibration.json").write_text(json.dumps(cal_s))
    report["splice"]["thresholds"] = {"pathogenic": cal_s["pathogenic_thresholds"], "benign": cal_s["benign_thresholds"]}
    report["splice"]["oof_strength_share"] = _strength_share(oofs, ys, cal_s)
    pd.DataFrame({"variation_id": sp["variation_id"], "gene": sp["gene"], "y": ys, "oof": oofs}).to_parquet(MODELS / "splice_oof.parquet", index=False)

    (MODELS / "insilico_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("n_train", "grouped_cv", "thresholds", "baselines")} for k, v in report.items()}, indent=1))
    return report


def _strength_share(scores, y, cal) -> dict:
    from allelion.models.calibrate import strength_for
    out = {}
    for label, mask in (("pathogenic", y == 1), ("benign", y == 0)):
        res = [strength_for(s, cal) for s in scores[mask]]
        keys = [f"{d}_{s}" if d else "none" for d, s in res]
        out[label] = pd.Series(keys).value_counts(normalize=True).round(4).to_dict()
    return out


if __name__ == "__main__":
    train()
