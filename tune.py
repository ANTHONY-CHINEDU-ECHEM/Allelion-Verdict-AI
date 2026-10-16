"""Tune the configurable rule thresholds on the development set only.

The search covers the thresholds the guidelines leave to the curator (BS1
frequency cut offs, the hot spot window, the PP2 and BP1 gene level rates).
The objective is three tier balanced accuracy, subject to a severe discordance
rate (pathogenic called benign or the reverse) of at most 1%. The chosen
values are written to data/models/rules.json with the full search log; the
blind set is not read here.
"""
from __future__ import annotations

import itertools
import json
import time

import pandas as pd

from allelion.acmg.engine import DEFAULT_RULES, Engine, Resources
from allelion.config import EVAL, KB, MODELS
from allelion.eval.concordance import metrics

GRID = {
    "bs1_af_dominant": [0.0005, 0.001, 0.003],
    "bs1_af_recessive": [0.005, 0.01, 0.02],
    "pm1_window": [5, 10],
    "pm1_min_plp": [3, 5, 8],
    "pp2_max_benign_frac": [0.05, 0.10],
    "pp2_min_plp": [5, 10],
    "bs2_hom_min": [2, 3, 5],
}


def dev_inputs(engine: Engine) -> tuple[list[dict], pd.Series]:
    v = pd.read_parquet(KB / "variants.parquet")
    dev = pd.read_parquet(EVAL / "dev_variants.parquet")
    rows = v[v["variation_id"].isin(dev["variation_id"])]
    rows = engine.score_batch(rows)
    truth = dev.set_index("variation_id")["truth5"].reindex(rows["variation_id"]).values
    return rows.to_dict("records"), pd.Series(truth)


def run(records, truth, rules: dict) -> dict:
    eng = Engine(Resources(rules))
    pred = [eng.evaluate(r)["classification"] for r in records]
    return metrics(truth, pd.Series(pred))


def tune() -> dict:
    base_engine = Engine(Resources(dict(DEFAULT_RULES)))
    records, truth = dev_inputs(base_engine)
    # share loaded resources across configurations
    shared = base_engine.r
    log = []
    keys = list(GRID)
    t0 = time.time()
    for combo in itertools.product(*[GRID[k] for k in keys]):
        rules = dict(DEFAULT_RULES)
        rules.update(dict(zip(keys, combo)))
        shared.rules = rules
        eng = Engine(shared)
        pred = [eng.evaluate(r)["classification"] for r in records]
        m = metrics(truth, pd.Series(pred))
        log.append({"rules": dict(zip(keys, combo)), "balanced_accuracy_3tier": m["balanced_accuracy_3tier"],
                    "accuracy_3tier": m["accuracy_3tier"], "exact_5tier": m["exact_5tier"],
                    "severe_discordance_rate": m["severe_discordance_rate"]})
    ok = [r for r in log if r["severe_discordance_rate"] <= 0.01] or log
    best = max(ok, key=lambda r: (r["balanced_accuracy_3tier"], r["exact_5tier"]))
    default_row = next((r for r in log if all(r["rules"][k] == DEFAULT_RULES[k] for k in keys)), None)
    out = {"rules": {**DEFAULT_RULES, **best["rules"]}, "selected": best, "default_config_result": default_row,
           "searched": len(log), "seconds": round(time.time() - t0, 1), "log": log, "n_dev": len(records)}
    (MODELS / "rules.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("selected", "default_config_result", "searched", "seconds")}, indent=1))
    return out


if __name__ == "__main__":
    tune()
