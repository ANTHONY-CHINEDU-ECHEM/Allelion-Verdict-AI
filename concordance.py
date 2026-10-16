"""Concordance metrics between predicted and reference classifications."""
from __future__ import annotations

import numpy as np
import pandas as pd

from allelion.config import CLASSES, THREE_TIER

TIER3 = ["P/LP", "VUS", "B/LB"]


def metrics(truth5: pd.Series, pred5: pd.Series) -> dict:
    t5, p5 = truth5.astype(str).values, pred5.astype(str).values
    t3 = np.array([THREE_TIER[x] for x in t5])
    p3 = np.array([THREE_TIER[x] for x in p5])
    n = len(t5)
    severe = ((t3 == "P/LP") & (p3 == "B/LB")) | ((t3 == "B/LB") & (p3 == "P/LP"))
    per_class = {}
    for c in TIER3:
        tp = int(((p3 == c) & (t3 == c)).sum())
        fp = int(((p3 == c) & (t3 != c)).sum())
        fn = int(((p3 != c) & (t3 == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[c] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "support": int((t3 == c).sum())}
    balanced = float(np.mean([per_class[c]["recall"] for c in TIER3 if per_class[c]["support"] > 0]))
    # definitive calls: share of reference P/LP and B/LB variants that receive a definitive call in the correct direction
    definitive_t = (t3 != "VUS")
    conf5 = pd.crosstab(pd.Categorical(t5, CLASSES), pd.Categorical(p5, CLASSES), dropna=False)
    conf5.index.name, conf5.columns.name = "reference", "predicted"
    within_one = np.mean([abs(CLASSES.index(a) - CLASSES.index(b)) <= 1 for a, b in zip(t5, p5)])
    return {
        "n": int(n),
        "exact_5tier": round(float((t5 == p5).mean()), 4),
        "within_one_tier": round(float(within_one), 4),
        "accuracy_3tier": round(float((t3 == p3).mean()), 4),
        "balanced_accuracy_3tier": round(balanced, 4),
        "macro_f1_3tier": round(float(np.mean([per_class[c]["f1"] for c in TIER3])), 4),
        "severe_discordance": int(severe.sum()),
        "severe_discordance_rate": round(float(severe.mean()), 5),
        "definitive_correct_rate": round(float(((t3 == p3) & definitive_t).sum() / max(definitive_t.sum(), 1)), 4),
        "per_class_3tier": per_class,
        "confusion_5tier": conf5.astype(int).values.tolist(),
        "classes": CLASSES,
    }
