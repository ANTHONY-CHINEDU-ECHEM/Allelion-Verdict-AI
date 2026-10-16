"""Calibration of a predictor score to ACMG/AMP evidence strength (local posterior method).

Follows the approach recommended by the ClinGen Sequence Variant Interpretation
working group (Pejaver and colleagues 2022): estimate the posterior probability
of pathogenicity as a function of score in a sliding local window, with the
benign and pathogenic sets reweighted so the implied prior equals 0.0441;
bootstrap the estimate; and take as the threshold for each evidence strength
the score beyond which the one sided 95% confidence bound of the posterior
clears the posterior that strength implies.

Positive likelihood ratio thresholds for prior 0.0441 (Pejaver 2022):
    Supporting 2.406, Moderate 5.790, Strong 33.53, Very strong 1124
"""
from __future__ import annotations

import numpy as np

from allelion.config import CALIBRATION_PRIOR

LR = {"Supporting": 2.406, "Moderate": 5.790, "Strong": 33.53, "VeryStrong": 1124.0}


def posterior_from_lr(lr: float, prior: float = CALIBRATION_PRIOR) -> float:
    return lr * prior / ((lr - 1.0) * prior + 1.0)


def local_posterior(scores: np.ndarray, labels: np.ndarray, grid: np.ndarray, prior: float = CALIBRATION_PRIOR,
                    min_frac: float = 0.03, min_n: int = 100) -> np.ndarray:
    """Posterior of pathogenicity at each grid point from the nearest window of variants."""
    order = np.argsort(scores)
    s, y = scores[order], labels[order]
    n = len(s)
    n_p, n_b = y.sum(), n - y.sum()
    w = (prior / (1 - prior)) * (n_b / max(n_p, 1))   # reweight pathogenic counts so the window prior is `prior`
    k = max(int(min_frac * n), min_n)
    out = np.empty(len(grid))
    for i, g in enumerate(grid):
        c = np.searchsorted(s, g)
        lo = max(0, c - k // 2)
        hi = min(n, lo + k)
        lo = max(0, hi - k)
        yy = y[lo:hi]
        p = yy.sum() * w
        b = len(yy) - yy.sum()
        out[i] = p / (p + b) if (p + b) > 0 else np.nan
    return out


def calibrate(scores: np.ndarray, labels: np.ndarray, n_boot: int = 200, seed: int = 7, n_grid: int = 200) -> dict:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    grid = np.quantile(scores, np.linspace(0.005, 0.995, n_grid))
    grid = np.unique(grid)
    point = local_posterior(scores, labels, grid)
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, len(grid)))
    for b in range(n_boot):
        idx = rng.integers(0, len(scores), len(scores))
        boots[b] = local_posterior(scores[idx], labels[idx], grid)
    lower = np.nanpercentile(boots, 5, axis=0)
    upper = np.nanpercentile(boots, 95, axis=0)

    path_thr, ben_thr = {}, {}
    for strength, lr in LR.items():
        post = posterior_from_lr(lr)
        ok = lower >= post
        # smallest grid score from which the bound stays above the threshold for every higher score
        idx = None
        for i in range(len(grid) - 1, -1, -1):
            if ok[i]:
                idx = i
            else:
                break
        path_thr[strength] = float(grid[idx]) if idx is not None else None
        post_b = posterior_from_lr(1.0 / lr)
        okb = upper <= post_b
        idx = None
        for i in range(len(grid)):
            if okb[i]:
                idx = i
            else:
                break
        ben_thr[strength] = float(grid[idx]) if idx is not None else None
    return {
        "prior": CALIBRATION_PRIOR,
        "lr": LR,
        "pathogenic_thresholds": path_thr,
        "benign_thresholds": ben_thr,
        "grid": grid.tolist(),
        "posterior": point.tolist(),
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "n": int(len(scores)),
        "n_pathogenic": int(labels.sum()),
    }


def strength_for(score: float, cal: dict) -> tuple[str | None, str | None]:
    """Return (direction, strength) for a score under a calibration, strongest first."""
    for s in ["VeryStrong", "Strong", "Moderate", "Supporting"]:
        t = cal["pathogenic_thresholds"].get(s)
        if t is not None and score >= t:
            return "pathogenic", s
    for s in ["VeryStrong", "Strong", "Moderate", "Supporting"]:
        t = cal["benign_thresholds"].get(s)
        if t is not None and score <= t:
            return "benign", s
    return None, None
