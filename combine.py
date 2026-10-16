"""Combining met criteria into a classification.

Two methods are always computed side by side:

1. The Bayesian point system (Tavtigian and colleagues 2018, 2020), which the
   ClinGen SVI working group recommends and which handles conflicting evidence
   without ad hoc rules: Supporting 1, Moderate 2, Strong 4, Very strong 8
   points (negative for benign criteria). Pathogenic at 10 or more, Likely
   pathogenic 6 to 9, Uncertain 0 to 5, Likely benign minus 1 to minus 6,
   Benign minus 7 or fewer. BA1 alone gives Benign.

2. The original combining rules of Richards and colleagues (2015), kept for
   comparison because many laboratories still report under them.

The Bayesian posterior is also reported: with prior 0.10 and
OddsPath = 350 ^ (points / 8), posterior = OddsPath x prior / ((OddsPath - 1) x prior + 1).
"""
from __future__ import annotations

from allelion.config import ODDS_VERY_STRONG, POINTS, PRIOR_P


def points_of(criteria: list[dict]) -> int:
    total = 0
    for c in criteria:
        if c["status"] != "met" or c["strength"] == "StandAlone":
            continue
        p = POINTS[c["strength"]]
        total += p if c["direction"] == "pathogenic" else -p
    return total


def class_from_points(points: int, ba1: bool) -> str:
    if ba1:
        return "Benign"
    if points >= 10:
        return "Pathogenic"
    if points >= 6:
        return "Likely pathogenic"
    if points >= 0:
        return "Uncertain significance"
    if points >= -6:
        return "Likely benign"
    return "Benign"


def posterior(points: int, prior: float = PRIOR_P) -> float:
    odds = ODDS_VERY_STRONG ** (points / 8.0)
    return odds * prior / ((odds - 1.0) * prior + 1.0)


def class_2015(criteria: list[dict]) -> str:
    met = [c for c in criteria if c["status"] == "met"]
    pvs = sum(1 for c in met if c["direction"] == "pathogenic" and c["strength"] == "VeryStrong")
    ps = sum(1 for c in met if c["direction"] == "pathogenic" and c["strength"] == "Strong")
    pm = sum(1 for c in met if c["direction"] == "pathogenic" and c["strength"] == "Moderate")
    pp = sum(1 for c in met if c["direction"] == "pathogenic" and c["strength"] == "Supporting")
    ba = any(c["direction"] == "benign" and c["strength"] == "StandAlone" for c in met)
    bs = sum(1 for c in met if c["direction"] == "benign" and c["strength"] in ("Strong", "VeryStrong"))
    bp = sum(1 for c in met if c["direction"] == "benign" and c["strength"] in ("Supporting", "Moderate"))

    pathogenic = (
        (pvs >= 1 and (ps >= 1 or pm >= 2 or (pm == 1 and pp == 1) or pp >= 2))
        or ps >= 2
        or (ps == 1 and (pm >= 3 or (pm == 2 and pp >= 2) or (pm == 1 and pp >= 4)))
    )
    likely_path = (
        (pvs >= 1 and pm == 1)
        or (ps == 1 and 1 <= pm <= 2)
        or (ps == 1 and pp >= 2)
        or pm >= 3
        or (pm == 2 and pp >= 2)
        or (pm == 1 and pp >= 4)
    )
    benign = ba or bs >= 2
    likely_benign = (bs == 1 and bp >= 1) or bp >= 2
    any_path = pathogenic or likely_path
    any_ben = benign or likely_benign
    if any_path and any_ben:
        return "Uncertain significance"
    if pathogenic:
        return "Pathogenic"
    if likely_path:
        return "Likely pathogenic"
    if benign:
        return "Benign"
    if likely_benign:
        return "Likely benign"
    return "Uncertain significance"
