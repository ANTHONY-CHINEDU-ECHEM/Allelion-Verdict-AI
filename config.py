"""Central paths and constants for Allelion Verdict AI.

Every layer reads from the directory written by the layer before it:

    data/raw        acquisition output, pinned and hashed
    data/kb         knowledge base (variants, genes, transcripts, phenotypes, literature, graph)
    data/eval       frozen development and blind evaluation sets
    data/models     trained models, calibration tables, indexes
    reports         evaluation summaries and figures
    web/public/bundle  exported in browser engine bundle
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("ALLELION_ROOT", Path(__file__).resolve().parents[1]))
DATA = ROOT / "data"
RAW = DATA / "raw"
KB = DATA / "kb"
EVAL = DATA / "eval"
MODELS = DATA / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
KNOWLEDGE = ROOT / "knowledge"
BUNDLE = ROOT / "web" / "public" / "bundle"
AUDIT_LOG = DATA / "audit" / "audit.jsonl"

for _p in (RAW, KB, EVAL, MODELS, REPORTS, FIGURES, AUDIT_LOG.parent):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 20261016

# ClinVar review status to gold stars (ClinVar convention at the time of the snapshot).
REVIEW_STARS = {
    "practice guideline": 4,
    "reviewed by expert panel": 3,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, conflicting interpretations": 1,
    "criteria provided, single submitter": 1,
    "no assertion criteria provided": 0,
    "no assertion provided": 0,
    "no interpretation for the single variant": 0,
}

# Five tier ACMG/AMP classes in a fixed order, pathogenic first.
CLASSES = ["Pathogenic", "Likely pathogenic", "Uncertain significance", "Likely benign", "Benign"]
CLASS_SHORT = {"Pathogenic": "P", "Likely pathogenic": "LP", "Uncertain significance": "VUS",
               "Likely benign": "LB", "Benign": "B"}
THREE_TIER = {"Pathogenic": "P/LP", "Likely pathogenic": "P/LP", "Uncertain significance": "VUS",
              "Likely benign": "B/LB", "Benign": "B/LB"}

# Bayesian point system (Tavtigian and colleagues 2018, 2020).
POINTS = {"VeryStrong": 8, "Strong": 4, "Moderate": 2, "Supporting": 1}
PRIOR_P = 0.10
ODDS_VERY_STRONG = 350.0

# Prior used by ClinGen SVI for in silico calibration (Pejaver and colleagues 2022).
CALIBRATION_PRIOR = 0.0441
