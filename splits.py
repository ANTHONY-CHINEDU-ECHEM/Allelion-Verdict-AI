"""Freeze the evaluation sets before any evidence table or model exists.

Blind set: every variant whose ClinVar assertion was made by an expert panel
or a practice guideline (three and four review stars) with a five tier
classification. These are the closest public stand in for a reference
classification, and they are scored once, after every component is frozen.

Development set: a stratified sample of two star variants (several
submitters, no conflict). Thresholds and design choices are tuned here only.

Evidence pool: every remaining variant with assertion criteria, a five tier
classification and no conflict. Only the pool feeds the allelic tables (PS1,
PM5), hotspot density (PM1), gene level rates (PP2, BP1) and the in silico
model. Neither evaluation set ever contributes evidence, so no variant can be
classified using its own label.

The canonical blind file is hashed and the digest written to
data/eval/blind_hash.txt; the evaluation refuses to run if the file changes.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pandas as pd

from allelion.config import EVAL, KB, SEED, THREE_TIER

PANEL_HINTS = [
    ("ENIGMA", "ENIGMA (BRCA1, BRCA2)"),
    ("InSiGHT", "InSiGHT (mismatch repair)"),
    ("CFTR2", "CFTR2"),
    ("RASopathy", "ClinGen RASopathy expert panel"),
    ("Cardiomyopathy", "ClinGen inherited cardiomyopathy expert panel"),
    ("PharmGKB", "PharmGKB"),
    ("CPIC", "CPIC guideline"),
]


def _panel(subs: list[dict]) -> str:
    names = " | ".join(s.get("submitter", "") for s in subs).lower()
    for hint, label in PANEL_HINTS:
        if hint.lower() in names:
            return label
    return "Other expert panel or guideline"


def canonical_blob(df: pd.DataFrame) -> bytes:
    cols = ["variation_id", "key", "gene", "truth5"]
    return df[cols].sort_values("variation_id").to_csv(index=False).encode()


def freeze() -> dict:
    v = pd.read_parquet(KB / "variants.parquet")
    labelled = v[v["clinvar_class"].notna()].copy()
    labelled["truth5"] = labelled["clinvar_class"]
    labelled["truth3"] = labelled["truth5"].map(THREE_TIER)

    blind = labelled[labelled["stars"] >= 3].copy()
    blind["panel"] = blind["submissions"].map(_panel)

    two_star = labelled[(labelled["stars"] == 2) & (labelled["conflicted"] == 0)]
    # Stratify by class and cap each gene so a handful of genes cannot dominate the tuning signal.
    rng_state = SEED
    parts = []
    per_class = {"Pathogenic": 700, "Likely pathogenic": 500, "Uncertain significance": 700, "Likely benign": 700, "Benign": 700}
    for cls, n in per_class.items():
        pool = two_star[two_star["truth5"] == cls]
        pool = pool.sample(frac=1.0, random_state=rng_state).groupby("gene").head(25)
        parts.append(pool.sample(min(n, len(pool)), random_state=rng_state))
    dev = pd.concat(parts)

    cols = ["variation_id", "key", "gene", "hgvs_c", "hgvs_p", "consequence", "truth5", "truth3", "stars"]
    blind_out = blind[cols + ["panel"]].sort_values("variation_id").reset_index(drop=True)
    dev_out = dev[cols].sort_values("variation_id").reset_index(drop=True)
    blind_out.to_parquet(EVAL / "blind_variants.parquet", index=False)
    dev_out.to_parquet(EVAL / "dev_variants.parquet", index=False)

    digest = hashlib.sha256(canonical_blob(blind_out)).hexdigest()
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (EVAL / "blind_hash.txt").write_text(f"{digest}  blind_variants  frozen {stamp}  n={len(blind_out)}\n")

    held = set(blind_out["variation_id"]) | set(dev_out["variation_id"])
    pool = labelled[(labelled["stars"] >= 1) & (labelled["conflicted"] == 0) & ~labelled["variation_id"].isin(held)]
    pool[["variation_id"]].to_parquet(EVAL / "pool_ids.parquet", index=False)

    summary = {
        "frozen_utc": stamp,
        "blind_sha256": digest,
        "blind_n": len(blind_out),
        "blind_classes": blind_out["truth5"].value_counts().to_dict(),
        "blind_panels": blind_out["panel"].value_counts().to_dict(),
        "blind_genes": int(blind_out["gene"].nunique()),
        "dev_n": len(dev_out),
        "dev_classes": dev_out["truth5"].value_counts().to_dict(),
        "dev_genes": int(dev_out["gene"].nunique()),
        "pool_n": len(pool),
        "pool_classes": pool["clinvar_class"].value_counts().to_dict(),
    }
    (EVAL / "splits_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return summary


def verify_blind() -> bool:
    blind = pd.read_parquet(EVAL / "blind_variants.parquet")
    recorded = (EVAL / "blind_hash.txt").read_text().split()[0]
    return hashlib.sha256(canonical_blob(blind)).hexdigest() == recorded


if __name__ == "__main__":
    freeze()
