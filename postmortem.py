"""Post hoc analysis of the single blind run.

This module never scores the blind set. It reads the predictions stored by
the one blind run (reports/blind_predictions.parquet) and slices them to
explain where the engine disagreed with the expert panels, writing
reports/blind_postmortem.json.

The main finding: ClinVar records many BRCA1 variants on the exon 11 skipping
isoforms (NM_007298.3, NM_007299.3). A frameshift inside exon 11 of the
canonical transcript NM_007294.3 is then written as a deep intronic change, so
the engine, which interprets the transcript ClinVar reports, sees an intron
variant, withholds PVS1 and may apply BP7.
"""
from __future__ import annotations

import json

import pandas as pd

from allelion.config import KB, REPORTS

ALT_ISOFORMS = ("NM_007297", "NM_007298", "NM_007299", "NM_007300")
CANONICAL = "NM_007294"
T3 = {"Pathogenic": "P/LP", "Likely pathogenic": "P/LP", "Uncertain significance": "VUS", "Likely benign": "B/LB", "Benign": "B/LB"}


def _summary(df: pd.DataFrame) -> dict:
    agree = (df["t3"] == df["p3"]).mean() if len(df) else None
    severe = (((df["t3"] == "P/LP") & (df["p3"] == "B/LB")) | ((df["t3"] == "B/LB") & (df["p3"] == "P/LP"))).sum()
    return {"n": int(len(df)), "accuracy_3tier": None if agree is None else round(float(agree), 4), "severe_discordance": int(severe),
            "severe_discordance_rate": round(float(severe / len(df)), 5) if len(df) else None}


def run() -> dict:
    b = pd.read_parquet(REPORTS / "blind_predictions.parquet")
    v = pd.read_parquet(KB / "variants.parquet", columns=["variation_id", "hgvs_c", "molecular_consequence"]).set_index("variation_id")
    b["transcript"] = b["variation_id"].map(v["hgvs_c"]).fillna("").str.split(":").str[0].str.split(".").str[0]
    b["mc"] = b["variation_id"].map(v["molecular_consequence"]).fillna("")
    b["t3"], b["p3"] = b["truth"].map(T3), b["pred"].map(T3)
    iso = b["transcript"].isin(ALT_ISOFORMS)
    canon_lof = iso & b["mc"].str.contains(CANONICAL) & b["mc"].str.contains("frameshift|nonsense|stop gained|splice donor|splice acceptor")
    sev = ((b["t3"] == "P/LP") & (b["p3"] == "B/LB")) | ((b["t3"] == "B/LB") & (b["p3"] == "P/LP"))
    out = {
        "note": "Slices of the single stored blind run. Nothing here re-scores the blind set.",
        "all": _summary(b),
        "brca1_alternate_isoform": _summary(b[iso]),
        "outside_alternate_isoform": _summary(b[~iso]),
        "alternate_isoform_with_canonical_truncating_annotation": {
            **_summary(b[canon_lof]),
            "reference": b.loc[canon_lof, "t3"].value_counts().to_dict(),
            "predicted": b.loc[canon_lof, "p3"].value_counts().to_dict(),
        },
        "severe_by_gene_and_consequence": (b[sev].groupby(["gene", "consequence"]).size().rename("n").reset_index().to_dict("records")),
        "severe_on_alternate_isoform": int((sev & iso).sum()),
        "reference_benign_predicted": b.loc[b["t3"] == "B/LB", "pred"].value_counts().to_dict(),
        "reference_benign_by_consequence": b.loc[b["t3"] == "B/LB", "consequence"].value_counts().head(6).to_dict(),
        "by_panel_outside_isoform": {p: _summary(g) for p, g in b[~iso].groupby("panel")},
    }
    (REPORTS / "blind_postmortem.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
