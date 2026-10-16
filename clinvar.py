"""Load the ClinVar snapshot, join both gnomAD tables and normalise every variant.

Output: data/kb/variants.parquet, one row per ClinVar allele with
    identifiers, gene, parsed HGVS, primary consequence, ClinVar assertion
    (five tier where possible), review stars, per submitter calls, traits,
    PubMed identifiers, and gnomAD exome and genome counts.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2

warnings.filterwarnings("ignore", message="All-NaN slice encountered")

from allelion.config import KB, RAW, REVIEW_STARS
from allelion.kb.hgvs import parse_c, parse_p, primary_consequence, refine_consequence

FIVE = {
    "pathogenic": "Pathogenic",
    "likely pathogenic": "Likely pathogenic",
    "uncertain significance": "Uncertain significance",
    "likely benign": "Likely benign",
    "benign": "Benign",
}
# Combined labels map to the less certain tier for five tier truth, and to the shared tier for three tier truth.
COMBINED = {
    "Pathogenic/Likely pathogenic": "Likely pathogenic",
    "Benign/Likely benign": "Likely benign",
}

POPS = ["AFR", "AMR", "ASJ", "EAS", "SAS", "FIN", "NFE", "OTH"]


def _gnomad(path, prefix: str) -> pd.DataFrame:
    cols = ["chrom", "pos", "ref", "alt", "Filter", "AC", "AN", "AF", "Hom", "AC_POPMAX", "AN_POPMAX", "AF_POPMAX", "POPMAX"]
    cols += [f"Hemi_{p}" for p in POPS] + [f"AC_{p}" for p in POPS] + [f"AN_{p}" for p in POPS]
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    # The source header repeats AC_AFR where AF_AFR belongs; keep the first (allele count) only.
    df = df.loc[:, ~df.columns.duplicated()]
    df = df[[c for c in cols if c in df.columns]].copy()
    num = [c for c in df.columns if c not in ("chrom", "pos", "ref", "alt", "Filter", "POPMAX")]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Hemi"] = df[[f"Hemi_{p}" for p in POPS]].sum(axis=1, min_count=1)
    # Filtering allele frequency per population: maximum AF across the major continental groups (not OTH, ASJ, FIN).
    afs = []
    for p in ["AFR", "AMR", "EAS", "SAS", "NFE"]:
        an = df[f"AN_{p}"].where(df[f"AN_{p}"] > 0)
        afs.append(df[f"AC_{p}"] / an)
    df["AF_grpmax"] = pd.concat(afs, axis=1).max(axis=1)
    df["AN_grpmax_min"] = pd.concat([df[f"AN_{p}"] for p in ["AFR", "AMR", "EAS", "SAS", "NFE"]], axis=1).min(axis=1)
    keep = ["chrom", "pos", "ref", "alt", "Filter", "AC", "AN", "AF", "Hom", "Hemi", "AF_POPMAX", "POPMAX", "AF_grpmax"]
    keep += [f"AC_{p}" for p in ["AFR", "AMR", "EAS", "SAS", "NFE"]] + [f"AN_{p}" for p in ["AFR", "AMR", "EAS", "SAS", "NFE"]]
    df = df[keep]
    df.columns = ["chrom", "pos", "ref", "alt"] + [f"{prefix}_{c}" for c in keep[4:]]
    return df


def build_variants() -> pd.DataFrame:
    cv = pd.read_csv(RAW / "clinvar_gnomad_exomes.b37.tsv.gz", sep="\t", dtype=str, low_memory=False)
    cv = cv.loc[:, ~cv.columns.duplicated()]
    keep = ["chrom", "pos", "ref", "alt", "variation_id", "allele_id", "symbol", "hgvs_c", "hgvs_p", "molecular_consequence",
            "clinical_significance", "clinical_significance_ordered", "review_status", "review_status_ordered",
            "last_evaluated", "submitters_ordered", "all_traits", "all_pmids", "disease_mechanism", "origin",
            "dates_ordered", "conflicted", "scv"]
    base = cv[keep].copy()
    ex = _gnomad(RAW / "clinvar_gnomad_exomes.b37.tsv.gz", "ex")
    ge = _gnomad(RAW / "clinvar_gnomad_genomes.b37.tsv.gz", "ge")
    df = base.merge(ex, on=["chrom", "pos", "ref", "alt"], how="left").merge(ge, on=["chrom", "pos", "ref", "alt"], how="left")
    df = df.drop_duplicates("variation_id").reset_index(drop=True)
    df["pos"] = pd.to_numeric(df["pos"]).astype(int)
    df["key"] = df["chrom"] + ":" + df["pos"].astype(str) + ":" + df["ref"] + ":" + df["alt"]
    df = df.rename(columns={"symbol": "gene"})

    # ClinVar assertion
    cs = df["clinical_significance"].fillna("")
    df["clinvar_class"] = cs.map(lambda s: FIVE.get(s.lower(), COMBINED.get(s)))
    df["clinvar_label"] = cs
    df["stars"] = df["review_status"].map(REVIEW_STARS).fillna(0).astype(int)
    df["conflicted"] = df["conflicted"].fillna("0").astype(int)

    # Submitter level calls (aligned lists in the source)
    def _calls(row):
        subs = (row["submitters_ordered"] or "").split(";") if isinstance(row["submitters_ordered"], str) else []
        sig = (row["clinical_significance_ordered"] or "").split(";") if isinstance(row["clinical_significance_ordered"], str) else []
        rev = (row["review_status_ordered"] or "").split(";") if isinstance(row["review_status_ordered"], str) else []
        dates = (row["dates_ordered"] or "").split(";") if isinstance(row["dates_ordered"], str) else []
        out = []
        for i, s in enumerate(subs):
            out.append({"submitter": s, "call": sig[i] if i < len(sig) else "", "review": rev[i] if i < len(rev) else "",
                        "date": dates[i] if i < len(dates) else ""})
        return out
    df["submissions"] = df.apply(_calls, axis=1)
    df["n_submitters"] = df["submissions"].map(len)

    # HGVS parsing and consequence
    pcs = df["hgvs_c"].map(parse_c)
    pps = df["hgvs_p"].map(parse_p)
    cons = [primary_consequence(m, h)[0] for m, h in zip(df["molecular_consequence"], df["hgvs_c"])]
    df["consequence"] = [refine_consequence(c, a, b) for c, a, b in zip(cons, pcs, pps)]
    df["transcript"] = [c.transcript if c else None for c in pcs]
    df["c_kind"] = [c.kind if c else None for c in pcs]
    df["c_region"] = [c.start.region if c else None for c in pcs]
    df["c_pos"] = [c.start.base if c else np.nan for c in pcs]
    df["c_end"] = [c.end.base if c else np.nan for c in pcs]
    df["c_offset"] = [c.start.offset if c else 0 for c in pcs]
    df["c_end_offset"] = [c.end.offset if c else 0 for c in pcs]
    df["splice_distance"] = [c.splice_distance if c else 0 for c in pcs]
    df["length_change"] = [c.length_change if c else np.nan for c in pcs]
    df["p_kind"] = [p.kind if p else None for p in pps]
    df["p_ref"] = [p.ref_aa if p else None for p in pps]
    df["p_pos"] = [p.pos if p and p.pos is not None else np.nan for p in pps]
    df["p_alt"] = [p.alt_aa if p else None for p in pps]
    df["p_end"] = [p.end_pos if p and p.end_pos is not None else np.nan for p in pps]
    df["protein_change"] = [p.short if p else "" for p in pps]

    # Population summary across exomes and genomes
    df["gn_AC"] = df[["ex_AC", "ge_AC"]].sum(axis=1, min_count=1)
    df["gn_AN"] = df[["ex_AN", "ge_AN"]].sum(axis=1, min_count=1)
    df["gn_AF"] = df["gn_AC"] / df["gn_AN"]
    df["gn_Hom"] = df[["ex_Hom", "ge_Hom"]].sum(axis=1, min_count=1)
    df["gn_Hemi"] = df[["ex_Hemi", "ge_Hemi"]].sum(axis=1, min_count=1)
    df["gn_grpmax"] = df[["ex_AF_grpmax", "ge_AF_grpmax"]].max(axis=1)
    df["gn_popmax"] = df[["ex_AF_POPMAX", "ge_AF_POPMAX"]].max(axis=1)
    # Filtering allele frequency (FAF95): the lower bound of the one sided 95% Poisson interval of the allele frequency in each
    # continental group, exomes and genomes combined; the variant's FAF95 is the maximum over groups (gnomAD convention,
    # bottlenecked groups excluded). Groups with fewer than 2000 alleles are ignored for the grpmax AF used by BA1.
    fafs, afs = [], []
    for p in ["AFR", "AMR", "EAS", "SAS", "NFE"]:
        ac = df[[f"ex_AC_{p}", f"ge_AC_{p}"]].sum(axis=1, min_count=1).fillna(0).values
        an = df[[f"ex_AN_{p}", f"ge_AN_{p}"]].sum(axis=1, min_count=1).fillna(0).values
        lam = np.where(ac > 0, chi2.ppf(0.05, 2 * np.maximum(ac, 1)) / 2.0, 0.0)
        fafs.append(np.where(an > 0, lam / np.maximum(an, 1), np.nan))
        afs.append(np.where(an >= 2000, ac / np.maximum(an, 1), np.nan))
    df["gn_faf95"] = np.nanmax(np.vstack(fafs), axis=0) if len(df) else np.nan
    df["gn_grpmax_2000"] = np.nanmax(np.vstack(afs), axis=0) if len(df) else np.nan
    df = df.drop(columns=[c for c in df.columns if c.startswith(("ex_AC_", "ge_AC_", "ex_AN_", "ge_AN_"))])
    df["gn_seen"] = df["gn_AC"].fillna(0) > 0
    df["gn_covered"] = df["ex_AN"].notna() | df["ge_AN"].notna()

    df["pmids"] = df["all_pmids"].fillna("").map(lambda s: [p for p in s.split(";") if p.isdigit()])
    df["traits"] = df["all_traits"].fillna("").map(lambda s: [t for t in s.split(";") if t and t != "not provided" and t != "not specified"])
    df["mechanism"] = df["disease_mechanism"].fillna("").map(lambda s: "LOF" if "loss of function" in s else ("GOF" if "gain of function" in s else ""))
    drop = ["submitters_ordered", "clinical_significance_ordered", "review_status_ordered", "dates_ordered", "all_pmids",
            "all_traits", "disease_mechanism"]
    df = df.drop(columns=drop)
    return df


def main() -> None:
    df = build_variants()
    out = KB / "variants.parquet"
    df.to_parquet(out, index=False)
    print(f"variants: {len(df):,} rows, {df['gene'].nunique():,} genes -> {out}")
    print(df["consequence"].value_counts().to_string())
    print(df["clinvar_class"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
