"""Feature construction for the in silico predictors.

The missense predictor sees only what a sequence based tool would see: the
amino acid substitution and its physicochemical change, where it falls in the
protein, the codon level nucleotide change, and gene level constraint that is
independent of ClinVar (ExAC pLI, LoFtool, curated haploinsufficiency, mouse
essentiality, mode of inheritance). It never sees the gene's name or any
statistic built from ClinVar labels, so its score does not restate the
hotspot (PM1), allelic (PS1, PM5) or gene level (PP2, BP1) evidence it will be
combined with.

The splice region predictor sees the intronic offset, the side of the exon,
the nucleotide change and the same gene constraint features.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

AAS = list("ACDEFGHIKLMNPQRSTVWY")

# Grantham (1974) properties: composition, polarity, volume
GRANTHAM_PROPS = {
    "S": (1.42, 9.2, 32), "R": (0.65, 10.5, 124), "L": (0.0, 4.9, 111), "P": (0.39, 8.0, 32.5), "T": (0.71, 8.6, 61),
    "A": (0.0, 8.1, 31), "V": (0.0, 5.9, 84), "G": (0.74, 9.0, 3), "I": (0.0, 5.2, 111), "F": (0.0, 5.2, 132),
    "Y": (0.20, 6.2, 136), "C": (2.75, 5.5, 55), "H": (0.58, 10.4, 96), "Q": (0.89, 10.5, 85), "N": (1.33, 11.6, 56),
    "K": (0.33, 11.3, 119), "D": (1.38, 13.0, 54), "E": (0.92, 12.3, 83), "M": (0.0, 5.7, 105), "W": (0.13, 5.4, 170),
}
KYTE_DOOLITTLE = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
                  "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}
CHARGE = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1}
AROMATIC = set("FWYH")

# BLOSUM62, rows and columns in the order of AAS
_B62 = """
 4  0 -2 -1 -2  0 -2 -1 -1 -1 -1 -2 -1 -1 -1  1  0  0 -3 -2
 0  9 -3 -4 -2 -3 -3 -1 -3 -1 -1 -3 -3 -3 -3 -1 -1 -1 -2 -2
-2 -3  6  2 -3 -1 -1 -3 -1 -4 -3  1 -1  0 -2  0 -1 -3 -4 -3
-1 -4  2  5 -3 -2  0 -3  1 -3 -2  0 -1  2  0  0 -1 -2 -3 -2
-2 -2 -3 -3  6 -3 -1  0 -3  0  0 -3 -4 -3 -3 -2 -2 -1  1  3
 0 -3 -1 -2 -3  6 -2 -4 -2 -4 -3  0 -2 -2 -2  0 -2 -3 -2 -3
-2 -3 -1  0 -1 -2  8 -3 -1 -3 -2  1 -2  0  0 -1 -2 -3 -2  2
-1 -1 -3 -3  0 -4 -3  4 -3  2  1 -3 -3 -3 -3 -2 -1  3 -3 -1
-1 -3 -1  1 -3 -2 -1 -3  5 -2 -1  0 -1  1  2  0 -1 -2 -3 -2
-1 -1 -4 -3  0 -4 -3  2 -2  4  2 -3 -3 -2 -2 -2 -1  1 -2 -1
-1 -1 -3 -2  0 -3 -2  1 -1  2  5 -2 -2  0 -1 -1 -1  1 -1 -1
-2 -3  1  0 -3  0  1 -3  0 -3 -2  6 -2  0  0  1  0 -3 -4 -2
-1 -3 -1 -1 -4 -2 -2 -3 -1 -3 -2 -2  7 -1 -2 -1 -1 -2 -4 -3
-1 -3  0  2 -3 -2  0 -3  1 -2  0  0 -1  5  1  0 -1 -2 -2 -1
-1 -3 -2  0 -3 -2  0 -3  2 -2 -1  0 -2  1  5 -1 -1 -3 -3 -2
 1 -1  0  0 -2  0 -1 -2  0 -2 -1  1 -1  0 -1  4  1 -2 -3 -2
 0 -1 -1 -1 -2 -2 -2 -1 -1 -1 -1  0 -1 -1 -1  1  5  0 -2 -2
 0 -1 -3 -2 -1 -3 -3  3 -2  1  1 -3 -2 -2 -3 -2  0  4 -3 -1
-3 -2 -4 -3  1 -2 -2 -3 -3 -2 -1 -4 -4 -2 -3 -3 -2 -3 11  2
-2 -2 -3 -2  3 -3  2 -1 -2 -1 -1 -2 -3 -1 -2 -2 -2 -1  2  7
"""
BLOSUM62 = {}
for _i, _row in enumerate(_B62.strip().split("\n")):
    for _j, _v in enumerate(_row.split()):
        BLOSUM62[(AAS[_i], AAS[_j])] = int(_v)


def grantham(a: str, b: str) -> float:
    if a not in GRANTHAM_PROPS or b not in GRANTHAM_PROPS:
        return np.nan
    c1, p1, v1 = GRANTHAM_PROPS[a]
    c2, p2, v2 = GRANTHAM_PROPS[b]
    return 50.723 * math.sqrt(1.833 * (c1 - c2) ** 2 + 0.1018 * (p1 - p2) ** 2 + 0.000399 * (v1 - v2) ** 2) / 1.0


TRANSITIONS = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}
GENE_FEATURES = ["pLI", "loftool", "haploinsufficient", "mouse_essential", "dominant", "recessive", "x_linked", "n_diseases", "log_protein_len"]


def gene_frame(genes: pd.DataFrame) -> pd.DataFrame:
    g = genes.set_index("gene")
    out = pd.DataFrame(index=g.index)
    out["pLI"] = g["pLI"]
    out["loftool"] = g["loftool"]
    for c in ["haploinsufficient", "mouse_essential", "dominant", "recessive", "x_linked"]:
        out[c] = g[c].astype(float)
    out["n_diseases"] = g["diseases"].map(len).astype(float)
    out["log_protein_len"] = np.log10(g["protein_len"].fillna(500).clip(lower=10))
    return out


MISSENSE_FEATURES = (
    ["blosum62", "grantham", "d_hydro", "abs_d_hydro", "d_volume", "abs_d_volume", "d_charge", "charge_change", "to_pro", "from_pro",
     "to_gly", "from_gly", "to_cys", "from_cys", "aromatic_change", "rel_pos", "log_pos", "log_dist_c", "codon_pos", "transition"]
    + [f"ref_{a}" for a in AAS] + [f"alt_{a}" for a in AAS] + GENE_FEATURES
)


def missense_features(df: pd.DataFrame, gf: pd.DataFrame) -> pd.DataFrame:
    """df needs: gene, p_ref, p_alt, p_pos, c_pos, hgvs_c (for nucleotide change)."""
    ref, alt = df["p_ref"].fillna("X"), df["p_alt"].fillna("X")
    X = pd.DataFrame(index=df.index)
    X["blosum62"] = [BLOSUM62.get((a, b), np.nan) for a, b in zip(ref, alt)]
    X["grantham"] = [grantham(a, b) for a, b in zip(ref, alt)]
    X["d_hydro"] = [KYTE_DOOLITTLE.get(b, np.nan) - KYTE_DOOLITTLE.get(a, np.nan) for a, b in zip(ref, alt)]
    X["abs_d_hydro"] = X["d_hydro"].abs()
    X["d_volume"] = [GRANTHAM_PROPS.get(b, (0, 0, np.nan))[2] - GRANTHAM_PROPS.get(a, (0, 0, np.nan))[2] for a, b in zip(ref, alt)]
    X["abs_d_volume"] = X["d_volume"].abs()
    X["d_charge"] = [CHARGE.get(b, 0.0) - CHARGE.get(a, 0.0) for a, b in zip(ref, alt)]
    X["charge_change"] = (X["d_charge"].abs() >= 0.9).astype(float)
    X["to_pro"] = (alt == "P").astype(float)
    X["from_pro"] = (ref == "P").astype(float)
    X["to_gly"] = (alt == "G").astype(float)
    X["from_gly"] = (ref == "G").astype(float)
    X["to_cys"] = (alt == "C").astype(float)
    X["from_cys"] = (ref == "C").astype(float)
    X["aromatic_change"] = [float((a in AROMATIC) != (b in AROMATIC)) for a, b in zip(ref, alt)]
    gene_feats = gf.reindex(df["gene"]).reset_index(drop=True)
    gene_feats.index = df.index
    plen = (10 ** gene_feats["log_protein_len"]).values
    pos = df["p_pos"].astype(float).values
    X["rel_pos"] = np.clip(pos / plen, 0, 1)
    X["log_pos"] = np.log10(np.maximum(pos, 1))
    X["log_dist_c"] = np.log10(np.maximum(plen - pos, 1))
    X["codon_pos"] = (df["c_pos"].astype(float) - 1) % 3
    nuc = df["hgvs_c"].fillna("").str.extract(r"([ACGT])>([ACGT])$")
    X["transition"] = [float((a, b) in TRANSITIONS) if isinstance(a, str) else np.nan for a, b in zip(nuc[0], nuc[1])]
    for a in AAS:
        X[f"ref_{a}"] = (ref == a).astype(float)
        X[f"alt_{a}"] = (alt == a).astype(float)
    for c in GENE_FEATURES:
        X[c] = gene_feats[c].values
    return X[MISSENSE_FEATURES]


SPLICE_FEATURES = ["offset", "abs_offset", "donor_side", "log_abs_offset", "transition", "ref_G", "ref_A", "ref_C", "ref_T",
                   "alt_G", "alt_A", "alt_C", "alt_T", "is_sub", "is_indel", "indel_len"] + GENE_FEATURES


def splice_features(df: pd.DataFrame, gf: pd.DataFrame) -> pd.DataFrame:
    """df needs: gene, c_offset, c_kind, hgvs_c, length_change."""
    X = pd.DataFrame(index=df.index)
    off = df["c_offset"].astype(float)
    X["offset"] = off
    X["abs_offset"] = off.abs()
    X["donor_side"] = (off > 0).astype(float)
    X["log_abs_offset"] = np.log10(off.abs().clip(lower=1))
    nuc = df["hgvs_c"].fillna("").str.extract(r"([ACGT])>([ACGT])$")
    X["transition"] = [float((a, b) in TRANSITIONS) if isinstance(a, str) else 0.0 for a, b in zip(nuc[0], nuc[1])]
    for b in "GACT":
        X[f"ref_{b}"] = (nuc[0] == b).astype(float)
        X[f"alt_{b}"] = (nuc[1] == b).astype(float)
    X["is_sub"] = (df["c_kind"] == "sub").astype(float)
    X["is_indel"] = (df["c_kind"] != "sub").astype(float)
    X["indel_len"] = df["length_change"].astype(float).abs().fillna(0)
    gene_feats = gf.reindex(df["gene"]).reset_index(drop=True)
    gene_feats.index = df.index
    for c in GENE_FEATURES:
        X[c] = gene_feats[c].values
    return X[SPLICE_FEATURES]
