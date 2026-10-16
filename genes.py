"""Gene table: constraint, inheritance, dosage sensitivity, diseases, exon model and ClinVar spectrum.

Gene level facts that do not depend on any variant label live here. Label
derived gene statistics (missense constraint from ClinVar, truncating
mechanism, PP2 and BP1 eligibility) are computed from the evidence pool in
allelion.acmg.tables so that evaluation variants never contribute to them.
"""
from __future__ import annotations

import gzip
import json

import numpy as np
import pandas as pd

from allelion.config import KB, RAW
from allelion.kb import transcripts as tx_mod


def _list(path) -> set[str]:
    return {l.strip().split("\t")[0] for l in open(path) if l.strip() and not l.startswith("#")}


def _orpha() -> pd.DataFrame:
    rows = []
    for line in open(RAW / "intervar" / "orpha.txt.utf8", encoding="utf8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2 or parts[0] == "Orpha":
            continue
        f = parts[1].split("|")
        if len(f) < 6:
            continue
        rows.append({"orpha": f[0], "name": f[1], "prevalence": f[2], "inheritance": f[3], "onset": f[4].replace("<br>", "; "),
                     "omim": [x for x in f[5].split() if x.isdigit()]})
    return pd.DataFrame(rows)


def build() -> pd.DataFrame:
    variants = pd.read_parquet(KB / "variants.parquet", columns=["gene", "clinvar_class", "consequence", "stars", "mechanism", "p_pos", "traits"])
    genes = sorted(set(variants["gene"].dropna()))
    g = pd.DataFrame({"gene": genes})

    pli = pd.read_csv(RAW / "vep" / "pLI_values.txt", sep="\t").drop_duplicates("gene").set_index("gene")["pLI"]
    lof = pd.read_csv(RAW / "vep" / "LoFtool_scores.txt", sep="\t")
    lof = lof.drop_duplicates(lof.columns[0]).set_index(lof.columns[0])[lof.columns[1]]
    g["pLI"] = g["gene"].map(pli)
    g["loftool"] = g["gene"].map(lof)

    gene_dis = json.loads((KB / "gene_diseases.json").read_text())
    dis = pd.read_parquet(KB / "diseases.parquet").set_index("disease_id")
    inh_lists = {"AR": _list(RAW / "gene_lists" / "all_ar.tsv"), "AD": _list(RAW / "gene_lists" / "all_ad.tsv")}
    hi = _list(RAW / "gene_lists" / "clingen_level3_genes_2018_09_13.tsv") | _list(RAW / "gene_lists" / "haploinsufficiency_severe_curated_2016.tsv")
    lof_tolerant = _list(RAW / "gene_lists" / "homozygous_lof_tolerant_twohit.tsv")
    essential = _list(RAW / "gene_lists" / "mgi_essential.tsv")
    iv_lof = _list(RAW / "intervar" / "PVS1.LOF.genes.hg19")
    iv_pp2 = _list(RAW / "intervar" / "PP2.genes.hg19")
    iv_bp1 = _list(RAW / "intervar" / "BP1.genes.hg19")
    adult = _list(RAW / "intervar" / "mim_adultonset.txt")

    orpha = _orpha()
    omim_to_orpha = {}
    for r in orpha.itertuples():
        for m in r.omim:
            omim_to_orpha.setdefault(m, []).append(r)

    rows = []
    for gene in genes:
        ds = gene_dis.get(gene, [])
        inh, names, onsets, prev = set(), [], set(), []
        for d in ds:
            if d in dis.index:
                inh |= set(dis.at[d, "inheritance"])
                names.append(dis.at[d, "name"])
            if d.startswith("OMIM:"):
                for o in omim_to_orpha.get(d[5:], []):
                    if o.prevalence and o.prevalence != "-":
                        prev.append(o.prevalence)
                    for x in o.onset.split("; "):
                        if x and x != "-":
                            onsets.add(x)
        for k, members in inh_lists.items():
            if gene in members:
                inh.add(k)
        adult_onset = any(d.startswith("OMIM:") and d[5:] in adult for d in ds)
        rows.append({
            "gene": gene,
            "diseases": ds,
            "disease_names": names,
            "inheritance": sorted(inh),
            "dominant": bool(inh & {"AD", "XLD"}),
            "recessive": bool(inh & {"AR", "XLR"}),
            "x_linked": bool(inh & {"XL", "XLR", "XLD"}),
            "haploinsufficient": gene in hi,
            "lof_tolerant_twohit": gene in lof_tolerant,
            "mouse_essential": gene in essential,
            "intervar_lof_gene": gene in iv_lof,
            "intervar_pp2_gene": gene in iv_pp2,
            "intervar_bp1_gene": gene in iv_bp1,
            "orphanet_prevalence": sorted(set(prev))[:4],
            "orphanet_onset": sorted(onsets),
            "adult_onset": adult_onset,
        })
    meta = pd.DataFrame(rows)
    g = g.merge(meta, on="gene", how="left")

    txs = tx_mod.load()
    g["exon_model"] = g["gene"].map(lambda x: txs[x].anchored if x in txs else False)
    g["n_exons"] = g["gene"].map(lambda x: len(txs[x].exons) if x in txs and txs[x].anchored else np.nan)
    g["cds_len"] = g["gene"].map(lambda x: txs[x].cds_len if x in txs and txs[x].anchored else np.nan)
    g["protein_len"] = g["gene"].map(variants.groupby("gene")["p_pos"].max())
    g["chrom"] = g["gene"].map(lambda x: txs[x].chrom if x in txs else None)

    # ClinVar spectrum over all assertions (display only; not used as evidence)
    spec = variants.dropna(subset=["clinvar_class"]).groupby(["gene", "clinvar_class"]).size().unstack(fill_value=0)
    for c in spec.columns:
        g[f"clinvar_{c.lower().replace(' ', '_')}"] = g["gene"].map(spec[c]).fillna(0).astype(int)
    g["clinvar_total"] = g["gene"].map(variants.groupby("gene").size()).fillna(0).astype(int)
    mech = variants[variants["mechanism"] != ""].groupby("gene")["mechanism"].agg(lambda s: s.value_counts().index[0])
    g["clinvar_mechanism"] = g["gene"].map(mech).fillna("")
    top_traits = variants.explode("traits").dropna(subset=["traits"]).groupby("gene")["traits"].agg(lambda s: s.value_counts().index[:5].tolist())
    g["top_traits"] = g["gene"].map(top_traits)
    g["top_traits"] = g["top_traits"].map(lambda x: x if isinstance(x, list) else [])
    g.to_parquet(KB / "genes.parquet", index=False)
    print(f"genes: {len(g):,}; with pLI {g['pLI'].notna().sum():,}; with diseases {(g['diseases'].map(len) > 0).sum():,}; "
          f"exon models {g['exon_model'].sum():,}; dominant {g['dominant'].sum():,}; recessive {g['recessive'].sum():,}")
    return g


if __name__ == "__main__":
    build()
