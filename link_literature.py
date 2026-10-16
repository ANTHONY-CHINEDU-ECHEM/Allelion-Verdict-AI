"""Link literature to genes and ClinVar variants.

For every document in the literature corpora: find gene symbols (case
sensitive dictionary match with an ambiguity stop list), extract variant
mentions with the trained extractor, normalise them, and match normalised
protein and coding changes to ClinVar variants in the genes the document
mentions. ClinVar's own PubMed citations add a second, independent link type.

Output: data/kb/literature_links.parquet with one row per (document, gene, variant, mention).
"""
from __future__ import annotations

import re

import pandas as pd

from allelion.config import KB, RAW
from allelion.models.extractor import extract, load_crf

# gene symbols that are also common words or acronyms in abstracts
STOP = {"CAT", "SET", "MET", "MAX", "REST", "WAS", "IMPACT", "CAPS", "SHE", "TIP", "SDS", "PCR", "DNA", "RNA", "SNP", "ACE", "ALL", "AML",
        "CLL", "CML", "MRI", "MS", "CF", "HD", "PD", "AD", "AR", "XL", "GC", "LPL", "CO", "CS", "ER", "PR", "TNF", "IL", "HLA", "ABO",
        "CAD", "MAP", "MARS", "CAP", "PIGS", "BAD", "BID", "ARC", "NHS", "FAT", "WT", "GAP", "SPR", "TH", "CD", "LD", "OR", "HR", "CI",
        "FAP", "PTH", "GH", "LH", "FSH", "TSH", "CRP", "ESR", "INS", "POR", "SON", "MAF", "HP", "PC", "CA", "MB", "TEC", "LCT"}


def gene_symbols() -> set[str]:
    genes = set(pd.read_parquet(KB / "genes.parquet")["gene"])
    uni = {l.strip() for l in open(RAW / "gene_lists" / "universe.tsv") if l.strip()}
    return {g for g in genes | uni if len(g) >= 3 and g not in STOP and re.fullmatch(r"[A-Z0-9]+", g) and not g.isdigit()}


_WORD = re.compile(r"[A-Za-z0-9]+")


def find_genes(text: str, symbols: set[str]) -> list[str]:
    seen = []
    for m in _WORD.finditer(text):
        w = m.group()
        if w in symbols and w not in seen:
            seen.append(w)
    return seen


def build() -> pd.DataFrame:
    docs = pd.read_parquet(KB / "literature_docs.parquet")
    v = pd.read_parquet(KB / "variants.parquet", columns=["variation_id", "gene", "protein_change", "hgvs_c", "key", "pmids"])
    by_prot = v[v["protein_change"] != ""].groupby(["gene", "protein_change"])["variation_id"].apply(list).to_dict()
    v["c_short"] = v["hgvs_c"].fillna("").str.split(":").str[-1]
    by_c = v[v["c_short"] != ""].groupby(["gene", "c_short"])["variation_id"].apply(list).to_dict()
    symbols = gene_symbols()
    crf = load_crf()
    rows = []
    for r in docs.itertuples():
        text = f"{r.title} {r.text}"
        genes = find_genes(text, symbols)
        for g in genes:
            rows.append({"doc_id": r.doc_id, "pmid": r.pmid, "gene": g, "variation_id": None, "mention": "", "norm": "", "link": "gene_mention"})
        for m in extract(text, crf):
            norm = m.get("norm")
            if not norm:
                continue
            hit = False
            for g in genes:
                ids = by_prot.get((g, norm), []) if m["kind"] == "protein" else by_c.get((g, norm), []) if m["kind"] == "dna" else []
                for vid in ids:
                    rows.append({"doc_id": r.doc_id, "pmid": r.pmid, "gene": g, "variation_id": vid, "mention": m["text"], "norm": norm,
                                 "link": "text_mined"})
                    hit = True
            if not hit:
                rows.append({"doc_id": r.doc_id, "pmid": r.pmid, "gene": genes[0] if len(genes) == 1 else "", "variation_id": None,
                             "mention": m["text"], "norm": norm, "link": "unlinked_mention"})
    # ClinVar citations that point at documents in the corpus
    corpus_pmids = set(docs["pmid"])
    cited = v[["variation_id", "gene", "pmids"]].explode("pmids").dropna()
    cited = cited[cited["pmids"].isin(corpus_pmids)]
    pm2doc = docs.groupby("pmid")["doc_id"].first().to_dict()
    for rr in cited.itertuples():
        rows.append({"doc_id": pm2doc[rr.pmids], "pmid": rr.pmids, "gene": rr.gene, "variation_id": rr.variation_id, "mention": "", "norm": "",
                     "link": "clinvar_citation"})
    out = pd.DataFrame(rows).drop_duplicates(["doc_id", "gene", "variation_id", "norm", "link"])
    out.to_parquet(KB / "literature_links.parquet", index=False)
    print(out["link"].value_counts().to_dict(), "| documents with a linked variant:",
          out[out["variation_id"].notna()]["doc_id"].nunique(), "| variants linked:", out["variation_id"].nunique())
    return out


if __name__ == "__main__":
    build()
