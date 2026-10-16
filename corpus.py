"""Passage corpus for the evidence copilot.

Passage kinds and their evidence level (shown with every citation):
    guidance    guideline digests written for this project, each with references      level: guideline
    gene        gene cards generated from the knowledge base (diseases, inheritance,
                constraint, pool spectrum, mechanism, exon model)                       level: curated database
    disease     disease cards from HPO annotations (inheritance, genes, features)       level: curated database
    phenotype   HPO term definitions                                                    level: ontology
    literature  abstracts and full text paragraphs from the variant corpora, chunked    level: literature
Variant level evidence is not indexed statically: when a question names a
variant, the copilot evaluates it with the criteria engine and cites the
criteria as numbered evidence (see allelion.copilot.copilot).
"""
from __future__ import annotations

import json

import pandas as pd

from allelion.acmg import tables as tables_mod
from allelion.acmg.engine import Resources

RULES = Resources().rules
from allelion.config import KB, KNOWLEDGE, MODELS
from allelion.kb import transcripts as tx_mod
from allelion.kb.link_literature import find_genes, gene_symbols
from allelion.retrieval.text import sentences

INHERITANCE_WORDS = {"AD": "autosomal dominant", "AR": "autosomal recessive", "XL": "X linked", "XLR": "X linked recessive",
                     "XLD": "X linked dominant", "MT": "mitochondrial", "SOM": "somatic", "POLY": "polygenic", "Y": "Y linked"}


def gene_card(g: dict, stats: dict, tx, disease_names: dict) -> str:
    sym = g["gene"]
    parts = [f"{sym}."]
    dn = [disease_names.get(d, d) for d in g["diseases"]][:8]
    if len(dn):
        parts.append(f"{sym} is associated with {len(g['diseases'])} disease records, including: {'; '.join(dn)}.")
    else:
        traits = list(g.get("top_traits") if g.get("top_traits") is not None else [])[:4]
        parts.append(f"No HPO disease annotation is recorded for {sym}." + (f" ClinVar conditions include: {'; '.join(traits)}." if traits else ""))
    inh = [INHERITANCE_WORDS.get(i, i) for i in g["inheritance"]]
    parts.append(f"Recorded inheritance for {sym}: {', '.join(inh) if inh else 'not recorded'}.")
    pli = "not available" if pd.isna(g["pLI"]) else f"{g['pLI']:.2f}"
    lt = "not available" if pd.isna(g["loftool"]) else f"{g['loftool']:.3g}"
    parts.append(f"Gene constraint for {sym}: ExAC pLI {pli}; LoFtool percentile {lt}.")
    parts.append(f"Curated haploinsufficiency for {sym}: {'yes' if g['haploinsufficient'] else 'no'}.")
    tp, mp, mb = stats.get("truncating_plp", 0), stats.get("missense_plp", 0), stats.get("missense_blb", 0)
    parts.append(f"Evidence pool spectrum for {sym}: {tp} pathogenic truncating variants, {mp} pathogenic missense variants, {mb} benign missense "
                 f"variants, {stats.get('synonymous_blb', 0)} benign synonymous variants and {stats.get('intronic_plp', 0)} pathogenic intronic variants.")
    lof, why = tables_mod.lof_mechanism(stats, g, RULES)
    parts.append(f"Loss of function is {'an established' if lof else 'not an established'} disease mechanism for {sym} in this knowledge base"
                 f"{', so PVS1 can apply' if lof else ', so PVS1 does not apply'}: {why}.")
    frac_b = mb / (mp + mb) if (mp + mb) else None
    if frac_b is not None:
        parts.append(f"Benign share of classified missense variants in {sym}: {frac_b:.0%}.")
    if tx is not None and tx.anchored:
        parts.append(f"The exon model for {sym} has {len(tx.exons)} exons anchored to the reporting transcript"
                     f"{f', coding length {tx.cds_len} bases ({tx.cds_len // 3 - 1} residues)' if tx.cds_len else ''}.")
    if len(g.get("orphanet_prevalence", [])):
        parts.append(f"Orphanet prevalence classes for {sym} disorders: {', '.join(g['orphanet_prevalence'])}.")
    if len(g.get("orphanet_onset", [])):
        parts.append(f"Orphanet onset for {sym} disorders: {', '.join(g['orphanet_onset'][:4])}.")
    return " ".join(parts)


def build() -> pd.DataFrame:
    rows = []
    cards = json.loads((KNOWLEDGE / "guidance_cards.json").read_text())
    for c in cards:
        rows.append({"source": f"guidance:{c['id']}", "kind": "guidance", "title": c["title"], "text": c["text"], "level": "guideline",
                     "genes": [], "codes": [t for t in c["tags"] if t[:2] in ("PV", "PS", "PM", "PP", "BA", "BS", "BP") and len(t) <= 4],
                     "refs": c["refs"]})
    genes = pd.read_parquet(KB / "genes.parquet")
    diseases = pd.read_parquet(KB / "diseases.parquet")
    dname = dict(zip(diseases["disease_id"], diseases["name"]))
    T = tables_mod.load()
    txs = tx_mod.load()
    for g in genes.to_dict("records"):
        if g["clinvar_total"] < 5 and not len(g["diseases"]):
            continue
        st = T.gene_stats.get(g["gene"], {})
        rows.append({"source": f"gene:{g['gene']}", "kind": "gene", "title": f"Gene card: {g['gene']}", "text": gene_card(g, st, txs.get(g["gene"]), dname),
                     "level": "curated database", "genes": [g["gene"]], "codes": [], "refs": ["ClinVar", "HPO", "ExAC", "evidence pool"]})
    gd = json.loads((KB / "gene_diseases.json").read_text())
    dis_genes: dict[str, list] = {}
    for gsym, ds in gd.items():
        for d in ds:
            dis_genes.setdefault(d, []).append(gsym)
    terms = pd.read_parquet(KB / "hpo_terms.parquet").set_index("hpo_id")
    tname, tic = terms["name"].to_dict(), terms["ic"].to_dict()
    for d in diseases.to_dict("records"):
        fq = json.loads(d["term_freq"]) if d.get("term_freq") else {}
        feats = sorted(d["terms"], key=lambda t: (-fq.get(t, 0.5), -tic.get(t, 0)))[:15]
        inh = [INHERITANCE_WORDS.get(i, i) for i in d["inheritance"]]
        gs = sorted(dis_genes.get(d["disease_id"], []))
        text = (f"{d['name']} ({d['disease_id']}). Inheritance: {', '.join(inh) if inh else 'not annotated'}. "
                f"{'Associated genes: ' + ', '.join(gs[:10]) + '. ' if gs else ''}"
                f"Characteristic features (most frequent first): {'; '.join(tname.get(t, t) for t in feats)}.")
        rows.append({"source": f"disease:{d['disease_id']}", "kind": "disease", "title": d["name"], "text": text, "level": "curated database",
                     "genes": gs[:10], "codes": [], "refs": ["HPO annotations"]})
    for hid, r in terms.iterrows():
        if not r["definition"]:
            continue
        syn = f" Synonyms: {', '.join(list(r['synonyms'])[:5])}." if len(r["synonyms"]) else ""
        par = [tname.get(p, p) for p in r["parents"]]
        rows.append({"source": f"hpo:{hid}", "kind": "phenotype", "title": f"{r['name']} ({hid})",
                     "text": f"{r['name']} ({hid}). Definition: {r['definition']}{syn} Broader terms: {', '.join(par)}.", "level": "ontology",
                     "genes": [], "codes": [], "refs": ["Human Phenotype Ontology"]})
    docs = pd.read_parquet(KB / "literature_docs.parquet")
    symbols = gene_symbols()
    for d in docs.to_dict("records"):
        sents = sentences(d["text"])
        chunk, chunks = [], []
        for s in sents:
            chunk.append(s)
            if sum(len(x.split()) for x in chunk) >= 110:
                chunks.append(" ".join(chunk))
                chunk = []
        if chunk:
            chunks.append(" ".join(chunk))
        for i, ch in enumerate(chunks or [""]):
            pm = d["pmid"]
            ref = f"PMID {pm}" if pm.isdigit() else pm
            rows.append({"source": f"lit:{d['doc_id']}", "kind": "literature", "title": d["title"], "text": ch, "level": "literature",
                         "genes": find_genes(d["title"] + " " + ch, symbols)[:8], "codes": [], "refs": [f"{ref} ({d['corpus']} corpus)"]})
    df = pd.DataFrame(rows)
    df.insert(0, "pid", range(len(df)))
    df.to_parquet(MODELS / "passages.parquet", index=False)
    stats = {"passages": len(df), "by_kind": df["kind"].value_counts().to_dict(), "words": int(df["text"].str.split().map(len).sum())}
    (MODELS / "passages_stats.json").write_text(json.dumps(stats, indent=1))
    print(stats)
    return df


if __name__ == "__main__":
    build()
