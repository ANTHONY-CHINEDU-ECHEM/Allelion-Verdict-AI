"""Knowledge graph with provenance on every edge.

Nodes: gene, disease, phenotype (HPO term), variant (ClinVar allele), trait (ClinVar condition name), publication.
Edges:
    gene  associated_with  disease          HPO gene to disease annotations
    disease  has_phenotype  phenotype       HPO disease annotations
    variant  in_gene  gene                  ClinVar
    variant  asserted_for  trait            ClinVar condition names
    trait  same_as  disease                 exact name match between ClinVar trait and HPO disease
    variant  cited_in  publication          ClinVar citations
    publication  mentions_gene  gene        dictionary match in the literature corpora
    publication  mentions_variant  variant  extractor plus normaliser plus gene context
"""
from __future__ import annotations

import json

import pandas as pd

from allelion.config import KB


def build() -> dict:
    genes = pd.read_parquet(KB / "genes.parquet")
    diseases = pd.read_parquet(KB / "diseases.parquet")
    terms = pd.read_parquet(KB / "hpo_terms.parquet")
    v = pd.read_parquet(KB / "variants.parquet", columns=["variation_id", "gene", "hgvs_c", "protein_change", "clinvar_class", "stars", "traits", "pmids"])
    links = pd.read_parquet(KB / "literature_links.parquet")
    gene_dis = json.loads((KB / "gene_diseases.json").read_text())

    nodes, edges = [], []
    for r in genes.itertuples():
        nodes.append({"id": f"gene:{r.gene}", "type": "gene", "label": r.gene})
    for r in diseases.itertuples():
        nodes.append({"id": f"disease:{r.disease_id}", "type": "disease", "label": r.name})
    used_terms = set()
    for r in diseases.itertuples():
        for t in r.terms:
            edges.append({"src": f"disease:{r.disease_id}", "rel": "has_phenotype", "dst": f"hpo:{t}", "source": "HPO annotations"})
            used_terms.add(t)
    tname = terms.set_index("hpo_id")["name"].to_dict()
    for t in used_terms:
        nodes.append({"id": f"hpo:{t}", "type": "phenotype", "label": tname.get(t, t)})
    for g, ds in gene_dis.items():
        for d in ds:
            edges.append({"src": f"gene:{g}", "rel": "associated_with", "dst": f"disease:{d}", "source": "HPO gene to disease"})
    v = v[v["gene"].notna()]
    for r in v.itertuples():
        nodes.append({"id": f"variant:{r.variation_id}", "type": "variant", "label": f"{r.gene} {str(r.hgvs_c).split(':')[-1]}"})
        edges.append({"src": f"variant:{r.variation_id}", "rel": "in_gene", "dst": f"gene:{r.gene}", "source": "ClinVar"})
    dname = {n.lower(): d for d, n in zip(diseases["disease_id"], diseases["name"])}
    traits = v[["variation_id", "traits"]].explode("traits").dropna()
    for t in traits["traits"].unique():
        nodes.append({"id": f"trait:{t}", "type": "trait", "label": t})
        if t.lower() in dname:
            edges.append({"src": f"trait:{t}", "rel": "same_as", "dst": f"disease:{dname[t.lower()]}", "source": "exact name match"})
    for r in traits.itertuples():
        edges.append({"src": f"variant:{r.variation_id}", "rel": "asserted_for", "dst": f"trait:{r.traits}", "source": "ClinVar"})
    cites = v[["variation_id", "pmids"]].explode("pmids").dropna()
    for p in cites["pmids"].unique():
        nodes.append({"id": f"pmid:{p}", "type": "publication", "label": f"PMID {p}"})
    for r in cites.itertuples():
        edges.append({"src": f"variant:{r.variation_id}", "rel": "cited_in", "dst": f"pmid:{r.pmids}", "source": "ClinVar"})
    for r in links.itertuples():
        if r.link == "gene_mention":
            edges.append({"src": f"pmid:{r.pmid}", "rel": "mentions_gene", "dst": f"gene:{r.gene}", "source": "literature corpora (dictionary)"})
        elif r.link == "text_mined" and r.variation_id:
            edges.append({"src": f"pmid:{r.pmid}", "rel": "mentions_variant", "dst": f"variant:{r.variation_id}", "source": f"literature corpora (extractor: {r.mention})"})
    for p in links["pmid"].unique():
        nodes.append({"id": f"pmid:{p}", "type": "publication", "label": f"PMID {p}"})
    n = pd.DataFrame(nodes).drop_duplicates("id")
    e = pd.DataFrame(edges).drop_duplicates(["src", "rel", "dst"])
    n.to_parquet(KB / "graph_nodes.parquet", index=False)
    e.to_parquet(KB / "graph_edges.parquet", index=False)
    stats = {"nodes": len(n), "edges": len(e), "node_types": n["type"].value_counts().to_dict(), "edge_types": e["rel"].value_counts().to_dict()}
    (KB / "graph_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    return stats


if __name__ == "__main__":
    build()
