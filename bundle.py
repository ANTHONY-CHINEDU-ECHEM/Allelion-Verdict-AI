"""Export the in browser engine bundle (web/public/bundle).

The React interface runs the whole system without a server from these files:
the criteria engine with its evidence tables, exon models and the three
LightGBM models (missense, splice region, learned verdict); the retrieval
pipeline (passages, SIF word vectors, dual encoder, reranker); the verifier
coefficients; phenotype ontology and disease profiles for PP4 matching; the
literature links; conflict records; and the evaluation summary.

Variant records are exported for every blind and development variant and for
all labelled or conflicting variants in a showcase gene set, so they can be
looked up with their gnomAD counts. Any other variant can be entered by HGVS.
"""
from __future__ import annotations

import gzip
import json
import pickle
import shutil

import lightgbm as lgb
import numpy as np
import pandas as pd

from allelion.acmg import tables as tables_mod
from allelion.acmg.engine import BA1_EXCEPTIONS, DEFAULT_STRENGTH, ORDER, TITLES, Resources
from allelion.config import BUNDLE, CLASSES, EVAL, KB, KNOWLEDGE, MODELS, REPORTS
from allelion.kb import transcripts as tx_mod
from allelion.kb.hgvs import parse_c, parse_p
from allelion.models.features import AAS, BLOSUM62, GRANTHAM_PROPS, KYTE_DOOLITTLE, MISSENSE_FEATURES, SPLICE_FEATURES

SHOWCASE = """APC MYH11 ACTA2 TMEM43 DSP PKP2 DSG2 DSC2 BRCA1 BRCA2 SCN5A RYR2 LMNA MYBPC3 MYH7 TNNT2 TNNI3 TPM1 MYL3 ACTC1 PRKAG2
MYL2 KCNQ1 KCNH2 LDLR APOB PCSK9 COL3A1 FBN1 TGFBR1 TGFBR2 SMAD3 MLH1 MSH2 MSH6 PMS2 MUTYH MEN1 RET NF2 PTEN RB1 SDHAF2 SDHB SDHC SDHD
STK11 TP53 TSC1 TSC2 VHL WT1 BMPR1A SMAD4 OTC PALB2 RYR1 CACNA1S ATP7B BTD GLA HFE TTR CFTR GJB2 PTPN11 SOS1 RAF1 BRAF KRAS HRAS
MAP2K1 MAP2K2 SHOC2 NF1 CDH1 CHEK2 ATM SCN1A DMD F8 F9 PAH GAA HBB SMN1 FMR1 MECP2 CDKL5 KCNQ2 STXBP1 SLC2A1 ABCA4 USH2A""".split()


def _clean(x):
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    if isinstance(x, (np.floating,)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return [_clean(y) for y in x.tolist()]
    return x


def _deep(o):
    if isinstance(o, dict):
        return {k: _deep(v) for k, v in o.items() if isinstance(k, str)}
    if isinstance(o, (list, tuple)):
        return [_deep(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_deep(v) for v in o.tolist()]
    return _clean(o)


def _write(name: str, obj) -> int:
    """Write gzip compressed JSON (the browser decompresses with DecompressionStream)."""
    p = BUNDLE / (name + ".gz")
    with gzip.open(p, "wb", compresslevel=9) as fh:
        fh.write(json.dumps(_deep(obj), separators=(",", ":"), allow_nan=False).encode())
    return p.stat().st_size


VAR_COLS = ["variation_id", "gene", "hgvs_c", "hgvs_p", "consequence", "gn_AC", "gn_AN", "gn_grpmax", "gn_grpmax_2000", "gn_faf95", "gn_Hom",
            "gn_Hemi", "clinvar_class", "stars", "conflicted", "set", "trait"]
PER_GENE = 300


def hydrate(rec: dict) -> dict:
    """Recreate the parsed HGVS fields from the compact record (the browser does the same with its own parser)."""
    c, p = parse_c(rec.get("hgvs_c")), parse_p(rec.get("hgvs_p"))
    rec.update({
        "c_kind": c.kind if c else None, "c_region": c.start.region if c else None, "c_pos": c.start.base if c else None,
        "c_end": c.end.base if c else None, "c_offset": c.start.offset if c else 0, "c_end_offset": c.end.offset if c else 0,
        "splice_distance": c.splice_distance if c else 0, "length_change": c.length_change if c else None,
        "p_kind": p.kind if p else None, "p_ref": p.ref_aa if p else None, "p_pos": p.pos if p and p.pos is not None else None,
        "p_alt": p.alt_aa if p else None, "p_end": p.end_pos if p and p.end_pos is not None else None, "protein_change": p.short if p else "",
    })
    return rec


def export_variants(sizes: dict) -> pd.DataFrame:
    v = pd.read_parquet(KB / "variants.parquet")
    blind = set(pd.read_parquet(EVAL / "blind_variants.parquet")["variation_id"])
    dev = set(pd.read_parquet(EVAL / "dev_variants.parquet")["variation_id"])
    pool = set(pd.read_parquet(EVAL / "pool_ids.parquet")["variation_id"])
    v["set"] = np.where(v["variation_id"].isin(blind), "blind", np.where(v["variation_id"].isin(dev), "dev",
                        np.where(v["variation_id"].isin(pool), "pool", "other")))
    show = v[v["gene"].isin(SHOWCASE) & ((v["stars"] >= 1) | (v["conflicted"] == 1)) & ~v["set"].isin(["blind", "dev"])]
    # per showcase gene: every conflicting variant up to the cap, then a class balanced sample of the rest
    parts = []
    for g, grp in show.groupby("gene"):
        conf = grp[grp["conflicted"] == 1].head(PER_GENE // 3)
        rest = grp.drop(conf.index)
        k = max(0, PER_GENE - len(conf))
        parts.append(conf)
        if len(rest) > k:
            rest = rest.groupby("clinvar_class", dropna=False, group_keys=False).apply(lambda x: x.sample(min(len(x), k // 5 + 1), random_state=7))
        parts.append(rest.head(k))
    sel = pd.concat([v[v["set"].isin(["blind", "dev"])], *parts]).copy()
    sel["trait"] = sel["traits"].map(lambda t: list(t)[0][:80] if len(t) else None)
    rows = [[_clean(x) if not isinstance(x, (list, np.ndarray)) else [_clean(y) for y in x] for x in r] for r in sel[VAR_COLS].itertuples(index=False)]
    half = len(rows) // 2
    sizes["variants_1.json"] = _write("variants_1.json", {"cols": VAR_COLS, "rows": rows[:half]})
    sizes["variants_2.json"] = _write("variants_2.json", {"cols": VAR_COLS, "rows": rows[half:]})
    # submissions for conflicting and expert panel variants (conflict explorer)
    conf = sel[(sel["conflicted"] == 1) | (sel["stars"] >= 3)]
    subs = {r.variation_id: [[s["submitter"][:40], s["call"], s["date"]] for s in r.submissions[:8]] for r in conf.itertuples()}
    sizes["submissions.json"] = _write("submissions.json", subs)
    return sel


def export_tables(sizes: dict, keep_ids: set) -> None:
    T = tables_mod.load()
    v = pd.read_parquet(KB / "variants.parquet", columns=["variation_id", "hgvs_c"])
    cshort = dict(zip(v["variation_id"], v["hgvs_c"].fillna("").str.split(":").str[-1]))
    cls_code = {c: i for i, c in enumerate(CLASSES)}
    missense: dict[str, list] = {}
    for (gene, pos), lst in T.by_residue.items():
        for pc, hc, cls, vid in lst:
            missense.setdefault(gene, []).append([pos, pc, cls_code[cls], vid, cshort.get(vid, "")])
    splice: dict[str, list] = {}
    for (gene, cpos, off), lst in T.by_splice_site.items():
        for hc, cls, vid in lst:
            splice.setdefault(gene, []).append([cpos, off, cshort.get(vid, ""), cls_code[cls], vid])
    index = {vid: [g, grp, cls_code.get(cls, 2), pos] for vid, (g, grp, cls, pos) in T.index.items() if vid in keep_ids}
    out = {"missense": missense, "splice": splice, "trunc": {g: a.tolist() for g, a in T.plp_truncating.items()},
           "inframe_blb": {g: a.tolist() for g, a in T.blb_inframe.items()}, "gene_stats": T.gene_stats}
    sizes["tables.json"] = _write("tables.json", out)
    sizes["pool_index.json"] = _write("pool_index.json", index)


def export_genes(sizes: dict) -> None:
    g = pd.read_parquet(KB / "genes.parquet")
    recs = {}
    for r in g.to_dict("records"):
        recs[r["gene"]] = {
            "pli": _clean(r["pLI"]), "lt": _clean(r["loftool"]), "hi": bool(r["haploinsufficient"]), "ess": bool(r["mouse_essential"]),
            "dom": bool(r["dominant"]), "rec": bool(r["recessive"]), "xl": bool(r["x_linked"]), "adult": bool(r["adult_onset"]),
            "dis": list(r["diseases"]), "inh": list(r["inheritance"]), "plen": _clean(r["protein_len"]), "chrom": r["chrom"],
            "cv": {k: int(r[f"clinvar_{k}"]) for k in ("pathogenic", "likely_pathogenic", "uncertain_significance", "likely_benign", "benign")},
            "total": int(r["clinvar_total"]), "mech": r["clinvar_mechanism"], "traits": list(r["top_traits"])[:5],
            "onset": list(r["orphanet_onset"])[:4], "prev": list(r["orphanet_prevalence"])[:3], "iv": [bool(r["intervar_lof_gene"]), bool(r["intervar_pp2_gene"]), bool(r["intervar_bp1_gene"])],
        }
    sizes["genes.json"] = _write("genes.json", recs)
    txs = tx_mod.load()
    tx = {k: {"strand": t.strand, "chrom": t.chrom, "cds": t.cds_len, "agree": t.agreement, "n": t.n_anchors,
              "exons": [[e["g_start"], e["g_end"], e["lc_start"], e["lc_end"], int(e.get("approx", False))] for e in t.exons]}
          for k, t in txs.items() if t.anchored}
    sizes["transcripts.json"] = _write("transcripts.json", tx)


def export_models(sizes: dict) -> None:
    for name in ("missense_model", "splice_model", "meta_model", "reranker"):
        b = lgb.Booster(model_file=str(MODELS / f"{name}.txt"))
        d = b.dump_model()
        trees = []
        for t in d["tree_info"]:
            trees.append(_compact_tree(t["tree_structure"]))
        sizes[f"{name}.json"] = _write(f"{name}.json", {"num_class": d.get("num_class", 1), "num_tree_per_iteration": d.get("num_tree_per_iteration", 1),
                                                         "objective": d.get("objective", ""), "features": d["feature_names"], "trees": trees})
    for name in ("missense_calibration", "splice_calibration"):
        c = json.loads((MODELS / f"{name}.json").read_text())
        sizes[f"{name}.json"] = _write(f"{name}.json", c)
    with open(MODELS / "verifier.pkl", "rb") as fh:
        lr = pickle.load(fh)
    from allelion.copilot.verifier import FEATURES as VF
    sizes["verifier.json"] = _write("verifier.json", {"features": VF, "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])})


def _compact_tree(node):
    """Flatten a LightGBM tree into arrays: [feature, threshold, default_left, missing_type, left, right] or leaf value."""
    nodes = []

    def walk(n):
        i = len(nodes)
        nodes.append(None)
        if "leaf_value" in n:
            nodes[i] = [n["leaf_value"]]
            return i
        mt = {"None": 0, "Zero": 1, "NaN": 2}[n.get("missing_type", "None")]
        nodes[i] = [n["split_feature"], n["threshold"], int(n["default_left"]), mt, 0, 0]
        nodes[i][4] = walk(n["left_child"])
        nodes[i][5] = walk(n["right_child"])
        return i
    walk(node)
    return nodes


def export_retrieval(sizes: dict) -> None:
    P = pd.read_parquet(MODELS / "passages.parquet")
    recs = [[r.source, r.kind, r.title, r.text, r.level, list(r.genes), list(r.codes), list(r.refs)] for r in P.itertuples()]
    third = len(recs) // 3
    for i, part in enumerate((recs[:third], recs[third:2 * third], recs[2 * third:]), 1):
        sizes[f"passages_{i}.json"] = _write(f"passages_{i}.json", part)
    vocab = json.loads((MODELS / "sif_vocab.json").read_text())
    words = sorted(vocab, key=vocab.get)
    V = np.load(MODELS / "sif_vectors.npy").astype(np.float16)
    with gzip.open(BUNDLE / "sif_vectors.f16.gz", "wb", compresslevel=9) as fh:
        fh.write(V.tobytes())
    sizes["sif_vectors.f16"] = (BUNDLE / "sif_vectors.f16.gz").stat().st_size
    sizes["sif.json"] = _write("sif.json", {"words": words, "weights": np.load(MODELS / "sif_weights.npy").round(6).tolist(),
                                             "pc": np.load(MODELS / "sif_pc.npy").tolist(), "dim": int(V.shape[1]),
                                             "Wq": np.load(MODELS / "dual_Wq.npy").round(6).tolist(), "Wp": np.load(MODELS / "dual_Wp.npy").round(6).tolist()})


def export_phenotypes(sizes: dict) -> None:
    t = pd.read_parquet(KB / "hpo_terms.parquet")
    sizes["hpo.json"] = _write("hpo.json", {r.hpo_id: [r.name, list(r.parents), round(float(r.ic), 4)] for r in t.itertuples()})
    d = pd.read_parquet(KB / "diseases.parquet")
    sizes["diseases.json"] = _write("diseases.json", {r.disease_id: [r.name, list(r.terms), list(r.inheritance)] for r in d.itertuples()})


def export_literature(sizes: dict) -> None:
    docs = pd.read_parquet(KB / "literature_docs.parquet")
    links = pd.read_parquet(KB / "literature_links.parquet")
    from allelion.models.extractor import extract, load_crf
    crf = load_crf()
    mentions = {}
    for r in docs.itertuples():
        text = f"{r.title} {r.text}"
        mentions[r.doc_id] = [[m["start"], m["end"], m.get("kind", ""), m.get("norm", "")] for m in extract(text, crf)]
    sizes["literature.json"] = _write("literature.json", {
        "docs": [[r.doc_id, r.pmid, r.corpus, r.split] for r in docs.itertuples()],
        "links": [[r.doc_id, r.gene, r.variation_id, r.mention, r.norm, r.link] for r in links.itertuples()],
        "mentions": mentions})


def export_meta(sizes: dict) -> None:
    res = Resources()
    meta = {"rules": res.rules, "classes": CLASSES, "order": ORDER, "titles": TITLES, "default_strength": DEFAULT_STRENGTH,
            "ba1_exceptions": [list(x) for x in BA1_EXCEPTIONS], "missense_features": MISSENSE_FEATURES, "splice_features": SPLICE_FEATURES,
            "blosum62": {f"{a}{b}": v for (a, b), v in BLOSUM62.items()}, "grantham_props": GRANTHAM_PROPS, "kyte_doolittle": KYTE_DOOLITTLE,
            "aas": AAS, "showcase_genes": SHOWCASE, "guidance": json.loads((KNOWLEDGE / "guidance_cards.json").read_text())}
    summ = REPORTS / "evaluation_summary.json"
    if summ.exists():
        meta["evaluation"] = json.loads(summ.read_text())
    meta["graph"] = json.loads((KB / "graph_stats.json").read_text())
    sizes["meta.json"] = _write("meta.json", meta)


def build() -> dict:
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir(parents=True)
    sizes: dict = {}
    sel = export_variants(sizes)
    export_tables(sizes, set(sel["variation_id"]))
    export_genes(sizes)
    export_models(sizes)
    export_retrieval(sizes)
    export_phenotypes(sizes)
    export_literature(sizes)
    export_meta(sizes)
    total = sum(sizes.values())
    manifest = {"files": sizes, "total_bytes": total, "variants": len(sel)}
    (BUNDLE / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: round(v / 1e6, 2) for k, v in sizes.items()}, indent=1), f"total {total / 1e6:.1f} MB, variants {len(sel):,}")
    return manifest


if __name__ == "__main__":
    build()
