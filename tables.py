"""Evidence tables derived from the evidence pool only.

Every table here is built from variants in data/eval/pool_ids.parquet (assertion
criteria, five tier class, no conflict, not in the blind or development sets).
A variant being classified is always removed from the tables before its own
criteria are evaluated (see EvidenceTables.without), so no variant ever
contributes evidence to itself even when it is in the pool.

Tables
  allelic      pathogenic missense by gene and residue, and by exact amino acid change (PS1, PM5)
  splice_sites pathogenic variants by gene and intronic position (same splice position, PS1 for splicing)
  density      sorted residue positions of pathogenic and benign missense per gene (PM1 hotspot)
  truncating   residue positions of pathogenic truncating variants per gene (PVS1 critical region)
  gene_stats   consequence by class counts per gene (PP2, BP1, PVS1 mechanism, BP3 and PM4 context)
"""
from __future__ import annotations

import bisect
import pickle
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from allelion.config import EVAL, KB, MODELS

PLP = {"Pathogenic", "Likely pathogenic"}
BLB = {"Benign", "Likely benign"}
TRUNC = {"nonsense", "frameshift", "splice_donor", "splice_acceptor", "start_loss"}


def consequence_group(c: str) -> str:
    if c in TRUNC:
        return "truncating"
    if c == "missense":
        return "missense"
    if c in ("inframe_indel", "stop_loss"):
        return "inframe"
    if c == "synonymous":
        return "synonymous"
    if c == "intron":
        return "intronic"
    return "other"


@dataclass
class EvidenceTables:
    # (gene, pos) -> list of (protein_change, hgvs_c, class, variation_id)
    by_residue: dict = field(default_factory=lambda: defaultdict(list))
    # (gene, protein_change) -> list of (hgvs_c, class, variation_id)
    by_change: dict = field(default_factory=lambda: defaultdict(list))
    # (gene, c_pos, offset) -> list of (hgvs_c, class, variation_id)
    by_splice_site: dict = field(default_factory=lambda: defaultdict(list))
    # gene -> sorted residue positions
    plp_missense: dict = field(default_factory=dict)
    blb_missense: dict = field(default_factory=dict)
    plp_truncating: dict = field(default_factory=dict)
    blb_inframe: dict = field(default_factory=dict)
    gene_stats: dict = field(default_factory=dict)
    # variation_id -> (gene, group, class, pos) so that a variant can be removed from its own evidence
    index: dict = field(default_factory=dict)

    # ---- queries (each takes the id of the variant being classified so that it is excluded) ----
    def same_change(self, gene: str, change: str, exclude: str | None, hgvs_c: str | None) -> list[tuple]:
        short = hgvs_c.split(":")[-1] if isinstance(hgvs_c, str) else ""
        return [r for r in self.by_change.get((gene, change), []) if r[2] != exclude and (r[0].split(":")[-1] if isinstance(r[0], str) else "") != short]

    def same_residue(self, gene: str, pos: int, exclude: str | None) -> list[tuple]:
        return [r for r in self.by_residue.get((gene, pos), []) if r[3] != exclude]

    def same_splice_site(self, gene: str, c_pos: int, offset: int, exclude: str | None, hgvs_c: str | None) -> list[tuple]:
        short = hgvs_c.split(":")[-1] if isinstance(hgvs_c, str) else ""
        return [r for r in self.by_splice_site.get((gene, c_pos, offset), []) if r[2] != exclude and (r[0].split(":")[-1] if isinstance(r[0], str) else "") != short]

    @staticmethod
    def _window(arr, lo, hi) -> int:
        if arr is None or len(arr) == 0:
            return 0
        return bisect.bisect_right(arr, hi) - bisect.bisect_left(arr, lo)

    def density(self, gene: str, pos: int, window: int, exclude: str | None) -> tuple[int, int]:
        p = self._window(self.plp_missense.get(gene), pos - window, pos + window)
        b = self._window(self.blb_missense.get(gene), pos - window, pos + window)
        own = self.index.get(exclude)
        if own and own[1] == "missense" and own[3] is not None and abs(own[3] - pos) <= window:
            if own[2] in PLP:
                p -= 1
            elif own[2] in BLB:
                b -= 1
        return max(p, 0), max(b, 0)

    def truncating_downstream(self, gene: str, pos: int, exclude: str | None) -> int:
        arr = self.plp_truncating.get(gene)
        if arr is None:
            return 0
        n = len(arr) - bisect.bisect_right(arr, pos)
        own = self.index.get(exclude)
        if own and own[1] == "truncating" and own[2] in PLP and own[3] is not None and own[3] > pos:
            n -= 1
        return max(n, 0)

    def benign_inframe_near(self, gene: str, pos: int, window: int, exclude: str | None) -> int:
        n = self._window(self.blb_inframe.get(gene), pos - window, pos + window)
        own = self.index.get(exclude)
        if own and own[1] == "inframe" and own[2] in BLB and own[3] is not None and abs(own[3] - pos) <= window:
            n -= 1
        return max(n, 0)

    def stats(self, gene: str, exclude: str | None = None) -> dict:
        s = dict(self.gene_stats.get(gene, {}))
        own = self.index.get(exclude)
        if own and own[0] == gene:
            key = f"{own[1]}_{'plp' if own[2] in PLP else 'blb' if own[2] in BLB else 'vus'}"
            if s.get(key, 0) > 0:
                s[key] -= 1
        return s


def lof_mechanism(stats: dict, g: dict, rules: dict) -> tuple[bool, str]:
    """Is loss of function an established disease mechanism for this gene? Shared by the engine and the gene cards."""
    n = stats.get("truncating_plp", 0)
    mp = stats.get("missense_plp", 0)
    if g.get("haploinsufficient"):
        return True, "the gene has curated evidence of haploinsufficiency"
    share = n / (n + mp) if (n + mp) else 0.0
    if mp >= rules.get("pvs1_missense_dominant_min", 20) and share < rules.get("pvs1_missense_dominant_frac", 0.10):
        return False, (f"pathogenic variation in this gene is predominantly missense ({mp} missense against {n} truncating), which points to "
                       f"gain of function or dominant negative disease rather than loss of function")
    if n >= rules.get("pvs1_min_trunc_plp", 3):
        return True, f"{n} truncating variants in this gene are classified pathogenic or likely pathogenic in the evidence pool"
    if g.get("recessive") and n >= 1:
        return True, f"loss of function is the mechanism for the recessive disorder ({n} pathogenic truncating variant in the pool)"
    return False, f"loss of function is not established for this gene ({n} pathogenic truncating variants in the pool, no curated haploinsufficiency)"


def build() -> EvidenceTables:
    v = pd.read_parquet(KB / "variants.parquet", columns=["variation_id", "gene", "consequence", "clinvar_class", "protein_change",
                                                        "p_pos", "hgvs_c", "c_pos", "c_offset", "c_region"])
    pool_ids = set(pd.read_parquet(EVAL / "pool_ids.parquet")["variation_id"])
    pool = v[v["variation_id"].isin(pool_ids)].copy()
    pool["group"] = pool["consequence"].map(consequence_group)
    t = EvidenceTables()
    for r in pool.itertuples(index=False):
        pos = None if np.isnan(r.p_pos) else int(r.p_pos)
        t.index[r.variation_id] = (r.gene, r.group, r.clinvar_class, pos)
        if r.group == "missense" and pos is not None:
            t.by_residue[(r.gene, pos)].append((r.protein_change, r.hgvs_c, r.clinvar_class, r.variation_id))
            if r.clinvar_class in PLP:
                t.by_change[(r.gene, r.protein_change)].append((r.hgvs_c, r.clinvar_class, r.variation_id))
        if r.clinvar_class in PLP and r.c_offset != 0 and not np.isnan(r.c_pos):
            t.by_splice_site[(r.gene, int(r.c_pos), int(r.c_offset))].append((r.hgvs_c, r.clinvar_class, r.variation_id))
    t.by_residue = dict(t.by_residue)
    t.by_change = dict(t.by_change)
    t.by_splice_site = dict(t.by_splice_site)
    mis = pool[(pool["group"] == "missense") & pool["p_pos"].notna()]
    t.plp_missense = {g: np.sort(x["p_pos"].astype(int).values) for g, x in mis[mis["clinvar_class"].isin(PLP)].groupby("gene")}
    t.blb_missense = {g: np.sort(x["p_pos"].astype(int).values) for g, x in mis[mis["clinvar_class"].isin(BLB)].groupby("gene")}
    tr = pool[(pool["group"] == "truncating") & pool["p_pos"].notna() & pool["clinvar_class"].isin(PLP)]
    t.plp_truncating = {g: np.sort(x["p_pos"].astype(int).values) for g, x in tr.groupby("gene")}
    inf = pool[(pool["group"] == "inframe") & pool["p_pos"].notna() & pool["clinvar_class"].isin(BLB)]
    t.blb_inframe = {g: np.sort(x["p_pos"].astype(int).values) for g, x in inf.groupby("gene")}
    pool["bucket"] = pool["clinvar_class"].map(lambda c: "plp" if c in PLP else "blb" if c in BLB else "vus")
    counts = pool.groupby(["gene", "group", "bucket"]).size()
    gs: dict = defaultdict(dict)
    for (g, grp, b), n in counts.items():
        gs[g][f"{grp}_{b}"] = int(n)
    t.gene_stats = dict(gs)
    with open(MODELS / "evidence_tables.pkl", "wb") as fh:
        pickle.dump(t, fh)
    print(f"evidence tables from {len(pool):,} pool variants: {len(t.by_residue):,} residues, {len(t.by_change):,} pathogenic changes, "
          f"{len(t.by_splice_site):,} pathogenic splice positions, {len(t.gene_stats):,} genes")
    return t


def load() -> EvidenceTables:
    with open(MODELS / "evidence_tables.pkl", "rb") as fh:
        return pickle.load(fh)


if __name__ == "__main__":
    from allelion.acmg import tables as _self  # pickle the class under its module path, not __main__

    _self.build()
