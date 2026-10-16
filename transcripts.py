"""Exon models anchored to coding coordinates.

The canonical exon models (GENCODE, GRCh37) carry genomic exon boundaries but
not coding coordinates, and the GENCODE canonical transcript is not always the
transcript a laboratory reports on. Every exonic ClinVar substitution supplies
a pair (genomic position, c. position) on the reporting transcript, so each
exon of the model can be anchored independently: within an exon the
difference between transcript position and c. position is constant, and the
modal difference over the anchors in that exon places the exon in c. space.
Exons of the model that the reporting transcript skips show up as a jump in
that difference equal to their length and are dropped; exons with no anchors
are kept only when the differences on both sides prove they are contiguous.

The result is a list of exons in linear c. coordinates (c.-1 becomes 0, the
5' UTR is negative, the 3' UTR continues past the stop codon) from which the
PVS1 logic reads: the exon holding a position, whether a premature stop
escapes nonsense mediated decay (last exon, or the last 50 bases of the
penultimate exon), the coding length of every exon and whether skipping it
keeps the reading frame, and the share of the protein a truncation removes.
"""
from __future__ import annotations

import bisect
import json
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from allelion.config import KB, RAW

AGREEMENT_MIN = 0.90
ANCHORS_MIN = 5


def linear_c(region: str | None, base: float, cds_len: int | None) -> int | None:
    """Map a c. position to a single linear axis (c.1 is 1, c.-1 is 0, c.*1 is cds_len + 1)."""
    if region is None or base is None or (isinstance(base, float) and np.isnan(base)):
        return None
    base = int(base)
    if region == "cds":
        return base
    if region == "utr5":
        return base + 1
    if region == "utr3":
        return (cds_len + base) if cds_len else None
    return None


@dataclass
class Transcript:
    gene: str
    chrom: str
    strand: str
    exons: list[dict]          # transcription order: g_start, g_end, lc_start, lc_end (1 based inclusive genomic)
    cds_len: int | None
    agreement: float
    n_anchors: int

    @property
    def anchored(self) -> bool:
        return bool(self.exons) and all(e["lc_start"] is not None for e in self.exons)

    def exon_of(self, lc: int) -> int | None:
        starts = [e["lc_start"] for e in self.exons]
        i = bisect.bisect_right(starts, lc) - 1
        if 0 <= i < len(self.exons) and self.exons[i]["lc_start"] <= lc <= self.exons[i]["lc_end"]:
            return i
        return None

    def nearest_exon(self, lc: int) -> int:
        d = [min(abs(lc - e["lc_start"]), abs(lc - e["lc_end"])) for e in self.exons]
        return int(np.argmin(d))

    def exon_frames(self) -> list[dict]:
        out = []
        for i, e in enumerate(self.exons):
            lo, hi = max(e["lc_start"], 1), min(e["lc_end"], self.cds_len or e["lc_end"])
            coding = max(0, hi - lo + 1)
            out.append({"exon": i + 1, "lc_start": e["lc_start"], "lc_end": e["lc_end"], "length": e["lc_end"] - e["lc_start"] + 1,
                        "coding": coding, "in_frame_skip": coding > 0 and coding % 3 == 0})
        return out

    def coding_exon_count(self) -> int:
        return sum(1 for f in self.exon_frames() if f["coding"] > 0)

    def nmd_escape(self, ptc_lc: int) -> tuple[bool, str]:
        """Nonsense mediated decay prediction for a premature stop at linear c. position ptc_lc."""
        i = self.exon_of(ptc_lc)
        n = len(self.exons)
        if n == 0:
            return False, "no exon model"
        if i is None:
            i = self.nearest_exon(ptc_lc)
        if i == n - 1:
            return True, f"the premature stop falls in the last exon (exon {n})"
        if i == n - 2 and self.exons[i]["lc_end"] - ptc_lc < 50:
            return True, f"the premature stop falls within the last 50 bases of the penultimate exon (exon {n - 1})"
        return False, f"the premature stop falls in exon {i + 1} of {n}, so the transcript is predicted to undergo decay"

    def to_dict(self) -> dict:
        return {"gene": self.gene, "chrom": self.chrom, "strand": self.strand, "exons": self.exons, "cds_len": self.cds_len,
                "agreement": self.agreement, "n_anchors": self.n_anchors}


def _load_models() -> dict[str, dict]:
    df = pd.read_csv(RAW / "spliceai" / "grch37.txt", sep="\t")
    df.columns = [c.lstrip("#") for c in df.columns]
    out = {}
    for r in df.itertuples(index=False):
        if r.NAME in out:
            continue
        starts = [int(x) for x in str(r.EXON_START).strip(",").split(",") if x]
        ends = [int(x) for x in str(r.EXON_END).strip(",").split(",") if x]
        ex = [(s + 1, e) for s, e in zip(starts, ends)]  # zero based half open -> one based inclusive
        if r.STRAND == "-":
            ex = ex[::-1]
        out[r.NAME] = {"chrom": str(r.CHROM), "strand": r.STRAND, "exons": ex}
    return out


def _t_of(g: int, exons: list[tuple[int, int]], strand: str) -> tuple[int, int] | None:
    t = 0
    for i, (s, e) in enumerate(exons):
        if s <= g <= e:
            return i, t + 1 + (g - s if strand == "+" else e - g)
        t += e - s + 1
    return None


def _anchor_gene(gene: str, model: dict, anchors: list[tuple[int, int]], cds_len_hint: int | None) -> Transcript | None:
    exons, strand = model["exons"], model["strand"]
    per_exon: dict[int, list[int]] = {}
    for g, lc in anchors:
        hit = _t_of(g, exons, strand)
        if hit:
            per_exon.setdefault(hit[0], []).append(hit[1] - lc)
    total = sum(len(v) for v in per_exon.values())
    if total < ANCHORS_MIN:
        return None
    deltas, agree = {}, 0
    for i, ds in per_exon.items():
        mode, cnt = Counter(ds).most_common(1)[0]
        deltas[i] = mode
        agree += cnt
    agreement = agree / total
    fail = Transcript(gene, model["chrom"], strand, [], None, round(agreement, 4), total)
    if agreement < AGREEMENT_MIN:
        return fail
    t, t_ranges = 0, []
    for s, e in exons:
        t_ranges.append((t + 1, t + e - s + 1))
        t += e - s + 1
    idx = sorted(deltas)
    placed = {i: [t_ranges[i][0] - deltas[i], t_ranges[i][1] - deltas[i], False] for i in idx}
    out_idx = []
    # leading exons before the first anchored exon: contiguous on the 5' side
    first, last = idx[0], idx[-1]
    for i in range(first - 1, -1, -1):
        nxt = placed[i + 1][0] if (i + 1) in placed else None
        ln = t_ranges[i][1] - t_ranges[i][0] + 1
        placed[i] = [nxt - ln, nxt - 1, False]
    for i in range(last + 1, len(exons)):
        prv = placed[i - 1][1]
        ln = t_ranges[i][1] - t_ranges[i][0] + 1
        placed[i] = [prv + 1, prv + ln, False]
    # between anchored exons: reconcile gaps (skipped exons, alternative splice sites)
    for a, b in zip(idx, idx[1:]):
        gap = placed[b][0] - placed[a][1] - 1
        between = list(range(a + 1, b))
        blen = sum(t_ranges[j][1] - t_ranges[j][0] + 1 for j in between)
        if gap == 0:
            continue                                   # exons in between are absent from the reporting transcript
        if gap > 0 and between and blen == gap:
            cur = placed[a][1]
            for j in between:
                ln = t_ranges[j][1] - t_ranges[j][0] + 1
                placed[j] = [cur + 1, cur + ln, False]
                cur += ln
            continue
        if abs(gap) <= 300 and not between:
            placed[a][1] = placed[b][0] - 1           # alternative splice site: trust the downstream anchors
            placed[a][2] = True
            continue
        if gap > 0 and between:
            cur = placed[a][1]                         # partially matching exons: spread the gap over them
            for k, j in enumerate(between):
                ln = gap - (cur - placed[a][1]) if k == len(between) - 1 else min(t_ranges[j][1] - t_ranges[j][0] + 1, gap - (cur - placed[a][1]) - (len(between) - k - 1))
                placed[j] = [cur + 1, cur + ln, True]
                cur += ln
            continue
        return fail
    keep = sorted(i for i in placed if i in idx or i < first or i > last or any(i in range(a + 1, b) and placed[i][0] > placed[a][1] for a, b in zip(idx, idx[1:])))
    out = []
    for i in keep:
        lo, hi, approx = placed[i]
        if hi < lo:
            continue
        s, e = exons[i]
        out.append({"g_start": s, "g_end": e, "lc_start": int(lo), "lc_end": int(hi), "approx": bool(approx)})
    out.sort(key=lambda x: x["lc_start"])
    for x, y in zip(out, out[1:]):
        if y["lc_start"] != x["lc_end"] + 1:
            return fail
    return Transcript(gene, model["chrom"], strand, out, cds_len_hint, round(agreement, 4), total)


def build() -> dict:
    v = pd.read_parquet(KB / "variants.parquet", columns=["gene", "pos", "c_kind", "c_offset", "c_pos", "c_region", "p_pos", "p_kind"])
    models = _load_models()
    sub = v[(v["c_kind"] == "sub") & (v["c_offset"] == 0) & (v["c_region"] == "cds") & v["c_pos"].notna()]
    # coding length from stop loss positions or the largest protein position (+ stop codon)
    plen = v.groupby("gene")["p_pos"].max()
    rows, stats = [], Counter()
    for gene, grp in sub.groupby("gene"):
        if gene not in models:
            stats["no_model"] += 1
            continue
        hint = int(plen[gene]) * 3 + 3 if gene in plen.index and not np.isnan(plen[gene]) else None
        tx = _anchor_gene(gene, models[gene], list(zip(grp["pos"].astype(int), grp["c_pos"].astype(int))), hint)
        if tx is None:
            stats["too_few_anchors"] += 1
            continue
        if tx.anchored:
            # make sure the coding end lies inside the model; otherwise drop the length hint
            if tx.cds_len and tx.exon_of(tx.cds_len) is None:
                tx.cds_len = None
            stats["anchored"] += 1
        else:
            stats["structure_mismatch"] += 1
        rows.append(tx.to_dict())
    (KB / "transcripts.json").write_text(json.dumps(rows))
    stats = dict(stats)
    stats["median_agreement"] = float(np.median([r["agreement"] for r in rows]))
    print(stats)
    return stats


def load() -> dict[str, Transcript]:
    rows = json.loads((KB / "transcripts.json").read_text())
    return {r["gene"]: Transcript(r["gene"], r["chrom"], r["strand"], r["exons"], r["cds_len"], r["agreement"], r["n_anchors"]) for r in rows}


if __name__ == "__main__":
    build()
