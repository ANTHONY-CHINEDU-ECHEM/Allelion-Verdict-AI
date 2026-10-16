"""The ACMG/AMP criteria engine.

Evaluates every criterion of Richards and colleagues (2015), as refined by the
ClinGen Sequence Variant Interpretation recommendations, for one variant, and
returns each criterion with its status (met, not met, not evaluated), its
strength, a plain language rationale and the evidence items it rests on.
Criteria that need case level data (de novo status, segregation, phase,
functional assays, case control counts, phenotype) are evaluated only when that
data is supplied; otherwise they are listed as not evaluated so the report
shows what a curator still has to look at.

PP5 and BP6 (reputable source) are never used, following the 2018 SVI
recommendation to retire them; the variant's own ClinVar assertion is never
an input.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from functools import cached_property

import lightgbm as lgb
import numpy as np
import pandas as pd

from allelion.acmg import tables as tables_mod
from allelion.acmg.combine import class_2015, class_from_points, points_of, posterior
from allelion.config import KB, MODELS
from allelion.kb import transcripts as tx_mod
from allelion.kb.hgvs import AA3, LOF_CONSEQUENCES, parse_c, parse_p
from allelion.models.calibrate import strength_for
from allelion.models.features import gene_frame, missense_features, splice_features

ORDER = ["PVS1", "PS1", "PS2", "PS3", "PS4", "PM1", "PM2", "PM3", "PM4", "PM5", "PM6", "PP1", "PP2", "PP3", "PP4",
         "BA1", "BS1", "BS2", "BS3", "BS4", "BP1", "BP2", "BP3", "BP4", "BP5", "BP7"]
DIRECTION = {c: ("benign" if c.startswith("B") else "pathogenic") for c in ORDER}
DEFAULT_STRENGTH = {"PVS1": "VeryStrong", "PS1": "Strong", "PS2": "Strong", "PS3": "Strong", "PS4": "Strong",
                    "PM1": "Moderate", "PM2": "Moderate", "PM3": "Moderate", "PM4": "Moderate", "PM5": "Moderate", "PM6": "Moderate",
                    "PP1": "Supporting", "PP2": "Supporting", "PP3": "Supporting", "PP4": "Supporting",
                    "BA1": "StandAlone", "BS1": "Strong", "BS2": "Strong", "BS3": "Strong", "BS4": "Strong",
                    "BP1": "Supporting", "BP2": "Supporting", "BP3": "Supporting", "BP4": "Supporting", "BP5": "Supporting", "BP7": "Supporting"}
TITLES = {
    "PVS1": "Null variant in a gene where loss of function causes disease",
    "PS1": "Same amino acid change (or same splice position) as an established pathogenic variant",
    "PS2": "Confirmed de novo in a patient with the disease",
    "PS3": "Well established functional study shows a damaging effect",
    "PS4": "Prevalence in affected individuals significantly higher than in controls",
    "PM1": "Located in a mutational hot spot or critical domain without benign variation",
    "PM2": "Absent from (or extremely rare in) population databases",
    "PM3": "For recessive disorders, detected in trans with a pathogenic variant",
    "PM4": "Protein length change (in frame indel or stop loss) outside a repeat region",
    "PM5": "Novel missense change at a residue where a different pathogenic missense change is established",
    "PM6": "Assumed de novo, without confirmation of parentage",
    "PP1": "Cosegregation with disease in multiple affected family members",
    "PP2": "Missense in a gene with a low rate of benign missense where missense is a common disease mechanism",
    "PP3": "Calibrated computational evidence supports a deleterious effect",
    "PP4": "Patient phenotype highly specific for a disease with a single genetic cause",
    "BA1": "Allele frequency above 5% in a large population database",
    "BS1": "Allele frequency greater than expected for the disorder",
    "BS2": "Observed in healthy adults in a state that should cause fully penetrant early onset disease",
    "BS3": "Well established functional study shows no damaging effect",
    "BS4": "Lack of segregation in affected members of a family",
    "BP1": "Missense in a gene where only truncating variants cause disease",
    "BP2": "Observed in trans with a pathogenic variant in a dominant gene, or in cis with a pathogenic variant",
    "BP3": "In frame indel in a repetitive region without known function",
    "BP4": "Calibrated computational evidence suggests no impact",
    "BP5": "Variant found in a case with an alternate molecular basis for disease",
    "BP7": "Synonymous (or deep intronic) variant with no predicted splice impact",
}

DEFAULT_RULES = {
    "ba1_af": 0.05,
    "bs1_af_dominant": 0.001,
    "bs1_af_recessive": 0.01,
    "bs1_af_unknown": 0.005,
    "pm2_af_recessive": 0.0001,
    "bs2_hom_min": 3,
    "pm1_window": 10,
    "pm1_min_plp": 5,
    "pm1_max_blb": 0,
    "pp2_min_plp": 10,
    "pp2_max_benign_frac": 0.10,
    "bp1_min_trunc": 10,
    "bp1_max_missense_frac": 0.05,
    "pvs1_min_trunc_plp": 3,
    "pvs1_missense_dominant_min": 20,
    "pvs1_missense_dominant_frac": 0.10,
    "pvs1_removed_frac_strong": 0.10,
    "bp3_window": 10,
    "pp3_pm1_cap_points": 4,
}

# Variants excluded from BA1 by the ClinGen SVI working group (partial list: well known common pathogenic alleles).
BA1_EXCEPTIONS = {("HFE", "C282Y"), ("HFE", "H63D"), ("GJB2", "V37I"), ("BTD", "D444H")}

STRENGTH_ORDER = ["Supporting", "Moderate", "Strong", "VeryStrong"]


def _down(strength: str, steps: int = 1) -> str:
    return STRENGTH_ORDER[max(0, STRENGTH_ORDER.index(strength) - steps)]


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _fmt_af(x) -> str:
    if _nan(x):
        return "not observed"
    if x == 0:
        return "0"
    if x >= 0.01:
        return f"{x:.3f}"
    return f"{x:.2e}"


@dataclass
class Criterion:
    code: str
    status: str = "not_met"              # met, not_met, not_evaluated
    strength: str = ""
    rationale: str = ""
    evidence: list = field(default_factory=list)

    @property
    def direction(self) -> str:
        return DIRECTION[self.code]

    def to_dict(self) -> dict:
        return {"code": self.code, "title": TITLES[self.code], "direction": self.direction, "status": self.status,
                "strength": self.strength or DEFAULT_STRENGTH[self.code], "default_strength": DEFAULT_STRENGTH[self.code],
                "label": self.label, "rationale": self.rationale, "evidence": self.evidence}

    @property
    def label(self) -> str:
        s = self.strength or DEFAULT_STRENGTH[self.code]
        if s == DEFAULT_STRENGTH[self.code]:
            return self.code
        return f"{self.code}_{s}"


class Resources:
    """Everything the engine needs, loaded once."""

    def __init__(self, rules: dict | None = None):
        self.rules = dict(DEFAULT_RULES)
        rp = MODELS / "rules.json"
        if rp.exists():
            self.rules.update(json.loads(rp.read_text()).get("rules", {}))
        if rules:
            self.rules.update(rules)

    @cached_property
    def genes(self) -> pd.DataFrame:
        return pd.read_parquet(KB / "genes.parquet").set_index("gene")

    @cached_property
    def gene_features(self) -> pd.DataFrame:
        return gene_frame(self.genes.reset_index())

    @cached_property
    def transcripts(self) -> dict:
        return tx_mod.load()

    @cached_property
    def tables(self):
        return tables_mod.load()

    @cached_property
    def missense_model(self):
        return lgb.Booster(model_file=str(MODELS / "missense_model.txt"))

    @cached_property
    def splice_model(self):
        return lgb.Booster(model_file=str(MODELS / "splice_model.txt"))

    @cached_property
    def missense_cal(self) -> dict:
        return json.loads((MODELS / "missense_calibration.json").read_text())

    @cached_property
    def splice_cal(self) -> dict:
        return json.loads((MODELS / "splice_calibration.json").read_text())

    def gene(self, symbol: str) -> dict:
        if symbol in self.genes.index:
            return self.genes.loc[symbol].to_dict()
        return {}


# ------------------------------------------------------------------------------------------------
# variant preparation
# ------------------------------------------------------------------------------------------------

def consequence_from_hgvs(c, p) -> str:
    if p is not None and p.kind in ("missense", "nonsense", "frameshift", "synonymous", "start_loss", "stop_loss"):
        return p.kind
    if p is not None and p.kind in ("inframe_del", "inframe_ins", "inframe_dup", "delins"):
        return "inframe_indel"
    if c is None:
        return "other"
    if c.start.offset:
        if c.splice_distance in (1, 2):
            return "splice_donor" if c.start.offset > 0 or c.end.offset > 0 else "splice_acceptor"
        return "intron"
    if c.start.region == "utr5":
        return "utr5"
    if c.start.region == "utr3":
        return "utr3"
    if c.kind in ("del", "ins", "dup", "delins"):
        return "frameshift" if c.frameshift else "inframe_indel"
    return "other"


def prepare(gene: str, hgvs_c: str, hgvs_p: str | None = None, population: dict | None = None, **extra) -> dict:
    """Build an engine input for a variant described by HGVS (used for variants that are not in the knowledge base)."""
    c = parse_c(hgvs_c)
    p = parse_p(hgvs_p) if hgvs_p else None
    if c is None:
        raise ValueError(f"could not parse coding HGVS {hgvs_c!r}")
    v = {
        "variation_id": extra.get("variation_id"), "gene": gene, "hgvs_c": hgvs_c, "hgvs_p": hgvs_p or "",
        "consequence": consequence_from_hgvs(c, p), "c_kind": c.kind, "c_region": c.start.region, "c_pos": c.start.base,
        "c_end": c.end.base, "c_offset": c.start.offset, "c_end_offset": c.end.offset, "splice_distance": c.splice_distance,
        "length_change": c.length_change, "p_kind": p.kind if p else None, "p_ref": p.ref_aa if p else None,
        "p_pos": p.pos if p and p.pos else np.nan, "p_alt": p.alt_aa if p else None, "p_end": p.end_pos if p and p.end_pos else np.nan,
        "protein_change": p.short if p else "",
    }
    pop = population or {}
    v.update({"gn_AC": pop.get("ac", np.nan), "gn_AN": pop.get("an", np.nan), "gn_grpmax": pop.get("grpmax_af", np.nan),
              "gn_grpmax_2000": pop.get("grpmax_af", np.nan), "gn_faf95": pop.get("faf95", pop.get("grpmax_af", np.nan)),
              "gn_popmax": pop.get("popmax_af", np.nan), "gn_Hom": pop.get("hom", np.nan), "gn_Hemi": pop.get("hemi", np.nan),
              "population_known": bool(pop)})
    v.update(extra)
    return v


def _fs_ptc(hgvs_p: str | None) -> int | None:
    """Length of the new reading frame to the premature stop, from p.Xaa123YyyfsTer17."""
    if not isinstance(hgvs_p, str):
        return None
    m = re.search(r"fs(?:Ter|\*)(\d+)", hgvs_p)
    return int(m.group(1)) if m else None


# ------------------------------------------------------------------------------------------------
# the engine
# ------------------------------------------------------------------------------------------------

class Engine:
    def __init__(self, resources: Resources | None = None):
        self.r = resources or Resources()

    # ---- scores ----
    def missense_score(self, v: dict) -> float | None:
        if "_missense_score" in v:
            return v["_missense_score"]
        if v.get("consequence") != "missense" or _nan(v.get("p_pos")) or v.get("p_ref") not in AA3.values() or v.get("p_alt") not in AA3.values():
            return None
        df = pd.DataFrame([{k: v.get(k) for k in ("gene", "p_ref", "p_alt", "p_pos", "c_pos", "hgvs_c")}])
        X = missense_features(df, self.r.gene_features)
        return float(self.r.missense_model.predict(X)[0])

    def splice_score(self, v: dict) -> float | None:
        if "_splice_score" in v:
            return v["_splice_score"]
        off = v.get("c_offset") or 0
        if v.get("c_region") != "cds" or not (3 <= abs(off) <= 50):
            return None
        df = pd.DataFrame([{k: v.get(k) for k in ("gene", "c_offset", "c_kind", "hgvs_c", "length_change")}])
        X = splice_features(df, self.r.gene_features)
        return float(self.r.splice_model.predict(X)[0])

    def score_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorised predictor scores for many knowledge base variants (adds _missense_score and _splice_score)."""
        df = df.copy()
        df["_missense_score"] = None
        df["_splice_score"] = None
        aa = set(AA3.values())
        m = (df["consequence"] == "missense") & df["p_pos"].notna() & df["p_ref"].isin(aa) & df["p_alt"].isin(aa)
        if m.any():
            X = missense_features(df[m], self.r.gene_features)
            df.loc[m, "_missense_score"] = self.r.missense_model.predict(X)
        s = (df["c_region"] == "cds") & df["c_offset"].abs().between(3, 50)
        if s.any():
            X = splice_features(df[s], self.r.gene_features)
            df.loc[s, "_splice_score"] = self.r.splice_model.predict(X)
        df["_missense_score"] = df["_missense_score"].astype(object).where(df["_missense_score"].notna(), None)
        df["_splice_score"] = df["_splice_score"].astype(object).where(df["_splice_score"].notna(), None)
        return df

    # ---- main entry ----
    def evaluate(self, v: dict, case: dict | None = None, use: set[str] | None = None) -> dict:
        case = case or {}
        g = self.r.gene(v.get("gene") or "")
        ex = v.get("variation_id")
        crit = {c: Criterion(c) for c in ORDER}
        notes: list[str] = []
        scores = {"missense": self.missense_score(v), "splice": self.splice_score(v)}

        self._population(v, g, crit)
        self._pvs1(v, g, crit, ex)
        self._allelic(v, crit, ex)
        self._region(v, crit, ex)
        self._gene_level(v, crit, ex)
        self._computational(v, crit, scores)
        self._case_level(v, g, case, crit)
        self._caps(crit, notes)

        if use is not None:  # ablations: drop criteria not in use
            for c in crit.values():
                if c.code not in use and c.status == "met":
                    c.status, c.rationale = "not_met", c.rationale + " (disabled in this run)"
        out = [c.to_dict() for c in crit.values()]
        ba1 = crit["BA1"].status == "met"
        pts = points_of(out)
        cls = class_from_points(pts, ba1)
        return {
            "variant": {k: (None if _nan(v.get(k)) else v.get(k)) for k in ("variation_id", "gene", "hgvs_c", "hgvs_p", "consequence", "protein_change", "key")},
            "criteria": out,
            "met": [c["label"] for c in out if c["status"] == "met"],
            "points": pts,
            "classification": cls,
            "classification_2015": class_2015(out),
            "posterior": round(posterior(pts), 4) if not ba1 else 0.0,
            "scores": {k: (round(s, 4) if s is not None else None) for k, s in scores.items()},
            "notes": notes,
            "not_evaluated": [c["code"] for c in out if c["status"] == "not_evaluated"],
        }

    # ---- population: BA1, BS1, PM2, BS2 ----
    def _population(self, v, g, crit):
        R = self.r.rules
        af_ba1 = v.get("gn_grpmax_2000")          # continental allele frequency, groups with at least 2000 alleles (BA1)
        if _nan(af_ba1):
            af_ba1 = v.get("gn_grpmax")
        af = v.get("gn_faf95")                    # filtering allele frequency (BS1, PM2)
        if _nan(af):
            af = v.get("gn_grpmax")
        ac = v.get("gn_AC")
        known = v.get("population_known", True)
        dominant, recessive = bool(g.get("dominant")), bool(g.get("recessive"))
        key = (v.get("gene"), v.get("protein_change"))
        an = v.get("gn_AN")
        hom0 = v.get("gn_Hom")
        detail = (f"filtering allele frequency {_fmt_af(af)}; highest continental frequency {_fmt_af(af_ba1)}; allele count "
                  f"{0 if _nan(ac) else int(ac)}{'' if _nan(an) else f' of {int(an)}'}; homozygotes {0 if _nan(hom0) else int(hom0)}")
        ev = [{"kind": "population", "ref": "gnomAD v2 exomes and genomes", "detail": detail}]
        if not known:
            for c in ("BA1", "BS1", "PM2", "BS2"):
                crit[c].status = "not_evaluated"
                crit[c].rationale = "No population frequency was supplied for this variant."
            return
        seen = not _nan(ac) and ac > 0
        if seen and not _nan(af_ba1) and af_ba1 > R["ba1_af"] and key not in BA1_EXCEPTIONS:
            c = crit["BA1"]
            c.status, c.strength = "met", "StandAlone"
            c.rationale = (f"The highest continental allele frequency in gnomAD (groups with at least 2000 alleles) is {_fmt_af(af_ba1)}, "
                           f"above the 5% stand alone benign threshold.")
            c.evidence = ev
            return
        thr = R["bs1_af_dominant"] if dominant else R["bs1_af_recessive"] if recessive else R["bs1_af_unknown"]
        mode = "dominant" if dominant else "recessive" if recessive else "undetermined inheritance"
        if seen and not _nan(af) and af > thr:
            c = crit["BS1"]
            c.status, c.strength = "met", "Strong"
            c.rationale = (f"The filtering allele frequency {_fmt_af(af)} exceeds {thr:g}, the largest frequency compatible with a "
                           f"pathogenic allele for a {mode} disorder under the configured prevalence and penetrance assumptions.")
            c.evidence = ev
        else:
            crit["BS1"].rationale = f"The filtering allele frequency {_fmt_af(af)} does not exceed {thr:g} ({mode})."
        if not seen:
            c = crit["PM2"]
            c.status, c.strength = "met", "Supporting"
            c.rationale = "Absent from gnomAD exomes and genomes. Applied at Supporting strength as recommended by ClinGen SVI (2020)."
            c.evidence = ev
        elif recessive and not dominant and not _nan(af) and af < R["pm2_af_recessive"]:
            c = crit["PM2"]
            c.status, c.strength = "met", "Supporting"
            c.rationale = f"For a recessive gene the frequency {_fmt_af(af)} is below {R['pm2_af_recessive']:g}, rare enough for PM2 at Supporting strength."
            c.evidence = ev
        else:
            crit["PM2"].rationale = f"Observed in gnomAD ({0 if _nan(ac) else int(ac)} alleles), so not absent."
        hom = v.get("gn_Hom")
        hemi = v.get("gn_Hemi")
        adult = bool(g.get("adult_onset"))
        if recessive and not dominant and not adult and not _nan(hom) and hom >= R["bs2_hom_min"] and crit["BA1"].status != "met":
            c = crit["BS2"]
            c.status, c.strength = "met", "Strong"
            c.rationale = (f"Observed homozygous in {int(hom)} gnomAD individuals, which is incompatible with a fully penetrant "
                           f"early onset recessive disorder (gnomAD excludes severe paediatric disease cohorts).")
            c.evidence = ev
        elif bool(g.get("x_linked")) and recessive and not dominant and not adult and not _nan(hemi) and hemi >= R["bs2_hom_min"]:
            c = crit["BS2"]
            c.status, c.strength = "met", "Strong"
            c.rationale = f"Observed hemizygous in {int(hemi)} gnomAD males for an X linked recessive gene."
            c.evidence = ev
        else:
            crit["BS2"].rationale = "No homozygous or hemizygous observations that rule out a penetrant recessive disorder."

    # ---- PVS1 ----
    def _lof_mechanism(self, gene: str, g: dict, ex) -> tuple[bool, str]:
        return tables_mod.lof_mechanism(self.r.tables.stats(gene, ex), g, self.r.rules)

    def _pvs1(self, v, g, crit, ex):
        cons = v.get("consequence")
        c = crit["PVS1"]
        if cons not in LOF_CONSEQUENCES:
            c.rationale = "Not a predicted null variant."
            return
        gene = v.get("gene")
        ok, why = self._lof_mechanism(gene, g, ex)
        ev = [{"kind": "mechanism", "ref": gene, "detail": why}]
        if not ok:
            c.rationale = f"The variant is a predicted null allele, but {why}."
            c.evidence = ev
            return
        tx = self.r.transcripts.get(gene)
        tx = tx if tx is not None and tx.anchored else None
        plen = g.get("protein_len")
        R = self.r.rules
        strength, detail = "VeryStrong", ""
        if cons in ("nonsense", "frameshift"):
            ppos = v.get("p_pos")
            if cons == "frameshift" and not _nan(ppos):
                extra = _fs_ptc(v.get("hgvs_p"))
                ptc = int(ppos) + (extra - 1 if extra else 0)
            else:
                ptc = None if _nan(ppos) else int(ppos)
            if ptc is None and not _nan(v.get("c_pos")):
                ptc = int(v["c_pos"]) // 3 + 1
            if tx is not None and ptc is not None:
                escape, nmd_text = tx.nmd_escape(ptc * 3)
                ev.append({"kind": "transcript", "ref": f"{gene} exon model", "detail": nmd_text})
                if not escape:
                    strength, detail = "VeryStrong", f"{nmd_text[0].upper() + nmd_text[1:]}."
                else:
                    down = self.r.tables.truncating_downstream(gene, ptc, ex)
                    removed = 1 - ptc / plen if plen and not _nan(plen) else 0
                    if down > 0:
                        strength = "Strong"
                        detail = (f"{nmd_text[0].upper() + nmd_text[1:]}, so the transcript escapes decay, but {down} pathogenic truncating "
                                  f"variants lie further downstream, so the removed region is critical.")
                        ev.append({"kind": "pool", "ref": gene, "detail": f"{down} pathogenic truncating variants downstream of residue {ptc}"})
                    elif removed >= R["pvs1_removed_frac_strong"]:
                        strength = "Strong"
                        detail = f"{nmd_text[0].upper() + nmd_text[1:]}; the truncation removes {removed:.0%} of the protein."
                    else:
                        strength = "Moderate"
                        detail = f"{nmd_text[0].upper() + nmd_text[1:]}; the truncation removes only {removed:.0%} of the protein."
            else:
                down = self.r.tables.truncating_downstream(gene, ptc or 0, ex) if ptc else 0
                if down > 0:
                    strength = "VeryStrong"
                    detail = (f"No reliable exon model; {down} pathogenic truncating variants lie downstream of residue {ptc}, "
                              f"so truncation at this position is expected to be damaging.")
                else:
                    strength = "Strong"
                    detail = "No reliable exon model and no downstream pathogenic truncation evidence, so PVS1 is reduced to Strong."
        elif cons in ("splice_donor", "splice_acceptor"):
            if tx is not None and not _nan(v.get("c_pos")):
                base = int(v["c_pos"])
                i = tx.exon_of(base)
                frames = tx.exon_frames()
                if i is not None:
                    f = frames[i]
                    removed = f["coding"] / tx.cds_len if tx.cds_len else 0
                    if not f["in_frame_skip"]:
                        strength = "VeryStrong"
                        detail = (f"Disrupts the canonical {cons.split('_')[1]} site of exon {i + 1}; skipping that exon "
                                  f"({f['coding']} coding bases) shifts the reading frame.")
                    else:
                        lo = (f["lc_start"] - 1) // 3 + 1
                        hi = f["lc_end"] // 3 + 1
                        p_in = self.r.tables.density(gene, (lo + hi) // 2, max(1, (hi - lo) // 2), ex)[0]
                        if p_in > 0 or removed >= R["pvs1_removed_frac_strong"]:
                            strength = "Strong"
                            detail = (f"Disrupts the canonical {cons.split('_')[1]} site of exon {i + 1}; skipping it keeps the frame but removes "
                                      f"{removed:.1%} of the coding sequence{', a region with pathogenic missense variation' if p_in else ''}.")
                        else:
                            strength = "Moderate"
                            detail = f"In frame skip of exon {i + 1} removes only {removed:.1%} of the coding sequence."
                    ev.append({"kind": "transcript", "ref": f"{gene} exon {i + 1}", "detail": f"{f['coding']} coding bases, in frame skip: {f['in_frame_skip']}"})
                else:
                    strength, detail = "Strong", "Canonical splice site variant; the adjacent exon could not be located, so the effect is not predicted precisely."
            else:
                strength, detail = "Strong", "Canonical splice site variant without a reliable exon model; applied at Strong."
        elif cons == "start_loss":
            strength, detail = "Moderate", "Initiation codon variant; an alternative downstream start codon cannot be excluded, so PVS1 is applied at Moderate."
        c.status, c.strength = "met", strength
        c.rationale = f"Predicted null variant and {why}. {detail}".strip()
        c.evidence = ev

    # ---- allelic: PS1, PM5 ----
    def _allelic(self, v, crit, ex):
        gene, cons = v.get("gene"), v.get("consequence")
        T = self.r.tables
        if cons == "missense" and v.get("protein_change"):
            same = T.same_change(gene, v["protein_change"], ex, v.get("hgvs_c"))
            if same:
                strongest = "Strong" if any(r[1] == "Pathogenic" for r in same) else "Moderate"
                c = crit["PS1"]
                c.status, c.strength = "met", strongest
                c.rationale = (f"The same amino acid change {v['protein_change']} is classified {same[0][1].lower()} for a different "
                               f"nucleotide change ({same[0][0]}).")
                c.evidence = [{"kind": "clinvar_pool", "ref": f"ClinVar {r[2]}", "detail": f"{r[0]}: {r[1]}"} for r in same[:5]]
            else:
                crit["PS1"].rationale = "No other nucleotide change producing this amino acid change is classified pathogenic in the pool."
            if crit["PS1"].status != "met" and not _nan(v.get("p_pos")):
                res = [r for r in T.same_residue(gene, int(v["p_pos"]), ex) if r[2] in tables_mod.PLP and r[0] != v["protein_change"]]
                alts = sorted({r[0] for r in res})
                if alts:
                    c = crit["PM5"]
                    c.status = "met"
                    c.strength = "Strong" if len(alts) >= 2 and any(r[2] == "Pathogenic" for r in res) else "Moderate"
                    if c.strength == "Moderate" and not any(r[2] == "Pathogenic" for r in res):
                        c.strength = "Supporting"
                    c.rationale = (f"A different missense change at the same residue is classified pathogenic or likely pathogenic "
                                   f"({', '.join(alts[:4])}).")
                    c.evidence = [{"kind": "clinvar_pool", "ref": f"ClinVar {r[3]}", "detail": f"{r[0]} ({r[1]}): {r[2]}"} for r in res[:5]]
                else:
                    crit["PM5"].rationale = "No other pathogenic missense change at this residue in the pool."
        elif cons in ("splice_donor", "splice_acceptor", "intron") and v.get("c_offset") and not _nan(v.get("c_pos")):
            same = T.same_splice_site(gene, int(v["c_pos"]), int(v["c_offset"]), ex, v.get("hgvs_c"))
            if same:
                c = crit["PS1"]
                c.status = "met"
                c.strength = "Moderate" if abs(int(v["c_offset"])) <= 2 else "Supporting"
                c.rationale = (f"A different change at the same splice position is classified {same[0][1].lower()} ({same[0][0]}); "
                               f"applied for splicing as recommended by ClinGen SVI (2023).")
                c.evidence = [{"kind": "clinvar_pool", "ref": f"ClinVar {r[2]}", "detail": f"{r[0]}: {r[1]}"} for r in same[:5]]
        else:
            crit["PS1"].rationale = crit["PM5"].rationale = "Not a missense or splice region variant."

    # ---- region: PM1, PM4, BP3 ----
    def _region(self, v, crit, ex):
        R, T = self.r.rules, self.r.tables
        gene, cons = v.get("gene"), v.get("consequence")
        if cons == "missense" and not _nan(v.get("p_pos")):
            pos = int(v["p_pos"])
            p, b = T.density(gene, pos, R["pm1_window"], ex)
            if p >= R["pm1_min_plp"] and b <= R["pm1_max_blb"]:
                c = crit["PM1"]
                c.status, c.strength = "met", "Moderate"
                c.rationale = (f"{p} pathogenic or likely pathogenic missense variants and {b} benign missense variants fall within "
                               f"{R['pm1_window']} residues of position {pos}, a mutational hot spot without benign variation.")
            elif p >= max(2, R["pm1_min_plp"] // 2) and b <= R["pm1_max_blb"]:
                c = crit["PM1"]
                c.status, c.strength = "met", "Supporting"
                c.rationale = f"{p} pathogenic missense variants and no benign missense variants within {R['pm1_window']} residues: a weaker hot spot."
            else:
                crit["PM1"].rationale = f"Within {R['pm1_window']} residues: {p} pathogenic and {b} benign missense variants (not a hot spot)."
            crit["PM1"].evidence = [{"kind": "pool_density", "ref": gene, "detail": f"residues {pos - R['pm1_window']} to {pos + R['pm1_window']}: {p} P/LP, {b} B/LB"}]
        else:
            crit["PM1"].rationale = "Applies to missense variants only."
        if cons in ("inframe_indel", "stop_loss"):
            pos = v.get("p_pos")
            near = T.benign_inframe_near(gene, int(pos), R["bp3_window"], ex) if not _nan(pos) else 0
            if near > 0 and cons == "inframe_indel":
                c = crit["BP3"]
                c.status, c.strength = "met", "Supporting"
                c.rationale = (f"{near} benign in frame indels lie within {R['bp3_window']} residues, which suggests a repetitive or "
                               f"tolerant region.")
                crit["PM4"].rationale = "Not applied: benign in frame variation nearby (BP3 applied instead)."
            else:
                c = crit["PM4"]
                c.status, c.strength = "met", "Moderate"
                lc = v.get("length_change")
                what = ("stop loss extends the protein" if cons == "stop_loss"
                        else f"in frame change of {abs(int(lc))} bases" if not _nan(lc) else "in frame change")
                c.rationale = f"Protein length change ({what}) with no benign in frame variation nearby."
        else:
            crit["PM4"].rationale = crit["BP3"].rationale = "Not an in frame indel or stop loss."

    # ---- gene level: PP2, BP1 ----
    def _gene_level(self, v, crit, ex):
        R = self.r.rules
        if v.get("consequence") != "missense":
            crit["PP2"].rationale = crit["BP1"].rationale = "Applies to missense variants only."
            return
        st = self.r.tables.stats(v.get("gene"), ex)
        mp, mb = st.get("missense_plp", 0), st.get("missense_blb", 0)
        tp = st.get("truncating_plp", 0)
        frac_b = mb / (mp + mb) if (mp + mb) else 1.0
        if mp >= R["pp2_min_plp"] and frac_b <= R["pp2_max_benign_frac"]:
            c = crit["PP2"]
            c.status, c.strength = "met", "Supporting"
            c.rationale = (f"{mp} missense variants in this gene are pathogenic and only {mb} benign ({frac_b:.0%} benign), so missense is a "
                           f"common mechanism with little benign missense variation.")
        else:
            crit["PP2"].rationale = f"Gene missense spectrum in the pool: {mp} pathogenic, {mb} benign ({frac_b:.0%} benign)."
        mis_share = mp / (mp + tp) if (mp + tp) else 1.0
        if tp >= R["bp1_min_trunc"] and mis_share <= R["bp1_max_missense_frac"]:
            c = crit["BP1"]
            c.status, c.strength = "met", "Supporting"
            c.rationale = (f"{tp} truncating but only {mp} missense variants are pathogenic ({mis_share:.1%} missense), so disease is caused "
                           f"mainly by truncating variants.")
        else:
            crit["BP1"].rationale = f"Pathogenic spectrum: {tp} truncating, {mp} missense ({mis_share:.0%} missense)."
        for k in ("PP2", "BP1"):
            crit[k].evidence = [{"kind": "pool_gene", "ref": v.get("gene"), "detail": f"missense P/LP {mp}, missense B/LB {mb}, truncating P/LP {tp}"}]

    # ---- computational: PP3, BP4, BP7 ----
    def _computational(self, v, crit, scores):
        cons = v.get("consequence")
        if scores["missense"] is not None:
            d, s = strength_for(scores["missense"], self.r.missense_cal)
            s = "Strong" if s == "VeryStrong" else s
            ev = [{"kind": "insilico", "ref": "missense model", "detail": f"score {scores['missense']:.3f} (calibrated, prior 0.0441)"}]
            if d == "pathogenic":
                crit["PP3"].status, crit["PP3"].strength = "met", s
                crit["PP3"].rationale = f"The calibrated missense predictor scores {scores['missense']:.3f}, which reaches {s} pathogenic evidence."
            elif d == "benign":
                crit["BP4"].status, crit["BP4"].strength = "met", s
                crit["BP4"].rationale = f"The calibrated missense predictor scores {scores['missense']:.3f}, which reaches {s} benign evidence."
            else:
                crit["PP3"].rationale = crit["BP4"].rationale = f"Missense score {scores['missense']:.3f} lies in the indeterminate range."
            crit["PP3"].evidence = crit["BP4"].evidence = ev
            return
        off = v.get("c_offset") or 0
        if scores["splice"] is not None:
            d, s = strength_for(scores["splice"], self.r.splice_cal)
            s = "Strong" if s == "VeryStrong" else s
            ev = [{"kind": "insilico", "ref": "splice region model", "detail": f"score {scores['splice']:.3f} at intronic offset {off:+d}"}]
            if d == "pathogenic":
                crit["PP3"].status, crit["PP3"].strength = "met", s
                crit["PP3"].rationale = f"The calibrated splice region predictor scores {scores['splice']:.3f} ({s} pathogenic) at offset {off:+d}."
            elif d == "benign":
                code = "BP7" if (off >= 7 or off <= -21) else "BP4"
                crit[code].status, crit[code].strength = "met", "Supporting" if code == "BP7" else s
                crit[code].rationale = f"The calibrated splice region predictor scores {scores['splice']:.3f}, predicting no splice impact at offset {off:+d}."
            else:
                crit["PP3"].rationale = crit["BP4"].rationale = f"Splice region score {scores['splice']:.3f} lies in the indeterminate range."
            for k in ("PP3", "BP4", "BP7"):
                crit[k].evidence = ev
            return
        if cons == "synonymous":
            near_edge = False
            tx = self.r.transcripts.get(v.get("gene"))
            if tx is not None and tx.anchored and not _nan(v.get("c_pos")):
                i = tx.exon_of(int(v["c_pos"]))
                if i is not None:
                    e = tx.exons[i]
                    near_edge = (e["lc_end"] - int(v["c_pos"]) < 3) or (int(v["c_pos"]) - e["lc_start"] < 1)
            if near_edge:
                crit["BP7"].rationale = "Synonymous change in the last three bases or first base of an exon, where splicing may be affected."
            else:
                crit["BP7"].status, crit["BP7"].strength = "met", "Supporting"
                crit["BP7"].rationale = "Synonymous change outside the exonic splice region; no splice impact is predicted."
            return
        if cons in ("intron",) and abs(off) > 50:
            crit["BP7"].status, crit["BP7"].strength = "met", "Supporting"
            crit["BP7"].rationale = f"Deep intronic position (offset {off:+d}) beyond the splice region with no predicted splice impact."
            return
        crit["PP3"].rationale = crit["BP4"].rationale = "No calibrated predictor covers this variant type."

    # ---- case level ----
    def _case_level(self, v, g, case, crit):
        def need(code, what):
            crit[code].status = "not_evaluated"
            crit[code].rationale = f"Needs {what}."

        # de novo (SVI point system)
        dn = case.get("de_novo") or []
        if dn:
            pts = 0.0
            for d in dn:
                confirmed = d.get("confirmed", False)
                specific = d.get("phenotype") == "specific"
                pts += (2.0 if specific else 1.0) if confirmed else (1.0 if specific else 0.5)
            s = "VeryStrong" if pts >= 4 else "Strong" if pts >= 2 else "Moderate" if pts >= 1 else "Supporting"
            code = "PS2" if any(d.get("confirmed") for d in dn) else "PM6"
            crit[code].status, crit[code].strength = "met", s
            crit[code].rationale = f"{len(dn)} de novo observation(s) totalling {pts:g} points under the ClinGen SVI de novo scale."
        else:
            need("PS2", "parental testing (confirmed de novo)")
            need("PM6", "de novo observations without confirmed parentage")
        # recessive in trans (SVI PM3 points)
        tr = case.get("in_trans") or []
        if tr:
            pts = 0.0
            for t in tr:
                partner = t.get("partner_class", "Pathogenic")
                phased = t.get("phase") == "trans"
                if t.get("homozygous"):
                    pts += 0.5
                elif phased:
                    pts += 1.0 if partner == "Pathogenic" else 0.5
                else:
                    pts += 0.5 if partner == "Pathogenic" else 0.25
            if pts >= 0.5:
                s = "VeryStrong" if pts >= 4 else "Strong" if pts >= 2 else "Moderate" if pts >= 1 else "Supporting"
                crit["PM3"].status, crit["PM3"].strength = "met", s
                crit["PM3"].rationale = f"{len(tr)} observation(s) with a pathogenic variant in a recessive gene, {pts:g} points on the SVI PM3 scale."
        else:
            need("PM3", "observations of the variant with a second pathogenic allele")
        # segregation
        meioses = case.get("segregation_meioses")
        if meioses:
            s = "Strong" if meioses >= 7 else "Moderate" if meioses >= 5 else "Supporting" if meioses >= 3 else None
            if s:
                crit["PP1"].status, crit["PP1"].strength = "met", s
                crit["PP1"].rationale = f"Cosegregates through {meioses} informative meioses (likelihood about 1 in {2 ** meioses})."
            else:
                crit["PP1"].rationale = f"Only {meioses} informative meioses, below the three needed for Supporting."
        else:
            need("PP1", "family segregation data")
        if case.get("nonsegregation"):
            crit["BS4"].status, crit["BS4"].strength = "met", "Strong"
            crit["BS4"].rationale = "Affected relatives do not carry the variant."
        else:
            need("BS4", "family segregation data")
        # functional
        fn = case.get("functional")
        if fn:
            code = "PS3" if fn.get("direction") == "damaging" else "BS3"
            crit[code].status, crit[code].strength = "met", fn.get("strength", "Strong" if code == "PS3" else "Strong")
            crit[code].rationale = f"Functional evidence: {fn.get('assay', 'validated assay')} ({crit[code].strength}, per the ClinGen SVI OddsPath framework)."
        else:
            need("PS3", "a validated functional assay")
            need("BS3", "a validated functional assay")
        # case control
        cc = case.get("case_control")
        if cc:
            orr, lo = cc.get("odds_ratio", 0), cc.get("ci_low", 0)
            if orr >= 5 and lo > 1:
                crit["PS4"].status, crit["PS4"].strength = "met", "Strong"
                crit["PS4"].rationale = f"Odds ratio {orr:g} with lower confidence limit {lo:g}."
        elif case.get("probands"):
            n = int(case["probands"])
            s = "Strong" if n >= 4 else "Moderate" if n >= 2 else "Supporting"
            crit["PS4"].status, crit["PS4"].strength = "met", s
            crit["PS4"].rationale = f"Observed in {n} unrelated affected probands and absent from controls (proband counting)."
        else:
            need("PS4", "case control or proband counts")
        # phenotype
        ph = case.get("phenotype_match")
        if ph is not None:
            if ph.get("p_value", 1) < 0.01 and ph.get("specific", False):
                crit["PP4"].status, crit["PP4"].strength = "met", "Supporting"
                crit["PP4"].rationale = f"Patient phenotype matches the gene's disease profile (similarity {ph.get('score', 0):.2f}, empirical p {ph['p_value']:.3g})."
            else:
                crit["PP4"].rationale = f"Phenotype match not specific enough (empirical p {ph.get('p_value', 1):.3g})."
        else:
            need("PP4", "patient phenotype terms")
        if case.get("alternate_cause"):
            crit["BP5"].status, crit["BP5"].strength = "met", "Supporting"
            crit["BP5"].rationale = "An alternate molecular cause of disease was found in this case."
        else:
            need("BP5", "case level molecular findings")
        if case.get("bp2"):
            crit["BP2"].status, crit["BP2"].strength = "met", "Supporting"
            crit["BP2"].rationale = case["bp2"]
        else:
            need("BP2", "phase data with a pathogenic variant")

    # ---- caps ----
    def _caps(self, crit, notes):
        R = self.r.rules
        from allelion.config import POINTS
        pp3, pm1 = crit["PP3"], crit["PM1"]
        if pp3.status == "met" and pm1.status == "met":
            total = POINTS[pp3.strength] + POINTS[pm1.strength]
            if total > R["pp3_pm1_cap_points"]:
                excess = total - R["pp3_pm1_cap_points"]
                new = POINTS[pm1.strength] - excess
                if new <= 0:
                    pm1.status = "not_met"
                    pm1.rationale += " Not counted: PP3 already carries the regional signal (PP3 plus PM1 capped at Strong)."
                else:
                    pm1.strength = {1: "Supporting", 2: "Moderate"}[new]
                notes.append("PP3 and PM1 together are capped at Strong (4 points), following ClinGen SVI guidance.")
        if crit["PVS1"].status == "met" and crit["PM4"].status == "met":
            crit["PM4"].status = "not_met"
            notes.append("PM4 is not combined with PVS1.")
        if crit["PVS1"].status == "met" and crit["PP3"].status == "met":
            crit["PP3"].status = "not_met"
            notes.append("PP3 is not combined with PVS1 for the same predicted effect.")
        if crit["BA1"].status == "met":
            crit["BS1"].status = "not_met"
