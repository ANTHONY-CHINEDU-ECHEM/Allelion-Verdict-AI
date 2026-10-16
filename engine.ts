// The ACMG/AMP criteria engine in the browser. A line by line port of allelion/acmg/engine.py;
// tests/test_parity.py and web/src/engine/parity.test.ts check both on a shared fixture.
import { AA3, LOF_CONSEQUENCES, consequenceFromHgvs, parseC, parseP } from "./hgvs";
import type { Bundle, Calibration, Variant } from "./data";
import { isNum } from "./data";
import { missenseFeatures, spliceFeatures } from "./features";
import { predictBinary, vector } from "./lgbm";
import * as T from "./tables";
import { exonFrames, exonOf, nmdEscape } from "./transcripts";
import { POINTS, class2015, classFromPoints, pointsOf, posterior, type CritOut } from "./combine";

export interface CaseEvidence {
  de_novo?: { confirmed: boolean; phenotype: "specific" | "consistent" }[];
  in_trans?: { partner_class: string; phase: "trans" | "unknown"; homozygous?: boolean }[];
  segregation_meioses?: number;
  nonsegregation?: boolean;
  functional?: { direction: "damaging" | "normal"; strength: string; assay?: string };
  case_control?: { odds_ratio: number; ci_low: number };
  probands?: number;
  phenotype_match?: { score: number; p_value: number; specific: boolean } | null;
  alternate_cause?: boolean;
  bp2?: string;
}

export interface Result {
  variant: Record<string, any>; criteria: CritOut[]; met: string[]; points: number; classification: string; classification_2015: string;
  posterior: number; scores: { missense: number | null; splice: number | null }; notes: string[]; not_evaluated: string[];
}

const nan = (x: any) => x === null || x === undefined || (typeof x === "number" && Number.isNaN(x));
const AA_SET = new Set(Object.values(AA3));

// ---- Python style formatting helpers ----
export function fmtAf(x: number | null | undefined): string {
  if (nan(x)) return "not observed";
  const v = x as number;
  if (v === 0) return "0";
  if (v >= 0.01) return v.toFixed(3);
  const [m, e] = v.toExponential(2).split("e");
  const ex = parseInt(e, 10);
  return `${m}e${ex < 0 ? "-" : "+"}${String(Math.abs(ex)).padStart(2, "0")}`;
}
const pct0 = (x: number) => `${Math.round(x * 100)}%`;
const pct1 = (x: number) => `${(x * 100).toFixed(1)}%`;
const fmtG = (x: number) => (Math.abs(x) < 1e-4 && x !== 0 ? (() => { const [m, e] = x.toExponential().split("e"); const ex = parseInt(e, 10); return `${m}e${ex < 0 ? "-" : "+"}${String(Math.abs(ex)).padStart(2, "0")}`; })() : String(Number(x.toPrecision(6))));
const sgn = (x: number) => (x >= 0 ? `+${x}` : `${x}`);
const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

export function strengthFor(score: number, cal: Calibration): [string | null, string | null] {
  for (const s of ["VeryStrong", "Strong", "Moderate", "Supporting"]) {
    const t = cal.pathogenic_thresholds[s];
    if (t !== null && t !== undefined && score >= t) return ["pathogenic", s];
  }
  for (const s of ["VeryStrong", "Strong", "Moderate", "Supporting"]) {
    const t = cal.benign_thresholds[s];
    if (t !== null && t !== undefined && score <= t) return ["benign", s];
  }
  return [null, null];
}

export function prepare(gene: string, hgvsC: string, hgvsP?: string | null, population?: { ac?: number; an?: number; grpmax_af?: number; faf95?: number; hom?: number; hemi?: number } | null): Variant {
  const c = parseC(hgvsC);
  const p = hgvsP ? parseP(hgvsP) : null;
  if (!c) throw new Error(`could not parse coding HGVS ${hgvsC}`);
  const pop = population || null;
  return {
    variation_id: null, gene, hgvs_c: hgvsC, hgvs_p: hgvsP || "", consequence: consequenceFromHgvs(c, p), c_kind: c.kind, c_region: c.start.region,
    c_pos: c.start.base, c_end: c.end.base, c_offset: c.start.offset, c_end_offset: c.end.offset, splice_distance: c.spliceDistance,
    length_change: c.lengthChange, p_kind: p ? p.kind : null, p_ref: p ? p.refAa : null, p_pos: p && p.pos ? p.pos : null, p_alt: p ? p.altAa : null,
    p_end: p && p.endPos ? p.endPos : null, protein_change: p ? p.short : "",
    gn_AC: pop?.ac ?? null, gn_AN: pop?.an ?? null, gn_grpmax: pop?.grpmax_af ?? null, gn_grpmax_2000: pop?.grpmax_af ?? null,
    gn_faf95: pop?.faf95 ?? pop?.grpmax_af ?? null, gn_Hom: pop?.hom ?? null, gn_Hemi: pop?.hemi ?? null, population_known: !!pop,
  };
}

class C {
  status: CritOut["status"] = "not_met"; strength = ""; rationale = ""; evidence: CritOut["evidence"] = [];
  constructor(public code: string) {}
}

export class Engine {
  constructor(public b: Bundle) {}
  get rules(): Record<string, number> { return this.b.meta.rules; }

  missenseScore(v: Variant): number | null {
    if (v.consequence !== "missense" || nan(v.p_pos) || !AA_SET.has(v.p_ref as string) || !AA_SET.has(v.p_alt as string)) return null;
    return predictBinary(this.b.missense, vector(this.b.missense, missenseFeatures(this.b, v)));
  }

  spliceScore(v: Variant): number | null {
    const off = v.c_offset || 0;
    if (v.c_region !== "cds" || !(Math.abs(off) >= 3 && Math.abs(off) <= 50)) return null;
    return predictBinary(this.b.splice, vector(this.b.splice, spliceFeatures(this.b, v)));
  }

  evaluate(v: Variant, kase: CaseEvidence = {}, use?: Set<string>): Result {
    const gene = this.b.genes[v.gene] as any || {};
    const ex = v.variation_id ?? null;
    const order: string[] = this.b.meta.order;
    const crit: Record<string, C> = {};
    order.forEach((c) => { crit[c] = new C(c); });
    const notes: string[] = [];
    const scores = { missense: this.missenseScore(v), splice: this.spliceScore(v) };
    this.population(v, gene, crit);
    this.pvs1(v, gene, crit, ex);
    this.allelic(v, crit, ex);
    this.region(v, crit, ex);
    this.geneLevel(v, crit, ex);
    this.computational(v, crit, scores);
    this.caseLevel(kase, crit);
    this.caps(crit, notes);
    if (use) {
      for (const c of Object.values(crit)) if (!use.has(c.code) && c.status === "met") { c.status = "not_met"; c.rationale += " (disabled in this run)"; }
    }
    const titles = this.b.meta.titles, dflt = this.b.meta.default_strength;
    const out: CritOut[] = order.map((code) => {
      const c = crit[code];
      const strength = c.strength || dflt[code];
      return {
        code, title: titles[code], direction: code.startsWith("B") ? "benign" : "pathogenic", status: c.status, strength, default_strength: dflt[code],
        label: strength === dflt[code] ? code : `${code}_${strength}`, rationale: c.rationale, evidence: c.evidence,
      };
    });
    const ba1 = crit.BA1.status === "met";
    const pts = pointsOf(out);
    return {
      variant: { variation_id: v.variation_id, gene: v.gene, hgvs_c: v.hgvs_c, hgvs_p: v.hgvs_p, consequence: v.consequence, protein_change: v.protein_change, key: v.key ?? null },
      criteria: out, met: out.filter((c) => c.status === "met").map((c) => c.label), points: pts, classification: classFromPoints(pts, ba1),
      classification_2015: class2015(out), posterior: ba1 ? 0 : Math.round(posterior(pts) * 1e4) / 1e4,
      scores: { missense: scores.missense === null ? null : Math.round(scores.missense * 1e4) / 1e4, splice: scores.splice === null ? null : Math.round(scores.splice * 1e4) / 1e4 },
      notes, not_evaluated: out.filter((c) => c.status === "not_evaluated").map((c) => c.code),
    };
  }

  private population(v: Variant, g: any, crit: Record<string, C>) {
    const R = this.rules;
    let afBa1 = v.gn_grpmax_2000;
    if (nan(afBa1)) afBa1 = v.gn_grpmax;
    let af = v.gn_faf95;
    if (nan(af)) af = v.gn_grpmax;
    const ac = v.gn_AC;
    const known = v.population_known === undefined ? true : v.population_known;
    const dominant = !!g.dom, recessive = !!g.rec;
    const exc = (this.b.meta.ba1_exceptions as string[][]).some(([gg, pc]) => gg === v.gene && pc === v.protein_change);
    const an = v.gn_AN, hom0 = v.gn_Hom;
    const detail = `filtering allele frequency ${fmtAf(af)}; highest continental frequency ${fmtAf(afBa1)}; allele count ${nan(ac) ? 0 : Math.trunc(ac as number)}${nan(an) ? "" : ` of ${Math.trunc(an as number)}`}; homozygotes ${nan(hom0) ? 0 : Math.trunc(hom0 as number)}`;
    const ev = [{ kind: "population", ref: "gnomAD v2 exomes and genomes", detail }];
    if (!known) {
      for (const c of ["BA1", "BS1", "PM2", "BS2"]) { crit[c].status = "not_evaluated"; crit[c].rationale = "No population frequency was supplied for this variant."; }
      return;
    }
    const seen = !nan(ac) && (ac as number) > 0;
    if (seen && !nan(afBa1) && (afBa1 as number) > R.ba1_af && !exc) {
      const c = crit.BA1;
      c.status = "met"; c.strength = "StandAlone";
      c.rationale = `The highest continental allele frequency in gnomAD (groups with at least 2000 alleles) is ${fmtAf(afBa1)}, above the 5% stand alone benign threshold.`;
      c.evidence = ev;
      return;
    }
    const thr = dominant ? R.bs1_af_dominant : recessive ? R.bs1_af_recessive : R.bs1_af_unknown;
    const mode = dominant ? "dominant" : recessive ? "recessive" : "undetermined inheritance";
    if (seen && !nan(af) && (af as number) > thr) {
      const c = crit.BS1;
      c.status = "met"; c.strength = "Strong";
      c.rationale = `The filtering allele frequency ${fmtAf(af)} exceeds ${fmtG(thr)}, the largest frequency compatible with a pathogenic allele for a ${mode} disorder under the configured prevalence and penetrance assumptions.`;
      c.evidence = ev;
    } else crit.BS1.rationale = `The filtering allele frequency ${fmtAf(af)} does not exceed ${fmtG(thr)} (${mode}).`;
    if (!seen) {
      const c = crit.PM2;
      c.status = "met"; c.strength = "Supporting";
      c.rationale = "Absent from gnomAD exomes and genomes. Applied at Supporting strength as recommended by ClinGen SVI (2020).";
      c.evidence = ev;
    } else if (recessive && !dominant && !nan(af) && (af as number) < R.pm2_af_recessive) {
      const c = crit.PM2;
      c.status = "met"; c.strength = "Supporting";
      c.rationale = `For a recessive gene the frequency ${fmtAf(af)} is below ${fmtG(R.pm2_af_recessive)}, rare enough for PM2 at Supporting strength.`;
      c.evidence = ev;
    } else crit.PM2.rationale = `Observed in gnomAD (${nan(ac) ? 0 : Math.trunc(ac as number)} alleles), so not absent.`;
    const hom = v.gn_Hom, hemi = v.gn_Hemi, adult = !!g.adult;
    if (recessive && !dominant && !adult && !nan(hom) && (hom as number) >= R.bs2_hom_min && crit.BA1.status !== "met") {
      const c = crit.BS2;
      c.status = "met"; c.strength = "Strong";
      c.rationale = `Observed homozygous in ${Math.trunc(hom as number)} gnomAD individuals, which is incompatible with a fully penetrant early onset recessive disorder (gnomAD excludes severe paediatric disease cohorts).`;
      c.evidence = ev;
    } else if (!!g.xl && recessive && !dominant && !adult && !nan(hemi) && (hemi as number) >= R.bs2_hom_min) {
      const c = crit.BS2;
      c.status = "met"; c.strength = "Strong";
      c.rationale = `Observed hemizygous in ${Math.trunc(hemi as number)} gnomAD males for an X linked recessive gene.`;
      c.evidence = ev;
    } else crit.BS2.rationale = "No homozygous or hemizygous observations that rule out a penetrant recessive disorder.";
  }

  private pvs1(v: Variant, g: any, crit: Record<string, C>, ex: string | null) {
    const cons = v.consequence;
    const c = crit.PVS1;
    if (!LOF_CONSEQUENCES.has(cons)) { c.rationale = "Not a predicted null variant."; return; }
    const gene = v.gene;
    const [ok, why] = T.lofMechanism(T.geneStats(this.b, gene, ex), g, this.rules);
    const ev: CritOut["evidence"] = [{ kind: "mechanism", ref: gene, detail: why }];
    if (!ok) { c.rationale = `The variant is a predicted null allele, but ${why}.`; c.evidence = ev; return; }
    const tx = this.b.transcripts[gene] || null;
    const plen = g.plen;
    const R = this.rules;
    let strength = "VeryStrong", detail = "";
    if (cons === "nonsense" || cons === "frameshift") {
      const ppos = v.p_pos;
      let ptc: number | null;
      if (cons === "frameshift" && !nan(ppos)) {
        const m = /fs(?:Ter|\*)(\d+)/.exec(v.hgvs_p || "");
        const extra = m ? parseInt(m[1], 10) : null;
        ptc = Math.trunc(ppos as number) + (extra ? extra - 1 : 0);
      } else ptc = nan(ppos) ? null : Math.trunc(ppos as number);
      if (ptc === null && !nan(v.c_pos)) ptc = Math.floor((v.c_pos as number) / 3) + 1;
      if (tx && ptc !== null) {
        const [escape, nmdText] = nmdEscape(tx, ptc * 3);
        ev.push({ kind: "transcript", ref: `${gene} exon model`, detail: nmdText });
        if (!escape) { strength = "VeryStrong"; detail = `${cap(nmdText)}.`; }
        else {
          const down = T.truncatingDownstream(this.b, gene, ptc, ex);
          const removed = plen && !nan(plen) ? 1 - ptc / plen : 0;
          if (down > 0) {
            strength = "Strong";
            detail = `${cap(nmdText)}, so the transcript escapes decay, but ${down} pathogenic truncating variants lie further downstream, so the removed region is critical.`;
            ev.push({ kind: "pool", ref: gene, detail: `${down} pathogenic truncating variants downstream of residue ${ptc}` });
          } else if (removed >= R.pvs1_removed_frac_strong) { strength = "Strong"; detail = `${cap(nmdText)}; the truncation removes ${pct0(removed)} of the protein.`; }
          else { strength = "Moderate"; detail = `${cap(nmdText)}; the truncation removes only ${pct0(removed)} of the protein.`; }
        }
      } else {
        const down = ptc ? T.truncatingDownstream(this.b, gene, ptc, ex) : 0;
        if (down > 0) { strength = "VeryStrong"; detail = `No reliable exon model; ${down} pathogenic truncating variants lie downstream of residue ${ptc}, so truncation at this position is expected to be damaging.`; }
        else { strength = "Strong"; detail = "No reliable exon model and no downstream pathogenic truncation evidence, so PVS1 is reduced to Strong."; }
      }
    } else if (cons === "splice_donor" || cons === "splice_acceptor") {
      if (tx && !nan(v.c_pos)) {
        const base = Math.trunc(v.c_pos as number);
        const i = exonOf(tx, base);
        const frames = exonFrames(tx);
        if (i !== null) {
          const f = frames[i];
          const removed = tx.cds ? f.coding / tx.cds : 0;
          const site = cons.split("_")[1];
          if (!f.in_frame_skip) { strength = "VeryStrong"; detail = `Disrupts the canonical ${site} site of exon ${i + 1}; skipping that exon (${f.coding} coding bases) shifts the reading frame.`; }
          else {
            const lo = Math.floor((f.lc_start - 1) / 3) + 1;
            const hi = Math.floor(f.lc_end / 3) + 1;
            const pIn = T.density(this.b, gene, Math.floor((lo + hi) / 2), Math.max(1, Math.floor((hi - lo) / 2)), ex)[0];
            if (pIn > 0 || removed >= R.pvs1_removed_frac_strong) {
              strength = "Strong";
              detail = `Disrupts the canonical ${site} site of exon ${i + 1}; skipping it keeps the frame but removes ${pct1(removed)} of the coding sequence${pIn ? ", a region with pathogenic missense variation" : ""}.`;
            } else { strength = "Moderate"; detail = `In frame skip of exon ${i + 1} removes only ${pct1(removed)} of the coding sequence.`; }
          }
          ev.push({ kind: "transcript", ref: `${gene} exon ${i + 1}`, detail: `${f.coding} coding bases, in frame skip: ${f.in_frame_skip ? "True" : "False"}` });
        } else { strength = "Strong"; detail = "Canonical splice site variant; the adjacent exon could not be located, so the effect is not predicted precisely."; }
      } else { strength = "Strong"; detail = "Canonical splice site variant without a reliable exon model; applied at Strong."; }
    } else if (cons === "start_loss") {
      strength = "Moderate"; detail = "Initiation codon variant; an alternative downstream start codon cannot be excluded, so PVS1 is applied at Moderate.";
    }
    c.status = "met"; c.strength = strength;
    c.rationale = `Predicted null variant and ${why}. ${detail}`.trim();
    c.evidence = ev;
  }

  private allelic(v: Variant, crit: Record<string, C>, ex: string | null) {
    const gene = v.gene, cons = v.consequence;
    const cls = this.b.meta.classes as string[];
    const short = (v.hgvs_c || "").split(":").pop() || "";
    if (cons === "missense" && v.protein_change) {
      const same = T.sameChange(this.b, gene, v.protein_change, ex, short);
      if (same.length) {
        const c = crit.PS1;
        c.status = "met"; c.strength = same.some((r) => r[2] === 0) ? "Strong" : "Moderate";
        c.rationale = `The same amino acid change ${v.protein_change} is classified ${cls[same[0][2]].toLowerCase()} for a different nucleotide change (${same[0][4]}).`;
        c.evidence = same.slice(0, 5).map((r) => ({ kind: "clinvar_pool", ref: `ClinVar ${r[3]}`, detail: `${r[4]}: ${cls[r[2]]}` }));
      } else crit.PS1.rationale = "No other nucleotide change producing this amino acid change is classified pathogenic in the pool.";
      if (crit.PS1.status !== "met" && !nan(v.p_pos)) {
        const res = T.sameResidue(this.b, gene, Math.trunc(v.p_pos as number), ex).filter((r) => T.PLP.has(r[2]) && r[1] !== v.protein_change);
        const alts = [...new Set(res.map((r) => r[1]))].sort();
        if (alts.length) {
          const c = crit.PM5;
          c.status = "met";
          const anyP = res.some((r) => r[2] === 0);
          c.strength = alts.length >= 2 && anyP ? "Strong" : "Moderate";
          if (c.strength === "Moderate" && !anyP) c.strength = "Supporting";
          c.rationale = `A different missense change at the same residue is classified pathogenic or likely pathogenic (${alts.slice(0, 4).join(", ")}).`;
          c.evidence = res.slice(0, 5).map((r) => ({ kind: "clinvar_pool", ref: `ClinVar ${r[3]}`, detail: `${r[1]} (${r[4]}): ${cls[r[2]]}` }));
        } else crit.PM5.rationale = "No other pathogenic missense change at this residue in the pool.";
      }
    } else if (["splice_donor", "splice_acceptor", "intron"].includes(cons) && v.c_offset && !nan(v.c_pos)) {
      const same = T.sameSpliceSite(this.b, gene, Math.trunc(v.c_pos as number), v.c_offset, ex, short);
      if (same.length) {
        const c = crit.PS1;
        c.status = "met"; c.strength = Math.abs(v.c_offset) <= 2 ? "Moderate" : "Supporting";
        c.rationale = `A different change at the same splice position is classified ${cls[same[0][3]].toLowerCase()} (${same[0][2]}); applied for splicing as recommended by ClinGen SVI (2023).`;
        c.evidence = same.slice(0, 5).map((r) => ({ kind: "clinvar_pool", ref: `ClinVar ${r[4]}`, detail: `${r[2]}: ${cls[r[3]]}` }));
      }
    } else { crit.PS1.rationale = crit.PM5.rationale = "Not a missense or splice region variant."; }
  }

  private region(v: Variant, crit: Record<string, C>, ex: string | null) {
    const R = this.rules, gene = v.gene, cons = v.consequence;
    if (cons === "missense" && !nan(v.p_pos)) {
      const pos = Math.trunc(v.p_pos as number);
      const [p, q] = T.density(this.b, gene, pos, R.pm1_window, ex);
      if (p >= R.pm1_min_plp && q <= R.pm1_max_blb) {
        const c = crit.PM1; c.status = "met"; c.strength = "Moderate";
        c.rationale = `${p} pathogenic or likely pathogenic missense variants and ${q} benign missense variants fall within ${R.pm1_window} residues of position ${pos}, a mutational hot spot without benign variation.`;
      } else if (p >= Math.max(2, Math.floor(R.pm1_min_plp / 2)) && q <= R.pm1_max_blb) {
        const c = crit.PM1; c.status = "met"; c.strength = "Supporting";
        c.rationale = `${p} pathogenic missense variants and no benign missense variants within ${R.pm1_window} residues: a weaker hot spot.`;
      } else crit.PM1.rationale = `Within ${R.pm1_window} residues: ${p} pathogenic and ${q} benign missense variants (not a hot spot).`;
      crit.PM1.evidence = [{ kind: "pool_density", ref: gene, detail: `residues ${pos - R.pm1_window} to ${pos + R.pm1_window}: ${p} P/LP, ${q} B/LB` }];
    } else crit.PM1.rationale = "Applies to missense variants only.";
    if (cons === "inframe_indel" || cons === "stop_loss") {
      const near = nan(v.p_pos) ? 0 : T.benignInframeNear(this.b, gene, Math.trunc(v.p_pos as number), R.bp3_window, ex);
      if (near > 0 && cons === "inframe_indel") {
        const c = crit.BP3; c.status = "met"; c.strength = "Supporting";
        c.rationale = `${near} benign in frame indels lie within ${R.bp3_window} residues, which suggests a repetitive or tolerant region.`;
        crit.PM4.rationale = "Not applied: benign in frame variation nearby (BP3 applied instead).";
      } else {
        const c = crit.PM4; c.status = "met"; c.strength = "Moderate";
        const lc = v.length_change;
        const what = cons === "stop_loss" ? "stop loss extends the protein" : !nan(lc) ? `in frame change of ${Math.abs(Math.trunc(lc as number))} bases` : "in frame change";
        c.rationale = `Protein length change (${what}) with no benign in frame variation nearby.`;
      }
    } else crit.PM4.rationale = crit.BP3.rationale = "Not an in frame indel or stop loss.";
  }

  private geneLevel(v: Variant, crit: Record<string, C>, ex: string | null) {
    const R = this.rules;
    if (v.consequence !== "missense") { crit.PP2.rationale = crit.BP1.rationale = "Applies to missense variants only."; return; }
    const st = T.geneStats(this.b, v.gene, ex);
    const mp = st.missense_plp || 0, mb = st.missense_blb || 0, tp = st.truncating_plp || 0;
    const fracB = mp + mb ? mb / (mp + mb) : 1.0;
    if (mp >= R.pp2_min_plp && fracB <= R.pp2_max_benign_frac) {
      const c = crit.PP2; c.status = "met"; c.strength = "Supporting";
      c.rationale = `${mp} missense variants in this gene are pathogenic and only ${mb} benign (${pct0(fracB)} benign), so missense is a common mechanism with little benign missense variation.`;
    } else crit.PP2.rationale = `Gene missense spectrum in the pool: ${mp} pathogenic, ${mb} benign (${pct0(fracB)} benign).`;
    const misShare = mp + tp ? mp / (mp + tp) : 1.0;
    if (tp >= R.bp1_min_trunc && misShare <= R.bp1_max_missense_frac) {
      const c = crit.BP1; c.status = "met"; c.strength = "Supporting";
      c.rationale = `${tp} truncating but only ${mp} missense variants are pathogenic (${pct1(misShare)} missense), so disease is caused mainly by truncating variants.`;
    } else crit.BP1.rationale = `Pathogenic spectrum: ${tp} truncating, ${mp} missense (${pct0(misShare)} missense).`;
    for (const k of ["PP2", "BP1"]) crit[k].evidence = [{ kind: "pool_gene", ref: v.gene, detail: `missense P/LP ${mp}, missense B/LB ${mb}, truncating P/LP ${tp}` }];
  }

  private computational(v: Variant, crit: Record<string, C>, scores: { missense: number | null; splice: number | null }) {
    const cons = v.consequence;
    if (scores.missense !== null) {
      let [d, s] = strengthFor(scores.missense, this.b.missenseCal);
      if (s === "VeryStrong") s = "Strong";
      const sc = scores.missense.toFixed(3);
      const ev = [{ kind: "insilico", ref: "missense model", detail: `score ${sc} (calibrated, prior 0.0441)` }];
      if (d === "pathogenic") { crit.PP3.status = "met"; crit.PP3.strength = s!; crit.PP3.rationale = `The calibrated missense predictor scores ${sc}, which reaches ${s} pathogenic evidence.`; }
      else if (d === "benign") { crit.BP4.status = "met"; crit.BP4.strength = s!; crit.BP4.rationale = `The calibrated missense predictor scores ${sc}, which reaches ${s} benign evidence.`; }
      else crit.PP3.rationale = crit.BP4.rationale = `Missense score ${sc} lies in the indeterminate range.`;
      crit.PP3.evidence = ev; crit.BP4.evidence = ev;
      return;
    }
    const off = v.c_offset || 0;
    if (scores.splice !== null) {
      let [d, s] = strengthFor(scores.splice, this.b.spliceCal);
      if (s === "VeryStrong") s = "Strong";
      const sc = scores.splice.toFixed(3);
      const ev = [{ kind: "insilico", ref: "splice region model", detail: `score ${sc} at intronic offset ${sgn(off)}` }];
      if (d === "pathogenic") { crit.PP3.status = "met"; crit.PP3.strength = s!; crit.PP3.rationale = `The calibrated splice region predictor scores ${sc} (${s} pathogenic) at offset ${sgn(off)}.`; }
      else if (d === "benign") {
        const code = off >= 7 || off <= -21 ? "BP7" : "BP4";
        crit[code].status = "met"; crit[code].strength = code === "BP7" ? "Supporting" : s!;
        crit[code].rationale = `The calibrated splice region predictor scores ${sc}, predicting no splice impact at offset ${sgn(off)}.`;
      } else crit.PP3.rationale = crit.BP4.rationale = `Splice region score ${sc} lies in the indeterminate range.`;
      for (const k of ["PP3", "BP4", "BP7"]) crit[k].evidence = ev;
      return;
    }
    if (cons === "synonymous") {
      let nearEdge = false;
      const tx = this.b.transcripts[v.gene];
      if (tx && !nan(v.c_pos)) {
        const cp = Math.trunc(v.c_pos as number);
        const i = exonOf(tx, cp);
        if (i !== null) { const e = tx.exons[i]; nearEdge = e[3] - cp < 3 || cp - e[2] < 1; }
      }
      if (nearEdge) crit.BP7.rationale = "Synonymous change in the last three bases or first base of an exon, where splicing may be affected.";
      else { crit.BP7.status = "met"; crit.BP7.strength = "Supporting"; crit.BP7.rationale = "Synonymous change outside the exonic splice region; no splice impact is predicted."; }
      return;
    }
    if (cons === "intron" && Math.abs(off) > 50) {
      crit.BP7.status = "met"; crit.BP7.strength = "Supporting";
      crit.BP7.rationale = `Deep intronic position (offset ${sgn(off)}) beyond the splice region with no predicted splice impact.`;
      return;
    }
    crit.PP3.rationale = crit.BP4.rationale = "No calibrated predictor covers this variant type.";
  }

  private caseLevel(k: CaseEvidence, crit: Record<string, C>) {
    const need = (code: string, what: string) => { crit[code].status = "not_evaluated"; crit[code].rationale = `Needs ${what}.`; };
    const lvl = (p: number) => (p >= 4 ? "VeryStrong" : p >= 2 ? "Strong" : p >= 1 ? "Moderate" : "Supporting");
    const dn = k.de_novo || [];
    if (dn.length) {
      let pts = 0;
      for (const d of dn) { const sp = d.phenotype === "specific"; pts += d.confirmed ? (sp ? 2 : 1) : (sp ? 1 : 0.5); }
      const code = dn.some((d) => d.confirmed) ? "PS2" : "PM6";
      crit[code].status = "met"; crit[code].strength = lvl(pts);
      crit[code].rationale = `${dn.length} de novo observation(s) totalling ${fmtG(pts)} points under the ClinGen SVI de novo scale.`;
    } else { need("PS2", "parental testing (confirmed de novo)"); need("PM6", "de novo observations without confirmed parentage"); }
    const tr = k.in_trans || [];
    if (tr.length) {
      let pts = 0;
      for (const t of tr) {
        const partner = t.partner_class || "Pathogenic";
        if (t.homozygous) pts += 0.5;
        else if (t.phase === "trans") pts += partner === "Pathogenic" ? 1 : 0.5;
        else pts += partner === "Pathogenic" ? 0.5 : 0.25;
      }
      if (pts >= 0.5) { crit.PM3.status = "met"; crit.PM3.strength = lvl(pts); crit.PM3.rationale = `${tr.length} observation(s) with a pathogenic variant in a recessive gene, ${fmtG(pts)} points on the SVI PM3 scale.`; }
    } else need("PM3", "observations of the variant with a second pathogenic allele");
    const mei = k.segregation_meioses;
    if (mei) {
      const s = mei >= 7 ? "Strong" : mei >= 5 ? "Moderate" : mei >= 3 ? "Supporting" : null;
      if (s) { crit.PP1.status = "met"; crit.PP1.strength = s; crit.PP1.rationale = `Cosegregates through ${mei} informative meioses (likelihood about 1 in ${2 ** mei}).`; }
      else crit.PP1.rationale = `Only ${mei} informative meioses, below the three needed for Supporting.`;
    } else need("PP1", "family segregation data");
    if (k.nonsegregation) { crit.BS4.status = "met"; crit.BS4.strength = "Strong"; crit.BS4.rationale = "Affected relatives do not carry the variant."; }
    else need("BS4", "family segregation data");
    const fn = k.functional;
    if (fn) {
      const code = fn.direction === "damaging" ? "PS3" : "BS3";
      crit[code].status = "met"; crit[code].strength = fn.strength || "Strong";
      crit[code].rationale = `Functional evidence: ${fn.assay || "validated assay"} (${crit[code].strength}, per the ClinGen SVI OddsPath framework).`;
    } else { need("PS3", "a validated functional assay"); need("BS3", "a validated functional assay"); }
    if (k.case_control) {
      const { odds_ratio: orr = 0, ci_low: lo = 0 } = k.case_control;
      if (orr >= 5 && lo > 1) { crit.PS4.status = "met"; crit.PS4.strength = "Strong"; crit.PS4.rationale = `Odds ratio ${fmtG(orr)} with lower confidence limit ${fmtG(lo)}.`; }
    } else if (k.probands) {
      const n = Math.trunc(k.probands);
      crit.PS4.status = "met"; crit.PS4.strength = n >= 4 ? "Strong" : n >= 2 ? "Moderate" : "Supporting";
      crit.PS4.rationale = `Observed in ${n} unrelated affected probands and absent from controls (proband counting).`;
    } else need("PS4", "case control or proband counts");
    const ph = k.phenotype_match;
    if (ph !== undefined && ph !== null) {
      if ((ph.p_value ?? 1) < 0.01 && ph.specific) {
        crit.PP4.status = "met"; crit.PP4.strength = "Supporting";
        crit.PP4.rationale = `Patient phenotype matches the gene's disease profile (similarity ${ph.score.toFixed(2)}, empirical p ${Number(ph.p_value.toPrecision(3))}).`;
      } else crit.PP4.rationale = `Phenotype match not specific enough (empirical p ${Number((ph.p_value ?? 1).toPrecision(3))}).`;
    } else need("PP4", "patient phenotype terms");
    if (k.alternate_cause) { crit.BP5.status = "met"; crit.BP5.strength = "Supporting"; crit.BP5.rationale = "An alternate molecular cause of disease was found in this case."; }
    else need("BP5", "case level molecular findings");
    if (k.bp2) { crit.BP2.status = "met"; crit.BP2.strength = "Supporting"; crit.BP2.rationale = k.bp2; }
    else need("BP2", "phase data with a pathogenic variant");
  }

  private caps(crit: Record<string, C>, notes: string[]) {
    const R = this.rules;
    const pp3 = crit.PP3, pm1 = crit.PM1;
    if (pp3.status === "met" && pm1.status === "met") {
      const total = POINTS[pp3.strength] + POINTS[pm1.strength];
      if (total > R.pp3_pm1_cap_points) {
        const newP = POINTS[pm1.strength] - (total - R.pp3_pm1_cap_points);
        if (newP <= 0) { pm1.status = "not_met"; pm1.rationale += " Not counted: PP3 already carries the regional signal (PP3 plus PM1 capped at Strong)."; }
        else pm1.strength = newP === 1 ? "Supporting" : "Moderate";
        notes.push("PP3 and PM1 together are capped at Strong (4 points), following ClinGen SVI guidance.");
      }
    }
    if (crit.PVS1.status === "met" && crit.PM4.status === "met") { crit.PM4.status = "not_met"; notes.push("PM4 is not combined with PVS1."); }
    if (crit.PVS1.status === "met" && crit.PP3.status === "met") { crit.PP3.status = "not_met"; notes.push("PP3 is not combined with PVS1 for the same predicted effect."); }
    if (crit.BA1.status === "met") crit.BS1.status = "not_met";
  }
}

export { isNum };
