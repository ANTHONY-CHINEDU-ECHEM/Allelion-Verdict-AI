// Predictor features. Mirrors allelion/models/features.py exactly (including NaN behaviour).
import type { Bundle, Variant } from "./data";

const AROMATIC = new Set(["F", "W", "Y", "H"]);
const CHARGE: Record<string, number> = { D: -1, E: -1, K: 1, R: 1, H: 0.1 };
const TRANSITIONS = new Set(["AG", "GA", "CT", "TC"]);
const pymod = (a: number, n: number) => ((a % n) + n) % n;

export function geneFeatures(b: Bundle, gene: string): Record<string, number> {
  const g = b.genes[gene];
  if (!g) {
    return { pLI: NaN, loftool: NaN, haploinsufficient: NaN, mouse_essential: NaN, dominant: NaN, recessive: NaN, x_linked: NaN, n_diseases: NaN, log_protein_len: NaN };
  }
  const plen = g.plen ?? 500;
  return {
    pLI: g.pli ?? NaN, loftool: g.lt ?? NaN, haploinsufficient: +g.hi, mouse_essential: +g.ess, dominant: +g.dom, recessive: +g.rec,
    x_linked: +g.xl, n_diseases: g.dis.length, log_protein_len: Math.log10(Math.max(plen, 10)),
  };
}

export function grantham(b: Bundle, a: string, c: string): number {
  const P = b.meta.grantham_props as Record<string, number[]>;
  if (!P[a] || !P[c]) return NaN;
  const [c1, p1, v1] = P[a];
  const [c2, p2, v2] = P[c];
  return (50.723 * Math.sqrt(1.833 * (c1 - c2) ** 2 + 0.1018 * (p1 - p2) ** 2 + 0.000399 * (v1 - v2) ** 2)) / 1.0;
}

function nucChange(hgvs: string | null | undefined): [string, string] | null {
  const m = /([ACGT])>([ACGT])$/.exec(hgvs || "");
  return m ? [m[1], m[2]] : null;
}

export function missenseFeatures(b: Bundle, v: Variant): Record<string, number> {
  const ref = v.p_ref || "X";
  const alt = v.p_alt || "X";
  const KD = b.meta.kyte_doolittle as Record<string, number>;
  const P = b.meta.grantham_props as Record<string, number[]>;
  const f: Record<string, number> = {};
  const bl = b.meta.blosum62[`${ref}${alt}`];
  f.blosum62 = bl === undefined ? NaN : bl;
  f.grantham = grantham(b, ref, alt);
  f.d_hydro = (KD[alt] ?? NaN) - (KD[ref] ?? NaN);
  f.abs_d_hydro = Math.abs(f.d_hydro);
  f.d_volume = (P[alt] ? P[alt][2] : NaN) - (P[ref] ? P[ref][2] : NaN);
  f.abs_d_volume = Math.abs(f.d_volume);
  f.d_charge = (CHARGE[alt] ?? 0) - (CHARGE[ref] ?? 0);
  f.charge_change = Math.abs(f.d_charge) >= 0.9 ? 1 : 0;
  f.to_pro = +(alt === "P"); f.from_pro = +(ref === "P"); f.to_gly = +(alt === "G"); f.from_gly = +(ref === "G");
  f.to_cys = +(alt === "C"); f.from_cys = +(ref === "C");
  f.aromatic_change = +(AROMATIC.has(ref) !== AROMATIC.has(alt));
  const gf = geneFeatures(b, v.gene);
  const plen = Math.pow(10, gf.log_protein_len);
  const pos = v.p_pos ?? NaN;
  f.rel_pos = Math.min(Math.max(pos / plen, 0), 1);
  if (Number.isNaN(pos / plen)) f.rel_pos = NaN;
  f.log_pos = Math.log10(Math.max(pos, 1));
  if (Number.isNaN(pos)) f.log_pos = NaN;
  const dc = plen - pos;
  f.log_dist_c = Number.isNaN(dc) ? NaN : Math.log10(Math.max(dc, 1));
  f.codon_pos = v.c_pos === null || v.c_pos === undefined ? NaN : pymod(v.c_pos - 1, 3);
  const nc = nucChange(v.hgvs_c);
  f.transition = nc ? +TRANSITIONS.has(nc[0] + nc[1]) : NaN;
  for (const a of b.meta.aas as string[]) { f[`ref_${a}`] = +(ref === a); f[`alt_${a}`] = +(alt === a); }
  return { ...f, ...gf };
}

export function spliceFeatures(b: Bundle, v: Variant): Record<string, number> {
  const off = v.c_offset ?? 0;
  const f: Record<string, number> = {};
  f.offset = off; f.abs_offset = Math.abs(off); f.donor_side = +(off > 0); f.log_abs_offset = Math.log10(Math.max(Math.abs(off), 1));
  const nc = nucChange(v.hgvs_c);
  f.transition = nc ? +TRANSITIONS.has(nc[0] + nc[1]) : 0;
  for (const x of ["G", "A", "C", "T"]) { f[`ref_${x}`] = +(nc !== null && nc[0] === x); f[`alt_${x}`] = +(nc !== null && nc[1] === x); }
  f.is_sub = +(v.c_kind === "sub"); f.is_indel = +(v.c_kind !== "sub");
  f.indel_len = v.length_change === null || v.length_change === undefined || Number.isNaN(v.length_change) ? 0 : Math.abs(v.length_change);
  return { ...f, ...geneFeatures(b, v.gene) };
}
