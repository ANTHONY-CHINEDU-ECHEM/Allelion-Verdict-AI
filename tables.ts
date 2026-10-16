// Evidence pool queries with self exclusion. Mirrors allelion/acmg/tables.py.
import type { Bundle } from "./data";

export const PLP = new Set([0, 1]); // class codes: 0 P, 1 LP, 2 VUS, 3 LB, 4 B
export const BLB = new Set([3, 4]);

function own(b: Bundle, exclude: string | null) {
  if (!exclude) return null;
  return b.poolIndex[exclude] ?? null; // [gene, group, classCode, pos]
}

export function sameChange(b: Bundle, gene: string, change: string, exclude: string | null, hgvsCShort: string | null) {
  return (b.tables.missense[gene] || []).filter((r) => r[1] === change && PLP.has(r[2]) && r[3] !== exclude && r[4] !== hgvsCShort);
}

export function sameResidue(b: Bundle, gene: string, pos: number, exclude: string | null) {
  return (b.tables.missense[gene] || []).filter((r) => r[0] === pos && r[3] !== exclude);
}

export function sameSpliceSite(b: Bundle, gene: string, cpos: number, offset: number, exclude: string | null, hgvsCShort: string | null) {
  return (b.tables.splice[gene] || []).filter((r) => r[0] === cpos && r[1] === offset && r[4] !== exclude && r[2] !== hgvsCShort);
}

export function density(b: Bundle, gene: string, pos: number, window: number, exclude: string | null): [number, number] {
  let p = 0, q = 0;
  for (const r of b.tables.missense[gene] || []) {
    if (r[0] < pos - window || r[0] > pos + window) continue;
    if (PLP.has(r[2])) p++;
    else if (BLB.has(r[2])) q++;
  }
  const o = own(b, exclude);
  if (o && o[1] === "missense" && o[3] !== null && Math.abs(o[3] - pos) <= window) {
    if (PLP.has(o[2])) p--;
    else if (BLB.has(o[2])) q--;
  }
  return [Math.max(p, 0), Math.max(q, 0)];
}

export function truncatingDownstream(b: Bundle, gene: string, pos: number, exclude: string | null): number {
  const arr = b.tables.trunc[gene];
  if (!arr) return 0;
  let n = arr.filter((x) => x > pos).length;
  const o = own(b, exclude);
  if (o && o[1] === "truncating" && PLP.has(o[2]) && o[3] !== null && o[3] > pos) n -= 1;
  return Math.max(n, 0);
}

export function benignInframeNear(b: Bundle, gene: string, pos: number, window: number, exclude: string | null): number {
  const arr = b.tables.inframe_blb[gene] || [];
  let n = arr.filter((x) => x >= pos - window && x <= pos + window).length;
  const o = own(b, exclude);
  if (o && o[1] === "inframe" && BLB.has(o[2]) && o[3] !== null && Math.abs(o[3] - pos) <= window) n -= 1;
  return Math.max(n, 0);
}

export function geneStats(b: Bundle, gene: string, exclude: string | null): Record<string, number> {
  const s = { ...(b.tables.gene_stats[gene] || {}) };
  const o = own(b, exclude);
  if (o && o[0] === gene) {
    const bucket = PLP.has(o[2]) ? "plp" : BLB.has(o[2]) ? "blb" : "vus";
    const key = `${o[1]}_${bucket}`;
    if ((s[key] || 0) > 0) s[key] -= 1;
  }
  return s;
}

export function lofMechanism(stats: Record<string, number>, g: any, rules: Record<string, number>): [boolean, string] {
  const n = stats.truncating_plp || 0;
  const mp = stats.missense_plp || 0;
  if (g && g.hi) return [true, "the gene has curated evidence of haploinsufficiency"];
  const share = n + mp ? n / (n + mp) : 0;
  if (mp >= (rules.pvs1_missense_dominant_min ?? 20) && share < (rules.pvs1_missense_dominant_frac ?? 0.1)) {
    return [false, `pathogenic variation in this gene is predominantly missense (${mp} missense against ${n} truncating), which points to gain of function or dominant negative disease rather than loss of function`];
  }
  if (n >= (rules.pvs1_min_trunc_plp ?? 3)) return [true, `${n} truncating variants in this gene are classified pathogenic or likely pathogenic in the evidence pool`];
  if (g && g.rec && n >= 1) return [true, `loss of function is the mechanism for the recessive disorder (${n} pathogenic truncating variant in the pool)`];
  return [false, `loss of function is not established for this gene (${n} pathogenic truncating variants in the pool, no curated haploinsufficiency)`];
}
