// Anchored exon models in linear c. coordinates. Mirrors allelion/kb/transcripts.py.
import type { Transcript } from "./data";

// exon row: [g_start, g_end, lc_start, lc_end, approx]
export function exonOf(tx: Transcript, lc: number): number | null {
  const ex = tx.exons;
  let lo = 0, hi = ex.length - 1, i = -1;
  while (lo <= hi) { const mid = (lo + hi) >> 1; if (ex[mid][2] <= lc) { i = mid; lo = mid + 1; } else hi = mid - 1; }
  if (i >= 0 && i < ex.length && ex[i][2] <= lc && lc <= ex[i][3]) return i;
  return null;
}

export function nearestExon(tx: Transcript, lc: number): number {
  let best = 0, bd = Infinity;
  tx.exons.forEach((e, i) => { const d = Math.min(Math.abs(lc - e[2]), Math.abs(lc - e[3])); if (d < bd) { bd = d; best = i; } });
  return best;
}

export interface ExonFrame { exon: number; lc_start: number; lc_end: number; length: number; coding: number; in_frame_skip: boolean }

export function exonFrames(tx: Transcript): ExonFrame[] {
  return tx.exons.map((e, i) => {
    const lo = Math.max(e[2], 1);
    const hi = Math.min(e[3], tx.cds ?? e[3]);
    const coding = Math.max(0, hi - lo + 1);
    return { exon: i + 1, lc_start: e[2], lc_end: e[3], length: e[3] - e[2] + 1, coding, in_frame_skip: coding > 0 && coding % 3 === 0 };
  });
}

export function nmdEscape(tx: Transcript, ptcLc: number): [boolean, string] {
  const n = tx.exons.length;
  if (n === 0) return [false, "no exon model"];
  let i = exonOf(tx, ptcLc);
  if (i === null) i = nearestExon(tx, ptcLc);
  if (i === n - 1) return [true, `the premature stop falls in the last exon (exon ${n})`];
  if (i === n - 2 && tx.exons[i][3] - ptcLc < 50) return [true, `the premature stop falls within the last 50 bases of the penultimate exon (exon ${n - 1})`];
  return [false, `the premature stop falls in exon ${i + 1} of ${n}, so the transcript is predicted to undergo decay`];
}
