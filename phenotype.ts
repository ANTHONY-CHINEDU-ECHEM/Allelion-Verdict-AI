// Phenotype specificity for PP4. Mirrors allelion/acmg/phenotype.py (same background of 300 genes).
import type { Bundle } from "./data";

export class PhenotypeMatcher {
  anc = new Map<string, Set<string>>();
  background: string[];
  disGeneCount = new Map<string, number>();
  constructor(public b: Bundle) {
    const genes = Object.keys(b.genes).filter((g) => b.genes[g].dis.some((d) => (b.diseases[d]?.[1] || []).length)).sort();
    const step = Math.max(1, Math.floor(genes.length / 300));
    this.background = genes.filter((_, i) => i % step === 0).slice(0, 300);
    for (const g of Object.values(b.genes)) for (const d of g.dis) this.disGeneCount.set(d, (this.disGeneCount.get(d) || 0) + 1);
  }

  ancestors(t: string): Set<string> {
    const hit = this.anc.get(t);
    if (hit) return hit;
    const out = new Set<string>([t]);
    for (const p of this.b.hpo[t]?.[1] || []) this.ancestors(p).forEach((x) => out.add(x));
    this.anc.set(t, out);
    return out;
  }

  ic(t: string): number { return this.b.hpo[t]?.[2] ?? 0; }

  resnik(a: string, c: string): number {
    const A = this.ancestors(a), C = this.ancestors(c);
    let best = 0;
    A.forEach((t) => { if (C.has(t)) best = Math.max(best, this.ic(t)); });
    return best;
  }

  bma(q: string[], t: string[]): number {
    if (!q.length || !t.length) return 0;
    let s = 0;
    for (const x of q) { let m = 0; for (const y of t) m = Math.max(m, this.resnik(x, y)); s += m; }
    return s / q.length;
  }

  geneScore(gene: string, terms: string[]): [number, string | null] {
    let best = 0, bd: string | null = null;
    for (const d of this.b.genes[gene]?.dis || []) {
      const dt = this.b.diseases[d]?.[1];
      if (!dt || !dt.length) continue;
      const s = this.bma(terms, dt);
      if (s > best) { best = s; bd = d; }
    }
    return [best, bd];
  }

  match(gene: string, terms: string[]) {
    const ts = terms.filter((t) => this.b.hpo[t]);
    const [score, d] = this.geneScore(gene, ts);
    const bg = this.background.map((g) => this.geneScore(g, ts)[0]);
    const p = (1 + bg.filter((s) => s >= score).length) / (1 + bg.length);
    return { score: Math.round(score * 1e4) / 1e4, p_value: Math.round(p * 1e4) / 1e4, best_disease: d, specific: !!d && (this.disGeneCount.get(d) || 0) <= 2, n_terms: ts.length };
  }

  rankGenes(terms: string[], limit = 25) {
    const ts = terms.filter((t) => this.b.hpo[t]);
    const scored = Object.keys(this.b.genes).filter((g) => this.b.genes[g].dis.length).map((g) => { const [s, d] = this.geneScore(g, ts); return { gene: g, score: s, disease: d }; });
    return scored.filter((x) => x.score > 0).sort((a, b) => b.score - a.score).slice(0, limit);
  }
}
