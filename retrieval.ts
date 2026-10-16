// Hybrid retrieval in the browser. Mirrors allelion/retrieval (query.py, bm25.py, dense.py, retriever.py).
import type { Bundle } from "./data";
import { parseC, parseP } from "./hgvs";
import { raw, vector } from "./lgbm";
import { tokenize } from "./text";

const K1 = 1.2, B = 0.75, RRF_K = 60;
const KINDS = ["guidance", "gene", "disease", "phenotype", "literature"];
const CODE_RE = /\b(PVS1|PS[1-4]|PM[1-6]|PP[1-5]|BA1|BS[1-4]|BP[1-7])\b/gi;
const HGVS_C_RE = /(?:[A-Z]{2}_\d+(?:\.\d+)?:)?c\.[-*]?\d+(?:[+-]\d+)?(?:_[-*]?\d+(?:[+-]\d+)?)?(?:[ACGT]+>[ACGT]+|del[ACGT]*ins[ACGT]+|del[ACGT]*|dup[ACGT]*|ins[ACGT]+)/g;
const AA = "Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val";
const HGVS_P_RE = new RegExp(`(?:p\\.)?\\(?(?:${AA}|Ter)\\d+(?:${AA}|Ter|=|fs\\w*|del|dup)\\)?`, "g");
const NOT_GENES = new Set(["DNA", "RNA", "SVI", "ACMG", "AMP", "PMID", "HPO", "OMIM", "UTR"]);
const INTENT_WORDS: Record<string, string[]> = {
  phenotype: ["define", "definition", "what does", "mean", "what is a", "what is an", "term"],
  disease: ["features", "phenotypes", "clinical", "characterise", "characterize", "symptoms", "syndrome", "disease", "inheritance of"],
  literature: ["study", "report", "paper", "abstract", "published", "described", "reported", "found", "patients", "family", "families"],
  guidance: ["criterion", "criteria", "guideline", "points", "strength", "should", "apply", "applied", "svi", "clingen", "acmg", "downgrade", "combine", "threshold", "rule", "recommend"],
  gene: ["gene", "constraint", "pli", "loftool", "mechanism", "haploinsufficiency", "spectrum", "exon", "evidence pool"],
};

export interface QueryInfo { text: string; genes: string[]; codes: string[]; hgvs_c: string[]; hgvs_p: string[]; pmids: string[]; hpo: string[]; diseases: string[]; intent: Record<string, number> }

export class Retrieval {
  n: number;
  vocab = new Map<string, number>();
  postings: { doc: Int32Array; w: Float32Array }[] = [];
  titleToks: Set<string>[] = [];
  fullToks: Set<string>[] = [];
  lengths: Float64Array;
  sifIdx = new Map<string, number>();
  Psif: Float32Array; Pdual: Float32Array;
  bySource = new Map<string, number[]>();
  byCode = new Map<string, number[]>();
  byGene = new Map<string, number[]>();
  hpoNames: [string, string][] = [];
  disNames: [string, string[]][] = [];

  constructor(public b: Bundle) {
    const P = b.passages;
    this.n = P.length;
    const toks = P.map((p) => tokenize(`${p[2]}. ${p[3]}`));
    const df = new Map<string, number>();
    toks.forEach((ts) => new Set(ts).forEach((w) => df.set(w, (df.get(w) || 0) + 1)));
    [...df.keys()].sort().forEach((w, i) => this.vocab.set(w, i));
    const idf = new Float64Array(this.vocab.size);
    for (const [w, i] of this.vocab) { const d = df.get(w)!; idf[i] = Math.log(1 + (this.n - d + 0.5) / (d + 0.5)); }
    const avg = toks.reduce((a, t) => a + t.length, 0) / this.n || 1;
    const lists: { d: number[]; w: number[] }[] = Array.from({ length: this.vocab.size }, () => ({ d: [], w: [] }));
    toks.forEach((ts, i) => {
      const tf = new Map<string, number>();
      ts.forEach((w) => tf.set(w, (tf.get(w) || 0) + 1));
      const norm = K1 * (1 - B + (B * ts.length) / avg);
      for (const [w, f] of tf) { const j = this.vocab.get(w)!; lists[j].d.push(i); lists[j].w.push((idf[j] * f * (K1 + 1)) / (f + norm)); }
    });
    this.postings = lists.map((l) => ({ doc: Int32Array.from(l.d), w: Float32Array.from(l.w) }));
    this.titleToks = P.map((p) => new Set(tokenize(p[2])));
    this.fullToks = P.map((p) => new Set(tokenize(`${p[2]} ${p[3]}`)));
    this.lengths = Float64Array.from(P.map((p) => Math.log1p(p[3].split(/\s+/).filter(Boolean).length)));
    b.sif.words.forEach((w, i) => this.sifIdx.set(w, i));
    const dim = b.sif.dim;
    this.Psif = new Float32Array(this.n * dim);
    P.forEach((p, i) => { const e = this.sifEmbed(`${p[2]}. ${p[3]}`); this.Psif.set(e, i * dim); });
    this.Pdual = new Float32Array(this.n * b.sif.Wp[0].length);
    for (let i = 0; i < this.n; i++) this.Pdual.set(this.project(this.Psif.subarray(i * dim, (i + 1) * dim), b.sif.Wp), i * b.sif.Wp[0].length);
    P.forEach((p, i) => {
      const push = (m: Map<string, number[]>, k: string) => { const a = m.get(k); if (a) a.push(i); else m.set(k, [i]); };
      push(this.bySource, p[0]);
      p[6].forEach((c) => push(this.byCode, c));
      p[5].forEach((g) => push(this.byGene, g));
    });
    for (const [id, [name]] of Object.entries(b.hpo)) { const n = name.toLowerCase().trim(); if (n.length >= 6) this.hpoNames.push([n, id]); }
    this.hpoNames.sort((a, c) => c[0].length - a[0].length);
    const dn = new Map<string, string[]>();
    for (const [id, [name]] of Object.entries(b.diseases)) {
      const n = name.toLowerCase().replace(/\s+/g, " ").trim();
      if (n.length >= 6 && n.includes(" ")) { const a = dn.get(n); if (a) a.push(id); else dn.set(n, [id]); }
    }
    this.disNames = [...dn.entries()].sort((a, c) => c[0].length - a[0].length);
  }

  sifEmbed(text: string): Float32Array {
    const { dim, weights, pc } = this.b.sif;
    const V = this.b.sifVectors;
    const out = new Float32Array(dim);
    const ids = tokenize(text).map((w) => this.sifIdx.get(w)).filter((x): x is number => x !== undefined);
    if (ids.length) {
      for (const id of ids) for (let d = 0; d < dim; d++) out[d] += V[id * dim + d] * weights[id];
      for (let d = 0; d < dim; d++) out[d] /= ids.length;
    }
    let dot = 0;
    for (let d = 0; d < dim; d++) dot += out[d] * pc[d];
    for (let d = 0; d < dim; d++) out[d] -= dot * pc[d];
    let nrm = 0;
    for (let d = 0; d < dim; d++) nrm += out[d] * out[d];
    nrm = Math.max(Math.sqrt(nrm), 1e-8);
    for (let d = 0; d < dim; d++) out[d] /= nrm;
    return out;
  }

  project(x: Float32Array, W: number[][]): Float32Array {
    const out = new Float32Array(W[0].length);
    for (let i = 0; i < x.length; i++) { const xi = x[i]; if (!xi) continue; const row = W[i]; for (let j = 0; j < out.length; j++) out[j] += xi * row[j]; }
    let n = 0;
    for (let j = 0; j < out.length; j++) n += out[j] * out[j];
    n = Math.max(Math.sqrt(n), 1e-8);
    for (let j = 0; j < out.length; j++) out[j] /= n;
    return out;
  }

  bm25Scores(q: string): Float64Array {
    const s = new Float64Array(this.n);
    const seen = new Set<number>();
    for (const w of tokenize(q)) {
      const j = this.vocab.get(w);
      if (j === undefined || seen.has(j)) continue;
      seen.add(j);
      const p = this.postings[j];
      for (let k = 0; k < p.doc.length; k++) s[p.doc[k]] += p.w[k];
    }
    return s;
  }

  static topK(s: ArrayLike<number>, k: number, mask?: Uint8Array | null, positiveOnly = true): [number, number][] {
    const idx: number[] = [];
    for (let i = 0; i < s.length; i++) if ((!mask || mask[i]) && (!positiveOnly || s[i] > 0)) idx.push(i);
    idx.sort((a, b) => s[b] - s[a] || a - b);
    return idx.slice(0, k).map((i) => [i, s[i]]);
  }

  understand(q: string): QueryInfo {
    const info: QueryInfo = { text: q, genes: [], codes: [], hgvs_c: [], hgvs_p: [], pmids: [], hpo: [], diseases: [], intent: {} };
    const bare = q.replace(/\[[A-Z_0-9]+\]/g, " ");
    for (const m of bare.matchAll(/\b[A-Z][A-Z0-9]{1,9}\b/g)) {
      const w = m[0];
      if (this.b.genes[w] && !info.genes.includes(w) && !/^(PVS1|PS[1-4]|PM[1-6]|PP[1-5]|BA1|BS[1-4]|BP[1-7])$/i.test(w) && !NOT_GENES.has(w)) info.genes.push(w);
    }
    info.codes = [...new Set([...q.matchAll(CODE_RE)].map((m) => m[1].toUpperCase()))].sort();
    info.hgvs_c = [...q.matchAll(HGVS_C_RE)].map((m) => m[0]).filter((m) => parseC(m));
    info.hgvs_p = [...q.matchAll(HGVS_P_RE)].map((m) => m[0]).filter((m) => parseP(m.startsWith("p.") ? m : "p." + m.replace(/^\(|\)$/g, "")));
    info.pmids = [...q.matchAll(/\bPMID[:\s]*(\d{6,9})\b/gi)].map((m) => m[1]);
    info.hpo = [...q.matchAll(/\bHP:\d{7}\b/g)].map((m) => m[0]);
    info.diseases = [...q.matchAll(/\b(?:OMIM|ORPHA):\d+\b/g)].map((m) => m[0]);
    const ql = (" " + q.toLowerCase().replace(/[^a-z0-9 ]+/g, " ") + " ").replace(/\s+/g, " ");
    for (const [name, hid] of this.hpoNames) {
      if (ql.includes(` ${name} `) && !info.hpo.includes(hid)) { info.hpo.push(hid); if (info.hpo.length >= 3) break; }
    }
    for (const [name, ids] of this.disNames) {
      const key = name.replace(/[^a-z0-9 ]+/g, " ").replace(/\s+/g, " ").trim();
      if (key.length >= 8 && ql.includes(` ${key} `)) {
        for (const d of ids) if (!info.diseases.includes(d)) info.diseases.push(d);
        if (info.diseases.length >= 4) break;
      }
    }
    const low = q.toLowerCase();
    for (const [k, words] of Object.entries(INTENT_WORDS)) info.intent[k] = words.filter((w) => low.includes(w)).length;
    if (info.codes.length) info.intent.guidance += 2;
    if (info.genes.length) info.intent.gene += 1;
    return info;
  }

  entityMask(qi: QueryInfo): Uint8Array | null {
    const idx = new Set<number>();
    const add = (a?: number[]) => a?.forEach((i) => idx.add(i));
    qi.genes.forEach((g) => { add(this.bySource.get(`gene:${g}`)); add(this.byGene.get(g)); });
    qi.codes.forEach((c) => add(this.byCode.get(c)));
    qi.diseases.forEach((d) => add(this.bySource.get(`disease:${d}`)));
    qi.hpo.forEach((h) => add(this.bySource.get(`hpo:${h}`)));
    qi.pmids.forEach((p) => ["tmvar", "seth", "mf", "osiris"].forEach((k) => add(this.bySource.get(`lit:${k}:${p}`))));
    if (!idx.size) return null;
    const m = new Uint8Array(this.n);
    idx.forEach((i) => { m[i] = 1; });
    return m;
  }

  kindMask(qi: QueryInfo): Uint8Array | null {
    let best = "", bv = -1;
    for (const k of ["phenotype", "disease", "literature", "guidance", "gene"]) if (qi.intent[k] > bv) { bv = qi.intent[k]; best = k; }
    if (bv <= 0) return null;
    return Uint8Array.from(this.b.passages.map((p) => (p[1] === best ? 1 : 0)));
  }

  candidates(q: string, qi: QueryInfo, depth = 100) {
    const bm = this.bm25Scores(q);
    const lists: Record<string, [number, number][]> = {};
    lists.bm25 = Retrieval.topK(bm, depth);
    const qs = this.sifEmbed(q);
    const dim = this.b.sif.dim, od = this.b.sif.Wp[0].length;
    const sSif = new Float64Array(this.n), sDual = new Float64Array(this.n);
    const qd = this.project(qs, this.b.sif.Wq);
    for (let i = 0; i < this.n; i++) {
      let a = 0; for (let d = 0; d < dim; d++) a += this.Psif[i * dim + d] * qs[d];
      let c = 0; for (let d = 0; d < od; d++) c += this.Pdual[i * od + d] * qd[d];
      sSif[i] = a; sDual[i] = c;
    }
    lists.sif = Retrieval.topK(sSif, depth, null, false);
    lists.dual = Retrieval.topK(sDual, depth, null, false);
    const em = this.entityMask(qi);
    lists.entity = em ? Retrieval.topK(bm, 50, em) : [];
    if (em && !lists.entity.length) { const ids: [number, number][] = []; em.forEach((v, i) => { if (v && ids.length < 50) ids.push([i, 0]); }); lists.entity = ids; }
    const km = this.kindMask(qi);
    lists.kind = km ? Retrieval.topK(bm, 50, km) : [];
    return { lists, bm, sSif, sDual };
  }

  static rrf(lists: Record<string, [number, number][]>, names: string[]): [number, number][] {
    const sc = new Map<number, number>();
    for (const n of names) (lists[n] || []).forEach(([i], r) => sc.set(i, (sc.get(i) || 0) + 1 / (RRF_K + r + 1)));
    return [...sc.entries()].sort((a, b) => b[1] - a[1] || a[0] - b[0]);
  }

  features(q: string, qi: QueryInfo, c: ReturnType<Retrieval["candidates"]>, cand: number[]): Record<string, number>[] {
    const ranks: Record<string, Map<number, number>> = {};
    for (const [n, l] of Object.entries(c.lists)) ranks[n] = new Map(l.map(([i], r) => [i, r]));
    let mx = 1e-6;
    for (let i = 0; i < c.bm.length; i++) if (c.bm[i] > mx) mx = c.bm[i];
    const fused = Retrieval.rrf(c.lists, ["bm25", "sif", "dual", "entity", "kind"]);
    const fusedScore = new Map(fused);
    const fusedRank = new Map(fused.map(([i], r) => [i, r]));
    const qt = new Set(tokenize(q));
    const P = this.b.passages;
    return cand.map((i) => {
      const src = P[i][0], kind = P[i][1];
      const tt = this.fullToks[i];
      let inter = 0; qt.forEach((t) => { if (tt.has(t)) inter++; });
      let tint = 0; qt.forEach((t) => { if (this.titleToks[i].has(t)) tint++; });
      const f: Record<string, number> = {
        bm25: c.bm[i], bm25_rank: ranks.bm25.get(i) ?? 200, bm25_norm: c.bm[i] / mx, sif: c.sSif[i], sif_rank: ranks.sif.get(i) ?? 200,
        dual: c.sDual[i], dual_rank: ranks.dual.get(i) ?? 200, entity_rank: ranks.entity.get(i) ?? 200, kind_rank: ranks.kind.get(i) ?? 200,
        rrf: fusedScore.get(i) ?? 0, rrf_rank: fusedRank.get(i) ?? 500, n_generators: Object.values(ranks).filter((m) => m.has(i)).length,
        q_coverage: inter / Math.max(qt.size, 1), title_coverage: tint / Math.max(qt.size, 1), len_log: this.lengths[i],
        gene_in_source: +qi.genes.some((g) => src === `gene:${g}`), gene_in_genes: +qi.genes.some((g) => P[i][5].includes(g)),
        code_match: +qi.codes.some((cc) => P[i][6].includes(cc)), disease_match: +qi.diseases.some((d) => src === `disease:${d}`),
        hpo_match: +qi.hpo.some((h) => src === `hpo:${h}`), pmid_match: +qi.pmids.some((p) => src.endsWith(`:${p}`)),
        n_query_genes: qi.genes.length, n_query_codes: qi.codes.length,
      };
      for (const k of KINDS) { f[`kind_${k}`] = +(kind === k); f[`intent_${k}`] = qi.intent[k] || 0; f[`intent_x_kind_${k}`] = (qi.intent[k] || 0) * +(kind === k); }
      return f;
    });
  }

  search(q: string, k = 8, method = "full", maxPerSource = 2): { query: QueryInfo; ids: number[]; ms: number } {
    const t0 = performance.now();
    const qi = this.understand(q);
    const c = this.candidates(q, qi);
    let ranked: number[];
    if (["bm25", "sif", "dual", "entity"].includes(method)) {
      ranked = c.lists[method].map(([i]) => i);
      if (method === "entity" && !ranked.length) ranked = c.lists.bm25.map(([i]) => i);
    } else if (method === "fused") ranked = Retrieval.rrf(c.lists, ["bm25", "sif", "dual", "entity", "kind"]).map(([i]) => i);
    else {
      const cand = Retrieval.rrf(c.lists, ["bm25", "sif", "dual", "entity", "kind"]).slice(0, 150).map(([i]) => i);
      const feats = this.features(q, qi, c, cand);
      const scores = feats.map((f) => raw(this.b.reranker, vector(this.b.reranker, f))[0]);
      ranked = cand.map((i, j) => [i, scores[j]] as [number, number]).sort((a, b) => b[1] - a[1]).map(([i]) => i);
    }
    const out: number[] = [];
    const seen = new Map<string, number>();
    for (const i of ranked) {
      const src = this.b.passages[i][0];
      if ((seen.get(src) || 0) >= maxPerSource) continue;
      seen.set(src, (seen.get(src) || 0) + 1);
      out.push(i);
      if (out.length >= k) break;
    }
    return { query: qi, ids: out, ms: Math.round((performance.now() - t0) * 10) / 10 };
  }
}
