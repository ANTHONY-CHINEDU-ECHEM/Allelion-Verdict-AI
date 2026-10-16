// Bundle loading and shared types for the in browser engine.
import type { Model } from "./lgbm";
import { parseC, parseP } from "./hgvs";

export interface Variant {
  variation_id: string | null; gene: string; hgvs_c: string; hgvs_p: string | null; consequence: string;
  c_kind: string | null; c_region: string | null; c_pos: number | null; c_end: number | null; c_offset: number; c_end_offset: number;
  splice_distance: number; length_change: number | null; p_kind: string | null; p_ref: string | null; p_pos: number | null;
  p_alt: string | null; p_end: number | null; protein_change: string; gn_AC: number | null; gn_AN: number | null; gn_grpmax: number | null;
  gn_grpmax_2000: number | null; gn_faf95: number | null; gn_Hom: number | null; gn_Hemi: number | null; clinvar_class?: string | null;
  stars?: number; conflicted?: number; key?: string | null; set?: string; trait?: string | null; population_known?: boolean;
}

export interface Gene {
  pli: number | null; lt: number | null; hi: boolean; ess: boolean; dom: boolean; rec: boolean; xl: boolean; adult: boolean;
  dis: string[]; inh: string[]; plen: number | null; chrom: string | null; cv: Record<string, number>; total: number; mech: string;
  traits: string[]; onset: string[]; prev: string[]; iv: boolean[];
}

export interface Transcript { strand: string; chrom: string; cds: number | null; agree: number; n: number; exons: number[][] }

export interface Tables {
  missense: Record<string, [number, string, number, string, string][]>;
  splice: Record<string, [number, number, string, number, string][]>;
  trunc: Record<string, number[]>;
  inframe_blb: Record<string, number[]>;
  gene_stats: Record<string, Record<string, number>>;
}

export interface Calibration { pathogenic_thresholds: Record<string, number | null>; benign_thresholds: Record<string, number | null>; grid: number[]; posterior: number[]; lower: number[]; upper: number[]; n: number; n_pathogenic: number; prior: number; lr: Record<string, number> }

export interface Bundle {
  meta: any;
  variants: Variant[];
  byId: Map<string, Variant>;
  genes: Record<string, Gene>;
  transcripts: Record<string, Transcript>;
  tables: Tables;
  poolIndex: Record<string, [string, string, number, number | null]>;
  missense: Model; splice: Model; metaModel: Model; reranker: Model;
  missenseCal: Calibration; spliceCal: Calibration;
  verifier: { features: string[]; coef: number[]; intercept: number };
  passages: [string, string, string, string, string, string[], string[], string[]][];
  sif: { words: string[]; weights: number[]; pc: number[]; dim: number; Wq: number[][]; Wp: number[][] };
  sifVectors: Float32Array;
  hpo: Record<string, [string, string[], number]>;
  diseases: Record<string, [string, string[], string[]]>;
  literature: { docs: string[][]; links: any[][]; mentions: Record<string, [number, number, string, string][]> };
  submissions: Record<string, string[][]>;
}

let cache: Promise<Bundle> | null = null;

/** Recreate the parsed HGVS fields of a compact bundle record (mirrors hydrate() in allelion/export/bundle.py). */
export function hydrate(o: any): Variant {
  const c = parseC(o.hgvs_c), p = parseP(o.hgvs_p);
  Object.assign(o, {
    c_kind: c ? c.kind : null, c_region: c ? c.start.region : null, c_pos: c ? c.start.base : null, c_end: c ? c.end.base : null,
    c_offset: c ? c.start.offset : 0, c_end_offset: c ? c.end.offset : 0, splice_distance: c ? c.spliceDistance : 0, length_change: c ? c.lengthChange : null,
    p_kind: p ? p.kind : null, p_ref: p ? p.refAa : null, p_pos: p && p.pos !== null ? p.pos : null, p_alt: p ? p.altAa : null,
    p_end: p && p.endPos !== null ? p.endPos : null, protein_change: p ? p.short : "",
  });
  return o as Variant;
}

async function gunzip(r: Response | Blob): Promise<ArrayBuffer> {
  const ds = new (globalThis as any).DecompressionStream("gzip");
  const body = r instanceof Blob ? r.stream() : r.body!;
  return await new Response(body.pipeThrough(ds)).arrayBuffer();
}

/** Fetch one gzip file of the bundle. Hosts that only serve text receive the
 * same gzip bytes base64 encoded (manifest "encoding": "gzip+base64"). */
async function fetchGz(base: string, name: string, encoding: string): Promise<ArrayBuffer> {
  if (encoding === "gzip+base64") {
    const r = await fetch(`${base}${name}.b64.txt`);
    if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
    const bin = atob((await r.text()).trim());
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return gunzip(new Blob([bytes]));
  }
  const r = await fetch(`${base}${name}`);
  if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
  return gunzip(r);
}
export type Progress = (msg: string, frac: number) => void;

function f16ToF32(buf: ArrayBuffer): Float32Array {
  const h = new Uint16Array(buf);
  const out = new Float32Array(h.length);
  for (let i = 0; i < h.length; i++) {
    const v = h[i];
    const s = v & 0x8000 ? -1 : 1;
    const e = (v >> 10) & 0x1f;
    const f = v & 0x3ff;
    out[i] = e === 0 ? s * Math.pow(2, -14) * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  return out;
}

export function loadBundle(base = "bundle/", progress?: Progress): Promise<Bundle> {
  if (cache) return cache;
  cache = (async () => {
    const files = ["meta", "variants_1", "variants_2", "genes", "transcripts", "tables", "pool_index", "missense_model", "splice_model",
      "meta_model", "reranker", "missense_calibration", "splice_calibration", "verifier", "passages_1", "passages_2", "passages_3", "sif",
      "hpo", "diseases", "literature", "submissions"];
    const got: Record<string, any> = {};
    let done = 0;
    let encoding = "gzip";
    try { const m = await fetch(`${base}manifest.json`); if (m.ok) encoding = (await m.json()).encoding || "gzip"; } catch { /* default encoding */ }
    await Promise.all(files.map(async (f) => {
      got[f] = JSON.parse(new TextDecoder().decode(await fetchGz(base, `${f}.json.gz`, encoding)));
      done += 1;
      progress?.(`Loaded ${f}`, done / (files.length + 1));
    }));
    const vb = await fetchGz(base, "sif_vectors.f16.gz", encoding);
    progress?.("Loaded word vectors", 1);
    const cols: string[] = got.variants_1.cols;
    const variants: Variant[] = [...got.variants_1.rows, ...got.variants_2.rows].map((row: any[]) => {
      const o: any = {};
      cols.forEach((c, i) => { o[c] = row[i]; });
      return hydrate(o);
    });
    const byId = new Map<string, Variant>();
    variants.forEach((v) => { if (v.variation_id) byId.set(v.variation_id, v); });
    return {
      meta: got.meta, variants, byId, genes: got.genes, transcripts: got.transcripts, tables: got.tables, poolIndex: got.pool_index,
      missense: got.missense_model, splice: got.splice_model, metaModel: got.meta_model, reranker: got.reranker,
      missenseCal: got.missense_calibration, spliceCal: got.splice_calibration, verifier: got.verifier,
      passages: [...got.passages_1, ...got.passages_2, ...got.passages_3], sif: got.sif, sifVectors: f16ToF32(vb),
      hpo: got.hpo, diseases: got.diseases, literature: got.literature, submissions: got.submissions,
    } as Bundle;
  })();
  return cache;
}

export const isNum = (x: any): x is number => typeof x === "number" && !Number.isNaN(x);
