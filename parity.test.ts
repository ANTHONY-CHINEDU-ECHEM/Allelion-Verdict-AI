// Parity between the Python engine and the browser engine on the shared fixture written by allelion/export/parity.py.
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { resolve } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { hydrate, type Bundle, type Variant } from "./data";
import { Engine, prepare } from "./engine";
import { parseC, parseP } from "./hgvs";
import { learnedVerdict } from "./meta";
import { PhenotypeMatcher } from "./phenotype";
import { Retrieval } from "./retrieval";
import { tokenize } from "./text";
import { verify } from "./copilot";

const root = resolve(__dirname, "../../public/bundle");
const J = (f: string) => JSON.parse(gunzipSync(readFileSync(resolve(root, `${f}.json.gz`))).toString("utf8"));
const fx = JSON.parse(readFileSync(resolve(__dirname, "../../test/parity_fixture.json"), "utf8"));

function f16(buf: Buffer): Float32Array {
  const h = new Uint16Array(buf.buffer, buf.byteOffset, buf.byteLength / 2);
  const out = new Float32Array(h.length);
  for (let i = 0; i < h.length; i++) {
    const v = h[i], s = v & 0x8000 ? -1 : 1, e = (v >> 10) & 0x1f, f = v & 0x3ff;
    out[i] = e === 0 ? s * Math.pow(2, -14) * (f / 1024) : e === 31 ? NaN : s * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  return out;
}

let b: Bundle;
beforeAll(() => {
  const v1 = J("variants_1"), v2 = J("variants_2");
  const variants: Variant[] = [...v1.rows, ...v2.rows].map((r: any[]) => { const o: any = {}; v1.cols.forEach((c: string, i: number) => { o[c] = r[i]; }); return hydrate(o); });
  b = {
    meta: J("meta"), variants, byId: new Map(variants.map((v) => [v.variation_id!, v])), genes: J("genes"), transcripts: J("transcripts"), tables: J("tables"),
    poolIndex: J("pool_index"), missense: J("missense_model"), splice: J("splice_model"), metaModel: J("meta_model"), reranker: J("reranker"),
    missenseCal: J("missense_calibration"), spliceCal: J("splice_calibration"), verifier: J("verifier"),
    passages: [...J("passages_1"), ...J("passages_2"), ...J("passages_3")], sif: J("sif"), sifVectors: f16(gunzipSync(readFileSync(resolve(root, "sif_vectors.f16.gz")))),
    hpo: J("hpo"), diseases: J("diseases"), literature: { docs: [], links: [], mentions: {} }, submissions: {},
  } as Bundle;
});

describe("text and HGVS", () => {
  it("tokenises identically", () => { for (const t of fx.tokenize) expect(tokenize(t.text)).toEqual(t.tokens); });
  it("parses coding HGVS identically", () => {
    for (const h of fx.hgvs_c) {
      const c = parseC(h.text);
      expect(c ? { kind: c.kind, region: c.start.region, base: c.start.base, offset: c.start.offset, length_change: c.lengthChange, splice_distance: c.spliceDistance } : null).toEqual(h.parsed);
    }
  });
  it("parses protein HGVS identically", () => {
    for (const h of fx.hgvs_p) { const p = parseP(h.text); expect(p ? { kind: p.kind, short: p.short } : null).toEqual(h.parsed); }
  });
});

describe("criteria engine", () => {
  it("agrees on every bundled fixture variant", () => {
    const e = new Engine(b);
    let mism = 0;
    for (const c of fx.variants) {
      const v = b.byId.get(c.variation_id)!;
      const r = e.evaluate(v);
      const x = c.expected;
      const ok = r.classification === x.classification && r.points === x.points && JSON.stringify(r.met) === JSON.stringify(x.met)
        && r.classification_2015 === x.classification_2015 && learnedVerdict(b, r, v).classification === x.learned;
      if (!ok) { mism++; if (mism < 6) console.log(c.variation_id, r.met, x.met, r.scores, x.scores, learnedVerdict(b, r, v).classification, x.learned); }
      for (const k of ["missense", "splice"] as const) if (x.scores[k] !== null) expect(Math.abs((r.scores[k] ?? -1) - x.scores[k])).toBeLessThan(2e-4);
      for (const [code, st] of Object.entries(x.status)) expect(r.criteria.find((cc) => cc.code === code)!.status).toBe(st);
    }
    expect(mism).toBe(0);
  });
  it("agrees on variants entered by HGVS", () => {
    const e = new Engine(b);
    for (const c of fx.custom) {
      const r = e.evaluate(prepare(c.gene, c.c, c.p, null));
      expect([c.gene, c.c, r.classification, r.points, r.met]).toEqual([c.gene, c.c, c.expected.classification, c.expected.points, c.expected.met]);
    }
  });
  it("agrees on case level evidence", () => {
    const e = new Engine(b);
    const v = prepare("FBN1", "c.3037G>A", "p.Gly1013Arg", null);
    for (const c of fx.case_level) { const r = e.evaluate(v, c.case); expect([r.met, r.points]).toEqual([c.expected.met, c.expected.points]); }
  });
  it("agrees on phenotype matches", () => {
    const m = new PhenotypeMatcher(b);
    for (const c of fx.phenotype) {
      const r = m.match(c.gene, c.terms);
      expect(Math.abs(r.score - c.expected.score)).toBeLessThan(2e-3);
      expect(r.specific).toBe(c.expected.specific);
      expect(Math.abs(r.p_value - c.expected.p_value)).toBeLessThan(0.01);
    }
  });
});

describe("copilot components", () => {
  it("verifier probabilities match", () => { for (const c of fx.verifier) expect(Math.abs(verify(b, c.sentence, c.evidence).probability - c.expected)).toBeLessThan(1e-3); });
  it("retrieval agrees on the top results", () => {
    const r = new Retrieval(b);
    let overlap = 0, total = 0, bm = 0;
    for (const q of fx.retrieval) {
      const got = r.search(q.q, 5).ids.map((i) => b.passages[i][0]);
      const gotBm = r.search(q.q, 5, "bm25").ids.map((i) => b.passages[i][0]);
      expect(got[0]).toBe(q.top5[0]);
      overlap += got.filter((s) => q.top5.includes(s)).length;
      bm += gotBm.filter((s) => q.bm25_top5.includes(s)).length;
      total += q.top5.length;
    }
    expect(bm / total).toBeGreaterThanOrEqual(0.98);
    expect(overlap / total).toBeGreaterThanOrEqual(0.9);
  });
});
