// Evidence copilot in the browser: guard, retrieval, variant evaluation, extractive writer, verifier, audit.
// Mirrors allelion/copilot (guard.py, writer.py, verifier.py, copilot.py).
import type { Bundle, Variant } from "./data";
import { Engine, prepare, type Result } from "./engine";
import { parseC, parseP } from "./hgvs";
import { Retrieval, type QueryInfo } from "./retrieval";
import { sentences, tokenize } from "./text";

// ------------------------------------------------------------------ guard
const REDACTIONS: [string, RegExp][] = [
  ["NHS_NUMBER", /(?:\bNHS(?:\s(?:no\.?|number))?[:\s#]*)?\b\d{3}[\s-]?\d{3}[\s-]?\d{4}\b/gi],
  ["EMAIL", /\b[\w.+-]+@[\w-]+\.[\w.-]+\b/g],
  ["PHONE", /(?:\+44\s?7\d{3}|\b07\d{3}|\b0\d{3,4})[\s-]?\d{3}[\s-]?\d{3,4}\b/g],
  ["DATE_OF_BIRTH", /\b(?:DOB|D\.O\.B\.?|date of birth|born on)[:\s]*\S+(?:\s\S+){0,2}/gi],
  ["DATE", /\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b|\b\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s\d{4}\b/gi],
  ["POSTCODE", /\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b/g],
  ["RECORD_NUMBER", /\b(?:MRN|hospital number|record number|patient id|case id)[:#\s]*[A-Z0-9-]{4,}\b/gi],
  ["NAME", /\b(?:Mr|Mrs|Ms|Miss|Dr|Prof)\.?\s[A-Z][a-z]+(?:\s[A-Z][a-z]+)?/g],
  ["AGE_OVER_89", /\b(?:aged?\s)?(?:9\d|1[0-1]\d)\s?(?:years?\sold|yo|year old)\b|\baged\s(?:9\d|1[0-1]\d)\b/gi],
];
const CRISIS = /\b(?:i want to|i'm going to|i am going to|thinking (?:about|of)|plan to)\s(?:kill myself|end my life|hurt myself|take my own life)\b|\bsuicid(?:e|al)\b.*\b(?:i|me|my)\b|\b(?:i|me|my)\b.*\bsuicid(?:e|al)\b/i;
const PERSONAL = /\b(?:should i|should my|should we|do i need|what should i do|am i going to|will i get|will my (?:child|son|daughter|baby)|my (?:result|test|report|baby|son|daughter|child|partner|wife|husband)\b.*\b(?:mean|surgery|mastectomy|treatment|terminate|pregnan))/i;
const INJECTION = /(?:ignore|disregard|forget)\s(?:all\s)?(?:the\s)?(?:previous|prior|above|earlier)\s(?:instructions?|prompts?|rules?)|\byou are now\b|\bsystem prompt\b|\bdeveloper mode\b|\bact as\b.{0,40}\bwithout\b|<\/?(?:system|assistant|instructions?)>/gi;
const DOMAIN = /\b(?:gene|genes|variant|variants|mutation|mutations|allele|pathogenic|benign|acmg|clinvar|clingen|criteri|pvs1|ps\d|pm\d|pp\d|ba1|bs\d|bp\d|exon|intron|splice|missense|nonsense|frameshift|syndrome|disease|disorder|phenotype|inheritance|dominant|recessive|penetrance|protein|hgvs|c\.\d|p\.[a-z]|gnomad|frequency|segregation|de novo|carrier|cancer|cardiomyopathy|deficiency|study|report|define|term|hpo|omim|orpha|predictor|in silico|calibrat|vus|uncertain)\b/i;

export const CRISIS_MESSAGE = "It sounds like you may be going through something very difficult. You do not have to face it alone. In the UK you can call Samaritans free on 116 123 at any time, or contact NHS 111. If you are in immediate danger, call 999. This tool cannot help with this, but the people at those services can.";
export const PERSONAL_NOTICE = "This tool summarises published evidence about variants and genes. It cannot advise on decisions about your own or a relative's care; a clinical genetics service or genetic counsellor can explain what a result means for you.";
export const OFF_TOPIC_MESSAGE = "This copilot answers questions about genetic variants, genes, inherited diseases, phenotypes and variant classification.";

export function redact(text: string): [string, string[]] {
  const found: string[] = [];
  const protectedVals: string[] = [];
  let t = text.replace(/(?:[cgpn]\.[^\s,;]+|rs\d+|HP:\d{7}|(?:OMIM|ORPHA):\d+|PMID[:\s]*\d+|N[MPRC]_\d+(?:\.\d+)?)/g, (m) => { protectedVals.push(m); return `\u0000${protectedVals.length - 1}\u0000`; });
  for (const [label, pat] of REDACTIONS) {
    pat.lastIndex = 0;
    if (pat.test(t)) { found.push(label); pat.lastIndex = 0; t = t.replace(pat, `[${label}]`); }
  }
  t = t.replace(/\u0000(\d+)\u0000/g, (_, i) => protectedVals[+i]);
  return [t, found];
}

export function neutralise(text: string): [string, boolean] {
  INJECTION.lastIndex = 0;
  const hit = INJECTION.test(text);
  INJECTION.lastIndex = 0;
  return [text.replace(INJECTION, "[removed instruction]"), hit];
}

export function guard(question: string) {
  const [red, found] = redact(question);
  const [clean, inj] = neutralise(red);
  const g = { text: clean, redactions: found, action: "answer", personal: false, injection: inj, message: "" };
  if (CRISIS.test(question)) return { ...g, action: "crisis", message: CRISIS_MESSAGE };
  if (!DOMAIN.test(clean) && !/\b[A-Z][A-Z0-9]{2,9}\b/.test(clean)) return { ...g, action: "off_topic", message: OFF_TOPIC_MESSAGE };
  if (PERSONAL.test(question)) return { ...g, personal: true, message: PERSONAL_NOTICE };
  return g;
}

// ------------------------------------------------------------------ verifier
const NUM = /(?<![A-Za-z:])\d+(?:\.\d+)?%?/g;
const GENE = /\b[A-Z][A-Z0-9]{2,9}\b/g;
const CODE = /\b(?:PVS1|PS[1-4]|PM[1-6]|PP[1-5]|BA1|BS[1-4]|BP[1-7])\b/g;
const HGVS = /\b[cp]\.[^\s,;)]+/g;
const NEG = /\b(?:not|no|never|none|without|absent|cannot|neither|nor|unlikely|does not|is not|are not)\b/i;
const CLASS_WORDS = ["pathogenic", "benign", "uncertain", "likely pathogenic", "likely benign"];
const NON_GENES = new Set(["DNA", "RNA", "PCR", "ACMG", "AMP", "SVI", "HPO", "OMIM", "ORPHA", "PMID", "UTR", "NMD", "VUS", "CNV", "MANE", "HGVS", "ENIGMA",
  "UK", "USA", "ID", "CI", "OR", "HP", "LR", "SNP", "THE", "AND", "FOR", "LOF", "GOF", "MAF", "CFTR2", "RASOPATHY"]);
const ANTONYMS: [string, string][] = [["dominant", "recessive"], ["increase", "decrease"], ["increased", "decreased"], ["higher", "lower"], ["pathogenic", "benign"],
  ["supporting", "strong"], ["absent", "present"], ["last", "first"], ["gain", "loss"], ["above", "below"], ["more", "less"], ["upstream", "downstream"],
  ["maternal", "paternal"], ["homozygous", "heterozygous"], ["met", "not met"]];

const bigrams = (t: string[]) => new Set(t.slice(1).map((w, i) => `${t[i]} ${w}`));
const nums = (s: string) => new Set([...s.matchAll(NUM)].map((m) => m[0].replace(/%$/, "")));
const has = (re: RegExp, s: string) => { re.lastIndex = 0; return re.test(s); };

export function verifierFeatures(sentence: string, evidence: string, cited = true): Record<string, number> {
  const st = tokenize(sentence), et = tokenize(evidence);
  const ss = new Set(st), es = new Set(et);
  let inter = 0; ss.forEach((t) => { if (es.has(t)) inter++; });
  const sb = bigrams(st), eb = bigrams(et);
  let bi = 0; sb.forEach((x) => { if (eb.has(x)) bi++; });
  const evS = sentences(evidence).length ? sentences(evidence) : [evidence];
  const spans = [...evS, ...evS.slice(1).map((b, i) => `${evS[i]} ${b}`)];
  let best = 0, bestS = "";
  for (const e of spans) {
    const et2 = new Set(tokenize(e));
    let o = 0; ss.forEach((t) => { if (et2.has(t)) o++; });
    const v = o / Math.max(ss.size, 1);
    if (v > best) { best = v; bestS = e; }
  }
  const evNums = nums(evidence);
  const numOk = [...nums(sentence)].every((n) => evNums.has(n));
  const genes = [...sentence.matchAll(GENE)].map((m) => m[0]).filter((g) => !NON_GENES.has(g) && !/^(PVS1|PS[1-4]|PM[1-6]|PP[1-5]|BA1|BS[1-4]|BP[1-7])$/.test(g));
  const geneOk = genes.every((g) => evidence.includes(g));
  const codeOk = [...sentence.matchAll(CODE)].every((m) => evidence.includes(m[0]));
  const hgvsOk = [...sentence.matchAll(HGVS)].every((m) => evidence.includes(m[0].replace(/\.$/, "")));
  const negAgree = bestS ? +(has(NEG, sentence) === has(NEG, bestS)) : 0;
  const sl = sentence.toLowerCase(), el = evidence.toLowerCase(), bl = bestS.toLowerCase();
  const classOk = CLASS_WORDS.filter((w) => new RegExp(`\\b${w}\\b`).test(sl)).every((w) => el.includes(w));
  let antonymOk = 1;
  for (const [a, b] of ANTONYMS) for (const [x, y] of [[a, b], [b, a]]) {
    if (new RegExp(`\\b${x}\\b`).test(sl) && !new RegExp(`\\b${x}\\b`).test(bl) && new RegExp(`\\b${y}\\b`).test(bl)) antonymOk = 0;
  }
  return {
    tok_recall: inter / Math.max(ss.size, 1), bigram_recall: bi / Math.max(sb.size, 1), best_sent: best, num_ok: +numOk, gene_ok: +geneOk, code_ok: +codeOk,
    hgvs_ok: +hgvsOk, neg_agree: negAgree, len_ratio: Math.min(st.length / Math.max(et.length, 1), 2), has_cite: +cited, class_ok: +classOk, antonym_ok: antonymOk,
  };
}

export function hardFail(f: Record<string, number>): string | null {
  if (!f.has_cite) return "no citation";
  if (!f.num_ok) return "a number is not in the cited evidence";
  if (!f.gene_ok) return "a gene is not in the cited evidence";
  if (!f.code_ok) return "a criterion code is not in the cited evidence";
  if (!f.hgvs_ok) return "a variant description is not in the cited evidence";
  if (!f.antonym_ok) return "a word contradicts the best matching evidence sentence";
  return null;
}

export function verify(b: Bundle, sentence: string, evidence: string | null) {
  const f = verifierFeatures(sentence, evidence ?? "", evidence !== null);
  let z = b.verifier.intercept;
  b.verifier.features.forEach((name, i) => { z += b.verifier.coef[i] * f[name]; });
  const p = 1 / (1 + Math.exp(-z));
  const reason = hardFail(f);
  const supported = reason === null && p >= 0.5;
  return { supported, probability: Math.round(p * 1e4) / 1e4, reason: reason || (supported ? "" : "low support score") };
}

// ------------------------------------------------------------------ writer
export interface Evidence { n: number; title: string; text: string; kind: string; source: string; level: string; refs: string[]; genes?: string[] }

function answerUnits(text: string): string[] {
  const out: string[] = [];
  let carry = "";
  for (const s of sentences(text)) {
    const cur = carry ? `${carry} ${s}`.trim() : s;
    if (cur.split(/\s+/).length < 6) { carry = cur; continue; }
    out.push(cur); carry = "";
  }
  if (carry) { if (out.length) out[out.length - 1] = `${out[out.length - 1]} ${carry}`; else out.push(carry); }
  return out;
}

const GENERIC = new Set(tokenize("clinical feature features phenotype phenotypes syndrome disease disorder gene genes variant variants mutation mutations report reported study studies patient patients describe described find found evidence apply applies applied criterion criteria strength mean meaning define definition annotated associated association type inheritance pattern recorded which what how does many"));

export function writeExtractive(question: string, qi: QueryInfo, evidence: Evidence[], maxSentences = 4): { text: string; cite: number }[] {
  const qt = new Set(tokenize(question));
  const focus = new Set([...qt].filter((t) => !GENERIC.has(t)));
  const ents = [...qi.genes, ...qi.codes].map((e) => e.toLowerCase());
  const cands: [number, number, number, string, number, Set<string>][] = [];
  evidence.forEach((ev, rank) => {
    answerUnits(ev.text).forEach((s, j) => {
      if (s.split(/\s+/).length < 5) return;
      const own = new Set(tokenize(s));
      const st = j === 0 ? new Set([...own, ...tokenize(ev.title)]) : own;
      let ovn = 0; qt.forEach((t) => { if (st.has(t)) ovn++; });
      const ov = ovn / Math.max(qt.size, 1);
      if (focus.size && ev.kind !== "engine") {
        const tt = new Set([...st, ...tokenize(ev.title)]);
        if (![...focus].some((t) => tt.has(t))) return;
      }
      const ent = ents.length ? ents.filter((e) => s.toLowerCase().includes(e)).length / ents.length : 0;
      const lead = ev.kind === "engine" && rank === 0 && j === 0 ? 0.15 : 0;
      cands.push([0.5 * ov + 0.2 * ent + 0.3 / (1 + rank) + lead, rank, j, s, ev.n, own]);
    });
  });
  if (!cands.length) return [];
  cands.sort((a, b) => b[0] - a[0]);
  const best = cands[0][0];
  const chosen: typeof cands = [];
  const per = new Map<number, number>();
  for (const c of cands) {
    if (c[0] < 0.5 * best || (per.get(c[4]) || 0) >= 2) continue;
    if (chosen.some((x) => { let i = 0; c[5].forEach((t) => { if (x[5].has(t)) i++; }); return i / Math.max(new Set([...c[5], ...x[5]]).size, 1) > 0.6; })) continue;
    chosen.push(c); per.set(c[4], (per.get(c[4]) || 0) + 1);
    if (chosen.length >= maxSentences) break;
  }
  chosen.sort((a, b) => a[1] - b[1] || a[2] - b[2]);
  return chosen.map((c) => ({ text: c[3], cite: c[4] }));
}

// ------------------------------------------------------------------ copilot
const LEVEL_RANK: Record<string, number> = { engine: 3, guideline: 3, "curated database": 2, ontology: 2, literature: 1 };

export interface Answer {
  question_redacted: string; redactions: string[]; action: string; notices: string[]; sentences: { text: string; cite: number; supported: boolean; probability: number; reason: string }[];
  evidence: Evidence[]; engine_result: Result | null; query?: QueryInfo; verified_share?: number; entity_coverage?: number; confidence: string; ms: number; writer: string;
}

export class Copilot {
  engine: Engine;
  constructor(public b: Bundle, public ret: Retrieval) { this.engine = new Engine(b); }

  resolveVariant(qi: QueryInfo): Variant | null {
    if (!qi.genes.length || !(qi.hgvs_c.length || qi.hgvs_p.length)) return null;
    for (const g of qi.genes) {
      const sub = this.b.variants.filter((v) => v.gene === g);
      for (const c of qi.hgvs_c) { const cs = c.split(":").pop(); const hit = sub.find((v) => (v.hgvs_c || "").split(":").pop() === cs); if (hit) return hit; }
      for (const p of qi.hgvs_p) {
        const pv = parseP(p.startsWith("p.") ? p : "p." + p.replace(/^\(|\)$/g, ""));
        if (pv && pv.short) { const hit = sub.find((v) => v.protein_change === pv.short); if (hit) return hit; }
      }
      if (qi.hgvs_c.length) { const c = qi.hgvs_c[0].split(":").pop()!; if (parseC(c)) return prepare(g, c, qi.hgvs_p[0] || null); }
    }
    return null;
  }

  variantEvidence(v: Variant): [Omit<Evidence, "n">[], Result] {
    const res = this.engine.evaluate(v);
    const name = `${v.gene} ${(v.hgvs_c || "").split(":").pop()}`;
    const met = res.criteria.filter((c) => c.status === "met");
    const head = `The Allelion criteria engine classifies ${name} (${String(res.variant.consequence).replace(/_/g, " ")}) as ${res.classification.toLowerCase()} with ${res.points} points. Criteria met: ${res.met.join(", ") || "none"}.`;
    const items: Omit<Evidence, "n">[] = [{ title: `Criteria engine summary for ${name}`, text: head, kind: "engine", source: `engine:${name}`, level: "engine", refs: ["Allelion criteria engine"] }];
    for (const c of met) items.push({ title: `${c.label}: ${c.title}`, text: `${c.label} applied for ${name}. ${c.rationale}`, kind: "engine", source: `engine:${name}:${c.code}`, level: "engine", refs: c.evidence.slice(0, 3).map((e) => e.ref) });
    const ne = res.criteria.filter((c) => c.status === "not_evaluated").map((c) => c.code);
    if (ne.length) items.push({ title: `Criteria awaiting case data for ${name}`, level: "engine", kind: "engine", source: `engine:${name}:pending`, text: `Criteria that need case level data for ${name} were not evaluated: ${ne.join(", ")}.`, refs: [] });
    return [items, res];
  }

  answer(question: string, k = 6): Answer {
    const t0 = performance.now();
    const g0 = guard(question);
    let g = g0;
    if (g0.action === "off_topic") {
      const qi0 = this.ret.understand(g0.text);
      if (qi0.genes.length || qi0.hpo.length || qi0.diseases.length || qi0.codes.length) g = { ...g0, action: "answer", message: "" };
    }
    const out: Answer = { question_redacted: g.text, redactions: g.redactions, action: g.action, notices: [], sentences: [], evidence: [], engine_result: null, confidence: "none", ms: 0, writer: "extractive" };
    if (g.action !== "answer") { out.notices.push(g.message); out.ms = Math.round(performance.now() - t0); audit(question, out); return out; }
    if (g.personal) out.notices.push(g.message);
    if (g.injection) out.notices.push("Instructions embedded in the question were removed before retrieval.");
    const qi = this.ret.understand(g.text);
    const ev: Omit<Evidence, "n">[] = [];
    const v = this.resolveVariant(qi);
    if (v) { const [items, res] = this.variantEvidence(v); ev.push(...items); out.engine_result = res; }
    const hits = this.ret.search(g.text, k);
    for (const i of hits.ids) {
      const p = this.b.passages[i];
      ev.push({ source: p[0], kind: p[1], title: p[2], text: neutralise(p[3])[0], level: p[4], genes: p[5], refs: p[7] });
    }
    const evidence: Evidence[] = ev.map((e, i) => ({ ...e, n: i + 1 }));
    const sents = writeExtractive(g.text, qi, evidence);
    const byN = new Map(evidence.map((e) => [e.n, e]));
    out.sentences = sents.map((s) => { const e = byN.get(s.cite); return { ...s, ...verify(this.b, s.text, e ? e.text : null) }; });
    const share = out.sentences.length ? out.sentences.filter((s) => s.supported).length / out.sentences.length : 0;
    const cited = out.sentences.map((s) => byN.get(s.cite)).filter((x): x is Evidence => !!x);
    const ents = [...qi.genes, ...qi.codes];
    const cov = ents.length ? ents.filter((e) => cited.some((c) => (c.text + c.title).includes(e))).length / ents.length : 1;
    const bestLevel = Math.max(0, ...cited.map((c) => LEVEL_RANK[c.level] ?? 1));
    out.confidence = share >= 0.9 && cov >= 0.5 && bestLevel >= 2 ? "high" : share >= 0.7 && bestLevel >= 1 ? "moderate" : "low";
    Object.assign(out, { query: qi, evidence, verified_share: Math.round(share * 1000) / 1000, entity_coverage: Math.round(cov * 1000) / 1000, ms: Math.round(performance.now() - t0) });
    audit(question, out);
    return out;
  }
}

// ------------------------------------------------------------------ audit (append only, this browser)
export interface AuditRecord { ts: string; question: string; sha256: string; action: string; writer: string; redactions: string[]; evidence: number; verified_share: number | null; confidence: string; classification: string | null }
const AUDIT_KEY = "allelion.audit.v1";

async function sha256(s: string): Promise<string> {
  try {
    const d = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s));
    return [...new Uint8Array(d)].map((x) => x.toString(16).padStart(2, "0")).join("");
  } catch { return ""; }
}

export const auditLog: AuditRecord[] = (() => { try { return JSON.parse(localStorage.getItem(AUDIT_KEY) || "[]"); } catch { return []; } })();

function audit(question: string, out: Answer) {
  const rec: AuditRecord = { ts: new Date().toISOString().slice(0, 19) + "Z", question: out.question_redacted, sha256: "", action: out.action, writer: out.writer,
    redactions: out.redactions, evidence: out.evidence.length, verified_share: out.verified_share ?? null, confidence: out.confidence, classification: out.engine_result?.classification ?? null };
  auditLog.push(rec);
  sha256(question).then((h) => { rec.sha256 = h; try { localStorage.setItem(AUDIT_KEY, JSON.stringify(auditLog.slice(-500))); } catch { /* storage unavailable */ } });
}
