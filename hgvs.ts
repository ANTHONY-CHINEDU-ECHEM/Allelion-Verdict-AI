// HGVS parsing for coding (c.) and protein (p.) descriptions. Mirrors allelion/kb/hgvs.py.

export const AA3: Record<string, string> = {
  Ala: "A", Arg: "R", Asn: "N", Asp: "D", Cys: "C", Gln: "Q", Glu: "E", Gly: "G", His: "H", Ile: "I", Leu: "L", Lys: "K", Met: "M",
  Phe: "F", Pro: "P", Ser: "S", Thr: "T", Trp: "W", Tyr: "Y", Val: "V", Ter: "*", Sec: "U", Xaa: "X",
};
export const AA1_TO_3: Record<string, string> = Object.fromEntries(Object.entries(AA3).map(([k, v]) => [v, k]));

export interface CPos { region: "cds" | "utr5" | "utr3"; base: number; offset: number }
export interface CVariant {
  transcript: string | null; start: CPos; end: CPos; kind: string; ref: string; alt: string;
  delLen: number | null; insLen: number | null; lengthChange: number | null; frameshift: boolean | null; spliceDistance: number;
}
export interface PVariant { transcript: string | null; kind: string; refAa: string; pos: number | null; altAa: string; endPos: number | null; short: string }

const POS = "([-*]?)(\\d+)(?:([+-])(\\d+))?";
const C_RE = new RegExp(
  `^(?:([A-Z]{2}_\\d+(?:\\.\\d+)?)(?:\\([^)]*\\))?:)?c\\.(${POS})(?:_(${POS}))?` +
  "([ACGTN]+>[ACGTN]+|del(?:[ACGTN]*|\\d+)ins(?:[ACGTN]+|\\d+)|del(?:[ACGTN]*|\\d+)|dup(?:[ACGTN]*|\\d+)|ins(?:[ACGTN]+|\\d+)|[ACGTN]?=|inv(?:[ACGTN]*|\\d+))?$",
);

function mkPos(sign: string | undefined, num: string, osign: string | undefined, onum: string | undefined): CPos {
  let base = parseInt(num, 10);
  let region: CPos["region"] = "cds";
  if (sign === "-") { region = "utr5"; base = -base; } else if (sign === "*") region = "utr3";
  let offset = 0;
  if (osign) offset = osign === "+" ? parseInt(onum!, 10) : -parseInt(onum!, 10);
  return { region, base, offset };
}

function span(a: CPos, b: CPos): number | null {
  if (a.offset || b.offset || a.region !== b.region) return null;
  return Math.abs(b.base - a.base) + 1;
}

export function parseC(text: string | null | undefined): CVariant | null {
  if (!text) return null;
  const m = C_RE.exec(text.trim());
  if (!m) return null;
  const start = mkPos(m[3], m[4], m[5], m[6]);
  const end = m[7] ? mkPos(m[8], m[9], m[10], m[11]) : start;
  const edit = m[12] || "";
  const sp = span(start, end);
  const n = (seq: string, dflt: number | null) => (/^\d+$/.test(seq) ? parseInt(seq, 10) : seq ? seq.length : dflt);
  let kind = "", ref = "", alt = "";
  let delLen: number | null = null, insLen: number | null = null;
  if (edit.includes(">")) { kind = "sub"; [ref, alt] = edit.split(">"); delLen = ref.length; insLen = alt.length; }
  else if (edit.startsWith("del") && edit.includes("ins")) {
    kind = "delins"; [ref, alt] = edit.slice(3).split("ins"); delLen = n(ref, sp); insLen = n(alt, null);
    if (/^\d+$/.test(ref)) ref = ""; if (/^\d+$/.test(alt)) alt = "";
  } else if (edit.startsWith("del")) { kind = "del"; ref = edit.slice(3); delLen = n(ref, sp); if (/^\d+$/.test(ref)) ref = ""; }
  else if (edit.startsWith("dup")) { kind = "dup"; ref = edit.slice(3); insLen = n(ref, sp); if (/^\d+$/.test(ref)) ref = ""; }
  else if (edit.startsWith("ins")) { kind = "ins"; alt = edit.slice(3); insLen = n(alt, null); if (/^\d+$/.test(alt)) alt = ""; }
  else if (edit.endsWith("=")) kind = "identity";
  else if (edit.startsWith("inv")) kind = "inv";
  else return null;
  let lengthChange: number | null = null;
  if (kind === "sub") lengthChange = 0;
  else if (kind === "del") lengthChange = delLen !== null ? -delLen : null;
  else if (kind === "dup" || kind === "ins") lengthChange = insLen;
  else if (kind === "delins" && delLen !== null && insLen !== null) lengthChange = insLen - delLen;
  const offs = [start, end].filter((p) => p.offset).map((p) => Math.abs(p.offset));
  return {
    transcript: m[1] || null, start, end, kind, ref, alt, delLen, insLen, lengthChange,
    frameshift: lengthChange === null ? null : lengthChange % 3 !== 0, spliceDistance: offs.length ? Math.min(...offs) : 0,
  };
}

const P_SIMPLE = /^([A-Z][a-z]{2}|\*|[A-Z])(\d+)([A-Z][a-z]{2}|\*|[A-Z]|=|\?)$/;
const P_FS = /^([A-Z][a-z]{2})(\d+)(?:[A-Z][a-z]{2})?fs(?:Ter|\*)?(\d+|\?)?$/;
const P_RANGE = /^([A-Z][a-z]{2})(\d+)(?:_([A-Z][a-z]{2})(\d+))?(del|dup|ins|delins)([A-Za-z*]*)$/;
const P_EXT = /^(Ter|\*)(\d+)([A-Z][a-z]{2})?ext/;

function aa(t: string | undefined): string {
  if (!t) return "?";
  if (AA3[t]) return AA3[t];
  if (t === "*" || t === "=") return t;
  return t.length === 1 ? t : "?";
}

function pv(transcript: string | null, kind: string, refAa: string, pos: number | null, altAa: string, endPos: number | null = null): PVariant {
  return { transcript, kind, refAa, pos, altAa, endPos, short: pos === null ? "" : `${refAa}${pos}${altAa}` };
}

export function parseP(text: string | null | undefined): PVariant | null {
  if (!text) return null;
  const m = /^(?:([A-Z]{2}_\d+(?:\.\d+)?):)?p\.\(?([^)]*)\)?$/.exec(text.trim());
  if (!m) return null;
  const tx = m[1] || null;
  let body = m[2].replace(/\++$/, "");
  body = body.replace(/^([A-Z][a-z]{2}\d+)[A-Z][a-z]{2}=$/, "$1=");
  const syn = /^([A-Z][a-z]{2})(\d+)_[A-Z][a-z]{2}\d+=$/.exec(body);
  if (syn) return pv(tx, "synonymous", aa(syn[1]), parseInt(syn[2], 10), "=");
  if (body === "?" || body === "0" || body === "=") return pv(tx, "unknown", "", null, body);
  let r: RegExpExecArray | null;
  if ((r = P_FS.exec(body))) return pv(tx, "frameshift", aa(r[1]), parseInt(r[2], 10), "fs");
  if ((r = P_EXT.exec(body))) return pv(tx, "stop_loss", "*", parseInt(r[2], 10), aa(r[3] || "Xaa"));
  if ((r = P_SIMPLE.exec(body))) {
    const ref = aa(r[1]); const pos = parseInt(r[2], 10); const alt = r[3];
    let alt1 = alt === "=" || alt === "?" ? alt : aa(alt);
    let kind: string;
    if (alt1 === "=" || alt1 === ref) { kind = "synonymous"; alt1 = "="; }
    else if (ref === "M" && pos === 1) kind = "start_loss";
    else if (alt1 === "*") kind = "nonsense";
    else if (ref === "*") kind = "stop_loss";
    else if (alt1 === "?") kind = "unknown";
    else kind = "missense";
    return pv(tx, kind, ref, pos, alt1);
  }
  if ((r = P_RANGE.exec(body))) {
    const ref = aa(r[1]); const pos = parseInt(r[2], 10); const end = r[4] ? parseInt(r[4], 10) : pos; const op = r[5];
    let kind = ({ del: "inframe_del", dup: "inframe_dup", ins: "inframe_ins", delins: "delins" } as Record<string, string>)[op];
    const tail = r[6] || "";
    if (tail.includes("Ter") || tail.includes("*")) kind = "nonsense";
    return pv(tx, kind, ref, pos, op, end);
  }
  return null;
}

export const LOF_CONSEQUENCES = new Set(["frameshift", "nonsense", "splice_donor", "splice_acceptor", "start_loss"]);

export function consequenceFromHgvs(c: CVariant | null, p: PVariant | null): string {
  if (p && ["missense", "nonsense", "frameshift", "synonymous", "start_loss", "stop_loss"].includes(p.kind)) return p.kind;
  if (p && ["inframe_del", "inframe_ins", "inframe_dup", "delins"].includes(p.kind)) return "inframe_indel";
  if (!c) return "other";
  if (c.start.offset) {
    if (c.spliceDistance === 1 || c.spliceDistance === 2) return c.start.offset > 0 || c.end.offset > 0 ? "splice_donor" : "splice_acceptor";
    return "intron";
  }
  if (c.start.region === "utr5") return "utr5";
  if (c.start.region === "utr3") return "utr3";
  if (["del", "ins", "dup", "delins"].includes(c.kind)) return c.frameshift ? "frameshift" : "inframe_indel";
  return "other";
}
