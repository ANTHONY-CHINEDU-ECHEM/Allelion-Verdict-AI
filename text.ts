// Text normalisation shared by every retrieval component. Mirrors allelion/retrieval/text.py.

const STOP = new Set(`a an and are as at be by can do does for from has have how i if in into is it its may more most no not of on or
should so such than that the their then there these this those to under was we were what when where which while who why will
with within without would you your about after also any been being both each other out over per same some up very only use used
using`.split(/\s+/));

const SPELL: Record<string, string> = {
  haemoglobin: "hemoglobin", anaemia: "anemia", oedema: "edema", tumour: "tumor", tumours: "tumors", foetal: "fetal", paediatric: "pediatric",
  haematological: "hematological", leukaemia: "leukemia", oestrogen: "estrogen", haemophilia: "hemophilia", haemochromatosis: "hemochromatosis",
  behaviour: "behavior", colour: "color", favour: "favor", normalised: "normalized", characterise: "characterize", characterised: "characterized",
  analyse: "analyze", centre: "center", fibre: "fiber", dyslipidaemia: "dyslipidemia", ischaemic: "ischemic", hypercholesterolaemia: "hypercholesterolemia",
  glycaemic: "glycemic", gynaecological: "gynecological",
};

const SUFFIXES = ["ational", "ization", "ations", "ation", "ities", "ness", "ments", "ment", "ings", "ing", "ies", "ied", "ers", "edly", "ed", "es", "ly", "s"];
const TOKEN = /[cgp]\.[\w*+\-_>=?]+|rs\d+|[a-z0-9]+(?:[>'][a-z0-9]+)?/gi;

export function stem(w: string): string {
  if (w.length <= 4 || /\d/.test(w)) return w;
  for (const suf of SUFFIXES) {
    if (w.endsWith(suf) && w.length - suf.length >= 4) {
      if (suf === "ies" || suf === "ied") return w.slice(0, -suf.length) + "y";
      return w.slice(0, -suf.length);
    }
  }
  return w;
}

export function tokenize(text: string | null | undefined): string[] {
  const out: string[] = [];
  if (!text) return out;
  for (const m of text.matchAll(TOKEN)) {
    let w = m[0].toLowerCase();
    w = SPELL[w] ?? w;
    if (STOP.has(w) || (w.length < 2 && !/^\d+$/.test(w))) continue;
    out.push(stem(w));
  }
  return out;
}

export function sentences(text: string): string[] {
  return text.trim().split(/(?<=[.!?])\s+(?=[A-Z0-9(])/).map((s) => s.trim()).filter(Boolean);
}
