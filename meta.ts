// Learned verdict over the engine's evidence. Mirrors allelion/models/meta.py featurise().
import type { Bundle, Variant } from "./data";
import { POINTS } from "./combine";
import type { Result } from "./engine";
import { predictMulti, vector } from "./lgbm";

const CONSEQ = ["missense", "synonymous", "intron", "frameshift", "nonsense", "splice_donor", "splice_acceptor", "inframe_indel", "utr5", "utr3", "start_loss", "stop_loss", "other"];
const nan = (x: any) => x === null || x === undefined || (typeof x === "number" && Number.isNaN(x));

export function learnedVerdict(b: Bundle, res: Result, v: Variant): { classification: string; probabilities: Record<string, number> } {
  const f: Record<string, number> = {};
  for (const c of b.meta.order as string[]) f[`pts_${c}`] = 0;
  for (const c of res.criteria) if (c.status === "met" && c.strength !== "StandAlone") { const p = POINTS[c.strength]; f[`pts_${c.code}`] = c.direction === "pathogenic" ? p : -p; }
  f.ba1 = +res.met.includes("BA1");
  f.points = res.points;
  f.missense_score = res.scores.missense ?? NaN;
  f.splice_score = res.scores.splice ?? NaN;
  const af = v.gn_faf95;
  f.log_af = !nan(af) && (af as number) > 0 ? Math.log10(af as number) : -7;
  f.seen = +(!nan(v.gn_AC) && (v.gn_AC as number) > 0);
  f.hom = nan(v.gn_Hom) ? 0 : (v.gn_Hom as number);
  const cons = CONSEQ.includes(v.consequence) ? v.consequence : "other";
  for (const c of CONSEQ) f[`cons_${c}`] = +(cons === c);
  const g = b.genes[v.gene];
  f.pLI = g?.pli ?? NaN; f.loftool = g?.lt ?? NaN;
  f.dominant = +!!g?.dom; f.recessive = +!!g?.rec; f.haploinsufficient = +!!g?.hi;
  const probs = predictMulti(b.metaModel, vector(b.metaModel, f));
  const classes: string[] = b.meta.classes;
  let best = 0;
  probs.forEach((p, i) => { if (p > probs[best]) best = i; });
  return { classification: classes[best], probabilities: Object.fromEntries(classes.map((c, i) => [c, Math.round(probs[i] * 1e4) / 1e4])) };
}
