// Combining criteria: Bayesian points and the 2015 rules. Mirrors allelion/acmg/combine.py.
export const POINTS: Record<string, number> = { VeryStrong: 8, Strong: 4, Moderate: 2, Supporting: 1 };

export interface CritOut {
  code: string; title: string; direction: "pathogenic" | "benign"; status: "met" | "not_met" | "not_evaluated"; strength: string;
  default_strength: string; label: string; rationale: string; evidence: { kind: string; ref: string; detail: string }[];
}

export function pointsOf(criteria: CritOut[]): number {
  let t = 0;
  for (const c of criteria) {
    if (c.status !== "met" || c.strength === "StandAlone") continue;
    const p = POINTS[c.strength];
    t += c.direction === "pathogenic" ? p : -p;
  }
  return t;
}

export function classFromPoints(p: number, ba1: boolean): string {
  if (ba1) return "Benign";
  if (p >= 10) return "Pathogenic";
  if (p >= 6) return "Likely pathogenic";
  if (p >= 0) return "Uncertain significance";
  if (p >= -6) return "Likely benign";
  return "Benign";
}

export function posterior(points: number, prior = 0.1): number {
  const odds = Math.pow(350, points / 8);
  return (odds * prior) / ((odds - 1) * prior + 1);
}

export function class2015(criteria: CritOut[]): string {
  const met = criteria.filter((c) => c.status === "met");
  const cnt = (dir: string, s: string[]) => met.filter((c) => c.direction === dir && s.includes(c.strength)).length;
  const pvs = cnt("pathogenic", ["VeryStrong"]), ps = cnt("pathogenic", ["Strong"]), pm = cnt("pathogenic", ["Moderate"]), pp = cnt("pathogenic", ["Supporting"]);
  const ba = met.some((c) => c.direction === "benign" && c.strength === "StandAlone");
  const bs = cnt("benign", ["Strong", "VeryStrong"]), bp = cnt("benign", ["Supporting", "Moderate"]);
  const pathogenic = (pvs >= 1 && (ps >= 1 || pm >= 2 || (pm === 1 && pp === 1) || pp >= 2)) || ps >= 2 ||
    (ps === 1 && (pm >= 3 || (pm === 2 && pp >= 2) || (pm === 1 && pp >= 4)));
  const lp = (pvs >= 1 && pm === 1) || (ps === 1 && pm >= 1 && pm <= 2) || (ps === 1 && pp >= 2) || pm >= 3 || (pm === 2 && pp >= 2) || (pm === 1 && pp >= 4);
  const benign = ba || bs >= 2;
  const lb = (bs === 1 && bp >= 1) || bp >= 2;
  if ((pathogenic || lp) && (benign || lb)) return "Uncertain significance";
  if (pathogenic) return "Pathogenic";
  if (lp) return "Likely pathogenic";
  if (benign) return "Benign";
  if (lb) return "Likely benign";
  return "Uncertain significance";
}
