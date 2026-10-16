// Small SVG charts drawn by hand, themed with CSS variables.
import type { CritOut } from "./engine/combine";
import { POINTS } from "./engine/combine";

export const CLS_SHORT: Record<string, string> = { Pathogenic: "P", "Likely pathogenic": "LP", "Uncertain significance": "VUS", "Likely benign": "LB", Benign: "B" };
const BANDS = [
  { lo: -14, hi: -7, c: "var(--b)", label: "Benign" },
  { lo: -7, hi: -1, c: "var(--lb)", label: "Likely benign" },
  { lo: -1, hi: 6, c: "var(--vus)", label: "Uncertain" },
  { lo: 6, hi: 10, c: "var(--lp)", label: "Likely pathogenic" },
  { lo: 10, hi: 18, c: "var(--p)", label: "Pathogenic" },
];

/** The evidence ruler: met criteria stacked as blocks on a points axis, benign to the left, pathogenic to the right. */
export function EvidenceRuler({ criteria, points, ba1 }: { criteria: CritOut[]; points: number; ba1: boolean }) {
  const W = 760, H = 146, lo = -14, hi = 18;
  const x = (p: number) => 20 + ((p - lo) / (hi - lo)) * (W - 40);
  const met = criteria.filter((c) => c.status === "met" && c.strength !== "StandAlone");
  const path = met.filter((c) => c.direction === "pathogenic");
  const ben = met.filter((c) => c.direction === "benign");
  let cur = 0;
  const pBlocks = path.map((c) => { const w = POINTS[c.strength]; const b = { c, from: cur, to: cur + w }; cur += w; return b; });
  cur = 0;
  const bBlocks = ben.map((c) => { const w = POINTS[c.strength]; const b = { c, from: cur - w, to: cur }; cur -= w; return b; });
  const marker = ba1 ? lo + 0.5 : Math.max(lo + 0.3, Math.min(hi - 0.3, points));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Evidence total ${points} points`}>
      {BANDS.map((b) => (
        <g key={b.label}>
          <rect x={x(b.lo)} y={92} width={x(b.hi) - x(b.lo)} height={8} fill={b.c} opacity={0.85} />
          <text x={(x(b.lo) + x(b.hi)) / 2} y={118} textAnchor="middle" fontSize={12} fill="var(--ink-2)">{b.label}</text>
        </g>
      ))}
      {[-14, -7, -1, 0, 6, 10, 18].map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={88} y2={104} stroke="var(--ink-3)" strokeWidth={t === 0 ? 1.5 : 1} />
          <text x={x(t)} y={138} textAnchor="middle" fontSize={11.5} fill="var(--ink-3)" fontFamily="var(--mono)">{t > 0 ? `+${t}` : t}</text>
        </g>
      ))}
      {[...pBlocks, ...bBlocks].map(({ c, from, to }, k) => {
        const w = x(to) - x(from);
        const wide = w > 52;
        const label = c.label.replace("_Supporting", " sup").replace("_Moderate", " mod").replace("_Strong", " str").replace("_VeryStrong", " vs");
        return (
          <g key={c.code}>
            <rect x={x(from) + 1} y={48} width={Math.max(w - 2, 2)} height={34} rx={3} fill={c.direction === "pathogenic" ? "var(--p)" : "var(--b)"} opacity={0.92}>
              <title>{`${c.label}: ${c.direction === "pathogenic" ? "+" : "-"}${POINTS[c.strength]} points`}</title>
            </rect>
            {wide ? <text x={(x(from) + x(to)) / 2} y={70} textAnchor="middle" fontSize={12} fill="#fff" fontFamily="var(--mono)">{label}</text>
              : <text x={(x(from) + x(to)) / 2} y={k % 2 ? 26 : 40} textAnchor="middle" fontSize={10.5} fill="var(--ink-2)" fontFamily="var(--mono)">{c.code}</text>}
          </g>);
      })}
      <line x1={x(marker)} x2={x(marker)} y1={44} y2={104} stroke="var(--ink)" strokeWidth={2} />
      <text x={x(marker) + 6} y={100} fontSize={13} fontWeight={700} fill="var(--ink)">{ba1 ? "BA1" : `${points > 0 ? "+" : ""}${points}`}</text>
    </svg>
  );
}

export function HBars({ data, max, color = "var(--ink)", fmt = (v: number) => String(v) }: { data: { label: string; value: number; color?: string }[]; max?: number; color?: string; fmt?: (v: number) => string }) {
  const m = max ?? Math.max(...data.map((d) => d.value), 1e-9);
  const W = 560, rowH = 26, lab = 230;
  return (
    <svg viewBox={`0 0 ${W} ${data.length * rowH + 4}`} role="img" style={{ width: "100%", maxWidth: W, height: "auto", display: "block" }}>
      {data.map((d, i) => (
        <g key={d.label} transform={`translate(0 ${i * rowH})`}>
          <text x={lab - 8} y={17} textAnchor="end" fontSize={12.5} fill="var(--ink-2)">{d.label}</text>
          <rect x={lab} y={5} width={Math.max(((W - lab - 60) * d.value) / m, 1)} height={15} rx={2} fill={d.color || color} />
          <text x={lab + Math.max(((W - lab - 60) * d.value) / m, 1) + 6} y={17} fontSize={12.5} fill="var(--ink)" fontFamily="var(--mono)">{fmt(d.value)}</text>
        </g>
      ))}
    </svg>
  );
}

/** Calibration curve: local posterior with its bootstrap band and the strength thresholds. */
export function CalibrationChart({ cal, title, mark }: { cal: any; title: string; mark?: number | null }) {
  const W = 620, H = 300, L = 48, R = 16, T = 16, B = 40;
  const g: number[] = cal.grid, p: number[] = cal.posterior, lo: number[] = cal.lower, up: number[] = cal.upper;
  const x = (v: number) => L + v * (W - L - R);
  const ymax = 1;
  const y = (v: number) => T + (1 - Math.min(v, ymax) / ymax) * (H - T - B);
  const path = (arr: number[]) => arr.map((v, i) => `${i ? "L" : "M"}${x(g[i]).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const band = path(up) + g.slice().reverse().map((gv, i) => `L${x(gv).toFixed(1)},${y(lo[lo.length - 1 - i]).toFixed(1)}`).join("") + "Z";
  const prior = cal.prior;
  const post = (lr: number) => (lr * prior) / ((lr - 1) * prior + 1);
  const lines = Object.entries(cal.lr as Record<string, number>).filter(([s]) => s !== "VeryStrong");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title} style={{ width: "100%", height: "auto" }}>
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (<g key={t}><line x1={L} x2={W - R} y1={y(t)} y2={y(t)} stroke="var(--rule)" /><text x={L - 6} y={y(t) + 4} fontSize={11} textAnchor="end" fill="var(--ink-3)">{t}</text></g>))}
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (<text key={t} x={x(t)} y={H - 20} fontSize={11} textAnchor="middle" fill="var(--ink-3)">{t}</text>))}
      <text x={(L + W - R) / 2} y={H - 4} fontSize={12} textAnchor="middle" fill="var(--ink-2)">Predictor score</text>
      <path d={band} fill="var(--ink-3)" opacity={0.18} />
      <path d={path(p)} fill="none" stroke="var(--ink)" strokeWidth={2} />
      {lines.map(([s, lr]) => (
        <g key={s}>
          <line x1={L} x2={W - R} y1={y(post(lr))} y2={y(post(lr))} stroke="var(--p)" strokeDasharray="4 4" opacity={0.7} />
          <text x={W - R - 2} y={y(post(lr)) - 4} fontSize={11} textAnchor="end" fill="var(--p)">PP3 {s}</text>
          <line x1={L} x2={W - R} y1={y(post(1 / lr))} y2={y(post(1 / lr))} stroke="var(--b)" strokeDasharray="4 4" opacity={0.7} />
        </g>
      ))}
      {Object.entries(cal.pathogenic_thresholds as Record<string, number | null>).filter(([, t]) => t !== null).map(([s, t]) => (
        <line key={`p${s}`} x1={x(t as number)} x2={x(t as number)} y1={T} y2={H - B} stroke="var(--p)" opacity={0.35} />
      ))}
      {Object.entries(cal.benign_thresholds as Record<string, number | null>).filter(([, t]) => t !== null).map(([s, t]) => (
        <line key={`b${s}`} x1={x(t as number)} x2={x(t as number)} y1={T} y2={H - B} stroke="var(--b)" opacity={0.35} />
      ))}
      {mark !== undefined && mark !== null && <circle cx={x(mark)} cy={y(p[Math.min(g.findIndex((v) => v >= mark), p.length - 1) >= 0 ? g.findIndex((v) => v >= mark) : p.length - 1])} r={6} fill="var(--vus)" stroke="var(--sheet)" strokeWidth={2} />}
      <text x={L + 4} y={T + 12} fontSize={12} fill="var(--ink-2)">Posterior probability of pathogenicity</text>
    </svg>
  );
}

/** Protein lollipop: pool missense classifications by residue, truncating pathogenic variants below the axis. */
export function Lollipop({ length, missense, trunc, highlight }: { length: number; missense: [number, string, number][]; trunc: number[]; highlight?: number | null }) {
  const W = 760, H = 190, L = 20, R = 20, axis = 110;
  const x = (p: number) => L + (p / Math.max(length, 1)) * (W - L - R);
  const col = ["var(--p)", "var(--lp)", "var(--vus)", "var(--lb)", "var(--b)"];
  const byPos = new Map<number, number[]>();
  missense.filter(([, , c]) => c !== 2).forEach(([p, , c]) => { const a = byPos.get(p) || []; a.push(c); byPos.set(p, a); });
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Classified variants along the protein" style={{ width: "100%", height: "auto" }}>
      <rect x={L} y={axis - 5} width={W - L - R} height={10} rx={3} fill="var(--rule-2)" />
      {[...byPos.entries()].map(([p, cs]) => cs.map((c, k) => (
        <g key={`${p}-${k}`}>
          <line x1={x(p)} x2={x(p)} y1={axis - 6} y2={axis - 18 - k * 7} stroke="var(--ink-3)" strokeWidth={0.6} />
          <circle cx={x(p)} cy={axis - 20 - k * 7} r={3} fill={col[c]} />
        </g>
      )))}
      {trunc.map((p, i) => (<line key={i} x1={x(p)} x2={x(p)} y1={axis + 7} y2={axis + 26} stroke="var(--p)" strokeWidth={1} opacity={0.6} />))}
      {highlight ? (<g><line x1={x(highlight)} x2={x(highlight)} y1={14} y2={axis + 34} stroke="var(--ink)" strokeWidth={1.5} strokeDasharray="3 3" /><text x={x(highlight)} y={12} textAnchor="middle" fontSize={12} fill="var(--ink)">this variant</text></g>) : null}
      <text x={L} y={H - 30} fontSize={11.5} fill="var(--ink-3)">1</text>
      <text x={W - R} y={H - 30} fontSize={11.5} fill="var(--ink-3)" textAnchor="end">{length}</text>
      <text x={L} y={H - 8} fontSize={12} fill="var(--ink-2)">Missense in the evidence pool above the axis; pathogenic truncating variants below</text>
    </svg>
  );
}

/** Exon map in coding coordinates with the variant position and the decay escape zone. */
export function ExonMap({ exons, cds, pos }: { exons: number[][]; cds: number | null; pos?: number | null }) {
  const W = 760, H = 86, L = 10, R = 10;
  const lo = exons[0][2], hi = exons[exons.length - 1][3];
  const x = (v: number) => L + ((v - lo) / Math.max(hi - lo, 1)) * (W - L - R);
  const n = exons.length;
  const zoneStart = n >= 2 ? exons[n - 2][3] - 50 : exons[n - 1][2];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Exon structure" style={{ width: "100%", height: "auto" }}>
      <rect x={x(Math.max(zoneStart, lo))} y={14} width={x(hi) - x(Math.max(zoneStart, lo))} height={34} fill="var(--vus-bg)" />
      <line x1={L} x2={W - R} y1={31} y2={31} stroke="var(--rule-2)" />
      {exons.map((e, i) => {
        const coding = cds ? !(e[3] < 1 || e[2] > cds) : true;
        return <rect key={i} x={x(e[2])} y={coding ? 20 : 25} width={Math.max(x(e[3]) - x(e[2]) - 1, 1.5)} height={coding ? 22 : 12} fill={coding ? "var(--ink-2)" : "var(--rule-2)"} rx={1.5}><title>{`Exon ${i + 1}: c.${e[2]} to c.${e[3]}`}</title></rect>;
      })}
      {pos !== undefined && pos !== null && pos >= lo && pos <= hi && (<g><line x1={x(pos)} x2={x(pos)} y1={8} y2={54} stroke="var(--p)" strokeWidth={2} /></g>)}
      <text x={L} y={70} fontSize={12} fill="var(--ink-2)">{n} exons; shaded: last exon and the final 50 bases of the penultimate exon, where a premature stop escapes decay</text>
    </svg>
  );
}

export function Confusion({ m, classes }: { m: number[][]; classes: string[] }) {
  const max = Math.max(...m.flat(), 1);
  return (
    <div className="scroll">
      <table>
        <thead><tr><th>Reference \ predicted</th>{classes.map((c) => <th key={c} className="num">{CLS_SHORT[c]}</th>)}</tr></thead>
        <tbody>{m.map((row, i) => (
          <tr key={i}><th>{classes[i]}</th>{row.map((v, j) => (
            <td key={j} className="num" style={{ background: `color-mix(in srgb, ${i === j ? "var(--ok)" : "var(--p)"} ${Math.round((v / max) * 45)}%, transparent)` }}>{v}</td>
          ))}</tr>))}
        </tbody>
      </table>
    </div>
  );
}
