import { Confusion, HBars } from "../charts";
import { useApp } from "../ctx";

const p = (x: number | undefined, d = 1) => (x === undefined || x === null ? "" : `${(x * 100).toFixed(d)}%`);

export default function Evaluation() {
  const { b } = useApp();
  const ev = b.meta.evaluation;
  if (!ev) return <p className="muted">No evaluation summary in this bundle.</p>;
  const blind = ev.classification_blind, dev = ev.classification_dev;
  const set = blind || dev;
  const retr = ev.retrieval;
  return (
    <div className="stack">
      <div>
        <h1>Evaluation</h1>
        <p className="lede">The blind set is every ClinVar variant classified by an expert panel or practice guideline, frozen and hashed before any evidence table or model existed and scored once. Tuning used only the development set. The engine sees no case data, while the panels did, so its pathogenic calls are deliberately conservative.</p>
      </div>
      <section className="panel scroll">
        <h2 style={{ marginTop: 0 }}>{blind ? `Blind set (${blind.rules_points.n.toLocaleString()} variants)` : "Development set"}</h2>
        <table>
          <thead><tr><th>Method</th><th className="num">Exact five tier</th><th className="num">Within one tier</th><th className="num">Three tier</th><th className="num">Balanced three tier</th><th className="num">Severe discordance</th><th className="num">P/LP precision</th><th className="num">B/LB precision</th></tr></thead>
          <tbody>{[["Point system (primary)", set.rules_points], ["2015 combining rules", set.rules_2015], ["Learned verdict", set.learned_meta]].map(([n, m]: any) => (
            <tr key={n}><td>{n}</td><td className="num">{p(m.exact_5tier)}</td><td className="num">{p(m.within_one_tier)}</td><td className="num">{p(m.accuracy_3tier)}</td><td className="num">{p(m.balanced_accuracy_3tier)}</td><td className="num">{p(m.severe_discordance_rate, 2)}</td><td className="num">{p(m.per_class_3tier["P/LP"].precision)}</td><td className="num">{p(m.per_class_3tier["B/LB"].precision)}</td></tr>))}</tbody>
        </table>
      </section>
      <div className="grid2">
        <section className="panel"><h3 style={{ marginTop: 0 }}>Confusion matrix, point system</h3><Confusion m={set.rules_points.confusion_5tier} classes={set.rules_points.classes} /></section>
        {blind?.by_panel && (
          <section className="panel scroll"><h3 style={{ marginTop: 0 }}>By expert panel</h3>
            <table><thead><tr><th>Panel</th><th className="num">n</th><th className="num">Three tier</th><th className="num">Exact</th><th className="num">Severe</th></tr></thead>
              <tbody>{Object.entries(blind.by_panel as Record<string, any>).map(([k, v]) => <tr key={k}><td>{k}</td><td className="num">{v.n}</td><td className="num">{p(v.accuracy_3tier)}</td><td className="num">{p(v.exact_5tier)}</td><td className="num">{v.severe_discordance}</td></tr>)}</tbody></table>
          </section>)}
      </div>
      <section className="panel">
        <h3 style={{ marginTop: 0 }}>Ablations: three tier agreement when a family of criteria is removed</h3>
        <HBars data={Object.entries(set.ablations as Record<string, any>).map(([k, v]) => ({ label: k, value: v.accuracy_3tier }))} max={1} fmt={(v) => p(v)} />
      </section>
      <section className="panel scroll">
        <h3 style={{ marginTop: 0 }}>Posterior reliability: share of reference pathogenic variants in each posterior band</h3>
        <table><thead><tr><th>Posterior band</th><th className="num">Variants</th><th className="num">Reference P/LP share</th></tr></thead>
          <tbody>{set.posterior_reliability.map((r: any) => <tr key={r.band}><td>{r.band}</td><td className="num">{r.n}</td><td className="num">{p(r.reference_pathogenic_share)}</td></tr>)}</tbody></table>
      </section>
      {retr && (
        <section className="panel scroll">
          <h3 style={{ marginTop: 0 }}>Copilot retrieval ({retr.blind ? "blind 30 questions" : "development 60 questions"})</h3>
          <table><thead><tr><th>Retriever</th><th className="num">Recall@5</th><th className="num">Strict recall@5</th><th className="num">MRR@10</th><th className="num">nDCG@10</th></tr></thead>
            <tbody>{Object.values((retr.blind || retr.dev) as Record<string, any>).map((m: any) => <tr key={m.label}><td>{m.label}</td><td className="num">{m["recall@5"].toFixed(3)}</td><td className="num">{m["strict_recall@5"].toFixed(3)}</td><td className="num">{m["mrr@10"].toFixed(3)}</td><td className="num">{m["ndcg@10"].toFixed(3)}</td></tr>)}</tbody></table>
          {retr.answers && <p className="small">Answers: {p(retr.answers.verified_share)} of sentences verified, {p(retr.answers.answers_citing_relevant_evidence)} cite relevant evidence, safety probes passed {retr.answers.safety_passed} of {retr.answers.safety_probes.length}.</p>}
        </section>)}
      {ev.insilico_report && (
        <section className="panel">
          <h3 style={{ marginTop: 0 }}>Missense predictor, gene grouped cross validation</h3>
          <HBars data={[{ label: "Allelion missense model", value: ev.insilico_report.missense.grouped_cv.auroc, color: "var(--ink)" },
            ...Object.entries(ev.insilico_report.missense.baselines as Record<string, any>).map(([k, v]) => ({ label: k, value: v.auroc, color: "var(--ink-3)" }))]} max={1} fmt={(v) => v.toFixed(3)} />
          <p className="small muted">AUROC; 0.5 is chance.</p>
        </section>)}
    </div>
  );
}
