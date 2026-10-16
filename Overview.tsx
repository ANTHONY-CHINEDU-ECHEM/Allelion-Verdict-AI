import { useMemo } from "react";
import { CLS_SHORT, EvidenceRuler } from "../charts";
import { useApp } from "../ctx";

const WORKSPACES: [string, string, string][] = [
  ["classify", "Classify a variant", "All 26 automatable and case level criteria with rationale, evidence, points and the Bayesian posterior, plus a learned verdict for comparison."],
  ["copilot", "Evidence copilot", "Questions answered from 37,881 passages; every sentence cites its evidence and is verified before it is shown."],
  ["genes", "Gene explorer", "Diseases, constraint, the classified spectrum, the protein map and the gene level criteria it implies."],
  ["conflicts", "Conflicting interpretations", "ClinVar disagreements side by side with the engine's evidence based verdict."],
  ["batch", "Batch triage", "Classify a list of variants and download the result."],
  ["phenotype", "Phenotype matcher", "Rank genes from HPO terms; the same score decides PP4."],
  ["calibration", "Calibration lab", "How predictor scores become evidence strengths."],
  ["literature", "Literature", "Variant mentions extracted from five corpora and linked to ClinVar."],
  ["graph", "Knowledge graph", "A gene's diseases, phenotypes, variants and publications."],
  ["evaluation", "Evaluation", "Blind expert panel concordance and every other metric."],
];

export default function Overview() {
  const { b, engine, go } = useApp();
  const demo = useMemo(() => b.variants.find((v) => v.gene === "MYH7" && (v.hgvs_c || "").endsWith("c.1208G>A")) || b.variants[0], [b]);
  const res = useMemo(() => engine.evaluate(demo), [engine, demo]);
  const ev = b.meta.evaluation;
  const blind = ev?.classification_blind;
  const kb = ev?.knowledge_base;
  return (
    <div className="stack">
      <section>
        <h1 style={{ fontSize: "clamp(30px, 4.6vw, 48px)", maxWidth: "20ch" }}>Every criterion, every number, traceable to its evidence.</h1>
        <p className="lede">Allelion Verdict AI classifies germline variants with the ACMG/AMP framework and the ClinGen refinements, using population frequency, anchored exon models, calibrated predictors and evidence from other classified variants. It runs entirely in this browser.</p>
      </section>
      <section className="panel hero" style={{ cursor: "pointer" }} onClick={() => go("classify", { id: demo.variation_id! })}>
        <p className="hero-variant"><span className="gene">{demo.gene}</span>{(demo.hgvs_c || "").split(":").pop()}</p>
        <div className="verdict"><span className={`verdict-word cls-${CLS_SHORT[res.classification]}`}>{res.classification}</span><span className="muted">{res.points} points from {res.met.join(", ") || "no criteria"}</span></div>
        <div className="ruler"><EvidenceRuler criteria={res.criteria} points={res.points} ba1={res.met.includes("BA1")} /></div>
        <p className="small muted" style={{ margin: 0 }}>Open this example in the classifier</p>
      </section>
      {blind && (
        <section className="grid3">
          <div className="panel stat"><b>{(blind.rules_points.accuracy_3tier * 100).toFixed(1)}%</b><span>three tier agreement with {blind.rules_points.n.toLocaleString()} expert panel classifications, scored once</span></div>
          <div className="panel stat"><b>{(blind.rules_points.severe_discordance_rate * 100).toFixed(2)}%</b><span>called in the opposite direction (pathogenic as benign or the reverse)</span></div>
          <div className="panel stat"><b>{(blind.rules_points.per_class_3tier["P/LP"].precision * 100).toFixed(1)}%</b><span>of pathogenic or likely pathogenic calls confirmed by the expert panels</span></div>
        </section>)}
      <section className="panel">
        <h2 style={{ marginTop: 0 }}>Workspaces</h2>
        {WORKSPACES.map(([k, t, d]) => (
          <div key={k} className="crit" style={{ gridTemplateColumns: "220px 1fr" }}>
            <button className="btn secondary" onClick={() => go(k)} style={{ justifySelf: "start" }}>{t}</button>
            <p style={{ margin: 0 }}>{d}</p>
          </div>))}
      </section>
      {kb && (
        <section className="panel">
          <h2 style={{ marginTop: 0 }}>Knowledge base</h2>
          <dl className="kv">
            <dt>ClinVar alleles</dt><dd>{kb.variants.toLocaleString()} across {kb.genes.toLocaleString()} genes, joined to gnomAD v2 exomes and genomes</dd>
            <dt>Exon models</dt><dd>{kb.genes_with_exon_model.toLocaleString()} genes anchored to their reporting transcript</dd>
            <dt>Diseases and phenotypes</dt><dd>{kb.diseases.toLocaleString()} disease profiles and {kb.hpo_terms.toLocaleString()} HPO terms</dd>
            <dt>Literature</dt><dd>{kb.literature_documents.toLocaleString()} documents with {kb.literature_variant_links.toLocaleString()} variant links</dd>
            <dt>Knowledge graph</dt><dd>{kb.graph.nodes.toLocaleString()} nodes and {kb.graph.edges.toLocaleString()} edges</dd>
            <dt>In this demo</dt><dd>{b.variants.length.toLocaleString()} variants with gnomAD counts; any other variant can be entered by HGVS</dd>
          </dl>
        </section>)}
      <p className="small muted">For research and education. Not a medical device, and not for individual clinical decisions.</p>
    </div>
  );
}
