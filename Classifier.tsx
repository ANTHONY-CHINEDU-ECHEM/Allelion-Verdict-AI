import { useEffect, useMemo, useState } from "react";
import { CalibrationChart, CLS_SHORT, EvidenceRuler, ExonMap, Lollipop } from "../charts";
import { CONSEQ_WORD, fmtAf, useApp, variantName } from "../ctx";
import type { Variant } from "../engine/data";
import { prepare, type CaseEvidence } from "../engine/engine";
import { learnedVerdict } from "../engine/meta";

const EXAMPLES = [
  ["BRCA1", "c.68_69delAG"], ["MYH7", "c.1208G>A"], ["CFTR", "c.1521_1523delCTT"], ["MLH1", "c.350C>T"],
  ["PTPN11", "c.922A>G"], ["BRCA2", "c.8754+5G>A"], ["MSH2", "c.942+3A>T"], ["LDLR", "c.1775G>A"],
];

function findVariant(vs: Variant[], gene: string, c: string) {
  return vs.find((v) => v.gene === gene && (v.hgvs_c || "").split(":").pop() === c);
}

export default function Classifier() {
  const { b, engine, params, go, phenotype } = useApp();
  const [query, setQuery] = useState("");
  const [focus, setFocus] = useState(false);
  const [variant, setVariant] = useState<Variant | null>(null);
  const [kase, setKase] = useState<CaseEvidence>({});
  const [tab, setTab] = useState<"met" | "not_met" | "not_evaluated">("met");
  const [custom, setCustom] = useState({ gene: "", c: "", p: "", faf: "", ac: "", an: "", hom: "" });
  const [error, setError] = useState("");
  const [hpoQuery, setHpoQuery] = useState("");
  const [terms, setTerms] = useState<string[]>([]);

  useEffect(() => {
    if (params.id && b.byId.get(params.id)) { setVariant(b.byId.get(params.id)!); setKase({}); setTerms([]); }
  }, [params.id, b]);
  useEffect(() => {
    if (!variant && !params.id) { const v = findVariant(b.variants, "MYH7", "c.1208G>A") || b.variants[0]; setVariant(v); }
  }, [b, variant, params.id]);

  const suggestions = useMemo(() => {
    const q = query.trim();
    if (q.length < 2) return [];
    const [g, ...rest] = q.split(/\s+/);
    const tail = rest.join(" ").toLowerCase();
    const G = g.toUpperCase();
    const out: Variant[] = [];
    for (const v of b.variants) {
      if (!v.gene.startsWith(G)) continue;
      const c = (v.hgvs_c || "").split(":").pop()!.toLowerCase();
      const p = (v.hgvs_p || "").split(":").pop()!.toLowerCase();
      if (!tail || c.includes(tail) || p.includes(tail) || v.protein_change.toLowerCase() === tail) out.push(v);
      if (out.length >= 30) break;
    }
    return out;
  }, [query, b]);

  const phenoMatch = useMemo(() => (variant && terms.length ? phenotype().match(variant.gene, terms) : null), [variant, terms, phenotype]);
  const res = useMemo(() => {
    if (!variant) return null;
    try { return engine.evaluate(variant, { ...kase, phenotype_match: phenoMatch }); } catch (e) { setError(String(e)); return null; }
  }, [variant, kase, engine, phenoMatch]);
  const learned = useMemo(() => (res && variant ? learnedVerdict(b, res, variant) : null), [res, variant, b]);

  const hpoHits = useMemo(() => {
    const q = hpoQuery.trim().toLowerCase();
    if (q.length < 3) return [];
    const out: [string, string][] = [];
    for (const [id, [name]] of Object.entries(b.hpo)) { if (name.toLowerCase().includes(q)) out.push([id, name]); if (out.length >= 12) break; }
    return out;
  }, [hpoQuery, b]);

  function runCustom() {
    setError("");
    try {
      const known = findVariant(b.variants, custom.gene.toUpperCase(), custom.c.trim());
      if (known && !custom.faf && !custom.ac) { setVariant(known); return; }
      const pop = custom.faf || custom.ac ? { faf95: custom.faf ? +custom.faf : undefined, grpmax_af: custom.faf ? +custom.faf : undefined, ac: custom.ac ? +custom.ac : 0, an: custom.an ? +custom.an : undefined, hom: custom.hom ? +custom.hom : 0 } : null;
      setVariant(prepare(custom.gene.toUpperCase().trim(), custom.c.trim(), custom.p.trim() || null, pop));
      setKase({});
    } catch (e: any) { setError(`${e.message}. Use a coding description such as c.1208G>A or c.68_69delAG.`); }
  }

  if (!variant || !res) return <p className="muted">{error || "Loading variant"}</p>;
  const short = CLS_SHORT[res.classification];
  const tx = b.transcripts[variant.gene];
  const g = b.genes[variant.gene];
  const ba1 = res.met.includes("BA1");
  const shown = res.criteria.filter((c) => c.status === tab);
  const counts = { met: res.criteria.filter((c) => c.status === "met").length, not_met: res.criteria.filter((c) => c.status === "not_met").length, not_evaluated: res.criteria.filter((c) => c.status === "not_evaluated").length };
  const set = <K extends keyof CaseEvidence>(k: K, v: CaseEvidence[K]) => setKase({ ...kase, [k]: v });

  return (
    <div className="stack">
      <div>
        <h1>Classify a variant</h1>
        <p className="lede">Every ACMG/AMP criterion is evaluated from population data, exon structure, calibrated predictors and the evidence pool. The variant's own ClinVar classification is never used as evidence.</p>
        <div style={{ position: "relative", maxWidth: 620 }}>
          <input type="text" aria-label="Search variants by gene and HGVS" placeholder="Search by gene and HGVS, for example MLH1 c.350" value={query} style={{ width: "100%" }}
            onChange={(e) => setQuery(e.target.value)} onFocus={() => setFocus(true)} onBlur={() => setTimeout(() => setFocus(false), 150)} />
          {focus && suggestions.length > 0 && (
            <div className="suggest" role="listbox">
              {suggestions.map((v) => (
                <button key={v.variation_id!} onMouseDown={() => { setVariant(v); setKase({}); setTerms([]); setQuery(""); go("classify", { id: v.variation_id! }); }}>
                  <b>{v.gene}</b> <span className="mono">{(v.hgvs_c || "").split(":").pop()}</span> <span className="muted">{(v.hgvs_p || "").split(":").pop()} {v.set === "blind" ? "(blind set)" : ""}</span>
                </button>))}
            </div>)}
        </div>
        <div className="row" style={{ marginTop: 10 }}>
          {EXAMPLES.filter(([gg, c]) => findVariant(b.variants, gg, c)).map(([gg, c]) => (
            <button key={gg + c} className="chip" onClick={() => { const v = findVariant(b.variants, gg, c)!; setVariant(v); setKase({}); setTerms([]); go("classify", { id: v.variation_id! }); }}>
              <b>{gg}</b> <span className="mono">{c}</span>
            </button>))}
        </div>
      </div>

      <section className="panel hero" aria-live="polite">
        <p className="hero-variant"><span className="gene">{variant.gene}</span>{(variant.hgvs_c || "").split(":").pop()}</p>
        <p className="hero-sub">
          {[(variant.hgvs_c || "").includes(":") ? variant.hgvs_c!.split(":")[0] : "entered by HGVS", (variant.hgvs_p || "").split(":").pop(),
            `${CONSEQ_WORD[variant.consequence] || variant.consequence} variant`,
            variant.set === "blind" ? "in the frozen blind set" : variant.set === "dev" ? "in the development set" : ""].filter(Boolean).join(", ")}
        </p>
        <div className="verdict">
          <span className={`verdict-word cls-${short}`}>{res.classification}</span>
          <span className="muted">{res.points} points; posterior probability of pathogenicity {ba1 ? "not applicable (BA1)" : res.posterior.toFixed(3)}</span>
        </div>
        <div className="ruler"><EvidenceRuler criteria={res.criteria} points={res.points} ba1={ba1} /></div>
        <div className="row small" style={{ gap: 18 }}>
          <span>2015 combining rules: <b>{res.classification_2015}</b></span>
          {learned && <span>Learned verdict: <b>{learned.classification}</b> <span className="muted">({(learned.probabilities[learned.classification] * 100).toFixed(0)}%)</span></span>}
          {variant.clinvar_class && <span>ClinVar reference, not used as evidence: <b>{variant.clinvar_class}</b> <span className="muted">({variant.stars} {variant.stars === 1 ? "star" : "stars"})</span></span>}
        </div>
        {res.notes.length > 0 && <p className="small muted" style={{ marginBottom: 0 }}>{res.notes.join(" ")}</p>}
      </section>

      <div className="grid2">
        <section className="panel">
          <div className="tabs" role="tablist">
            {(["met", "not_met", "not_evaluated"] as const).map((t) => (
              <button key={t} role="tab" aria-selected={tab === t} onClick={() => setTab(t)}>
                {t === "met" ? "Met" : t === "not_met" ? "Not met" : "Awaiting case data"} ({counts[t]})
              </button>))}
          </div>
          {shown.length === 0 && <p className="muted">{tab === "met" ? "No criterion is met with the evidence available." : "None."}</p>}
          {shown.map((c) => (
            <div key={c.code} className={`crit ${c.direction === "pathogenic" ? "path" : "ben"}`}>
              <div className="crit-code">{c.label}<span className="pts">{c.status === "met" ? (c.strength === "StandAlone" ? "stand alone" : (() => { const n = ({ Supporting: 1, Moderate: 2, Strong: 4, VeryStrong: 8 } as any)[c.strength]; return `${c.direction === "pathogenic" ? "+" : "-"}${n} ${n === 1 ? "point" : "points"}`; })()) : c.default_strength.replace("VeryStrong", "very strong").toLowerCase()}</span></div>
              <div>
                <div className="crit-title">{c.title}</div>
                <p>{c.rationale}</p>
                {c.evidence.length > 0 && <div className="ev">{c.evidence.map((e, i) => <div key={i}>{e.ref}: {e.detail}</div>)}</div>}
              </div>
            </div>))}
        </section>

        <div className="stack">
          <section className="panel">
            <h3 style={{ marginTop: 0 }}>Population</h3>
            {variant.population_known === false ? <p className="muted small">No population data supplied, so BA1, BS1, BS2 and PM2 are not evaluated. Enter frequencies below to include them.</p> : (
              <dl className="kv">
                <dt>Filtering allele frequency</dt><dd>{fmtAf(variant.gn_faf95)}</dd>
                <dt>Highest continental frequency</dt><dd>{fmtAf(variant.gn_grpmax_2000 ?? variant.gn_grpmax)}</dd>
                <dt>Allele count</dt><dd>{variant.gn_AC ?? 0}{variant.gn_AN ? ` of ${variant.gn_AN}` : ""}</dd>
                <dt>Homozygotes</dt><dd>{variant.gn_Hom ?? 0}</dd>
                <dt>Source</dt><dd>gnomAD v2 exomes and genomes</dd>
              </dl>)}
          </section>
          {(res.scores.missense !== null || res.scores.splice !== null) && (
            <section className="panel">
              <h3 style={{ marginTop: 0 }}>{res.scores.missense !== null ? "Missense predictor" : "Splice region predictor"}</h3>
              <p className="small muted" style={{ marginTop: 0 }}>Score {(res.scores.missense ?? res.scores.splice)!.toFixed(3)}; the curve is the calibrated posterior with its 90% bootstrap band.</p>
              <CalibrationChart cal={res.scores.missense !== null ? b.missenseCal : b.spliceCal} title="Calibration" mark={res.scores.missense ?? res.scores.splice} />
            </section>)}
          <section className="panel">
            <h3 style={{ marginTop: 0 }}>Gene context: {variant.gene}</h3>
            {g ? (
              <dl className="kv">
                <dt>Inheritance</dt><dd>{g.inh.join(", ") || "not recorded"}</dd>
                <dt>Constraint</dt><dd>pLI {g.pli?.toFixed(2) ?? "n/a"}, LoFtool {g.lt?.toPrecision(3) ?? "n/a"}</dd>
                <dt>Haploinsufficiency</dt><dd>{g.hi ? "curated" : "not curated"}</dd>
                <dt>Diseases</dt><dd>{g.dis.length} records</dd>
              </dl>) : <p className="muted">This gene is not in the knowledge base.</p>}
            <button className="btn secondary" style={{ marginTop: 10 }} onClick={() => go("genes", { gene: variant.gene })}>Open gene explorer</button>
          </section>
        </div>
      </div>

      {tx && (
        <section className="panel">
          <h3 style={{ marginTop: 0 }}>Exon model and position</h3>
          <ExonMap exons={tx.exons} cds={tx.cds} pos={variant.c_region === "cds" ? variant.c_pos : null} />
          {variant.consequence === "missense" && g?.plen && <Lollipop length={g.plen} missense={(b.tables.missense[variant.gene] || []).map((r) => [r[0], r[1], r[2]])} trunc={b.tables.trunc[variant.gene] || []} highlight={variant.p_pos} />}
        </section>)}

      <section className="panel">
        <h2 style={{ marginTop: 0 }}>Add case evidence</h2>
        <p className="small muted">Criteria that need patient or family data stay unevaluated until you add it here. The classification updates as you type.</p>
        <div className="grid3">
          <div className="field"><label htmlFor="dn">Confirmed de novo observations</label>
            <input id="dn" type="number" min={0} value={(kase.de_novo || []).filter((d) => d.confirmed).length}
              onChange={(e) => { const n = +e.target.value; const assumed = (kase.de_novo || []).filter((d) => !d.confirmed); set("de_novo", [...Array.from({ length: n }, () => ({ confirmed: true, phenotype: "consistent" as const })), ...assumed]); }} /></div>
          <div className="field"><label htmlFor="dna">Assumed de novo observations</label>
            <input id="dna" type="number" min={0} value={(kase.de_novo || []).filter((d) => !d.confirmed).length}
              onChange={(e) => { const n = +e.target.value; const conf = (kase.de_novo || []).filter((d) => d.confirmed); set("de_novo", [...conf, ...Array.from({ length: n }, () => ({ confirmed: false, phenotype: "consistent" as const }))]); }} /></div>
          <div className="field"><label htmlFor="seg">Informative meioses with cosegregation</label>
            <input id="seg" type="number" min={0} value={kase.segregation_meioses ?? 0} onChange={(e) => set("segregation_meioses", +e.target.value || undefined)} /></div>
          <div className="field"><label htmlFor="trans">In trans with a pathogenic variant (probands)</label>
            <input id="trans" type="number" min={0} value={(kase.in_trans || []).length} onChange={(e) => set("in_trans", Array.from({ length: +e.target.value }, () => ({ partner_class: "Pathogenic", phase: "trans" as const })))} /></div>
          <div className="field"><label htmlFor="prob">Unrelated affected probands (PS4 counting)</label>
            <input id="prob" type="number" min={0} value={kase.probands ?? 0} onChange={(e) => set("probands", +e.target.value || undefined)} /></div>
          <div className="field"><label htmlFor="fn">Functional assay</label>
            <select id="fn" value={kase.functional ? `${kase.functional.direction}:${kase.functional.strength}` : ""} onChange={(e) => { const [d, s] = e.target.value.split(":"); set("functional", e.target.value ? { direction: d as any, strength: s, assay: "validated functional assay" } : undefined); }}>
              <option value="">None</option>
              <option value="damaging:Supporting">Damaging, supporting</option><option value="damaging:Moderate">Damaging, moderate</option><option value="damaging:Strong">Damaging, strong</option>
              <option value="normal:Supporting">Normal function, supporting</option><option value="normal:Strong">Normal function, strong</option>
            </select></div>
        </div>
        <div className="row" style={{ marginTop: 12 }}>
          <label className="chip"><input type="checkbox" checked={!!kase.nonsegregation} onChange={(e) => set("nonsegregation", e.target.checked)} /> Affected relative without the variant (BS4)</label>
          <label className="chip"><input type="checkbox" checked={!!kase.alternate_cause} onChange={(e) => set("alternate_cause", e.target.checked)} /> Alternate molecular cause found (BP5)</label>
        </div>
        <h3>Patient phenotype for PP4</h3>
        <div style={{ position: "relative", maxWidth: 480 }}>
          <input type="text" aria-label="Search phenotype terms" placeholder="Search HPO terms, for example arachnodactyly" value={hpoQuery} onChange={(e) => setHpoQuery(e.target.value)} style={{ width: "100%" }} />
          {hpoHits.length > 0 && <div className="suggest">{hpoHits.map(([id, name]) => <button key={id} onClick={() => { setTerms([...new Set([...terms, id])]); setHpoQuery(""); }}>{name} <span className="muted mono">{id}</span></button>)}</div>}
        </div>
        <div className="row" style={{ marginTop: 8 }}>
          {terms.map((t) => <button key={t} className="chip" aria-pressed="true" onClick={() => setTerms(terms.filter((x) => x !== t))}>{b.hpo[t]?.[0]} (remove)</button>)}
        </div>
        {phenoMatch && <p className="small">Similarity {phenoMatch.score.toFixed(2)} to {phenoMatch.best_disease ? b.diseases[phenoMatch.best_disease]?.[0] : "no disease"}; empirical p {phenoMatch.p_value} against 300 background genes; {phenoMatch.specific ? "a disease with a single genetic cause" : "a disease with several genetic causes, so PP4 does not apply"}.</p>}
        <button className="btn secondary" onClick={() => { setKase({}); setTerms([]); }}>Clear case evidence</button>
      </section>

      <section className="panel">
        <h2 style={{ marginTop: 0 }}>Classify a variant that is not listed</h2>
        <div className="grid3">
          <div className="field"><label htmlFor="cg">Gene</label><input id="cg" type="text" value={custom.gene} onChange={(e) => setCustom({ ...custom, gene: e.target.value })} placeholder="TP53" /></div>
          <div className="field"><label htmlFor="cc">Coding change</label><input id="cc" type="text" value={custom.c} onChange={(e) => setCustom({ ...custom, c: e.target.value })} placeholder="c.743G>A" /></div>
          <div className="field"><label htmlFor="cp">Protein change (optional)</label><input id="cp" type="text" value={custom.p} onChange={(e) => setCustom({ ...custom, p: e.target.value })} placeholder="p.Arg248Gln" /></div>
          <div className="field"><label htmlFor="cf">Filtering allele frequency (optional)</label><input id="cf" type="text" value={custom.faf} onChange={(e) => setCustom({ ...custom, faf: e.target.value })} placeholder="0.00002" /></div>
          <div className="field"><label htmlFor="ca">Allele count and number (optional)</label><div className="row"><input id="ca" type="text" value={custom.ac} onChange={(e) => setCustom({ ...custom, ac: e.target.value })} placeholder="3" style={{ width: 80 }} /><input type="text" aria-label="Allele number" value={custom.an} onChange={(e) => setCustom({ ...custom, an: e.target.value })} placeholder="251000" style={{ width: 120 }} /></div></div>
          <div className="field"><label htmlFor="ch">Homozygotes (optional)</label><input id="ch" type="text" value={custom.hom} onChange={(e) => setCustom({ ...custom, hom: e.target.value })} placeholder="0" /></div>
        </div>
        <div className="row" style={{ marginTop: 12 }}><button className="btn" onClick={runCustom} disabled={!custom.gene || !custom.c}>Classify</button>{error && <span className="small" style={{ color: "var(--bad)" }}>{error}</span>}</div>
      </section>
    </div>
  );
}
