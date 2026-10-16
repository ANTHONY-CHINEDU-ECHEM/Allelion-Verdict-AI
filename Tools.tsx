// Conflict explorer, batch triage, phenotype matcher, calibration lab, literature, knowledge graph and audit log.
import { useMemo, useState } from "react";
import { CalibrationChart, CLS_SHORT } from "../charts";
import { useApp, variantName } from "../ctx";
import { auditLog } from "../engine/copilot";
import { prepare } from "../engine/engine";
import { learnedVerdict } from "../engine/meta";

const CALL_CLASS: Record<string, string> = { pathogenic: "P", "likely pathogenic": "LP", "uncertain significance": "VUS", "likely benign": "LB", benign: "B" };

export function Conflicts() {
  const { b, engine, go } = useApp();
  const [gene, setGene] = useState("");
  const rows = useMemo(() => {
    const out = [];
    for (const v of b.variants) {
      if (v.conflicted !== 1 || (gene && v.gene !== gene.toUpperCase())) continue;
      const subs = b.submissions[v.variation_id!] || [];
      const res = engine.evaluate(v);
      out.push({ v, subs, res });
      if (out.length >= 150) break;
    }
    return out;
  }, [b, engine, gene]);
  return (
    <div className="stack">
      <div>
        <h1>Conflicting interpretations</h1>
        <p className="lede">Variants where ClinVar submitters disagree, shown with every submitted call next to the engine's evidence based verdict. Most conflicts sit between uncertain and likely benign, where frequency and predictor evidence can help settle the call.</p>
        <input type="text" aria-label="Filter by gene" placeholder="Filter by gene, for example MSH6" value={gene} onChange={(e) => setGene(e.target.value)} />
      </div>
      <section className="panel scroll">
        <table>
          <thead><tr><th>Variant</th><th>Submitted calls</th><th>Engine</th><th>Criteria met</th></tr></thead>
          <tbody>{rows.map(({ v, subs, res }) => (
            <tr key={v.variation_id!} className="clickable" onClick={() => go("classify", { id: v.variation_id! })}>
              <td className="mono">{variantName(v)}</td>
              <td>{subs.slice(0, 6).map((s, i) => <span key={i} className={`tag bg-${CALL_CLASS[s[1]] || "VUS"}`} style={{ marginRight: 4, opacity: CALL_CLASS[s[1]] ? 1 : 0.5 }} title={`${s[0]} (${s[3]})`}>{CALL_CLASS[s[1]] || "other"}</span>)}</td>
              <td><span className={`cls-${CLS_SHORT[res.classification]}`}><b>{res.classification}</b></span> <span className="muted small">{res.points} pts</span></td>
              <td className="mono small">{res.met.join(" ")}</td>
            </tr>))}</tbody>
        </table>
        {rows.length === 0 && <p className="muted">No conflicting variants for this gene in the demo bundle.</p>}
      </section>
    </div>
  );
}

export function Batch() {
  const { b, engine } = useApp();
  const [text, setText] = useState("MLH1 c.350C>T\nBRCA2 c.8754+5G>A\nCFTR c.1521_1523delCTT\nTP53 c.743G>A p.Arg248Gln\nMSH6 c.3261dupC\nMYH7 c.1208G>A");
  const [rows, setRows] = useState<any[]>([]);
  function run() {
    const out = [];
    for (const line of text.split("\n").map((l) => l.trim()).filter(Boolean).slice(0, 500)) {
      const [gene, c, p] = line.replace(/[,\t]/g, " ").split(/\s+/);
      try {
        const known = b.variants.find((v) => v.gene === gene && (v.hgvs_c || "").split(":").pop() === c);
        const v = known || prepare(gene, c, p || null, null);
        const res = engine.evaluate(v);
        out.push({ line, v, res, learned: learnedVerdict(b, res, v).classification, known: !!known });
      } catch (e: any) { out.push({ line, error: e.message }); }
    }
    setRows(out);
  }
  const [copied, setCopied] = useState("");
  function csv() {
    const head = "input,classification,points,criteria_met,learned_verdict,clinvar_reference,population_data\n";
    return head + rows.map((r) => r.error ? `"${r.line}",error,,,,,` : `"${r.line}",${r.res.classification},${r.res.points},"${r.res.met.join(" ")}",${r.learned},${r.v.clinvar_class || ""},${r.known ? "gnomAD" : "not supplied"}`).join("\n");
  }
  function copy() {
    navigator.clipboard.writeText(csv()).then(() => setCopied("Copied"), () => setCopied("Copy blocked; select the table instead"));
    setTimeout(() => setCopied(""), 2500);
  }
  function download() {
    const url = URL.createObjectURL(new Blob([csv()], { type: "text/csv" }));
    const a = document.createElement("a"); a.href = url; a.download = "allelion_batch.csv"; a.click(); URL.revokeObjectURL(url);
  }
  return (
    <div className="stack">
      <div>
        <h1>Batch triage</h1>
        <p className="lede">Paste one variant per line as gene, coding change and optional protein change. Variants found in the bundle use their gnomAD counts; others are classified without population data, so the frequency criteria stay open.</p>
      </div>
      <section className="panel">
        <textarea rows={7} value={text} onChange={(e) => setText(e.target.value)} aria-label="Variants, one per line" className="mono" />
        <div className="row" style={{ marginTop: 10 }}><button className="btn" onClick={run}>Classify all</button>{rows.length > 0 && <button className="btn secondary" onClick={download}>Download CSV</button>}{rows.length > 0 && <button className="btn secondary" onClick={copy}>{copied || "Copy CSV"}</button>}</div>
      </section>
      {rows.length > 0 && (
        <section className="panel scroll">
          <table><thead><tr><th>Input</th><th>Engine</th><th className="num">Points</th><th>Criteria met</th><th>Learned verdict</th><th>ClinVar reference</th></tr></thead>
            <tbody>{rows.map((r, i) => r.error ? <tr key={i}><td className="mono">{r.line}</td><td colSpan={5} style={{ color: "var(--bad)" }}>{r.error}</td></tr> : (
              <tr key={i}><td className="mono">{r.line}</td><td className={`cls-${CLS_SHORT[r.res.classification]}`}><b>{r.res.classification}</b></td><td className="num">{r.res.points}</td><td className="mono small">{r.res.met.join(" ")}</td><td>{r.learned}</td><td>{r.v.clinvar_class || <span className="muted">not in the demo bundle</span>}</td></tr>))}</tbody></table>
        </section>)}
    </div>
  );
}

export function Phenotype() {
  const { b, phenotype, go } = useApp();
  const [q, setQ] = useState("");
  const [terms, setTerms] = useState<string[]>(["HP:0001166", "HP:0001083", "HP:0002616"]);
  const hits = useMemo(() => { const s = q.trim().toLowerCase(); if (s.length < 3) return []; const o: [string, string][] = []; for (const [id, [n]] of Object.entries(b.hpo)) { if (n.toLowerCase().includes(s)) o.push([id, n]); if (o.length >= 12) break; } return o; }, [q, b]);
  const ranked = useMemo(() => (terms.length ? phenotype().rankGenes(terms, 20) : []), [terms, phenotype]);
  return (
    <div className="stack">
      <div>
        <h1>Phenotype matcher</h1>
        <p className="lede">Rank genes by how well a patient's HPO terms match their disease profiles (best match average of the information content of shared ancestors). In the classifier the same score, with an empirical p value against 300 background genes, decides PP4.</p>
        <div style={{ position: "relative", maxWidth: 480 }}>
          <input type="text" aria-label="Search phenotype terms" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Add a term, for example seizure" style={{ width: "100%" }} />
          {hits.length > 0 && <div className="suggest">{hits.map(([id, n]) => <button key={id} onClick={() => { setTerms([...new Set([...terms, id])]); setQ(""); }}>{n} <span className="muted mono">{id}</span></button>)}</div>}
        </div>
        <div className="row" style={{ marginTop: 8 }}>{terms.map((t) => <button key={t} className="chip" aria-pressed="true" onClick={() => setTerms(terms.filter((x) => x !== t))}>{b.hpo[t]?.[0]} (remove)</button>)}</div>
      </div>
      <section className="panel scroll">
        <table><thead><tr><th>Gene</th><th className="num">Similarity</th><th>Best matching disease</th></tr></thead>
          <tbody>{ranked.map((r) => <tr key={r.gene} className="clickable" onClick={() => go("genes", { gene: r.gene })}><td><b>{r.gene}</b></td><td className="num">{r.score.toFixed(2)}</td><td>{r.disease ? b.diseases[r.disease]?.[0] : ""}</td></tr>)}</tbody></table>
        {!terms.length && <p className="muted">Add at least one phenotype term.</p>}
      </section>
    </div>
  );
}

export function Calibration() {
  const { b } = useApp();
  const [score, setScore] = useState(0.75);
  const fmt = (t: Record<string, number | null>) => Object.entries(t).filter(([, v]) => v !== null).map(([k, v]) => `${k} ${(v as number).toFixed(3)}`).join(", ") || "none reached";
  return (
    <div className="stack">
      <div>
        <h1>Calibration lab</h1>
        <p className="lede">How predictor scores become evidence strengths. Out of fold scores from gene grouped cross validation are turned into a local posterior probability of pathogenicity, reweighted to the ClinGen prior of 0.0441. A strength is awarded only where the one sided 95% bound of the posterior clears the threshold that strength implies.</p>
      </div>
      <div className="grid2">
        <section className="panel">
          <h3 style={{ marginTop: 0 }}>Missense predictor</h3>
          <CalibrationChart cal={b.missenseCal} title="Missense calibration" mark={score} />
          <p className="small">Pathogenic thresholds: {fmt(b.missenseCal.pathogenic_thresholds)}. Benign thresholds: {fmt(b.missenseCal.benign_thresholds)}. Trained on {b.missenseCal.n.toLocaleString()} pool variants.</p>
          <label className="small" htmlFor="sc">Try a score: {score.toFixed(2)}</label>
          <input id="sc" type="range" min={0} max={1} step={0.01} value={score} onChange={(e) => setScore(+e.target.value)} style={{ width: "100%" }} />
        </section>
        <section className="panel">
          <h3 style={{ marginTop: 0 }}>Splice region predictor (intronic offsets 3 to 50)</h3>
          <CalibrationChart cal={b.spliceCal} title="Splice calibration" />
          <p className="small">Pathogenic thresholds: {fmt(b.spliceCal.pathogenic_thresholds)}. Benign thresholds: {fmt(b.spliceCal.benign_thresholds)}. Trained on {b.spliceCal.n.toLocaleString()} pool variants.</p>
        </section>
      </div>
    </div>
  );
}

export function Literature() {
  const { b, go } = useApp();
  const [q, setQ] = useState("BRCA1");
  const [open, setOpen] = useState<string | null>(null);
  const passByDoc = useMemo(() => { const m = new Map<string, string[]>(); b.passages.forEach((p) => { if (p[1] === "literature") { const id = p[0].slice(4); const a = m.get(id) || []; a.push(p[3]); m.set(id, a); } }); return m; }, [b]);
  const titles = useMemo(() => { const m = new Map<string, string>(); b.passages.forEach((p) => { if (p[1] === "literature") m.set(p[0].slice(4), p[2]); }); return m; }, [b]);
  const docs = useMemo(() => {
    const s = q.trim().toLowerCase();
    return b.literature.docs.filter(([id]) => !s || (titles.get(id) || "").toLowerCase().includes(s) || (passByDoc.get(id) || []).join(" ").toLowerCase().includes(s)).slice(0, 60);
  }, [q, b, titles, passByDoc]);
  const links = useMemo(() => { const m = new Map<string, any[]>(); for (const l of b.literature.links) if (l[5] === "text_mined" && l[2]) { const a = m.get(l[0]) || []; a.push(l); m.set(l[0], a); } return m; }, [b]);
  function highlight(id: string) {
    const text = `${titles.get(id) || ""} ${(passByDoc.get(id) || []).join(" ")}`;
    const ms = (b.literature.mentions[id] || []).filter((m) => m[1] <= text.length);
    const out: any[] = []; let cur = 0;
    ms.forEach(([s, e, kind, norm], i) => { if (s < cur) return; out.push(text.slice(cur, s)); out.push(<mark key={i} title={`${kind}${norm ? `: ${norm}` : ""}`}>{text.slice(s, e)}</mark>); cur = e; });
    out.push(text.slice(cur));
    return out;
  }
  return (
    <div className="stack">
      <div>
        <h1>Literature</h1>
        <p className="lede">2,168 abstracts and full text paragraphs from five annotated variant corpora. Variant mentions found by the extractor are highlighted; normalised mentions that match a ClinVar variant in a gene named in the same document become links.</p>
        <input type="text" aria-label="Search literature" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search titles and text" />
      </div>
      <section className="panel">
        {docs.map(([id, pmid, corpus]) => (
          <div key={id} className="evid">
            <button className="chip" style={{ float: "right" }} onClick={() => setOpen(open === id ? null : id)}>{open === id ? "Hide text" : "Show text"}</button>
            <b>{titles.get(id)}</b>
            <div className="small muted">{/^\d+$/.test(pmid) ? `PMID ${pmid}` : pmid}, {corpus} corpus</div>
            {(links.get(id) || []).length > 0 && <div className="row small" style={{ marginTop: 4 }}>Linked variants: {[...new Map((links.get(id) || []).map((l) => [l[2], l])).values()].slice(0, 6).map((l) => (
              <button key={l[2]} className="chip" onClick={() => b.byId.get(l[2]) ? go("classify", { id: l[2] }) : go("genes", { gene: l[1] })}>{l[1]} {l[4]}</button>))}</div>}
            {open === id && <p className="small" style={{ lineHeight: 1.7 }}>{highlight(id)}</p>}
          </div>))}
        {docs.length === 0 && <p className="muted">No document matches. Try a gene symbol or a disease name.</p>}
      </section>
    </div>
  );
}

export function Graph() {
  const { b, go } = useApp();
  const [gene, setGene] = useState("FBN1");
  const g = b.genes[gene.toUpperCase()];
  const G = gene.toUpperCase();
  const dis = (g?.dis || []).slice(0, 8);
  const phen = useMemo(() => {
    const c = new Map<string, number>();
    dis.forEach((d) => (b.diseases[d]?.[1] || []).forEach((t) => c.set(t, (c.get(t) || 0) + 1)));
    return [...c.entries()].sort((a, x) => x[1] - a[1] || (b.hpo[x[0]]?.[2] || 0) - (b.hpo[a[0]]?.[2] || 0)).slice(0, 12).map(([t]) => t);
  }, [dis, b]);
  const pubs = useMemo(() => [...new Set(b.literature.links.filter((l) => l[1] === G).map((l) => l[0]))].slice(0, 8), [b, G]);
  const vars = useMemo(() => b.variants.filter((v) => v.gene === G && (v.clinvar_class === "Pathogenic")).slice(0, 8), [b, G]);
  // Columns read left to right: phenotype terms, the diseases that share them, the gene, then its variants and publications.
  const rowH = 26, top = 34;
  const right = [...vars.map((v) => ({ kind: "v" as const, v })), ...pubs.map((p) => ({ kind: "p" as const, p }))];
  const rows = Math.max(phen.length, dis.length, right.length, 3);
  const W = 940, H = top + rows * rowH + 12;
  const colY = (n: number, i: number) => top + ((rows - n) * rowH) / 2 + i * rowH + rowH / 2;
  const xPhen = 232, xDis = 336, xDisEnd = 556, xGene = 640, xRight = 726;
  const gy = top + (rows * rowH) / 2;
  const disY = new Map(dis.map((d, i) => [d, colY(dis.length, i)]));
  const lab = (s: string, n = 28) => (s.length > n ? s.slice(0, n - 1) + "..." : s);
  return (
    <div className="stack">
      <div>
        <h1>Knowledge graph</h1>
        <p className="lede">A gene's neighbourhood: diseases from HPO annotations, their most shared phenotype terms, pathogenic variants and the publications that mention the gene. The full graph holds {b.meta.graph.nodes.toLocaleString()} nodes and {b.meta.graph.edges.toLocaleString()} edges, each with its source.</p>
        <form className="row" onSubmit={(e) => e.preventDefault()}><input type="text" aria-label="Gene" value={gene} onChange={(e) => setGene(e.target.value)} /></form>
      </div>
      {!g ? <p className="muted">Unknown gene.</p> : (
        <section className="panel">
          <div className="scroll">
          <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", minWidth: 640, height: "auto", display: "block" }} role="img" aria-label={`Neighbourhood of ${G}`}>
            {[["Phenotype terms", xPhen, "end"], ["Diseases", xDis - 6, "start"], ["Variants and publications", xRight - 6, "start"]].map(([t, x, a]) => (
              <text key={t as string} x={x as number} y={16} fontSize={11.5} fontWeight={700} textAnchor={a as any} fill="var(--ink-3)">{t}</text>))}
            {phen.map((t, i) => dis.filter((d) => (b.diseases[d]?.[1] || []).includes(t)).map((d) => (
              <line key={t + d} x1={xPhen} y1={colY(phen.length, i)} x2={xDis - 7} y2={disY.get(d)!} stroke="var(--rule-2)" />)))}
            {dis.map((d) => <line key={d} x1={xDisEnd} y1={disY.get(d)!} x2={xGene - 26} y2={gy} stroke="var(--rule-2)" />)}
            {right.map((r, i) => <line key={i} x1={xGene + 26} y1={gy} x2={xRight - 8} y2={colY(right.length, i)} stroke="var(--rule-2)" strokeDasharray={r.kind === "p" ? "3 3" : undefined} />)}
            {phen.map((t, i) => { const y = colY(phen.length, i); return (
              <g key={t}><circle cx={xPhen} cy={y} r={4} fill="var(--lb)" /><text x={xPhen - 10} y={y + 4} fontSize={11.5} textAnchor="end" fill="var(--ink-2)">{lab(b.hpo[t]?.[0] || t, 30)}</text></g>); })}
            {dis.map((d) => { const y = disY.get(d)!; return (
              <g key={d}><rect x={xDis - 6} y={y - 6} width={12} height={12} rx={2} fill="var(--vus)" /><text x={xDis + 12} y={y + 4} fontSize={11.5} fill="var(--ink)">{lab(b.diseases[d]?.[0] || d, 30)}</text></g>); })}
            {right.map((r, i) => { const y = colY(right.length, i); return r.kind === "v" ? (
              <g key={r.v.variation_id ?? i} style={{ cursor: "pointer" }} onClick={() => go("classify", { id: r.v.variation_id ?? "" })}>
                <circle cx={xRight} cy={y} r={6} fill="var(--p)" /><text x={xRight + 12} y={y + 4} fontSize={11.5} fill="var(--ink)" fontFamily="var(--mono)">{lab((r.v.hgvs_c || "").split(":").pop() || "", 24)}</text></g>
            ) : (
              <g key={r.p}><rect x={xRight - 5} y={y - 5} width={10} height={10} fill="var(--b)" transform={`rotate(45 ${xRight} ${y})`} /><text x={xRight + 12} y={y + 4} fontSize={11.5} fill="var(--ink-2)">PMID {r.p.split(":").pop()}</text></g>); })}
            <circle cx={xGene} cy={gy} r={26} fill="var(--ink)" /><text x={xGene} y={gy + 5} textAnchor="middle" fontSize={13.5} fontWeight={700} fill="var(--sheet)">{G}</text>
          </svg>
          </div>
          <div className="legend"><span><i style={{ background: "var(--vus)" }} />disease</span><span><i style={{ background: "var(--lb)" }} />phenotype term</span><span><i style={{ background: "var(--p)" }} />pathogenic variant (click to classify)</span><span><i style={{ background: "var(--b)" }} />publication</span></div>
        </section>)}
    </div>
  );
}

export function Audit() {
  const rows = [...auditLog].reverse();
  return (
    <div className="stack">
      <div>
        <h1>Audit log</h1>
        <p className="lede">Every copilot question is recorded after redaction, with a SHA256 hash of the original text, the writer used, the share of verified sentences and the confidence. In the demo the log lives in this browser only; the server keeps an append only file.</p>
      </div>
      <section className="panel scroll">
        {rows.length === 0 ? <p className="muted">No questions yet. Ask the evidence copilot something and it will appear here.</p> : (
          <table><thead><tr><th>Time (UTC)</th><th>Question as stored</th><th>Outcome</th><th>Verified</th><th>Hash of original</th></tr></thead>
            <tbody>{rows.map((r, i) => <tr key={i}><td className="small">{r.ts.replace("T", " ")}</td><td>{r.question}</td><td>{r.action === "answer" ? `${r.confidence} confidence${r.classification ? `, ${r.classification}` : ""}` : r.action.replace("_", " ")}</td><td className="num">{r.verified_share === null ? "" : `${Math.round(r.verified_share * 100)}%`}</td><td className="mono small">{r.sha256.slice(0, 16)}</td></tr>)}</tbody></table>)}
      </section>
    </div>
  );
}
