import { useMemo, useState } from "react";
import { ExonMap, HBars, Lollipop } from "../charts";
import { useApp, variantName } from "../ctx";
import { lofMechanism } from "../engine/tables";

const INH: Record<string, string> = { AD: "autosomal dominant", AR: "autosomal recessive", XL: "X linked", XLR: "X linked recessive", XLD: "X linked dominant", MT: "mitochondrial", SOM: "somatic", POLY: "polygenic", Y: "Y linked" };

export default function Genes() {
  const { b, params, go } = useApp();
  const [q, setQ] = useState(params.gene || "BRCA1");
  const gene = (params.gene || q).toUpperCase();
  const g = b.genes[gene];
  const stats = b.tables.gene_stats[gene] || {};
  const tx = b.transcripts[gene];
  const [mech, why] = g ? lofMechanism(stats, g, b.meta.rules) : [false, ""];
  const list = useMemo(() => b.variants.filter((v) => v.gene === gene && v.clinvar_class).slice(0, 400), [b, gene]);
  const matches = useMemo(() => (q.length >= 2 ? Object.keys(b.genes).filter((x) => x.startsWith(q.toUpperCase())).slice(0, 12) : []), [q, b]);
  const R = b.meta.rules;
  const mp = stats.missense_plp || 0, mb = stats.missense_blb || 0, tp = stats.truncating_plp || 0;
  const pp2 = mp >= R.pp2_min_plp && (mp + mb ? mb / (mp + mb) : 1) <= R.pp2_max_benign_frac;
  const bp1 = tp >= R.bp1_min_trunc && (mp + tp ? mp / (mp + tp) : 1) <= R.bp1_max_missense_frac;

  return (
    <div className="stack">
      <div>
        <h1>Gene explorer</h1>
        <p className="lede">Disease associations, constraint, the classified variant spectrum in the evidence pool and the gene level criteria they imply.</p>
        <form className="row" onSubmit={(e) => { e.preventDefault(); go("genes", { gene: q.toUpperCase() }); }}>
          <input type="text" aria-label="Gene symbol" value={q} onChange={(e) => setQ(e.target.value)} />
          <button className="btn">Open</button>
          {matches.filter((m) => m !== q.toUpperCase()).slice(0, 8).map((m) => <button key={m} type="button" className="chip" onClick={() => { setQ(m); go("genes", { gene: m }); }}>{m}</button>)}
        </form>
      </div>
      {!g ? <p className="muted">No gene called {gene} in the knowledge base. Gene symbols are case sensitive HGNC symbols such as MLH1.</p> : (
        <>
          <section className="panel">
            <h2 style={{ marginTop: 0 }}>{gene}</h2>
            <div className="grid2">
              <dl className="kv">
                <dt>Inheritance</dt><dd>{g.inh.map((i) => INH[i] || i).join(", ") || "not recorded"}</dd>
                <dt>pLI</dt><dd>{g.pli?.toFixed(3) ?? "not available"}</dd>
                <dt>LoFtool percentile</dt><dd>{g.lt?.toPrecision(3) ?? "not available"}</dd>
                <dt>Haploinsufficiency</dt><dd>{g.hi ? "curated" : "not curated"}</dd>
                <dt>Protein length</dt><dd>{g.plen ?? "unknown"} residues</dd>
                <dt>Exon model</dt><dd>{tx ? `${tx.exons.length} exons anchored by ${tx.n} variants (${(tx.agree * 100).toFixed(1)}% agreement)` : "no reliable model"}</dd>
                <dt>Onset</dt><dd>{g.onset.join(", ") || "not recorded"}</dd>
              </dl>
              <dl className="kv">
                <dt>PVS1</dt><dd>{mech ? "can apply" : "does not apply"}: {why}</dd>
                <dt>PP2</dt><dd>{pp2 ? "applies to missense" : "does not apply"} ({mp} pathogenic, {mb} benign missense)</dd>
                <dt>BP1</dt><dd>{bp1 ? "applies to missense" : "does not apply"} ({tp} pathogenic truncating)</dd>
              </dl>
            </div>
            <h3>Diseases</h3>
            <p className="small">{g.dis.slice(0, 14).map((d) => b.diseases[d]?.[0] || d).join("; ") || "No HPO disease annotation."}</p>
          </section>
          <div className="grid2">
            <section className="panel">
              <h3 style={{ marginTop: 0 }}>ClinVar spectrum (all assertions)</h3>
              <HBars data={[
                { label: "Pathogenic", value: g.cv.pathogenic, color: "var(--p)" }, { label: "Likely pathogenic", value: g.cv.likely_pathogenic, color: "var(--lp)" },
                { label: "Uncertain significance", value: g.cv.uncertain_significance, color: "var(--vus)" }, { label: "Likely benign", value: g.cv.likely_benign, color: "var(--lb)" },
                { label: "Benign", value: g.cv.benign, color: "var(--b)" }]} />
            </section>
            <section className="panel">
              <h3 style={{ marginTop: 0 }}>Evidence pool by consequence</h3>
              <HBars data={Object.entries(stats).sort((a, c) => c[1] - a[1]).slice(0, 10).map(([k, v]) => ({ label: k.replace("_plp", ", pathogenic").replace("_blb", ", benign").replace("_vus", ", uncertain"), value: v, color: k.endsWith("plp") ? "var(--p)" : k.endsWith("blb") ? "var(--b)" : "var(--vus)" }))} />
            </section>
          </div>
          {g.plen && <section className="panel"><h3 style={{ marginTop: 0 }}>Protein map</h3><Lollipop length={g.plen} missense={(b.tables.missense[gene] || []).map((r) => [r[0], r[1], r[2]])} trunc={b.tables.trunc[gene] || []} /></section>}
          {tx && <section className="panel"><h3 style={{ marginTop: 0 }}>Exon model</h3><ExonMap exons={tx.exons} cds={tx.cds} /></section>}
          <section className="panel">
            <h3 style={{ marginTop: 0 }}>Variants available in the demo ({list.length}{list.length === 400 ? ", first 400" : ""})</h3>
            {list.length === 0 ? <p className="muted">No bundled variants for this gene; enter one by HGVS in the classifier.</p> : (
              <div className="scroll" style={{ maxHeight: 420 }}>
                <table><thead><tr><th>Variant</th><th>Protein</th><th>Consequence</th><th>ClinVar</th><th>Set</th></tr></thead>
                  <tbody>{list.map((v) => (
                    <tr key={v.variation_id!} className="clickable" onClick={() => go("classify", { id: v.variation_id! })}>
                      <td className="mono">{variantName(v)}</td><td className="mono">{(v.hgvs_p || "").split(":").pop()}</td><td>{v.consequence}</td><td>{v.clinvar_class}</td><td>{v.set}</td>
                    </tr>))}</tbody></table>
              </div>)}
          </section>
        </>)}
    </div>
  );
}
