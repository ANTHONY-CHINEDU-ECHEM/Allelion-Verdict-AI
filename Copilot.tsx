import { useEffect, useRef, useState } from "react";
import { CLS_SHORT } from "../charts";
import { useApp } from "../ctx";
import { Copilot, type Answer } from "../engine/copilot";

const SAMPLES = [
  "What evidence applies to MYH7 c.1208G>A?",
  "At what strength should PM2 be applied after the 2020 SVI recommendation?",
  "When does a premature stop escape nonsense mediated decay?",
  "What are the clinical features of Marfan syndrome?",
  "Which report describes the F11 Asp506Gly mutation?",
  "Is loss of function an established mechanism for PTPN11?",
];

export default function CopilotPage() {
  const { b, retrieval, go } = useApp();
  const [cp, setCp] = useState<Copilot | null>(null);
  const [status, setStatus] = useState("Building the search index in your browser");
  const [q, setQ] = useState("");
  const [ans, setAns] = useState<Answer | null>(null);
  const [active, setActive] = useState<number | null>(null);
  const evRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let live = true;
    const t0 = performance.now();
    retrieval().then((r) => { if (live) { setCp(new Copilot(b, r)); setStatus(`Index ready in ${((performance.now() - t0) / 1000).toFixed(1)} s`); } });
    return () => { live = false; };
  }, [b, retrieval]);

  function ask(text: string) {
    if (!cp || !text.trim()) return;
    setQ(text);
    setActive(null);
    setTimeout(() => setAns(cp.answer(text)), 0);
  }

  const showCite = (n: number) => {
    setActive(n);
    evRef.current?.querySelector(`[data-n="${n}"]`)?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  };

  return (
    <div className="stack">
      <div>
        <h1>Evidence copilot</h1>
        <p className="lede">Ask about criteria, genes, diseases, phenotype terms, variants or the literature corpus. Each sentence cites a numbered evidence item and is checked against it; anything the check cannot support is underlined. Nothing you type leaves your browser.</p>
        <form className="row" onSubmit={(e) => { e.preventDefault(); ask(q); }}>
          <input type="text" aria-label="Question" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Ask a question about variant interpretation" style={{ flex: "1 1 360px" }} />
          <button className="btn" disabled={!cp}>Ask</button>
        </form>
        <p className="small muted">{status}</p>
        <div className="row">{SAMPLES.map((s) => <button key={s} className="chip" disabled={!cp} onClick={() => ask(s)}>{s}</button>)}</div>
      </div>

      {ans && (
        <div className="grid2">
          <section className="panel">
            {ans.notices.map((n, i) => <p key={i} className="notice">{n}</p>)}
            {ans.action === "answer" && (
              <>
                <div className="answer">
                  {ans.sentences.length === 0 && <p className="muted">The evidence found does not answer this question. Try naming a gene, criterion or disease.</p>}
                  {ans.sentences.map((s, i) => (
                    <p key={i}><span className={s.supported ? "" : "unsupported"} title={s.supported ? `Verified (support ${s.probability})` : `Not verified: ${s.reason}`}>{s.text}</span>
                      <button className="cite" onClick={() => showCite(s.cite)} aria-label={`Show evidence ${s.cite}`}>{s.cite}</button></p>))}
                </div>
                <p className="small muted">Confidence {ans.confidence}; {Math.round((ans.verified_share ?? 0) * 100)}% of sentences verified; answered in {ans.ms} ms with the extractive writer.</p>
                {ans.engine_result && (
                  <p className="small">The engine classified the variant as <b className={`cls-${CLS_SHORT[ans.engine_result.classification]}`}>{ans.engine_result.classification}</b>.{" "}
                    {ans.engine_result.variant.variation_id && <button className="btn secondary" onClick={() => go("classify", { id: String(ans.engine_result!.variant.variation_id) })}>Open in the classifier</button>}</p>)}
                {ans.query && (
                  <dl className="kv small" style={{ marginTop: 12 }}>
                    <dt>Genes</dt><dd>{ans.query.genes.join(", ") || "none"}</dd>
                    <dt>Criteria</dt><dd>{ans.query.codes.join(", ") || "none"}</dd>
                    <dt>Variants</dt><dd>{[...ans.query.hgvs_c, ...ans.query.hgvs_p].join(", ") || "none"}</dd>
                    <dt>Diseases and terms</dt><dd>{[...ans.query.diseases, ...ans.query.hpo].map((x) => b.diseases[x]?.[0] || b.hpo[x]?.[0] || x).join(", ") || "none"}</dd>
                    {ans.redactions.length > 0 && <><dt>Redacted</dt><dd>{ans.redactions.join(", ").toLowerCase().replace(/_/g, " ")}</dd></>}
                  </dl>)}
              </>)}
          </section>
          <section className="panel" ref={evRef}>
            <h3 style={{ marginTop: 0 }}>Evidence</h3>
            {ans.evidence.map((e) => (
              <div key={e.n} data-n={e.n} className={`evid ${active === e.n ? "active" : ""}`}>
                <b>[{e.n}] {e.title}</b>
                <div className="small muted">{e.level}{e.refs?.length ? `, ${e.refs.slice(0, 2).join("; ")}` : ""}</div>
                <p>{e.text.length > 600 ? e.text.slice(0, 600) + " ..." : e.text}</p>
              </div>))}
          </section>
        </div>)}
    </div>
  );
}
