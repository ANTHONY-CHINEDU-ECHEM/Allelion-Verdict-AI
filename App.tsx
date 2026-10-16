import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AppCtx } from "./ctx";
import { loadBundle, type Bundle } from "./engine/data";
import { Engine } from "./engine/engine";
import { PhenotypeMatcher } from "./engine/phenotype";
import { Retrieval } from "./engine/retrieval";
import Classifier from "./pages/Classifier";
import CopilotPage from "./pages/Copilot";
import Evaluation from "./pages/Evaluation";
import Genes from "./pages/Genes";
import Overview from "./pages/Overview";
import { Audit, Batch, Calibration, Conflicts, Graph, Literature, Phenotype } from "./pages/Tools";

const NAV: [string, string, string][] = [
  ["Interpret", "overview", "Overview"], ["Interpret", "classify", "Classify a variant"], ["Interpret", "copilot", "Evidence copilot"],
  ["Interpret", "batch", "Batch triage"], ["Explore", "genes", "Gene explorer"], ["Explore", "conflicts", "Conflicts"],
  ["Explore", "phenotype", "Phenotype matcher"], ["Explore", "literature", "Literature"], ["Explore", "graph", "Knowledge graph"],
  ["Assure", "calibration", "Calibration lab"], ["Assure", "evaluation", "Evaluation"], ["Assure", "audit", "Audit log"],
];
const PAGES: Record<string, () => JSX.Element> = {
  overview: Overview, classify: Classifier, copilot: CopilotPage, batch: Batch, genes: Genes, conflicts: Conflicts,
  phenotype: Phenotype, literature: Literature, graph: Graph, calibration: Calibration, evaluation: Evaluation, audit: Audit,
};

function parseHash(): [string, Record<string, string>] {
  const h = location.hash.replace(/^#\/?/, "");
  const [page, qs] = h.split("?");
  const params: Record<string, string> = {};
  new URLSearchParams(qs || "").forEach((v, k) => { params[k] = v; });
  return [PAGES[page] ? page : "overview", params];
}

export default function App() {
  const [b, setB] = useState<Bundle | null>(null);
  const [prog, setProg] = useState({ msg: "Loading the knowledge base", frac: 0 });
  const [err, setErr] = useState("");
  const [[page, params], setRoute] = useState(parseHash());
  const [theme, setTheme] = useState<string>(() => { try { return localStorage.getItem("allelion.theme") || ""; } catch { return ""; } });
  const retr = useRef<Promise<Retrieval> | null>(null);
  const pheno = useRef<PhenotypeMatcher | null>(null);

  useEffect(() => { loadBundle("bundle/", (msg, frac) => setProg({ msg, frac })).then(setB).catch((e) => setErr(String(e))); }, []);
  useEffect(() => { const f = () => setRoute(parseHash()); window.addEventListener("hashchange", f); return () => window.removeEventListener("hashchange", f); }, []);
  useEffect(() => {
    if (theme) document.documentElement.setAttribute("data-theme", theme); else document.documentElement.removeAttribute("data-theme");
    try { localStorage.setItem("allelion.theme", theme); } catch { /* storage unavailable */ }
  }, [theme]);

  const go = useCallback((p: string, q?: Record<string, string>) => {
    const qs = q ? "?" + new URLSearchParams(q).toString() : "";
    location.hash = `/${p}${qs}`;
    window.scrollTo({ top: 0 });
  }, []);
  const engine = useMemo(() => (b ? new Engine(b) : null), [b]);
  const retrieval = useCallback(() => {
    if (!retr.current) retr.current = new Promise((res) => setTimeout(() => res(new Retrieval(b!)), 30));
    return retr.current;
  }, [b]);
  const phenotype = useCallback(() => { if (!pheno.current) pheno.current = new PhenotypeMatcher(b!); return pheno.current; }, [b]);

  if (err) return <div className="loading"><div><b>The bundle could not be loaded.</b><p>{err}</p></div></div>;
  if (!b || !engine) return (
    <div className="loading"><div><b>Allelion Verdict AI</b><p>{prog.msg}</p><div className="bar"><i style={{ width: `${Math.round(prog.frac * 100)}%` }} /></div></div></div>);
  const Page = PAGES[page];
  let group = "";
  return (
    <AppCtx.Provider value={{ b, engine, retrieval, phenotype, go, params }}>
      <div className="shell">
        <aside className="rail">
          <div className="brand">
            <svg className="brand-mark" viewBox="0 0 30 30" aria-hidden="true">
              <rect x="2" y="17" width="5" height="11" rx="1" fill="var(--b)" /><rect x="8.5" y="12" width="5" height="16" rx="1" fill="var(--lb)" />
              <rect x="15" y="8" width="5" height="20" rx="1" fill="var(--vus)" /><rect x="21.5" y="2" width="5" height="26" rx="1" fill="var(--p)" />
            </svg>
            <div><b>Allelion Verdict</b><span>Variant interpretation workbench</span></div>
          </div>
          <nav className="nav" aria-label="Workspaces">
            {NAV.map(([g, k, label]) => {
              const head = g !== group ? <div key={`h${g}`} className="nav-group">{g}</div> : null;
              group = g;
              return [head, <button key={k} aria-current={page === k ? "page" : undefined} onClick={() => go(k)}>{label}</button>];
            })}
          </nav>
          <div className="rail-foot">
            Runs in your browser from a {Math.round(b.variants.length / 1000)} thousand variant bundle.
            <br /><button className="theme-toggle" onClick={() => setTheme(theme === "dark" ? "light" : theme === "light" ? "" : "dark")}>Theme: {theme || "system"}</button>
          </div>
        </aside>
        <main><Page /></main>
      </div>
    </AppCtx.Provider>
  );
}
