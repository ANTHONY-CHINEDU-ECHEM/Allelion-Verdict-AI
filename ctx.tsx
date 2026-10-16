import { createContext, useContext } from "react";
import type { Bundle, Variant } from "./engine/data";
import type { Engine } from "./engine/engine";
import type { Retrieval } from "./engine/retrieval";
import type { PhenotypeMatcher } from "./engine/phenotype";

export interface Ctx {
  b: Bundle;
  engine: Engine;
  retrieval: () => Promise<Retrieval>;
  phenotype: () => PhenotypeMatcher;
  go: (page: string, params?: Record<string, string>) => void;
  params: Record<string, string>;
}

export const AppCtx = createContext<Ctx | null>(null);
export const useApp = () => useContext(AppCtx)!;

export const variantName = (v: Variant) => `${v.gene} ${(v.hgvs_c || "").split(":").pop()}`;
export const fmtAf = (x: number | null | undefined) => (x === null || x === undefined || Number.isNaN(x) ? "not observed" : x === 0 ? "0" : x >= 0.01 ? x.toFixed(3) : x.toExponential(2));
export const pct = (x: number, d = 1) => `${(x * 100).toFixed(d)}%`;
export const CONSEQ_WORD: Record<string, string> = {
  missense: "missense", synonymous: "synonymous", frameshift: "frameshift", nonsense: "nonsense", splice_donor: "canonical splice donor",
  splice_acceptor: "canonical splice acceptor", intron: "intronic", utr5: "5 prime UTR", utr3: "3 prime UTR", inframe_indel: "in frame indel",
  start_loss: "initiation codon", stop_loss: "stop loss", upstream: "upstream", downstream: "downstream", noncoding: "noncoding", other: "other",
};
