"""Train the LambdaMART reranker on synthetic queries generated from the knowledge base.

Queries are produced from templates over passage facts (never from the
evaluation questions): criterion and topic questions for guidance cards,
inheritance, mechanism, constraint and disease questions for gene cards,
feature and gene questions for disease cards, definition questions for
phenotype terms, and partial title questions for literature. Labels are graded:
2 for the passage the query was generated from (or any passage of the same
source), 1 for passages that share the query's gene, disease or code, 0
otherwise. Queries are split 80/20 into training and validation sets.
"""
from __future__ import annotations

import json
import random
import re

import lightgbm as lgb
import numpy as np
import pandas as pd

from allelion.config import MODELS, SEED
from allelion.retrieval.query import understand
from allelion.retrieval.retriever import FEATURES, Retriever

GENE_T = ["What is the inheritance of {g}?", "Is loss of function an established mechanism for {g}?", "Which diseases are associated with {g}?",
          "What is the gene constraint of {g}?", "How many pathogenic missense variants does {g} have?", "Does PVS1 apply to {g}?",
          "What is the benign share of missense variants in {g}?", "Tell me about the {g} gene"]
DIS_T = ["What are the features of {d}?", "Which genes cause {d}?", "What is the inheritance of {d}?", "Which phenotypes are annotated for {d}?"]
HPO_T = ["What is {h}?", "Define {h}.", "What does the term {h} mean?"]
LIT_T = ["Which study reported {t}?", "Is there a report on {t}?", "What was found in the study of {t}?"]


def _guidance_queries(p, rng):
    title = p["title"]
    out = [f"Explain {title.lower()}", f"What does the guidance say about {title.split(':')[-1].strip().lower()}?"]
    words = [w for w in re.findall(r"[A-Za-z0-9]+", p["text"]) if len(w) > 5]
    if len(words) > 6:
        out.append(" ".join(rng.sample(words, 6)))
    return out


def synth(ret: Retriever, n_per_kind: int = 800) -> list[dict]:
    rng = random.Random(SEED)
    P = ret.passages
    qs = []
    for kind, n in (("guidance", 76), ("gene", n_per_kind), ("disease", n_per_kind), ("phenotype", n_per_kind // 2), ("literature", n_per_kind)):
        sub = P[P["kind"] == kind]
        sub = sub.sample(min(n, len(sub)), random_state=SEED)
        for p in sub.to_dict("records"):
            if kind == "guidance":
                texts = _guidance_queries(p, rng)
            elif kind == "gene":
                g = p["source"].split(":", 1)[1]
                texts = [rng.choice(GENE_T).format(g=g)]
            elif kind == "disease":
                texts = [rng.choice(DIS_T).format(d=p["title"])]
            elif kind == "phenotype":
                texts = [rng.choice(HPO_T).format(h=p["title"].split(" (")[0])]
            else:
                tw = p["title"].split()
                if len(tw) < 4:
                    continue
                s = rng.randint(0, max(0, len(tw) - 7))
                texts = [rng.choice(LIT_T).format(t=" ".join(tw[s:s + rng.randint(5, 9)]).rstrip(".,:"))]
            for t in texts:
                qs.append({"q": t, "source": p["source"], "kind": kind, "genes": list(p["genes"])})
    rng.shuffle(qs)
    return qs


def label(ret: Retriever, q: dict, i: int) -> int:
    src = ret.source[i]
    if src == q["source"]:
        return 2
    if q["kind"] == "gene" and q["source"].split(":", 1)[1] in ret.genes[i]:
        return 1
    if q["kind"] == "disease" and ret.kind[i] == "gene" and any(g in ret.genes[i] for g in q["genes"]):
        return 1
    return 0


def train() -> dict:
    ret = Retriever()
    qs = synth(ret)
    X_all, y_all, groups, qid = [], [], [], []
    for n, q in enumerate(qs):
        qi = understand(q["q"])
        lists, raw = ret.candidates(q["q"], qi)
        fused = ret.rrf(lists, ["bm25", "sif", "dual", "entity", "kind"])
        cand = [i for i, _ in fused[:100]]
        # make sure the source passage is in the candidate pool for training only when a generator found it
        X = ret.features(q["q"], qi, lists, raw, cand)
        y = [label(ret, q, i) for i in cand]
        if max(y) == 0:
            continue
        X_all.append(X)
        y_all.extend(y)
        groups.append(len(cand))
        qid.append(n)
    X = pd.concat(X_all)
    y = np.array(y_all)
    nq = len(groups)
    cut = int(nq * 0.8)
    off = np.cumsum([0] + groups)
    tr_idx = np.arange(off[cut])
    va_idx = np.arange(off[cut], off[-1])
    params = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[5, 10], learning_rate=0.05, num_leaves=31, min_data_in_leaf=50,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=SEED, num_threads=2)
    dtr = lgb.Dataset(X.iloc[tr_idx], y[tr_idx], group=groups[:cut])
    dva = lgb.Dataset(X.iloc[va_idx], y[va_idx], group=groups[cut:])
    evals = {}
    m = lgb.train(params, dtr, 600, valid_sets=[dva], callbacks=[lgb.early_stopping(40, verbose=False), lgb.record_evaluation(evals)])
    m.save_model(str(MODELS / "reranker.txt"), num_iteration=m.best_iteration)
    imp = pd.Series(m.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    rep = {"synthetic_queries": len(qs), "queries_with_positive_candidate": nq, "training_rows": int(len(tr_idx)),
           "validation_queries": nq - cut, "best_iteration": m.best_iteration,
           "validation_ndcg@5": round(evals["valid_0"]["ndcg@5"][m.best_iteration - 1], 4),
           "validation_ndcg@10": round(evals["valid_0"]["ndcg@10"][m.best_iteration - 1], 4),
           "importance_gain": {k: round(float(v), 1) for k, v in imp.items()}}
    (MODELS / "reranker_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "importance_gain"}, indent=1))
    return rep


if __name__ == "__main__":
    train()
