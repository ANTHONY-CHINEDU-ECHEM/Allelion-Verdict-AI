"""Build the retrieval indexes: passages, BM25, SIF vectors and the dual encoder."""
from __future__ import annotations

import json
import time

from allelion.config import MODELS
from allelion.retrieval import corpus, dense
from allelion.retrieval.bm25 import BM25


def build() -> dict:
    t = time.time()
    p = corpus.build()
    texts = (p["title"] + ". " + p["text"]).tolist()
    bm = BM25.fit(texts)
    bm.save()
    d = dense.build(p["title"].tolist(), p["text"].tolist())
    rep = {"passages": len(p), "bm25_terms": len(bm.vocab), "dense": d, "seconds": round(time.time() - t, 1)}
    (MODELS / "index_report.json").write_text(json.dumps(rep, indent=1))
    print(rep)
    return rep


if __name__ == "__main__":
    build()
