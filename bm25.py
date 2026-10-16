"""BM25 over the passage corpus, stored as a sparse term weight matrix.

The matrix holds the full BM25 term weight of every (passage, term) pair, so a
query score is a sparse matrix times query vector product. The same matrix is
exported for the in browser engine.
"""
from __future__ import annotations

import json
import math
from collections import Counter

import numpy as np
import scipy.sparse as sp

from allelion.config import MODELS
from allelion.retrieval.text import tokenize

K1, B = 1.2, 0.75


class BM25:
    def __init__(self, vocab: dict[str, int], weights: sp.csr_matrix, idf: np.ndarray):
        self.vocab, self.W, self.idf = vocab, weights.tocsc(), idf

    @classmethod
    def fit(cls, texts: list[str]) -> "BM25":
        toks = [tokenize(t) for t in texts]
        df = Counter()
        for ts in toks:
            df.update(set(ts))
        vocab = {w: i for i, w in enumerate(sorted(df))}
        n = len(toks)
        idf = np.array([math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5)) for w in sorted(df)], dtype=np.float32)
        avg = np.mean([len(t) for t in toks]) or 1.0
        rows, cols, vals = [], [], []
        for i, ts in enumerate(toks):
            tf = Counter(ts)
            norm = K1 * (1 - B + B * len(ts) / avg)
            for w, f in tf.items():
                j = vocab[w]
                rows.append(i)
                cols.append(j)
                vals.append(idf[j] * f * (K1 + 1) / (f + norm))
        W = sp.csr_matrix((np.array(vals, dtype=np.float32), (rows, cols)), shape=(n, len(vocab)))
        return cls(vocab, W, idf)

    def query_vec(self, q: str) -> sp.csr_matrix:
        idx = [self.vocab[w] for w in tokenize(q) if w in self.vocab]
        c = Counter(idx)
        v = sp.csr_matrix((np.ones(len(c), dtype=np.float32), (np.zeros(len(c), dtype=int), list(c.keys()))), shape=(1, len(self.vocab)))
        return v

    def scores(self, q: str) -> np.ndarray:
        qv = self.query_vec(q)
        if qv.nnz == 0:
            return np.zeros(self.W.shape[0], dtype=np.float32)
        return np.asarray(self.W @ qv.T.toarray()).ravel()

    def top(self, q: str, k: int = 100, mask: np.ndarray | None = None) -> list[tuple[int, float]]:
        s = self.scores(q)
        if mask is not None:
            s = np.where(mask, s, 0.0)
        if not s.any():
            return []
        k = min(k, int((s > 0).sum()))
        idx = np.argpartition(-s, k - 1)[:k]
        idx = idx[np.argsort(-s[idx])]
        return [(int(i), float(s[i])) for i in idx]

    def save(self) -> None:
        sp.save_npz(MODELS / "bm25_weights.npz", self.W.tocsr())
        (MODELS / "bm25_vocab.json").write_text(json.dumps(self.vocab))
        np.save(MODELS / "bm25_idf.npy", self.idf)

    @classmethod
    def load(cls) -> "BM25":
        return cls(json.loads((MODELS / "bm25_vocab.json").read_text()), sp.load_npz(MODELS / "bm25_weights.npz"), np.load(MODELS / "bm25_idf.npy"))
