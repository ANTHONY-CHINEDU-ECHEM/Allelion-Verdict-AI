"""Dense retrieval trained on the corpus itself, on CPU.

1. Skip gram word2vec (gensim) over every passage.
2. Smooth inverse frequency (SIF) sentence vectors: frequency weighted average
   of word vectors with the first principal component removed (Arora 2017).
3. A dual encoder of two linear towers on top of SIF vectors, trained with a
   symmetric in batch InfoNCE loss on pairs mined from the corpus (passage
   title with passage text, and a random sentence with the rest of its
   passage). Gradients are written out analytically and optimised with Adam in
   NumPy, so the whole model is two matrices that the browser can multiply.
"""
from __future__ import annotations

import json
import random
from collections import Counter

import numpy as np
from gensim.models import Word2Vec

from allelion.config import MODELS, SEED
from allelion.retrieval.text import sentences, tokenize

DIM, OUT = 100, 128
SIF_A = 1e-3


class SIF:
    def __init__(self, vocab: dict[str, int], vectors: np.ndarray, weights: np.ndarray, pc: np.ndarray):
        self.vocab, self.V, self.w, self.pc = vocab, vectors, weights, pc

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.V.shape[1]), dtype=np.float32)
        for i, t in enumerate(texts):
            ids = [self.vocab[w] for w in tokenize(t) if w in self.vocab]
            if ids:
                out[i] = (self.V[ids] * self.w[ids, None]).sum(0) / len(ids)
        out -= (out @ self.pc)[:, None] * self.pc[None, :]
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(n, 1e-8)


def train_sif(texts: list[str]) -> SIF:
    toks = [tokenize(t) for t in texts]
    w2v = Word2Vec(toks, vector_size=DIM, window=5, min_count=3, sg=1, negative=8, epochs=5, workers=2, seed=SEED)
    vocab = {w: i for i, w in enumerate(w2v.wv.index_to_key)}
    V = w2v.wv.vectors.astype(np.float32)
    cnt = Counter(w for ts in toks for w in ts)
    total = sum(cnt.values())
    weights = np.array([SIF_A / (SIF_A + cnt[w] / total) for w in w2v.wv.index_to_key], dtype=np.float32)
    tmp = SIF(vocab, V, weights, np.zeros(DIM, dtype=np.float32))
    sample = random.Random(SEED).sample(texts, min(20000, len(texts)))
    E = tmp.embed(sample)
    _, _, vt = np.linalg.svd(E - E.mean(0), full_matrices=False)
    return SIF(vocab, V, weights, vt[0].astype(np.float32))


class DualEncoder:
    def __init__(self, Wq: np.ndarray, Wp: np.ndarray):
        self.Wq, self.Wp = Wq, Wp

    @staticmethod
    def _norm(x):
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)

    def q(self, X):
        return self._norm(X @ self.Wq)

    def p(self, X):
        return self._norm(X @ self.Wp)


def mine_pairs(titles: list[str], texts: list[str], rng: random.Random) -> list[tuple[str, int]]:
    pairs = []
    for i, (t, x) in enumerate(zip(titles, texts)):
        pairs.append((t, i))
        ss = sentences(x)
        if len(ss) >= 3:
            pairs.append((rng.choice(ss), i))
    rng.shuffle(pairs)
    return pairs


def train_dual(sif: SIF, titles: list[str], texts: list[str], P: np.ndarray, epochs: int = 3, batch: int = 256, lr: float = 2e-3,
               temp: float = 0.05) -> tuple[DualEncoder, list[float]]:
    rng = random.Random(SEED)
    nrng = np.random.default_rng(SEED)
    pairs = mine_pairs(titles, texts, rng)
    Q = sif.embed([q for q, _ in pairs])
    idx = np.array([i for _, i in pairs])
    Wq = (np.eye(DIM, OUT) + 0.01 * nrng.standard_normal((DIM, OUT))).astype(np.float32)
    Wp = Wq.copy()
    m = {k: np.zeros_like(Wq) for k in ("q", "p")}
    v = {k: np.zeros_like(Wq) for k in ("q", "p")}
    t, losses = 0, []
    for ep in range(epochs):
        order = nrng.permutation(len(pairs))
        for s in range(0, len(order) - batch + 1, batch):
            b = order[s:s + batch]
            xq, xp = Q[b], P[idx[b]]
            hq, hp = xq @ Wq, xp @ Wp
            nq = np.linalg.norm(hq, axis=1, keepdims=True) + 1e-8
            np_ = np.linalg.norm(hp, axis=1, keepdims=True) + 1e-8
            zq, zp = hq / nq, hp / np_
            S = zq @ zp.T / temp
            # symmetric InfoNCE
            def softmax(a):
                a = a - a.max(1, keepdims=True)
                e = np.exp(a)
                return e / e.sum(1, keepdims=True)
            A, Bm = softmax(S), softmax(S.T)
            n = len(b)
            loss = -(np.log(np.diag(A) + 1e-12).mean() + np.log(np.diag(Bm) + 1e-12).mean()) / 2
            I = np.eye(n, dtype=np.float32)
            dS = ((A - I) + (Bm - I).T) / (2 * n) / temp
            dzq, dzp = dS @ zp, dS.T @ zq
            # back through the L2 normalisation
            dhq = (dzq - zq * (dzq * zq).sum(1, keepdims=True)) / nq
            dhp = (dzp - zp * (dzp * zp).sum(1, keepdims=True)) / np_
            grads = {"q": xq.T @ dhq, "p": xp.T @ dhp}
            t += 1
            for k, W in (("q", Wq), ("p", Wp)):
                m[k] = 0.9 * m[k] + 0.1 * grads[k]
                v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
                W -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
            losses.append(float(loss))
    return DualEncoder(Wq, Wp), losses


def build(titles: list[str], texts: list[str]) -> dict:
    full = [f"{a}. {b}" for a, b in zip(titles, texts)]
    sif = train_sif(full)
    P = sif.embed(full)
    de, losses = train_dual(sif, titles, texts, P)
    np.save(MODELS / "sif_vectors.npy", sif.V.astype(np.float16))
    np.save(MODELS / "sif_weights.npy", sif.w)
    np.save(MODELS / "sif_pc.npy", sif.pc)
    (MODELS / "sif_vocab.json").write_text(json.dumps(sif.vocab))
    np.save(MODELS / "passage_sif.npy", P.astype(np.float16))
    np.save(MODELS / "dual_Wq.npy", de.Wq)
    np.save(MODELS / "dual_Wp.npy", de.Wp)
    np.save(MODELS / "passage_dual.npy", de.p(P).astype(np.float16))
    curve = [float(np.mean(losses[i:i + 20])) for i in range(0, len(losses), 20)]
    (MODELS / "dual_training.json").write_text(json.dumps({"loss_curve": curve, "steps": len(losses)}))
    return {"vocab": len(sif.vocab), "steps": len(losses), "loss_start": curve[0], "loss_end": curve[-1]}


def load() -> tuple[SIF, DualEncoder, np.ndarray, np.ndarray]:
    sif = SIF(json.loads((MODELS / "sif_vocab.json").read_text()), np.load(MODELS / "sif_vectors.npy").astype(np.float32),
              np.load(MODELS / "sif_weights.npy"), np.load(MODELS / "sif_pc.npy"))
    de = DualEncoder(np.load(MODELS / "dual_Wq.npy"), np.load(MODELS / "dual_Wp.npy"))
    return sif, de, np.load(MODELS / "passage_sif.npy").astype(np.float32), np.load(MODELS / "passage_dual.npy").astype(np.float32)
