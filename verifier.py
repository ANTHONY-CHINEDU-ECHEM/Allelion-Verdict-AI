"""Sentence level claim verifier.

Scores whether a sentence is supported by the evidence item it cites. Eleven
features compare the sentence with the cited text; a logistic regression
trained on perturbed corpus sentences turns them into a probability, and hard
rules override it: a number, gene symbol, criterion code or HGVS description
absent from the cited evidence, or a missing or invalid citation, always makes
the sentence unsupported.

Training data: sentences sampled from the passage corpus, each paired with its
own passage (supported) in verbatim and lightly paraphrased form, and in seven
corrupted forms (wrong citation, gene swapped, number changed, negation
flipped, criterion code swapped, phenotype or disease word swapped, class word
swapped).
"""
from __future__ import annotations

import json
import pickle
import random
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from allelion.config import MODELS, SEED
from allelion.retrieval.text import sentences, tokenize

NUM = re.compile(r"(?<![A-Za-z:])\d+(?:\.\d+)?%?")
GENE = re.compile(r"\b[A-Z][A-Z0-9]{2,9}\b")
CODE = re.compile(r"\b(?:PVS1|PS[1-4]|PM[1-6]|PP[1-5]|BA1|BS[1-4]|BP[1-7])\b")
HGVS = re.compile(r"\b[cp]\.[^\s,;)]+")
NEG = re.compile(r"\b(?:not|no|never|none|without|absent|cannot|neither|nor|unlikely|does not|is not|are not)\b", re.I)
CLASS_WORDS = ["pathogenic", "benign", "uncertain", "likely pathogenic", "likely benign"]
FEATURES = ["tok_recall", "bigram_recall", "best_sent", "num_ok", "gene_ok", "code_ok", "hgvs_ok", "neg_agree", "len_ratio", "has_cite", "class_ok",
            "antonym_ok"]
ANTONYMS = [("dominant", "recessive"), ("increase", "decrease"), ("increased", "decreased"), ("higher", "lower"), ("pathogenic", "benign"),
            ("supporting", "strong"), ("absent", "present"), ("last", "first"), ("gain", "loss"), ("above", "below"), ("more", "less"),
            ("upstream", "downstream"), ("maternal", "paternal"), ("homozygous", "heterozygous"), ("met", "not met")]
NON_GENES = {"DNA", "RNA", "PCR", "ACMG", "AMP", "SVI", "HPO", "OMIM", "ORPHA", "PMID", "UTR", "NMD", "VUS", "CNV", "MANE", "HGVS", "ENIGMA",
             "UK", "USA", "ID", "CI", "OR", "HP", "LR", "SNP", "THE", "AND", "FOR", "LOF", "GOF", "MAF", "CFTR2", "RASOPATHY"}


def _bigrams(t):
    return set(zip(t, t[1:]))


def _numbers(s):
    return {n.rstrip("%") for n in NUM.findall(s)}


def features(sentence: str, evidence: str, cited: bool = True) -> dict:
    st, et = tokenize(sentence), tokenize(evidence)
    ss, es = set(st), set(et)
    tok_recall = len(ss & es) / max(len(ss), 1)
    sb = _bigrams(st)
    bigram_recall = len(sb & _bigrams(et)) / max(len(sb), 1)
    best, best_s = 0.0, ""
    ev_s = sentences(evidence) or [evidence]
    spans = ev_s + [f"{a} {b}" for a, b in zip(ev_s, ev_s[1:])]
    for e in spans:
        o = len(ss & set(tokenize(e))) / max(len(ss), 1)
        if o > best:
            best, best_s = o, e
    ev_nums = _numbers(evidence)
    nums = _numbers(sentence)
    num_ok = float(all(n in ev_nums for n in nums))
    genes = {g for g in GENE.findall(sentence) if g not in NON_GENES and not CODE.fullmatch(g)}
    gene_ok = float(all(g in evidence for g in genes))
    code_ok = float(all(c in evidence for c in CODE.findall(sentence)))
    hgvs_ok = float(all(h.rstrip(".") in evidence for h in HGVS.findall(sentence)))
    neg_agree = float(bool(NEG.search(sentence)) == bool(NEG.search(best_s))) if best_s else 0.0
    sl = sentence.lower()
    el = evidence.lower()
    class_ok = float(all(w in el for w in CLASS_WORDS if re.search(rf"\b{w}\b", sl)))
    bl = best_s.lower()
    antonym_ok = 1.0
    for a, b in ANTONYMS:
        for x, y in ((a, b), (b, a)):
            if re.search(rf"\b{x}\b", sl) and not re.search(rf"\b{x}\b", bl) and re.search(rf"\b{y}\b", bl):
                antonym_ok = 0.0
    return {"tok_recall": tok_recall, "bigram_recall": bigram_recall, "best_sent": best, "num_ok": num_ok, "gene_ok": gene_ok,
            "code_ok": code_ok, "hgvs_ok": hgvs_ok, "neg_agree": neg_agree, "len_ratio": min(len(st) / max(len(et), 1), 2.0),
            "has_cite": float(cited), "class_ok": class_ok, "antonym_ok": antonym_ok}


def hard_fail(f: dict) -> str | None:
    if not f["has_cite"]:
        return "no citation"
    if not f["num_ok"]:
        return "a number is not in the cited evidence"
    if not f["gene_ok"]:
        return "a gene is not in the cited evidence"
    if not f["code_ok"]:
        return "a criterion code is not in the cited evidence"
    if not f["hgvs_ok"]:
        return "a variant description is not in the cited evidence"
    if not f["antonym_ok"]:
        return "a word contradicts the best matching evidence sentence"
    return None


class Verifier:
    def __init__(self, model: LogisticRegression):
        self.model = model

    def score(self, sentence: str, evidence: str | None) -> dict:
        f = features(sentence, evidence or "", evidence is not None)
        p = float(self.model.predict_proba(pd.DataFrame([f])[FEATURES])[0, 1])
        reason = hard_fail(f)
        supported = reason is None and p >= 0.5
        return {"supported": supported, "probability": round(p, 4), "reason": reason or ("" if supported else "low support score"), "features": f}

    @classmethod
    def load(cls) -> "Verifier":
        with open(MODELS / "verifier.pkl", "rb") as fh:
            return cls(pickle.load(fh))


# ------------------------------------------------------------------ training data
SWAP_PHRASES = [("dominant", "recessive"), ("recessive", "dominant"), ("increase", "decrease"), ("higher", "lower"), ("lower", "higher"),
                ("pathogenic", "benign"), ("benign", "pathogenic"), ("supporting", "strong"), ("strong", "supporting"), ("absent", "present"),
                ("last exon", "first exon"), ("gain of function", "loss of function"), ("loss of function", "gain of function")]
SYN = {"shows": "demonstrates", "associated": "linked", "variant": "change", "patients": "individuals", "reported": "described",
       "frequency": "rate", "study": "report", "significant": "notable", "evidence": "support", "common": "frequent"}


def _paraphrase(s, rng):
    words = s.split()
    out = [SYN.get(w.lower(), w) if rng.random() < 0.5 else w for w in words]
    if len(out) > 8 and rng.random() < 0.5:
        drop = rng.randrange(1, len(out) - 1)
        if not re.search(r"\d|[A-Z]{3}", out[drop]):
            out.pop(drop)
    return " ".join(out)


def _corrupt(s, kind, rng, genes, codes):
    if kind == "gene":
        gs = [g for g in GENE.findall(s) if g not in NON_GENES]
        if not gs:
            return None
        return s.replace(gs[0], rng.choice([g for g in genes if g != gs[0]]), 1)
    if kind == "number":
        ns = NUM.findall(s)
        if not ns:
            return None
        n = ns[0].rstrip("%")
        try:
            new = str(int(float(n)) + rng.choice([1, 2, 3, 5, 10]))
        except ValueError:
            return None
        return s.replace(n, new, 1)
    if kind == "negation":
        if NEG.search(s):
            return NEG.sub("", s, count=1).replace("  ", " ")
        m = re.search(r"\b(is|are|was|were|can|does|do|has|have)\b", s)
        return s[:m.end()] + " not" + s[m.end():] if m else None
    if kind == "code":
        cs = CODE.findall(s)
        if not cs:
            return None
        return s.replace(cs[0], rng.choice([c for c in codes if c != cs[0]]), 1)
    if kind == "swap":
        for a, b in SWAP_PHRASES:
            if re.search(rf"\b{a}\b", s, re.I):
                return re.sub(rf"\b{a}\b", b, s, count=1, flags=re.I)
        return None
    return None


def train(n_sent: int = 6000) -> dict:
    rng = random.Random(SEED)
    P = pd.read_parquet(MODELS / "passages.parquet")
    genes = sorted({g for gs in P["genes"] for g in gs})
    codes = ["PVS1", "PS1", "PS2", "PS3", "PS4", "PM1", "PM2", "PM3", "PM4", "PM5", "PM6", "PP1", "PP2", "PP3", "PP4", "BA1", "BS1", "BS2", "BS3", "BS4",
             "BP1", "BP2", "BP3", "BP4", "BP5", "BP7"]
    texts = P["text"].tolist()
    weights = P["kind"].map({"guidance": 20, "gene": 3, "disease": 1, "phenotype": 1, "literature": 3}).values.astype(float)
    picks = rng.choices(range(len(P)), weights=weights, k=n_sent)
    rows = []
    for i in picks:
        ss = [s for s in sentences(texts[i]) if 6 <= len(s.split()) <= 45]
        if not ss:
            continue
        s = rng.choice(ss)
        ev = texts[i]
        rows.append({**features(s, ev), "y": 1, "kind": "verbatim"})
        rows.append({**features(_paraphrase(s, rng), ev), "y": 1, "kind": "paraphrase"})
        j = rng.randrange(len(P))
        if j != i:
            rows.append({**features(s, texts[j]), "y": 0, "kind": "wrong_citation"})
        for kind in ("gene", "number", "negation", "code", "swap"):
            c = _corrupt(s, kind, rng, genes, codes)
            if c and c != s:
                rows.append({**features(c, ev), "y": 0, "kind": kind})
    df = pd.DataFrame(rows)
    rng2 = np.random.default_rng(SEED)
    test = rng2.random(len(df)) < 0.25
    model = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    model.fit(df.loc[~test, FEATURES], df.loc[~test, "y"])
    with open(MODELS / "verifier.pkl", "wb") as fh:
        pickle.dump(model, fh)
    te = df[test].copy()
    te["p"] = model.predict_proba(te[FEATURES])[:, 1]
    te["hard"] = [hard_fail(r) is not None for r in te[FEATURES].to_dict("records")]
    te["pred"] = (te["p"] >= 0.5) & ~te["hard"]
    per_kind = {k: {"n": int(len(g)), "accuracy": round(float((g["pred"] == (g["y"] == 1)).mean()), 4)} for k, g in te.groupby("kind")}
    rep = {"train_sentences": int((~test).sum()), "test_sentences": int(test.sum()), "auroc": round(float(roc_auc_score(te["y"], te["p"])), 4),
           "accuracy_with_hard_rules": round(float((te["pred"] == (te["y"] == 1)).mean()), 4), "per_perturbation": per_kind,
           "coefficients": dict(zip(FEATURES, [round(float(c), 3) for c in model.coef_[0]])), "intercept": round(float(model.intercept_[0]), 3)}
    (MODELS / "verifier_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    train()
