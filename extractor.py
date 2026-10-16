"""Variant mention extraction and normalisation from biomedical text.

Three components:

1. A rule based extractor (regular expressions in the spirit of MutationFinder
   and tmVar's patterns) used as the baseline and as the normaliser.
2. A conditional random field span tagger trained on the tmVar training split
   and 80% of SETH (split by PubMed identifier), tested on the tmVar test split
   and the remaining SETH abstracts, which it never sees.
3. A normaliser that turns a mention into a canonical form: protein changes as
   one letter wild type, position, mutant (D506G), coding changes as
   c.position ref > alt, and dbSNP identifiers.

The document level test uses the MutationFinder test set, whose gold standard
lists the normalised protein mutations in each abstract.
"""
from __future__ import annotations

import json
import pickle
import random
import re
from collections import Counter

import pandas as pd
import sklearn_crfsuite

from allelion.config import KB, MODELS, SEED
from allelion.kb.hgvs import AA3

AA_NAMES = {
    "alanine": "A", "arginine": "R", "asparagine": "N", "aspartate": "D", "aspartic acid": "D", "cysteine": "C", "glutamine": "Q",
    "glutamate": "E", "glutamic acid": "E", "glycine": "G", "histidine": "H", "isoleucine": "I", "leucine": "L", "lysine": "K",
    "methionine": "M", "phenylalanine": "F", "proline": "P", "serine": "S", "threonine": "T", "tryptophan": "W", "tyrosine": "Y",
    "valine": "V",
}
AA3_UP = {k.upper(): v for k, v in AA3.items()}
AA1 = set("ACDEFGHIKLMNPQRSTVWY")
_AA3 = "(?:Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val|Ter|Stop|Xaa)"
_AA1 = "[ACDEFGHIKLMNPQRSTVWY]"
_NAME = "(?:" + "|".join(sorted(AA_NAMES, key=len, reverse=True)) + ")"

PATTERNS = [
    ("protein", re.compile(rf"\bp\.\(?{_AA3}\d+(?:{_AA3}|\*|=|fs\S*|del|dup)\)?")),
    ("protein", re.compile(rf"\bp\.{_AA1}\d+(?:{_AA1}|\*|X|fs\S*)\b")),
    ("protein", re.compile(rf"\b{_AA3}[\s-]?\d+[\s-]?(?:{_AA3}|\*|stop)\b")),
    ("protein", re.compile(rf"\b{_AA3}(?:\s?(?:->|>|→|to)\s?){_AA3}\s?(?:at\s)?(?:codon|position|residue)?\s?\d+\b")),
    ("protein", re.compile(rf"\b{_NAME}[\s-]?\d+[\s-]?(?:to[\s-])?{_NAME}\b", re.I)),
    ("protein", re.compile(rf"\b{_NAME}\s(?:to|by|for|->)\s{_NAME}\s(?:substitution\s)?(?:at\s)?(?:codon|position|residue)\s\d+\b", re.I)),
    ("protein", re.compile(rf"(?<![A-Za-z0-9]){_AA1}\d{{1,4}}{_AA1}(?![A-Za-z0-9])")),
    ("dna", re.compile(r"\bc\.[-*]?\d+(?:[+-]\d+)?(?:_[-*]?\d+(?:[+-]\d+)?)?(?:[ACGT]+>[ACGT]+|del[ACGT]*ins[ACGT]+|del[ACGT]*|dup[ACGT]*|ins[ACGT]+)")),
    ("dna", re.compile(r"\bg\.\d+(?:_\d+)?(?:[ACGT]>[ACGT]|del[ACGT]*|dup[ACGT]*|ins[ACGT]+)")),
    ("dna", re.compile(r"(?<![A-Za-z0-9.])[-+]?\d+(?:[+-]\d+)?[ACGT]\s?>\s?[ACGT](?![A-Za-z])")),
    ("dna", re.compile(r"(?<![A-Za-z0-9])IVS\d+[+-]\d+[ACGT]>[ACGT]")),
    ("snp", re.compile(r"\brs\d{3,}\b")),
]


def normalise(text: str) -> dict | None:
    t = text.strip().strip(".,;()[]")
    m = re.match(r"^rs(\d+)$", t)
    if m:
        return {"kind": "snp", "norm": f"rs{m.group(1)}"}
    m = re.match(r"^(?:p\.)?\(?(" + _AA3 + r")[\s-]?(\d+)[\s-]?(" + _AA3 + r"|\*|stop|=)\)?$", t, re.I)
    if m:
        wt = AA3_UP.get(m.group(1).upper(), "?")
        mt = m.group(3)
        mt1 = "*" if mt.lower() in ("*", "stop", "ter") else ("=" if mt == "=" else AA3_UP.get(mt.upper(), "?"))
        return {"kind": "protein", "norm": f"{wt}{m.group(2)}{mt1}"}
    m = re.match(r"^(?:p\.)?(" + _AA1 + r")(\d{1,5})(" + _AA1 + r"|\*|X)$", t)
    if m:
        return {"kind": "protein", "norm": f"{m.group(1)}{m.group(2)}{'*' if m.group(3) in '*X' else m.group(3)}"}
    m = re.match(r"^(" + _AA3 + r")\s?(?:->|>|→|to)\s?(" + _AA3 + r")\s?(?:at\s)?(?:codon|position|residue)?\s?(\d+)$", t, re.I)
    if m:
        return {"kind": "protein", "norm": f"{AA3_UP.get(m.group(1).upper(), '?')}{m.group(3)}{AA3_UP.get(m.group(2).upper(), '?')}"}
    m = re.match(r"^(" + _NAME + r")[\s-]?(\d+)[\s-]?(?:to[\s-])?(" + _NAME + r")$", t, re.I)
    if m:
        return {"kind": "protein", "norm": f"{AA_NAMES[m.group(1).lower()]}{m.group(2)}{AA_NAMES[m.group(3).lower()]}"}
    m = re.match(r"^(" + _NAME + r")\s(?:to|by|for|->)\s(" + _NAME + r")\s(?:substitution\s)?(?:at\s)?(?:codon|position|residue)\s(\d+)$", t, re.I)
    if m:
        return {"kind": "protein", "norm": f"{AA_NAMES[m.group(1).lower()]}{m.group(3)}{AA_NAMES[m.group(2).lower()]}"}
    m = re.match(r"^c\.([-*]?\d+(?:[+-]\d+)?)([ACGT])>([ACGT])$", t)
    if m:
        return {"kind": "dna", "norm": f"c.{m.group(1)}{m.group(2)}>{m.group(3)}"}
    m = re.match(r"^c\.(.+)$", t)
    if m:
        return {"kind": "dna", "norm": t.replace(" ", "")}
    m = re.match(r"^([-+]?\d+(?:[+-]\d+)?)([ACGT])\s?>\s?([ACGT])$", t)
    if m:
        return {"kind": "dna", "norm": f"c.{m.group(1).lstrip('+')}{m.group(2)}>{m.group(3)}"}
    return None


def rule_extract(text: str) -> list[dict]:
    spans = []
    for kind, pat in PATTERNS:
        for m in pat.finditer(text):
            s, e = m.span()
            if any(not (e <= a or s >= b) for a, b, _ in spans):
                continue
            spans.append((s, e, kind))
    out = []
    for s, e, kind in sorted(spans):
        frag = text[s:e]
        # discard single letter protein forms that look like cell lines or identifiers (for example A549 has no mutant residue)
        if kind == "protein" and re.fullmatch(r"[A-Z]\d+[A-Z]", frag) and len(frag) < 4:
            continue
        out.append({"start": s, "end": e, "text": frag, "kind": kind, **({"norm": n["norm"]} if (n := normalise(frag)) else {})})
    return out


# ------------------------------------------------------------------ CRF
_TOK = re.compile(r"[A-Za-z0-9.*+>_\-→=?]+|[^\sA-Za-z0-9]")


def tokens(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group()) for m in _TOK.finditer(text)]


def _shape(w: str) -> str:
    s = re.sub(r"[A-Z]", "A", w)
    s = re.sub(r"[a-z]", "a", s)
    s = re.sub(r"[0-9]", "0", s)
    return re.sub(r"(.)\1+", r"\1\1", s)


def _wfeat(w: str, prefix: str) -> dict:
    lw = w.lower()
    return {
        f"{prefix}lower": lw[:20], f"{prefix}shape": _shape(w)[:12], f"{prefix}digit": any(c.isdigit() for c in w),
        f"{prefix}gt": ">" in w, f"{prefix}c.": lw.startswith("c."), f"{prefix}p.": lw.startswith("p."), f"{prefix}rs": bool(re.fullmatch(r"rs\d+", lw)),
        f"{prefix}aa3": bool(re.match(_AA3, w)), f"{prefix}aaname": lw in AA_NAMES, f"{prefix}ivs": lw.startswith("ivs"),
        f"{prefix}p3": lw[:3], f"{prefix}s3": lw[-3:], f"{prefix}len": min(len(w), 12),
        f"{prefix}aa1num": bool(re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]\d+[ACDEFGHIKLMNPQRSTVWY*X]", w)),
        f"{prefix}del": "del" in lw or "ins" in lw or "dup" in lw or "fs" in lw,
    }


def sent_features(toks: list[str], rule_tags: list[int]) -> list[dict]:
    feats = []
    for i, w in enumerate(toks):
        f = {"bias": 1.0, "rule": rule_tags[i]}
        f.update(_wfeat(w, ""))
        for o in (-2, -1, 1, 2):
            j = i + o
            if 0 <= j < len(toks):
                f.update({k: v for k, v in _wfeat(toks[j], f"{o}:").items() if k.endswith(("lower", "shape", "digit", "aa3", "aaname"))})
            else:
                f[f"{o}:pad"] = True
        feats.append(f)
    return feats


def _bio(text: str, spans: list[tuple[int, int]]) -> tuple[list[str], list[str], list[tuple[int, int]], list[int]]:
    toks = tokens(text)
    rule = rule_extract(text)
    labels, rtags = [], []
    for s, e, _ in toks:
        lab = "O"
        for a, b in spans:
            if s >= a and e <= b:
                lab = "B" if s == a or not labels or labels[-1] == "O" else "I"
                break
        labels.append(lab)
        rtags.append(int(any(s >= r["start"] and e <= r["end"] for r in rule)))
    return [t[2] for t in toks], labels, [(t[0], t[1]) for t in toks], rtags


def decode(offs, labels) -> list[tuple[int, int]]:
    spans, cur = [], None
    for (s, e), lab in zip(offs, labels):
        if lab == "B" or (lab == "I" and cur is None):
            if cur:
                spans.append(cur)
            cur = [s, e]
        elif lab == "I":
            cur[1] = e
        else:
            if cur:
                spans.append(cur)
            cur = None
    if cur:
        spans.append(cur)
    return [tuple(x) for x in spans]


def _doc_text(r) -> str:
    return r["title"] + " " + r["text"] if r["corpus"] == "tmVar" else r["text"]


def _prf(tp, fp, fn) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0, "tp": tp, "fp": fp, "fn": fn}


def _span_eval(gold: dict, pred: dict, overlap: bool) -> dict:
    tp = fp = fn = 0
    for d in gold.keys() | pred.keys():
        g, p = set(gold.get(d, [])), set(pred.get(d, []))
        if overlap:
            matched_g = {x for x in g if any(not (x[1] <= y[0] or x[0] >= y[1]) for y in p)}
            matched_p = {y for y in p if any(not (x[1] <= y[0] or x[0] >= y[1]) for x in g)}
            tp += len(matched_p)
            fp += len(p - matched_p)
            fn += len(g - matched_g)
        else:
            tp += len(g & p)
            fp += len(p - g)
            fn += len(g - p)
    return _prf(tp, fp, fn)


def train() -> dict:
    docs = pd.read_parquet(KB / "literature_docs.parquet")
    ments = pd.read_parquet(KB / "literature_mentions.parquet")
    span_m = ments[(ments["start"] >= 0) & ments["type"].isin(["DNAMutation", "ProteinMutation", "SNP"])]
    by_doc = span_m.groupby("doc_id").apply(lambda g: list(zip(g["start"], g["end"])), include_groups=False).to_dict()
    seth = docs[docs["corpus"] == "SETH"]["doc_id"].tolist()
    random.Random(SEED).shuffle(seth)
    seth_test = set(seth[: len(seth) // 5])
    train_ids = set(docs[(docs["corpus"] == "tmVar") & (docs["split"] == "train")]["doc_id"]) | (set(seth) - seth_test)
    test_ids = set(docs[(docs["corpus"] == "tmVar") & (docs["split"] == "test")]["doc_id"]) | seth_test
    drows = docs.set_index("doc_id")

    def make(ids):
        X, y, meta = [], [], []
        for d in sorted(ids):
            r = drows.loc[d]
            text = _doc_text({"title": r["title"], "text": r["text"], "corpus": r["corpus"]})
            toks, labs, offs, rt = _bio(text, by_doc.get(d, []))
            X.append(sent_features(toks, rt))
            y.append(labs)
            meta.append((d, offs, text))
        return X, y, meta

    Xtr, ytr, _ = make(train_ids)
    Xte, yte, mte = make(test_ids)
    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.05, c2=0.05, max_iterations=150, all_possible_transitions=True)
    crf.fit(Xtr, ytr)
    with open(MODELS / "variant_crf.pkl", "wb") as fh:
        pickle.dump(crf, fh)
    pred = crf.predict(Xte)

    gold, crf_spans, rule_spans = {}, {}, {}
    for (d, offs, text), labs in zip(mte, pred):
        gold[d] = by_doc.get(d, [])
        crf_spans[d] = decode(offs, labs)
        rule_spans[d] = [(r["start"], r["end"]) for r in rule_extract(text)]
    report = {
        "train_documents": len(train_ids), "test_documents": len(test_ids),
        "train_mentions": int(sum(len(by_doc.get(d, [])) for d in train_ids)), "test_mentions": int(sum(len(gold[d]) for d in gold)),
        "crf_exact": _span_eval(gold, crf_spans, False), "crf_overlap": _span_eval(gold, crf_spans, True),
        "rules_exact": _span_eval(gold, rule_spans, False), "rules_overlap": _span_eval(gold, rule_spans, True),
    }
    # per corpus
    for corp in ("tmVar", "SETH"):
        ids = [d for d in gold if drows.loc[d, "corpus"] == corp]
        report[f"crf_exact_{corp}"] = _span_eval({d: gold[d] for d in ids}, {d: crf_spans[d] for d in ids}, False)
        report[f"rules_exact_{corp}"] = _span_eval({d: gold[d] for d in ids}, {d: rule_spans[d] for d in ids}, False)

    # document level normalisation on the MutationFinder test set (never used in training)
    mf = docs[(docs["corpus"] == "MutationFinder") & (docs["split"] == "test")]
    mf_gold = ments[ments["doc_id"].isin(mf["doc_id"])].groupby("doc_id")["norm"].apply(set).to_dict()
    tp = fp = fn = 0
    tp_r = fp_r = fn_r = 0
    for r in mf.itertuples():
        text = r.title + " " + r.text
        g = {x.upper() for x in mf_gold.get(r.doc_id, set())}
        rules_norm = {x["norm"].upper() for x in rule_extract(text) if x.get("kind") == "protein" and x.get("norm")}
        found = extract(text, crf)
        crf_norm = {x["norm"].upper() for x in found if x.get("kind") == "protein" and x.get("norm")}
        tp += len(g & crf_norm); fp += len(crf_norm - g); fn += len(g - crf_norm)
        tp_r += len(g & rules_norm); fp_r += len(rules_norm - g); fn_r += len(g - rules_norm)
    report["mutationfinder_doc_level_crf_plus_rules"] = _prf(tp, fp, fn)
    report["mutationfinder_doc_level_rules"] = _prf(tp_r, fp_r, fn_r)
    report["mutationfinder_test_documents"] = len(mf)
    (MODELS / "extractor_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return report


def load_crf():
    with open(MODELS / "variant_crf.pkl", "rb") as fh:
        return pickle.load(fh)


def extract(text: str, crf=None) -> list[dict]:
    """CRF spans, normalised; rule matches the CRF missed are added when they normalise cleanly."""
    crf = crf or load_crf()
    toks, _, offs, rt = _bio(text, [])
    labs = crf.predict_single(sent_features(toks, rt))
    out = []
    for s, e in decode(offs, labs):
        frag = text[s:e]
        n = normalise(frag)
        out.append({"start": s, "end": e, "text": frag, "kind": n["kind"] if n else "unknown", **({"norm": n["norm"]} if n else {})})
    for r in rule_extract(text):
        if r.get("norm") and not any(not (r["end"] <= o["start"] or r["start"] >= o["end"]) for o in out):
            out.append(r)
    return sorted(out, key=lambda x: x["start"])


if __name__ == "__main__":
    train()
