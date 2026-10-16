"""Text normalisation shared by every retrieval component (mirrored in web/src/engine/text.ts).

Tokens keep variant notation intact (c.1517a>g, p.d506g, rs1801133), criterion
codes (pvs1, pm2) and gene symbols, fold British and American spellings that
matter in genetics, and apply a light suffix stemmer. Stop words are dropped.
"""
from __future__ import annotations

import re

STOP = set("""a an and are as at be by can do does for from has have how i if in into is it its may more most no not of on or
should so such than that the their then there these this those to under was we were what when where which while who why will
with within without would you your about after also any been being both each other out over per same some up very only use used
using""".split())

SPELL = {"haemoglobin": "hemoglobin", "anaemia": "anemia", "oedema": "edema", "tumour": "tumor", "tumours": "tumors", "foetal": "fetal",
         "paediatric": "pediatric", "haematological": "hematological", "leukaemia": "leukemia", "oestrogen": "estrogen",
         "haemophilia": "hemophilia", "haemochromatosis": "hemochromatosis", "behaviour": "behavior", "colour": "color",
         "favour": "favor", "normalised": "normalized", "characterise": "characterize", "characterised": "characterized",
         "analyse": "analyze", "centre": "center", "fibre": "fiber", "dyslipidaemia": "dyslipidemia", "ischaemic": "ischemic",
         "hypercholesterolaemia": "hypercholesterolemia", "glycaemic": "glycemic", "gynaecological": "gynecological"}

_TOKEN = re.compile(r"[cgp]\.[\w*+\-_>=?]+|rs\d+|[a-z0-9]+(?:[>'][a-z0-9]+)?", re.I)


def stem(w: str) -> str:
    if len(w) <= 4 or any(c.isdigit() for c in w):
        return w
    for suf in ("ational", "ization", "ations", "ation", "ities", "ness", "ments", "ment", "ings", "ing", "ies", "ied", "ers", "edly",
                "ed", "es", "ly", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            if suf in ("ies", "ied"):
                return w[: -len(suf)] + "y"
            return w[: -len(suf)]
    return w


def tokenize(text: str) -> list[str]:
    out = []
    for m in _TOKEN.finditer(text or ""):
        w = m.group().lower()
        w = SPELL.get(w, w)
        if w in STOP or len(w) < 2 and not w.isdigit():
            continue
        out.append(stem(w))
    return out


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", text.strip())
    return [p.strip() for p in parts if p.strip()]
