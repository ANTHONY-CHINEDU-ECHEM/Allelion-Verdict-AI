"""Literature corpora: documents and gold variant mention annotations.

Sources (all from the nala corpora collection, pinned in the manifest):
  tmVar      500 PubMed abstracts, span level DNA, protein and SNP mentions with normalised forms, fixed train and test split
  SETH       630 abstracts, span level SNP mentions (brat standoff)
  MutationFinder  813 abstracts, document level gold mutation lists (devo and test sets)
  OSIRIS     105 abstracts about sequence variants (text only here)
  Variome    10 full text articles split into paragraphs (gene, disease and mutation entities)

Outputs: data/kb/literature_docs.parquet and data/kb/literature_mentions.parquet
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from allelion.config import KB, RAW

C = RAW / "corpora"


def _pubtator(path: Path, split: str) -> tuple[list, list]:
    docs, ments = [], []
    cur = {}
    for line in open(path, encoding="utf8"):
        line = line.rstrip("\n")
        if not line:
            if cur:
                docs.append(cur)
                cur = {}
            continue
        if "|t|" in line[:20]:
            pmid, _, title = line.split("|", 2)
            cur = {"doc_id": f"tmvar:{pmid}", "pmid": pmid, "corpus": "tmVar", "split": split, "title": title, "text": ""}
        elif "|a|" in line[:20]:
            cur["text"] = line.split("|", 2)[2]
        else:
            f = line.split("\t")
            if len(f) >= 5:
                ments.append({"doc_id": f"tmvar:{f[0]}", "start": int(f[1]), "end": int(f[2]), "text": f[3], "type": f[4],
                              "norm": f[5] if len(f) > 5 else ""})
    if cur:
        docs.append(cur)
    return docs, ments


def _seth() -> tuple[list, list]:
    docs, ments = [], []
    for line in open(C / "seth" / "corpus.txt", encoding="utf8"):
        pmid, text = line.rstrip("\n").split("\t", 1)
        # the SETH corpus stores title and abstract in one string; the title is the first sentence
        m = re.match(r"(.+?[.?!])\s", text)
        title = m.group(1) if m else text[:150]
        docs.append({"doc_id": f"seth:{pmid}", "pmid": pmid, "corpus": "SETH", "split": "", "title": title, "text": text})
        ann = C / "seth" / "annotations" / f"{pmid}.ann"
        if ann.exists():
            for a in open(ann, encoding="utf8"):
                f = a.rstrip("\n").split("\t")
                if len(f) < 3 or not f[0].startswith("T"):
                    continue
                typ, *span = f[1].split()
                if typ not in ("SNP", "Gene", "RS"):
                    continue
                try:
                    s, e = int(span[0]), int(span[-1])
                except ValueError:
                    continue
                ments.append({"doc_id": f"seth:{pmid}", "start": s, "end": e, "text": f[2], "type": typ, "norm": ""})
    return docs, ments


def _mutationfinder() -> tuple[list, list]:
    docs, ments = [], []
    base = C / "mutationfinder" / "cleaned corpus"
    for split, (txt, gold) in {"devo": ("devo_set.txt", "devo_gold_std.txt"), "test": ("test_set.txt", "test_gold_std.txt")}.items():
        for line in open(base / txt, encoding="utf8", errors="replace"):
            f = line.rstrip("\n").split("\t")
            if len(f) < 2:
                continue
            pmid, title = f[0], f[1]
            text = f[2] if len(f) > 2 else ""
            docs.append({"doc_id": f"mf:{pmid}", "pmid": pmid, "corpus": "MutationFinder", "split": split, "title": title, "text": text})
        for line in open(base / gold, encoding="utf8"):
            f = line.rstrip("\n").split("\t")
            for m in f[1:]:
                if m.strip():
                    ments.append({"doc_id": f"mf:{f[0]}", "start": -1, "end": -1, "text": m.strip(), "type": "ProteinMutation",
                                  "norm": m.strip()})
    return docs, ments


def _osiris() -> list:
    docs = []
    for p in sorted((C / "osiris").glob("article*.txt")):
        lines = [l.strip() for l in open(p, encoding="utf8", errors="replace").read().split("\n") if l.strip()]
        if len(lines) < 2:
            continue
        pmid, title, text = lines[0], lines[1], " ".join(lines[2:])
        docs.append({"doc_id": f"osiris:{pmid}", "pmid": pmid, "corpus": "OSIRIS", "split": "", "title": title, "text": text})
    return docs


def _variome() -> list:
    docs = []
    for p in sorted((C / "variome" / "data").glob("*.txt")):
        raw = open(p, encoding="utf8", errors="replace").read()
        lines = [l for l in raw.split("\n") if l.strip() and "IGNORE LINE" not in l]
        if not lines:
            continue
        pmcid, serial, *rest = p.stem.split("-")
        section = " ".join(rest[:-1]).replace("-", " ")
        docs.append({"doc_id": f"variome:{p.stem}", "pmid": f"PMC{pmcid}", "corpus": "Variome", "split": "",
                     "title": f"{lines[0][:120]} ({section})", "text": " ".join(lines[1:]) if len(lines) > 1 else lines[0]})
    return docs


def build() -> dict:
    d1, m1 = _pubtator(C / "tmvar" / "train.PubTator.txt", "train")
    d2, m2 = _pubtator(C / "tmvar" / "test.PubTator.txt", "test")
    d3, m3 = _seth()
    d4, m4 = _mutationfinder()
    docs = pd.DataFrame(d1 + d2 + d3 + d4 + _osiris() + _variome())
    docs = docs.drop_duplicates("doc_id")
    ments = pd.DataFrame(m1 + m2 + m3 + m4)
    docs.to_parquet(KB / "literature_docs.parquet", index=False)
    ments.to_parquet(KB / "literature_mentions.parquet", index=False)
    stats = {"documents": len(docs), "by_corpus": docs["corpus"].value_counts().to_dict(),
             "unique_pmids": int(docs["pmid"].nunique()), "mentions": len(ments),
             "mention_types": ments["type"].value_counts().to_dict(),
             "words": int(docs["text"].str.split().map(len).sum() + docs["title"].str.split().map(len).sum())}
    print(stats)
    return stats


if __name__ == "__main__":
    build()
