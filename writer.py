"""Answer writers. Every sentence must cite one numbered evidence item.

ExtractiveWriter: selects and orders verbatim sentences from the evidence by
query overlap, entity matches and evidence rank, with a redundancy filter.
Needs no model and is what the evaluation measures.

OllamaWriter: a local language model through Ollama (any chat model;
llama3.1:8b by default) prompted to answer only from the numbered evidence and
to end every sentence with its citation. Used when ALLELION_LLM=ollama and a
server answers on OLLAMA_URL; its sentences go through the same verifier.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

from allelion.copilot.guard import neutralise
from allelion.retrieval.text import sentences, tokenize

CITE = re.compile(r"\[(\d+)\]\s*$")


def answer_units(text: str) -> list[str]:
    """Sentences, with fragments shorter than six words joined to the sentence that follows them."""
    out, carry = [], ""
    for s in sentences(text):
        cur = f"{carry} {s}".strip() if carry else s
        if len(cur.split()) < 6:
            carry = cur
            continue
        out.append(cur)
        carry = ""
    if carry:
        if out:
            out[-1] = f"{out[-1]} {carry}"
        else:
            out.append(carry)
    return out


GENERIC = set(tokenize("clinical feature features phenotype phenotypes syndrome disease disorder gene genes variant variants mutation "
                       "mutations report reported study studies patient patients describe described find found evidence apply applies "
                       "applied criterion criteria strength mean meaning define definition annotated associated association type "
                       "inheritance pattern recorded which what how does many"))


class ExtractiveWriter:
    name = "extractive"

    def write(self, question: str, qinfo: dict, evidence: list[dict], max_sentences: int = 4) -> list[dict]:
        qt = set(tokenize(question))
        focus = qt - GENERIC
        ents = [e.lower() for e in qinfo.get("genes", []) + qinfo.get("codes", [])]
        cands = []
        for rank, ev in enumerate(evidence):
            units = answer_units(ev["text"])
            for j, s in enumerate(units):
                if len(s.split()) < 5:
                    continue
                st = set(tokenize(s)) | set(tokenize(ev.get("title", ""))) if j == 0 else set(tokenize(s))
                ov = len(qt & st) / max(len(qt), 1)
                if focus and ev.get("kind") != "engine" and not (focus & (st | set(tokenize(ev.get("title", ""))))):
                    continue
                ent = sum(1 for e in ents if e in s.lower()) / max(len(ents), 1) if ents else 0.0
                lead = 0.15 if ev.get("kind") == "engine" and rank == 0 and j == 0 else 0.0
                score = 0.5 * ov + 0.2 * ent + 0.3 / (1 + rank) + lead
                cands.append((score, rank, j, s, ev["n"], set(tokenize(s))))
        if not cands:
            return []
        cands.sort(key=lambda x: -x[0])
        best = cands[0][0]
        chosen, per_item = [], {}
        for score, rank, j, s, n, st in cands:
            if score < 0.5 * best or per_item.get(n, 0) >= 2:
                continue
            if any(len(st & c[5]) / max(len(st | c[5]), 1) > 0.6 for c in chosen):
                continue
            chosen.append((score, rank, j, s, n, st))
            per_item[n] = per_item.get(n, 0) + 1
            if len(chosen) >= max_sentences:
                break
        chosen.sort(key=lambda x: (x[1], x[2]))
        return [{"text": s, "cite": n} for _, _, _, s, n, _ in chosen]


class OllamaWriter:
    name = "ollama"

    def __init__(self, model: str | None = None, url: str | None = None):
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
        self.url = (url or os.environ.get("OLLAMA_URL", "http://localhost:11434")).rstrip("/")

    def available(self) -> bool:
        try:
            urllib.request.urlopen(f"{self.url}/api/tags", timeout=2)
            return True
        except Exception:
            return False

    def write(self, question: str, qinfo: dict, evidence: list[dict], max_sentences: int = 5) -> list[dict]:
        ev = "\n".join(f"[{e['n']}] {neutralise(e['title'])[0]}: {neutralise(e['text'])[0]}" for e in evidence)
        system = ("You answer questions about genetic variant interpretation using only the numbered evidence. Write at most "
                  f"{max_sentences} short sentences. End every sentence with exactly one citation like [2]. Do not add facts, numbers, "
                  "genes or recommendations that are not in the cited item. Never give individual medical advice. If the evidence does "
                  "not answer the question, say so in one sentence citing the closest item.")
        body = json.dumps({"model": self.model, "stream": False, "options": {"temperature": 0},
                           "messages": [{"role": "system", "content": system}, {"role": "user", "content": f"Evidence:\n{ev}\n\nQuestion: {question}"}]}).encode()
        req = urllib.request.Request(f"{self.url}/api/chat", data=body, headers={"Content-Type": "application/json"})
        out = json.load(urllib.request.urlopen(req, timeout=120))["message"]["content"]
        res = []
        for s in sentences(out):
            m = CITE.search(s)
            res.append({"text": CITE.sub("", s).strip(), "cite": int(m.group(1)) if m else None})
        return res[:max_sentences]


def get_writer():
    if os.environ.get("ALLELION_LLM", "").lower() == "ollama":
        w = OllamaWriter()
        if w.available():
            return w
    return ExtractiveWriter()
