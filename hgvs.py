"""HGVS parsing for coding (c.) and protein (p.) descriptions.

The parser is deliberately small and strict: it understands the forms that
appear in clinical reports (substitutions, deletions, duplications,
insertions, deletion insertions, intronic offsets, UTR positions and the
common protein forms) and returns None rather than guessing when a string is
outside that grammar. The TypeScript engine (web/src/engine/hgvs.ts) implements
the same grammar; tests/test_parity.py checks both on a shared fixture.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

AA3 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E", "Gly": "G",
    "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S",
    "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V", "Ter": "*", "Sec": "U", "Xaa": "X",
}
AA1 = {v: k for k, v in AA3.items()}

# position: optional UTR prefix (- or *), integer, optional intronic offset (+n or -n)
_POS = r"([-*]?)(\d+)(?:([+-])(\d+))?"
_C_RE = re.compile(
    rf"^(?:(?P<tx>[A-Z]{{2}}_\d+(?:\.\d+)?)(?:\([^)]*\))?:)?c\.(?P<p1>{_POS})(?:_(?P<p2>{_POS}))?"
    r"(?P<edit>[ACGTN]+>[ACGTN]+|del(?:[ACGTN]*|\d+)ins(?:[ACGTN]+|\d+)|del(?:[ACGTN]*|\d+)|dup(?:[ACGTN]*|\d+)|ins(?:[ACGTN]+|\d+)|[ACGTN]?=|inv(?:[ACGTN]*|\d+))?$"
)
_P_RE = re.compile(
    r"^(?:(?P<tx>[A-Z]{2}_\d+(?:\.\d+)?):)?p\.\(?(?P<body>[^)]*)\)?$"
)


@dataclass
class CPos:
    region: str      # cds, utr5, utr3
    base: int        # c. coordinate (negative for 5' UTR, positive past stop for 3' UTR as *n)
    offset: int      # intronic offset, 0 when exonic

    @property
    def intronic(self) -> bool:
        return self.offset != 0


@dataclass
class CVariant:
    transcript: str | None
    start: CPos
    end: CPos
    kind: str        # sub, del, dup, ins, delins, identity, inv
    ref: str
    alt: str
    del_len: int | None
    ins_len: int | None

    @property
    def length_change(self) -> int | None:
        if self.kind == "sub":
            return 0
        if self.kind in ("del",):
            return -self.del_len if self.del_len is not None else None
        if self.kind in ("dup", "ins"):
            return self.ins_len
        if self.kind == "delins" and self.del_len is not None and self.ins_len is not None:
            return self.ins_len - self.del_len
        return None

    @property
    def frameshift(self) -> bool | None:
        lc = self.length_change
        if lc is None:
            return None
        return lc % 3 != 0

    @property
    def splice_distance(self) -> int:
        """Distance in bases from the nearest exon boundary for intronic positions (0 when exonic)."""
        offs = [abs(p.offset) for p in (self.start, self.end) if p.offset]
        return min(offs) if offs else 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(length_change=self.length_change, frameshift=self.frameshift, splice_distance=self.splice_distance)
        return d


def _parse_pos(sign: str, num: str, osign: str | None, onum: str | None) -> CPos:
    base = int(num)
    region = "cds"
    if sign == "-":
        region, base = "utr5", -base
    elif sign == "*":
        region = "utr3"
    offset = 0
    if osign:
        offset = int(onum) if osign == "+" else -int(onum)
    return CPos(region, base, offset)


def _span(a: CPos, b: CPos) -> int | None:
    """Number of bases between two exonic coding positions, inclusive."""
    if a.offset or b.offset or a.region != b.region:
        return None
    return abs(b.base - a.base) + 1


def parse_c(text: str) -> CVariant | None:
    if not text or not isinstance(text, str):
        return None
    m = _C_RE.match(text.strip())
    if not m:
        return None
    g = m.groups()
    # groups: tx, p1, s1, n1, os1, on1, p2, s2, n2, os2, on2, edit
    start = _parse_pos(g[2], g[3], g[4], g[5])
    end = _parse_pos(g[7], g[8], g[9], g[10]) if g[6] else start
    edit = m.group("edit") or ""
    ref = alt = ""
    del_len = ins_len = None
    span = _span(start, end)
    def _n(seq: str, default):
        return int(seq) if seq.isdigit() else (len(seq) if seq else default)

    if ">" in edit:
        kind = "sub"
        ref, alt = edit.split(">")
        del_len, ins_len = len(ref), len(alt)
    elif edit.startswith("del") and "ins" in edit:
        kind = "delins"
        ref, alt = edit[3:].split("ins")
        del_len, ins_len = _n(ref, span), _n(alt, None)
        ref, alt = ("" if ref.isdigit() else ref), ("" if alt.isdigit() else alt)
    elif edit.startswith("del"):
        kind = "del"
        ref = edit[3:]
        del_len = _n(ref, span)
        ref = "" if ref.isdigit() else ref
    elif edit.startswith("dup"):
        kind = "dup"
        ref = edit[3:]
        ins_len = _n(ref, span)
        ref = "" if ref.isdigit() else ref
    elif edit.startswith("ins"):
        kind = "ins"
        alt = edit[3:]
        ins_len = _n(alt, None)
        alt = "" if alt.isdigit() else alt
    elif edit.endswith("="):
        kind = "identity"
    elif edit.startswith("inv"):
        kind = "inv"
    else:
        return None
    return CVariant(m.group("tx"), start, end, kind, ref, alt, del_len, ins_len)


@dataclass
class PVariant:
    transcript: str | None
    kind: str          # missense, nonsense, synonymous, frameshift, inframe_del, inframe_ins, inframe_dup, delins, start_loss, stop_loss, unknown
    ref_aa: str        # one letter
    pos: int | None
    alt_aa: str        # one letter, * for stop, = for synonymous
    end_pos: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def short(self) -> str:
        if self.pos is None:
            return ""
        return f"{self.ref_aa}{self.pos}{self.alt_aa}"


_P_SIMPLE = re.compile(r"^([A-Z][a-z]{2}|\*|[A-Z])(\d+)([A-Z][a-z]{2}|\*|[A-Z]|=|\?)$")
_P_FS = re.compile(r"^([A-Z][a-z]{2})(\d+)(?:[A-Z][a-z]{2})?fs(?:Ter|\*)?(\d+|\?)?$")
_P_RANGE = re.compile(r"^([A-Z][a-z]{2})(\d+)(?:_([A-Z][a-z]{2})(\d+))?(del|dup|ins|delins)([A-Za-z*]*)$")
_P_EXT = re.compile(r"^(Ter|\*)(\d+)([A-Z][a-z]{2})?ext")


def _aa(token: str) -> str:
    if token in AA3:
        return AA3[token]
    if token in ("*", "="):
        return token
    return token if len(token) == 1 else "?"


def parse_p(text: str) -> PVariant | None:
    if not text or not isinstance(text, str):
        return None
    m = _P_RE.match(text.strip())
    if not m:
        return None
    tx, body = m.group("tx"), m.group("body").rstrip("+")
    body = re.sub(r"^([A-Z][a-z]{2}\d+)[A-Z][a-z]{2}=$", r"\1=", body)
    if re.match(r"^[A-Z][a-z]{2}\d+_[A-Z][a-z]{2}\d+=$", body):
        mm = re.match(r"^([A-Z][a-z]{2})(\d+)_", body)
        return PVariant(tx, "synonymous", _aa(mm.group(1)), int(mm.group(2)), "=")
    if body in ("?", "0", "="):
        return PVariant(tx, "unknown", "", None, body)
    if (fm := _P_FS.match(body)):
        return PVariant(tx, "frameshift", _aa(fm.group(1)), int(fm.group(2)), "fs")
    if (em := _P_EXT.match(body)):
        return PVariant(tx, "stop_loss", "*", int(em.group(2)), _aa(em.group(3) or "Xaa"))
    if (sm := _P_SIMPLE.match(body)):
        ref, pos, alt = _aa(sm.group(1)), int(sm.group(2)), sm.group(3)
        alt1 = _aa(alt) if alt not in ("=", "?") else alt
        if alt1 == "=" or alt1 == ref:
            kind = "synonymous"
            alt1 = "="
        elif ref == "M" and pos == 1:
            kind = "start_loss"
        elif alt1 == "*":
            kind = "nonsense"
        elif ref == "*":
            kind = "stop_loss"
        elif alt1 == "?":
            kind = "unknown"
        else:
            kind = "missense"
        return PVariant(tx, kind, ref, pos, alt1)
    if (rm := _P_RANGE.match(body)):
        ref = _aa(rm.group(1))
        pos = int(rm.group(2))
        end = int(rm.group(4)) if rm.group(4) else pos
        op = rm.group(5)
        kind = {"del": "inframe_del", "dup": "inframe_dup", "ins": "inframe_ins", "delins": "delins"}[op]
        tail = rm.group(6) or ""
        if "Ter" in tail or "*" in tail:
            kind = "nonsense"
        return PVariant(tx, kind, ref, pos, op, end)
    return None


CONSEQUENCE_MAP = {
    "missense variant": "missense",
    "nonsense": "nonsense",
    "frameshift variant": "frameshift",
    "splice donor variant": "splice_donor",
    "splice acceptor variant": "splice_acceptor",
    "inframe_variant": "inframe_indel",
    "synonymous variant": "synonymous",
    "intron variant": "intron",
    "5 prime UTR variant": "utr5",
    "3 prime UTR variant": "utr3",
    "2KB upstream variant": "upstream",
    "500B downstream variant": "downstream",
    "non-coding transcript variant": "noncoding",
    "stop lost": "stop_loss",
}
SEVERITY = ["frameshift", "nonsense", "splice_donor", "splice_acceptor", "start_loss", "stop_loss", "inframe_indel",
            "missense", "synonymous", "intron", "utr5", "utr3", "upstream", "downstream", "noncoding", "other"]
LOF_CONSEQUENCES = {"frameshift", "nonsense", "splice_donor", "splice_acceptor", "start_loss"}


def primary_consequence(molecular_consequence: str | float, hgvs_c: str | float) -> tuple[str, str | None]:
    """Choose the consequence on the transcript named in hgvs_c, falling back to the most severe coding one."""
    if not isinstance(molecular_consequence, str) or not molecular_consequence or molecular_consequence == "-":
        return "other", None
    tx = None
    if isinstance(hgvs_c, str) and ":" in hgvs_c:
        tx = hgvs_c.split(":", 1)[0]
    entries = []
    for part in molecular_consequence.split(";"):
        bits = part.rsplit(":", 1)
        if len(bits) != 2:
            continue
        left, term = bits
        conseq = CONSEQUENCE_MAP.get(term.strip(), "other")
        t = left.split(":", 1)[0]
        entries.append((t, conseq, left))
    if not entries:
        return "other", None
    if tx:
        for t, c, left in entries:
            if t == tx:
                return c, left
    coding = [e for e in entries if e[0].startswith("NM_")] or entries
    coding.sort(key=lambda e: SEVERITY.index(e[1]))
    return coding[0][1], coding[0][2]


def refine_consequence(conseq: str, c: CVariant | None, p: PVariant | None) -> str:
    """Repair consequence labels using the parsed descriptions (start loss, in frame types, splice region)."""
    if p is not None:
        if p.kind == "start_loss":
            return "start_loss"
        if p.kind == "stop_loss":
            return "stop_loss"
        if conseq in ("other", "noncoding") and p.kind in ("missense", "nonsense", "frameshift", "synonymous"):
            return p.kind
    if c is not None and conseq == "intron" and c.splice_distance in (1, 2):
        return "splice_donor" if (c.start.offset > 0) else "splice_acceptor"
    return conseq
