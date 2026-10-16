"""Input guard: runs before retrieval and before anything is logged (mirrored in web/src/engine/guard.ts).

1. Redacts personal identifiers: NHS numbers, other long identifiers, dates of
   birth and other dates, emails, phone numbers, UK postcodes, hospital record
   numbers, titled names and ages above 89.
2. Routes first person crisis language to crisis support instead of retrieval.
3. Flags requests for individual medical decisions; the answer carries a fixed
   notice and stays at the level of variant evidence.
4. Declines questions with no genetics or medicine content.
5. Neutralises prompt injection patterns in the question and in any passage
   placed in front of a language model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

REDACTIONS = [
    ("NHS_NUMBER", re.compile(r"(?:\bNHS(?:\s(?:no\.?|number))?[:\s#]*)?\b\d{3}[\s-]?\d{3}[\s-]?\d{4}\b", re.I)),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("PHONE", re.compile(r"(?:\+44\s?7\d{3}|\b07\d{3}|\b0\d{3,4})[\s-]?\d{3}[\s-]?\d{3,4}\b")),
    ("DATE_OF_BIRTH", re.compile(r"\b(?:DOB|D\.O\.B\.?|date of birth|born on)[:\s]*\S+(?:\s\S+){0,2}", re.I)),
    ("DATE", re.compile(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b|\b\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s\d{4}\b", re.I)),
    ("POSTCODE", re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b")),
    ("RECORD_NUMBER", re.compile(r"\b(?:MRN|hospital number|record number|patient id|case id)[:#\s]*[A-Z0-9-]{4,}\b", re.I)),
    ("NAME", re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof)\.?\s[A-Z][a-z]+(?:\s[A-Z][a-z]+)?")),
    ("AGE_OVER_89", re.compile(r"\b(?:aged?\s)?(?:9\d|1[0-1]\d)\s?(?:years?\sold|yo|year old)\b|\baged\s(?:9\d|1[0-1]\d)\b", re.I)),
]

CRISIS = re.compile(r"\b(?:i want to|i'm going to|i am going to|thinking (?:about|of)|plan to)\s(?:kill myself|end my life|hurt myself|take my own life)\b|"
                    r"\bsuicid(?:e|al)\b.*\b(?:i|me|my)\b|\b(?:i|me|my)\b.*\bsuicid(?:e|al)\b", re.I)
PERSONAL = re.compile(r"\b(?:should i|should my|should we|do i need|what should i do|am i going to|will i get|will my (?:child|son|daughter|baby)|"
                      r"my (?:result|test|report|baby|son|daughter|child|partner|wife|husband)\b.*\b(?:mean|surgery|mastectomy|treatment|terminate|pregnan))", re.I)
INJECTION = re.compile(r"(?:ignore|disregard|forget)\s(?:all\s)?(?:the\s)?(?:previous|prior|above|earlier)\s(?:instructions?|prompts?|rules?)|"
                       r"\byou are now\b|\bsystem prompt\b|\bdeveloper mode\b|\bact as\b.{0,40}\bwithout\b|</?(?:system|assistant|instructions?)>", re.I)
DOMAIN = re.compile(r"\b(?:gene|genes|variant|variants|mutation|mutations|allele|pathogenic|benign|acmg|clinvar|clingen|criteri|pvs1|ps\d|pm\d|pp\d|"
                    r"ba1|bs\d|bp\d|exon|intron|splice|missense|nonsense|frameshift|syndrome|disease|disorder|phenotype|inheritance|"
                    r"dominant|recessive|penetrance|protein|hgvs|c\.\d|p\.[a-z]|gnomad|frequency|segregation|de novo|carrier|cancer|"
                    r"cardiomyopathy|deficiency|study|report|define|term|hpo|omim|orpha|predictor|in silico|calibrat|vus|uncertain)\b", re.I)

CRISIS_MESSAGE = ("It sounds like you may be going through something very difficult. You do not have to face it alone. In the UK you can call "
                  "Samaritans free on 116 123 at any time, or contact NHS 111. If you are in immediate danger, call 999. This tool cannot "
                  "help with this, but the people at those services can.")
PERSONAL_NOTICE = ("This tool summarises published evidence about variants and genes. It cannot advise on decisions about your own or a "
                   "relative's care; a clinical genetics service or genetic counsellor can explain what a result means for you.")
OFF_TOPIC_MESSAGE = "This copilot answers questions about genetic variants, genes, inherited diseases, phenotypes and variant classification."


@dataclass
class GuardResult:
    text: str
    redactions: list[str] = field(default_factory=list)
    action: str = "answer"          # answer, crisis, off_topic
    personal: bool = False
    injection: bool = False
    message: str = ""


def redact(text: str) -> tuple[str, list[str]]:
    found = []
    # protect HGVS and identifiers that look like dates or phone numbers
    protected = {}

    def _protect(m):
        k = f"\x00{len(protected)}\x00"
        protected[k] = m.group()
        return k
    t = re.sub(r"(?:[cgpn]\.[^\s,;]+|rs\d+|HP:\d{7}|(?:OMIM|ORPHA):\d+|PMID[:\s]*\d+|N[MPRC]_\d+(?:\.\d+)?)", _protect, text)
    for label, pat in REDACTIONS:
        if pat.search(t):
            found.append(label)
            t = pat.sub(f"[{label}]", t)
    for k, v in protected.items():
        t = t.replace(k, v)
    return t, found


def neutralise(text: str) -> tuple[str, bool]:
    hit = bool(INJECTION.search(text))
    return INJECTION.sub("[removed instruction]", text), hit


def check(question: str) -> GuardResult:
    red, found = redact(question)
    clean, inj = neutralise(red)
    g = GuardResult(text=clean, redactions=found, injection=inj)
    if CRISIS.search(question):
        g.action, g.message = "crisis", CRISIS_MESSAGE
        return g
    if not DOMAIN.search(clean) and not re.search(r"\b[A-Z][A-Z0-9]{2,9}\b", clean):
        g.action, g.message = "off_topic", OFF_TOPIC_MESSAGE
        return g
    if PERSONAL.search(question):
        g.personal, g.message = True, PERSONAL_NOTICE
    return g
