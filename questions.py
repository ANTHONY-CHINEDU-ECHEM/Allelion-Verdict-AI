"""Copilot evaluation questions, written before the retriever existed.

Each question names its relevant evidence by source identifier, not by
passage number (passages did not exist when the questions were written):
    guidance:Gxx       a guidance card
    gene:SYMBOL        a gene card
    disease:ID         a disease card (OMIM or ORPHA)
    hpo:HP:nnnnnnn     a phenotype term card
    lit:<doc_id>       any passage of a literature document
Grade 2 relevance: a passage from a target source. Grade 1: any passage that
contains every phrase in `required` (case insensitive).

60 development questions guided the work. 30 blind questions were written at
the same time, hashed (data/eval/questions_blind_hash.txt) and scored once
after every component was frozen.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from allelion.config import EVAL

DEV = [
    # criteria and guidance
    ("When should PVS1 be downgraded for a nonsense variant near the end of a gene?", ["guidance:G02", "guidance:G03", "guidance:G62"], ["decay"]),
    ("What is the 50 nucleotide rule for nonsense mediated decay?", ["guidance:G03"], ["50"]),
    ("At what strength should PM2 be applied after the 2020 SVI recommendation?", ["guidance:G11"], ["PM2", "supporting"]),
    ("Why were PP5 and BP6 retired?", ["guidance:G19"], ["PP5"]),
    ("How many points does a strong criterion carry in the Bayesian point system?", ["guidance:G27"], ["points"]),
    ("What point total is needed for a likely pathogenic classification?", ["guidance:G27", "guidance:G69"], ["six"]),
    ("What allele frequency gives stand alone benign evidence under BA1?", ["guidance:G20"], ["five percent"]),
    ("Which variants are exceptions to BA1?", ["guidance:G20"], ["HFE"]),
    ("How is the maximum credible allele frequency calculated for BS1?", ["guidance:G21", "guidance:G64"], ["maximum credible"]),
    ("When can PS1 be used for splicing variants?", ["guidance:G06", "guidance:G46"], ["PS1", "splic"]),
    ("How many informative meioses are needed for moderate segregation evidence?", ["guidance:G15"], ["meioses"]),
    ("What likelihood ratio does a predictor need for strong PP3 evidence?", ["guidance:G17"], ["33"]),
    ("Can PM1 and PP3 be combined at full strength?", ["guidance:G10", "guidance:G72"], ["PM1", "PP3"]),
    ("How are de novo observations scored with the SVI points scale?", ["guidance:G07"], ["de novo", "points"]),
    ("How does the PM3 points system count phase unknown observations?", ["guidance:G12"], ["PM3"]),
    ("What does BP7 require for intronic variants?", ["guidance:G26", "guidance:G46"], ["BP7"]),
    ("Why is BS2 unreliable for late onset disorders?", ["guidance:G22", "guidance:G51"], ["BS2"]),
    ("How should initiation codon variants be weighted under PVS1?", ["guidance:G05"], ["initiation codon"]),
    ("What makes an in frame deletion qualify for BP3 instead of PM4?", ["guidance:G24", "guidance:G13", "guidance:G63"], ["repetitive"]),
    ("What is the posterior probability threshold for likely pathogenic?", ["guidance:G28"], ["posterior"]),
    ("What does the ClinVar review status of three stars mean?", ["guidance:G33"], ["expert panel"]),
    ("How should functional assay strength be decided for PS3?", ["guidance:G08"], ["OddsPath"]),
    ("Why should a single predictor be chosen in advance?", ["guidance:G66", "guidance:G17"], ["predictor"]),
    ("Are copy number variants covered by the sequence variant criteria?", ["guidance:G53"], ["copy number"]),
    ("How does the ENIGMA consortium classify BRCA1 variants?", ["guidance:G37"], ["ENIGMA"]),
    # genes
    ("What diseases are associated with CFTR and what is the inheritance?", ["gene:CFTR", "guidance:G41"], ["CFTR"]),
    ("Is loss of function an established mechanism for MYH7?", ["gene:MYH7", "guidance:G40"], ["MYH7"]),
    ("What is the gene constraint of BRCA2?", ["gene:BRCA2"], ["BRCA2", "pLI"]),
    ("How many pathogenic truncating variants does MLH1 have in the evidence pool?", ["gene:MLH1"], ["MLH1", "truncating"]),
    ("Is PP2 applicable to missense variants in PTPN11?", ["gene:PTPN11", "guidance:G16", "guidance:G39"], ["PTPN11"]),
    ("Which disorders are linked to the FBN1 gene?", ["gene:FBN1"], ["FBN1"]),
    ("What inheritance pattern is recorded for ATM?", ["gene:ATM"], ["ATM"]),
    ("Does TP53 have curated haploinsufficiency?", ["gene:TP53"], ["TP53"]),
    ("How many exons are in the anchored exon model for SCN5A?", ["gene:SCN5A"], ["SCN5A", "exon"]),
    ("What is the missense spectrum for KCNQ1 in the evidence pool?", ["gene:KCNQ1"], ["KCNQ1", "missense"]),
    ("Which diseases are associated with LDLR?", ["gene:LDLR"], ["LDLR"]),
    ("What is the constraint of the DMD gene?", ["gene:DMD"], ["DMD"]),
    ("Is BP1 applicable to missense variants in APC?", ["gene:APC", "guidance:G16"], ["APC"]),
    ("What conditions are associated with SCN1A?", ["gene:SCN1A"], ["SCN1A"]),
    ("Which genes cause Noonan syndrome according to the RASopathy panel?", ["guidance:G39", "disease:OMIM:163950"], ["Noonan"]),
    # diseases and phenotypes
    ("What are the clinical features of Marfan syndrome?", ["disease:OMIM:154700"], ["Marfan"]),
    ("Which phenotypes characterise Wolman disease?", ["disease:OMIM:620151", "disease:ORPHA:75233"], ["Wolman"]),
    ("What is the inheritance of hemochromatosis type 1?", ["disease:OMIM:235200", "guidance:G20"], ["hemochromatosis"]),
    ("Which features are annotated for Duchenne muscular dystrophy?", ["disease:OMIM:310200"], ["Duchenne"]),
    ("What does the phenotype term arachnodactyly mean?", ["hpo:HP:0001166"], ["arachnodactyly"]),
    ("Define ectopia lentis.", ["hpo:HP:0001083"], ["lens"]),
    ("What are the features of Holt-Oram syndrome?", ["disease:OMIM:142900", "disease:ORPHA:392"], ["Holt"]),
    ("Which phenotypes are listed for Brugada syndrome 1?", ["disease:OMIM:601144"], ["Brugada"]),
    ("What is macrocephaly?", ["hpo:HP:0000256"], ["macrocephaly"]),
    ("What inheritance is annotated for Rett syndrome?", ["disease:OMIM:312750"], ["Rett"]),
    # literature
    ("Which report describes the F11 Asp506Gly mutation in a Korean woman?", ["lit:tmvar:22016685"], ["Asp506Gly"]),
    ("Is there a study of SLC6A8 deficiency in X linked mental retardation?", ["lit:seth:15154114"], ["SLC6A8"]),
    ("What was found about a cryptic splice mutation in PKP2?", ["lit:seth:17041889"], ["PKP2"]),
    ("Which paper reported LIPA Tyr22X in Wolman disease?", ["lit:seth:8956047"], ["LIPA"]),
    ("What did the study of CAPN3 transcripts in LGMD2A patients show?", ["lit:tmvar:17157502"], ["CAPN3"]),
    ("Which study examined manifesting carriers of DMD mutations?", ["lit:tmvar:20630757"], ["manifesting carriers"]),
    ("Does a RAD51 polymorphism modify breast cancer risk in BRCA2 carriers?", ["lit:osiris:15138485"], ["RAD51"]),
    ("What variants of the SCN5A gene were studied in Han Chinese?", ["lit:osiris:15161528"], ["SCN5A"]),
    ("Which report described a new mutation in the ARSA gene with adult onset metachromatic leukodystrophy?", ["lit:tmvar:15710861"], ["ARSA"]),
    ("What did the study of the WTX gene in Wilms tumours find?", ["lit:tmvar:18391980"], ["WTX"]),
]

BLIND = [
    ("How should a canonical splice variant be graded if skipping the exon keeps the reading frame?", ["guidance:G04"], ["frame"]),
    ("What prior probability does the ClinGen calibration of in silico tools use?", ["guidance:G17"], ["0.0441"]),
    ("Which criteria are deprecated by ClinGen SVI?", ["guidance:G19"], ["BP6"]),
    ("What is the filtering allele frequency?", ["guidance:G30", "guidance:G21"], ["filtering"]),
    ("How many points make a variant benign in the point system?", ["guidance:G27"], ["minus seven"]),
    ("When does a premature stop escape nonsense mediated decay?", ["guidance:G03"], ["last exon"]),
    ("What evidence does PP4 require?", ["guidance:G18"], ["PP4"]),
    ("How do the 2015 combining rules handle conflicting pathogenic and benign evidence?", ["guidance:G29", "guidance:G75"], ["uncertain"]),
    ("Why should gene disease validity be checked before classifying a variant?", ["guidance:G31"], ["validity"]),
    ("What is type two circularity in predictor evaluation?", ["guidance:G45"], ["circularity"]),
    ("How does the InSiGHT group classify mismatch repair variants?", ["guidance:G38"], ["InSiGHT"]),
    ("Can the germline criteria be used for tumour only findings?", ["guidance:G54"], ["somatic"]),
    ("What diseases are associated with PTEN?", ["gene:PTEN"], ["PTEN"]),
    ("Is loss of function the mechanism for MSH2?", ["gene:MSH2"], ["MSH2"]),
    ("What is the pLI of the NF1 gene?", ["gene:NF1"], ["NF1"]),
    ("Which inheritance is recorded for GJB2?", ["gene:GJB2"], ["GJB2"]),
    ("How many benign missense variants does RYR1 have in the evidence pool?", ["gene:RYR1"], ["RYR1"]),
    ("What conditions are linked to COL1A1?", ["gene:COL1A1"], ["COL1A1"]),
    ("Is KRAS a gene where PVS1 applies?", ["gene:KRAS", "guidance:G39", "guidance:G42"], ["KRAS"]),
    ("Which phenotypes describe Tay-Sachs disease?", ["disease:OMIM:272800", "disease:ORPHA:845"], ["Tay"]),
    ("What are the features of neurofibromatosis type 1?", ["disease:OMIM:162200"], ["neurofibromatosis"]),
    ("What inheritance is annotated for Fabry disease?", ["disease:OMIM:301500", "disease:ORPHA:324"], ["Fabry"]),
    ("What does global developmental delay mean as a phenotype term?", ["hpo:HP:0001263"], ["developmental"]),
    ("Which phenotypes are annotated for Wilson disease?", ["disease:OMIM:277900", "disease:ORPHA:905"], ["Wilson"]),
    ("What is an atrial septal defect?", ["hpo:HP:0001631"], ["septal"]),
    ("Which study reported haploinsufficiency for FKBP6 in Williams-Beuren syndrome?", ["lit:tmvar:15770126"], ["FKBP6"]),
    ("Were two novel TSH beta subunit mutations described in congenital central hypothyroidism?", ["lit:tmvar:20534762"], ["TSH"]),
    ("Is genetic variation near HNF4A associated with type 2 diabetes?", ["lit:osiris:15047633"], ["hepatocyte nuclear factor"]),
    ("Which study looked at APOA5 variants and coronary heart disease progression?", ["lit:osiris:14729863"], ["APOA5"]),
    ("Was SQSTM1 evaluated in Belgian patients with Paget disease?", ["lit:osiris:15164150"], ["SQSTM1"]),
]


def _rows(items, prefix):
    return [{"id": f"{prefix}{i + 1:02d}", "question": q, "targets": t, "required": r} for i, (q, t, r) in enumerate(items)]


def write() -> dict:
    dev, blind = _rows(DEV, "D"), _rows(BLIND, "B")
    (EVAL / "questions_dev.json").write_text(json.dumps(dev, indent=1))
    blob = json.dumps(blind, sort_keys=True).encode()
    (EVAL / "questions_blind.json").write_text(json.dumps(blind, indent=1))
    digest = hashlib.sha256(blob).hexdigest()
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (EVAL / "questions_blind_hash.txt").write_text(f"{digest}  questions_blind  written {stamp}  n={len(blind)}\n")
    print(f"dev {len(dev)}, blind {len(blind)}, blind sha256 {digest[:16]}")
    return {"dev": len(dev), "blind": len(blind), "sha256": digest}


def verify_blind() -> bool:
    blind = json.loads((EVAL / "questions_blind.json").read_text())
    rec = (EVAL / "questions_blind_hash.txt").read_text().split()[0]
    return hashlib.sha256(json.dumps(blind, sort_keys=True).encode()).hexdigest() == rec


if __name__ == "__main__":
    write()
