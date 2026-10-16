import pandas as pd
import pytest

from allelion.acmg.engine import Engine, prepare
from allelion.config import EVAL, KB


@pytest.fixture(scope="module")
def eng():
    return Engine()


def test_no_variant_uses_its_own_label(eng):
    """Evaluating a pool variant must give the same result whatever its own ClinVar class is."""
    v = pd.read_parquet(KB / "variants.parquet")
    pool = set(pd.read_parquet(EVAL / "pool_ids.parquet")["variation_id"])
    sub = v[v["variation_id"].isin(pool) & (v["consequence"] == "missense") & v["clinvar_class"].isin(["Pathogenic", "Benign"])].head(40)
    for r in eng.score_batch(sub).to_dict("records"):
        a = eng.evaluate(r)
        flipped = dict(r, clinvar_class="Benign" if r["clinvar_class"] == "Pathogenic" else "Pathogenic", clinvar_label="flipped")
        b = eng.evaluate(flipped)
        assert a["met"] == b["met"] and a["points"] == b["points"]


def test_null_variant_in_lof_gene(eng):
    r = eng.evaluate(prepare("BRCA2", "c.9976A>T", "p.Lys3326Ter"))
    assert r["variant"]["consequence"] == "nonsense"
    # c.9976 lies in the last exon of BRCA2, so PVS1 must not be very strong
    assert not any(m == "PVS1" for m in r["met"])


def test_population_not_supplied_leaves_frequency_criteria_open(eng):
    r = eng.evaluate(prepare("TP53", "c.743G>A", "p.Arg248Gln"))
    assert {"BA1", "BS1", "PM2", "BS2"} <= set(r["not_evaluated"])


def test_ba1(eng):
    v = prepare("MLH1", "c.655A>G", "p.Ile219Val", {"ac": 40000, "an": 250000, "grpmax_af": 0.3, "faf95": 0.29, "hom": 3000})
    r = eng.evaluate(v)
    assert "BA1" in r["met"] and r["classification"] == "Benign"


def test_case_evidence_moves_the_class(eng):
    v = prepare("FBN1", "c.3037G>A", "p.Gly1013Arg")
    base = eng.evaluate(v)["points"]
    more = eng.evaluate(v, {"de_novo": [{"confirmed": True, "phenotype": "specific"}], "segregation_meioses": 5})["points"]
    assert more == base + 4 + 2


def test_mechanism_rules(eng):
    r = eng.evaluate(prepare("MYH7", "c.2770G>T", "p.Glu924Ter"))
    pvs1 = next(c for c in r["criteria"] if c["code"] == "PVS1")
    assert pvs1["status"] == "not_met" and "missense" in pvs1["rationale"]
