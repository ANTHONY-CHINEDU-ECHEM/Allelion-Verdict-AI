import pandas as pd

from allelion.config import EVAL
from allelion.eval import questions, splits


def test_blind_variant_hash():
    assert splits.verify_blind()


def test_blind_question_hash():
    assert questions.verify_blind()


def test_sets_are_disjoint_from_the_pool():
    pool = set(pd.read_parquet(EVAL / "pool_ids.parquet")["variation_id"])
    blind = set(pd.read_parquet(EVAL / "blind_variants.parquet")["variation_id"])
    dev = set(pd.read_parquet(EVAL / "dev_variants.parquet")["variation_id"])
    assert not (pool & blind) and not (pool & dev) and not (blind & dev)
