import numpy as np
import pandas as pd
import pytest

from evaluation.stats import (paired_test, vs_reference, holm, standard_claims,
                              streme_gap, reliability)


def _df(effect=0.2, n=30, seed=0):
    """Toy results: pseudocounts add `effect` to overlap_rate on every
    dataset, shuffled bg adds nothing; STREME sits at 0.9."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        base = rng.random() * 0.5
        for alg in ("greedy", "randomized", "gibbs"):
            for pc in ("on", "off"):
                for bgm in ("uniform", "shuffled"):
                    rows.append(dict(dataset=f"d{i}", algorithm=alg, bg_model=bgm,
                                     pseudocount=pc, seed=0, data_bg="uniform",
                                     strength=[0.2, 0.6, 1.0][i % 3],
                                     overlap_rate=base + (effect if pc == "on" else 0)
                                     + rng.normal(0, 0.01)))
        rows.append(dict(dataset=f"d{i}", algorithm="streme", bg_model="-",
                         pseudocount=None, seed=0, data_bg="uniform",
                         strength=[0.2, 0.6, 1.0][i % 3], overlap_rate=0.9))
    return pd.DataFrame(rows)


def test_paired_test_detects_real_effect():
    r = paired_test(_df(), "overlap_rate", "pseudocount", "off", "on",
                    where="algorithm == 'gibbs'")
    assert r["n"] == 60                    # 30 datasets x 2 bg_models
    assert r["mean_diff"] == pytest.approx(0.2, abs=0.01)
    assert r["p"] < 1e-6 and r["effect_r"] == pytest.approx(1.0)
    assert r["wins"] == 60


def test_paired_test_no_effect_is_not_significant():
    r = paired_test(_df(), "overlap_rate", "bg_model", "uniform", "shuffled",
                    where="algorithm == 'gibbs'")
    assert r["p"] > 0.05 and abs(r["effect_r"]) < 0.5


def test_identical_values_give_p_1():
    df = _df(effect=0.0)
    df["overlap_rate"] = 0.5
    r = paired_test(df, "overlap_rate", "pseudocount", "off", "on")
    assert r["p"] == 1.0 and r["ties"] == r["n"]


def test_vs_reference_pairs_on_dataset():
    r = vs_reference(_df(), "overlap_rate", "gibbs",
                     where="pseudocount == 'on' and bg_model == 'shuffled'")
    assert r["n"] == 30 and r["mean_diff"] < 0      # STREME (0.9) is better


def test_holm_known_values():
    adj = holm([0.01, 0.04, 0.03, np.nan])
    # sorted 0.01, 0.03, 0.04 -> x3, x2, x1 -> 0.03, 0.06, 0.06 (monotone)
    assert np.allclose(adj[:3], [0.03, 0.06, 0.06])
    assert np.isnan(adj[3])


def test_standard_claims_table():
    t = standard_claims(_df())
    assert len(t) == 3 + 3 + 3              # pc, bg (1 data_bg), vs STREME
    pc = t[t["claim"].str.contains("pseudocounts")]
    bg = t[t["claim"].str.contains("shuffled bg")]
    assert pc["significant"].all() and not bg["significant"].any()


def test_streme_gap_fraction():
    g = streme_gap(_df()).set_index("algorithm")
    # pc adds 0.2 of a gap of about 0.65 -> roughly 30% closed
    assert 0.2 < g.loc["gibbs", "fraction_closed"] < 0.45


def test_reliability_threshold():
    df = pd.DataFrame(dict(algorithm="gibbs", pseudocount="on", bg_model="uniform",
                           data_bg="uniform", strength=[0.2, 0.6, 1.0],
                           overlap_rate=[0.1, 0.85, 0.95]))
    r = reliability(df, level=0.8)
    assert r["reliable_from_strength"].iloc[0] == 0.6
