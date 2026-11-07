import numpy as np
import pandas as pd
import pytest

from evaluation.evaluate import RESULT_COLUMNS
from evaluation.baselines import oracle_scan, random_motif
from evaluation.plots import (load_results, mean_ci, summary_table,
                              plot_metric, make_all, METRICS)


def _fake_results(tmp_path):
    """Small results CSV with every kind of row the plots must handle."""
    rng = np.random.default_rng(0)
    rows = []
    for strength in (0.2, 0.6, 1.0):
        for bg in ("uniform", "at_rich"):
            for rep in range(3):
                for alg in ("greedy", "gibbs"):
                    for pc in (True, False):
                        for bgm in ("uniform", "shuffled"):
                            rows.append(dict(algorithm=alg, bg_model=bgm,
                                             pseudocount=pc, data_bg=bg,
                                             strength=strength, replicate=rep))
                for ref in ("streme", "oracle_scan", "random"):
                    rows.append(dict(algorithm=ref, data_bg=bg,
                                     strength=strength, replicate=rep))
    df = pd.DataFrame(rows).reindex(columns=RESULT_COLUMNS)
    for m in METRICS:
        df[m] = rng.random(len(df))
    path = tmp_path / "results.csv"
    df.to_csv(path, index=False)
    return path


def test_mean_ci_known_numbers():
    df = pd.DataFrame(dict(g=["a"] * 4, x=[1.0, 2.0, 3.0, 4.0]))
    out = mean_ci(df, "g", "x").iloc[0]
    assert out["mean"] == 2.5 and out["n"] == 4
    assert out["ci"] == pytest.approx(1.96 * np.std([1, 2, 3, 4], ddof=1) / 2)


def test_load_results_normalises_pseudocount(tmp_path):
    df = load_results(_fake_results(tmp_path))
    assert set(df["pseudocount"].dropna()) == {"on", "off"}
    assert (df.loc[df["algorithm"] == "streme", "bg_model"] == "-").all()


def test_summary_table_one_row_per_condition(tmp_path):
    df = load_results(_fake_results(tmp_path))
    t = summary_table(df)
    # 2 algs x 2 pc x 2 bg_model x 2 data_bg x 3 strengths + 3 refs x 2 x 3
    assert len(t) == 2 * 2 * 2 * 2 * 3 + 3 * 2 * 3
    assert "overlap_rate_mean" in t and "overlap_rate_ci95" in t


def test_make_all_writes_every_figure(tmp_path):
    paths = make_all([_fake_results(tmp_path)], tmp_path / "figs")
    assert len(paths) == len(METRICS) + 1
    for p in paths:
        assert (tmp_path / "figs" / p.split("figs")[-1].lstrip("\\/")).stat().st_size > 0


def test_plot_skips_unknown_algorithm(tmp_path, capsys):
    df = load_results(_fake_results(tmp_path))
    df.loc[df["algorithm"] == "gibbs", "algorithm"] = "mystery"
    plot_metric(df, "pearson_true", tmp_path / "p.png")
    assert "mystery" in capsys.readouterr().out


def test_baselines_shapes():
    seqs = ["ACGT" * 25] * 5
    T = np.full((4, 6), 0.25)
    T[:, 0] = [1, 0, 0, 0]
    meta = dict(true_matrix=T.tolist(), width=6)
    o = oracle_scan(seqs, meta)
    r = random_motif(seqs, 6, seed=1)
    assert len(o["positions"]) == 5 and len(r["positions"]) == 5
    assert r["profile"].shape == (4, 6)
    assert all(0 <= p <= 94 for p in r["positions"])
