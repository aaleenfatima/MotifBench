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


def _with_real(tmp_path):
    """The fake results plus real-site rows at strength 1.0 with an absurd
    metric value, so leaking into the synthetic curves would be obvious."""
    df = pd.read_csv(_fake_results(tmp_path))
    real = df[df["strength"] == 1.0].copy()
    real["data_type"] = "real_sites"
    real["dataset"] = "real_" + real["dataset"].astype(str)
    real["motif_id"] = "MA0003.4"
    for m in ("pearson_true", "hamming_norm", "overlap_rate", "exact_rate"):
        real[m] = 0.999
    df["data_type"] = "synthetic"
    path = tmp_path / "with_real.csv"
    pd.concat([df, real]).to_csv(path, index=False)
    return path


def test_found_entropy_is_per_position(tmp_path):
    pd.DataFrame(dict(found_ic=[0.0, 20.0], found_width=[10, 10],
                      pseudocount=[True, False], bg_model=["a", "b"],
                      data_type=["synthetic", "synthetic"])
                 ).to_csv(tmp_path / "x.csv", index=False)
    out = load_results(tmp_path / "x.csv")
    assert list(out["found_entropy"]) == [2.0, 0.0]   # noise, fully specific


def test_real_rows_kept_out_of_strength_summary(tmp_path):
    df = load_results(_with_real(tmp_path))
    t = summary_table(df)
    syn = t[(t["data_type"] == "synthetic") & (t["strength"] == 1.0)]
    assert (syn["overlap_rate_mean"] < 0.999).all()
    assert (t["data_type"] == "real_sites").any()


def test_make_all_writes_every_figure(tmp_path):
    paths = make_all([_with_real(tmp_path)], tmp_path / "figs")
    assert len(paths) == len(METRICS) + 2          # + real_data.png + table
    assert any(p.endswith("real_data.png") for p in paths)
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
