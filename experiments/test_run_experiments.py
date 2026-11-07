"""Tests for experiments/run_experiments.py.  From the repo root:

    python -m pytest experiments -q

They build one small synthetic dataset in a temp folder, so they need the
JASPAR files in data/jaspar/ but not STREME.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data"))   # data/generate.py uses flat imports

import generate  # noqa: E402
from evaluation.evaluate import RESULT_COLUMNS, append_results  # noqa: E402
from experiments import run_experiments as rx  # noqa: E402


@pytest.fixture
def dataset(tmp_path):
    return generate.make_dataset("MA0001.3", 0.8, "at_rich", 0, outdir=str(tmp_path / "syn"))


@pytest.fixture
def small_budget(monkeypatch):
    monkeypatch.setitem(rx.BUDGET, "randomized", {"n_restarts": 3})
    monkeypatch.setitem(rx.BUDGET, "gibbs", {"n_restarts": 2, "n_iterations": 50})


def run(dataset, tmp_path):
    return rx.run_dataset((dataset, str(tmp_path / "no_streme")))


def test_one_dataset_gives_every_method_and_setting(dataset, small_budget, tmp_path):
    rows, errors = run(dataset, tmp_path)
    algs = [r["algorithm"] for r in rows]
    assert len(rows) == 14                       # 3 algs x 2 bg x 2 pc + 2 baselines
    assert {a: algs.count(a) for a in set(algs)} == {
        "greedy": 4, "randomized": 4, "gibbs": 4, "oracle_scan": 1, "random": 1}
    combos = {(r["algorithm"], r["bg_model"], r["pseudocount"]) for r in rows
              if r["algorithm"] in rx.ALGORITHMS}
    assert len(combos) == 12                     # no setting duplicated or missing
    assert all(set(r) == set(RESULT_COLUMNS) for r in rows)


def test_missing_streme_output_is_reported_not_fatal(dataset, small_budget, tmp_path):
    rows, errors = run(dataset, tmp_path)
    assert len(errors) == 1 and "no output yet" in errors[0]
    assert "streme" not in [r["algorithm"] for r in rows]


def test_same_seed_for_every_setting_of_a_dataset(dataset, small_budget, tmp_path):
    rows, _ = run(dataset, tmp_path)
    assert len({r["seed"] for r in rows if r["algorithm"] in rx.ALGORITHMS}) == 1


def test_run_is_reproducible(dataset, small_budget, tmp_path):
    a, _ = run(dataset, tmp_path)
    b, _ = run(dataset, tmp_path)
    drop = lambda rows: [{k: v for k, v in r.items() if k != "runtime_s"} for r in rows]
    assert drop(a) == drop(b)


def test_metrics_are_in_range(dataset, small_budget, tmp_path):
    rows, _ = run(dataset, tmp_path)
    for r in rows:
        assert 0 <= r["exact_rate"] <= r["overlap_rate"] <= 1
        assert 0 <= r["mean_overlap"] <= 1
        assert 0 <= r["hamming"] <= r["true_width"]


def test_finished_datasets_supports_resume(dataset, small_budget, tmp_path):
    out = str(tmp_path / "results.csv")
    assert rx.finished_datasets(out) == set()            # file does not exist
    open(out, "w").close()
    assert rx.finished_datasets(out) == set()            # empty file
    rows, _ = run(dataset, tmp_path)
    append_results(out, rows)
    assert rx.finished_datasets(out) == {os.path.basename(dataset)}