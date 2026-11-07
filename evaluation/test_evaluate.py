import csv
import numpy as np
import pytest

from evaluation.metrics import profile_from_sites, revcomp_str
from evaluation.evaluate import (RESULT_COLUMNS, evaluate, append_results,
                                 load_dataset)

# A tiny hand-made "dataset" in the same shape as Laiba's meta.json
SITES = ["GATTACAGGC", "GATTACAGGC", "GATTTCAGGC", "GATTACAGGA"]
TRUE_POS = [3, 10, 0, 7]


def _meta():
    T = profile_from_sites(SITES, pseudocount=0.1)
    return dict(name="toy_rep0", motif_id="TOY", strength_fraction=1.0,
                true_ic=12.3, background="uniform", replicate=0, width=10,
                positions=TRUE_POS, true_matrix=T.tolist(),
                jaspar_matrix=T.tolist())


def _found(sites, positions):
    return dict(profile=profile_from_sites(sites, pseudocount=0.1),
                positions=positions)


def test_row_has_exactly_the_agreed_columns():
    row = evaluate(_found(SITES, TRUE_POS), _meta(), algorithm="gibbs")
    assert list(row) == RESULT_COLUMNS


def test_perfect_answer_scores_perfectly():
    row = evaluate(_found(SITES, TRUE_POS), _meta(), algorithm="oracle",
                   bg_model="uniform", pseudocount=True, seed=3)
    assert row["hamming"] == 0
    assert row["pearson_true"] == pytest.approx(1.0)
    assert row["exact_rate"] == 1.0 and row["overlap_rate"] == 1.0
    assert row["algorithm"] == "oracle" and row["seed"] == 3
    assert row["found_consensus"] == "GATTACAGGC"


def test_reverse_strand_answer_still_matches():
    rc_sites = [revcomp_str(s) for s in SITES]
    row = evaluate(_found(rc_sites, TRUE_POS), _meta(), algorithm="streme")
    assert row["hamming"] == 0
    assert row["align_strand"] == "-"
    assert row["pearson_true"] == pytest.approx(1.0)


def test_missing_sites_lower_recovery():
    row = evaluate(_found(SITES, [3, None, None, 7]), _meta(), algorithm="x")
    assert row["exact_rate"] == 0.5


def test_floats_are_rounded():
    row = evaluate(_found(SITES, [5, 12, 2, 9]), _meta(), algorithm="x")
    assert row["mean_overlap"] == 0.8          # not 0.7999999999999998


def test_append_results_writes_header_once(tmp_path):
    path = tmp_path / "out" / "results.csv"
    row = evaluate(_found(SITES, TRUE_POS), _meta(), algorithm="gibbs")
    append_results(path, [row])
    append_results(path, [row, row])
    with open(path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert list(rows[0]) == RESULT_COLUMNS


def test_load_dataset(tmp_path):
    (tmp_path / "seqs.fasta").write_text(">seq0\nACGT\n>seq1\nTTTT\n")
    (tmp_path / "meta.json").write_text('{"name": "x"}')
    seqs, meta = load_dataset(tmp_path)
    assert seqs == ["ACGT", "TTTT"] and meta["name"] == "x"
