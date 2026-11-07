import numpy as np
import pytest

from evaluation.metrics import (check_profile, revcomp_str, revcomp_profile,
                                consensus, profile_from_sites,
                                column_entropy, information_content,
                                hamming_aligned, pearson_aligned)


# ---- Section 1: core helpers ----------------------------------------------

def test_revcomp_str():
    assert revcomp_str("AACG") == "CGTT"
    assert revcomp_str("acgt") == "ACGT"          # ACGT is its own revcomp
    assert revcomp_str(revcomp_str("GATTACA")) == "GATTACA"


def test_revcomp_profile_matches_string_revcomp():
    P = profile_from_sites(["AACG"])
    assert consensus(revcomp_profile(P)) == revcomp_str("AACG")


def test_revcomp_profile_twice_is_identity():
    rng = np.random.default_rng(0)
    P = rng.random((4, 7))
    P /= P.sum(axis=0)
    assert np.allclose(revcomp_profile(revcomp_profile(P)), P)


def test_consensus():
    P = profile_from_sites(["ACGT", "ACGA", "TCGT"])
    assert consensus(P) == "ACGT"


def test_profile_from_sites_counts():
    P = profile_from_sites(["AA", "AC"])
    assert np.allclose(P[:, 0], [1, 0, 0, 0])
    assert np.allclose(P[:, 1], [0.5, 0.5, 0, 0])


def test_profile_from_sites_pseudocount_removes_zeros():
    P = profile_from_sites(["AA", "AC"], pseudocount=1)
    assert (P > 0).all()
    assert np.allclose(P[:, 0], [3/6, 1/6, 1/6, 1/6])


def test_ic_extremes():
    conserved = profile_from_sites(["ACGT"])       # one site -> fully conserved
    uniform = np.full((4, 4), 0.25)
    assert information_content(conserved) == pytest.approx(8.0)   # 2 bits x 4
    assert information_content(uniform) == pytest.approx(0.0)
    assert np.allclose(column_entropy(uniform), 2.0)


def test_check_profile_rejects_bad_input():
    with pytest.raises(ValueError):
        check_profile(np.ones((3, 5)) / 3)         # wrong number of rows
    with pytest.raises(ValueError):
        check_profile(np.ones((4, 5)))             # columns don't sum to 1


# ---- Section 2: alignment, Hamming, Pearson -------------------------------

TRUE = "GATTACAGGC"


def _random_profile(k, seed):
    P = np.random.default_rng(seed).random((4, k))
    return P / P.sum(axis=0)


def test_hamming_perfect_match():
    h = hamming_aligned(TRUE, TRUE)
    assert h["distance"] == 0 and h["offset"] == 0 and h["strand"] == "+"


def test_hamming_counts_mismatches():
    assert hamming_aligned("GATTACAGGA", TRUE)["distance"] == 1


def test_hamming_shifted_by_one():
    # found motif = true shifted right by 1: overlap is perfect, 1 true
    # position is uncovered -> distance 1, offset 1
    h = hamming_aligned("ATTACAGGCT", TRUE)
    assert h["distance"] == 1 and h["offset"] == 1


def test_hamming_reverse_strand():
    h = hamming_aligned(revcomp_str(TRUE), TRUE)
    assert h["distance"] == 0 and h["strand"] == "-"


def test_hamming_shorter_found_motif():
    # STREME-style: found motif is a shorter piece of the true one
    h = hamming_aligned("TTACAGGC", TRUE)          # 8 of 10 bases
    assert h["distance"] == 2 and h["offset"] == 2


def test_pearson_perfect_match():
    T = _random_profile(10, seed=1)
    p = pearson_aligned(T, T)
    assert p["r"] == pytest.approx(1.0) and p["offset"] == 0


def test_pearson_shifted_match():
    T = _random_profile(10, seed=1)
    F = T[:, 2:]                                   # drop first 2 columns
    p = pearson_aligned(F, T)
    assert p["r"] == pytest.approx(1.0) and p["offset"] == 2


def test_pearson_reverse_strand():
    T = _random_profile(10, seed=1)
    p = pearson_aligned(revcomp_profile(T), T)
    assert p["r"] == pytest.approx(1.0) and p["strand"] == "-"


def test_pearson_random_motif_is_lower():
    T = profile_from_sites([TRUE], pseudocount=0.1)      # strong motif
    R = _random_profile(10, seed=7)
    assert pearson_aligned(R, T)["r"] < 0.6


def test_pearson_uniform_gives_nan():
    U = np.full((4, 10), 0.25)
    assert np.isnan(pearson_aligned(U, U)["r"])
