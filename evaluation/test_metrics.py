import numpy as np
import pytest

from evaluation.metrics import (check_profile, revcomp_str, revcomp_profile,
                                consensus, profile_from_sites,
                                column_entropy, information_content)


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
