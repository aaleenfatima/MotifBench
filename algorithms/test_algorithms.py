"""Unit tests for algorithms/motif_search.py.  Run from the repo root:

    python -m pytest algorithms -q
"""

import numpy as np
import pytest

from algorithms.motif_search import ALGORITHMS, NUCLEOTIDES

# Small budgets so the tests run fast.
FAST = {
    "greedy": {},
    "randomized": {"n_restarts": 30},
    "gibbs": {"n_restarts": 5, "n_iterations": 300},
}

# Gibbs can get stuck one base off the true frame with a small budget (about
# 1 run in 4 at 5 x 1000 on the dataset below), so the recovery test uses a
# larger budget and a fixed seed.
STRONG = {**FAST, "gibbs": {"n_restarts": 20, "n_iterations": 1000}}

MOTIF = "GATTACAG"


def planted_dataset(n_seqs=15, length=80, motif=MOTIF, seed=123):
    """Random sequences with the identical motif planted at a random position."""
    rng = np.random.default_rng(seed)
    seqs, starts = [], []
    for _ in range(n_seqs):
        s = list("".join(rng.choice(list(NUCLEOTIDES), length)))
        p = int(rng.integers(0, length - len(motif) + 1))
        s[p:p + len(motif)] = motif
        seqs.append("".join(s))
        starts.append(p)
    return seqs, starts


def run(name, seqs, k, budget=FAST, **kw):
    return ALGORITHMS[name](seqs, k, **{**budget[name], **kw})


@pytest.mark.parametrize("name", list(ALGORITHMS))
@pytest.mark.parametrize("pseudocount", [True, False])
def test_strong_motif_is_recovered(name, pseudocount):
    seqs, starts = planted_dataset()
    out = run(name, seqs, len(MOTIF), budget=STRONG, pseudocount=pseudocount, seed=1)
    assert out["consensus"] == MOTIF
    assert out["positions"] == starts


@pytest.mark.parametrize("name", list(ALGORITHMS))
@pytest.mark.parametrize("background", ["uniform", "shuffled"])
@pytest.mark.parametrize("pseudocount", [True, False])
def test_output_format(name, background, pseudocount):
    seqs, _ = planted_dataset(n_seqs=6, length=40)
    k = 6
    out = run(name, seqs, k, background=background, pseudocount=pseudocount)
    assert set(out) >= {"positions", "profile", "consensus"}
    assert len(out["positions"]) == len(seqs)
    assert all(0 <= p <= len(s) - k for p, s in zip(out["positions"], seqs))
    assert out["profile"].shape == (4, k)
    assert np.allclose(out["profile"].sum(axis=0), 1.0)
    assert len(out["consensus"]) == k and set(out["consensus"]) <= set(NUCLEOTIDES)
    assert np.isfinite(out["score"])


@pytest.mark.parametrize("name", list(ALGORITHMS))
def test_same_seed_is_reproducible(name):
    seqs, _ = planted_dataset(n_seqs=8, length=50)
    a = run(name, seqs, 7, background="shuffled", seed=5)
    b = run(name, seqs, 7, background="shuffled", seed=5)
    assert a["positions"] == b["positions"]
    assert np.array_equal(a["profile"], b["profile"])


def test_pseudocount_false_has_zeros_and_true_does_not():
    seqs, _ = planted_dataset()
    raw = run("randomized", seqs, len(MOTIF), pseudocount=False, seed=2)
    laplace = run("randomized", seqs, len(MOTIF), pseudocount=True, seed=2)
    assert (raw["profile"] == 0).any()
    assert (laplace["profile"] > 0).all()


@pytest.mark.parametrize("name", list(ALGORITHMS))
def test_all_windows_zero_probability_does_not_crash(name):
    # With raw counts, the second sequence has no base seen in the first
    # k-mer at some columns, so every window can score -inf.
    seqs = ["AAAAAA", "CCCCCC", "GGGGGG", "TTTTTT"]
    out = run(name, seqs, 3, pseudocount=False)
    assert len(out["positions"]) == 4 and np.isfinite(out["score"])


@pytest.mark.parametrize("name", list(ALGORITHMS))
def test_sequences_of_different_length(name):
    seqs, _ = planted_dataset(n_seqs=8, length=60)
    seqs = [s[: 40 + 3 * i] if MOTIF in s[: 40 + 3 * i] else s for i, s in enumerate(seqs)]
    out = run(name, seqs, len(MOTIF))
    assert all(p <= len(s) - len(MOTIF) for p, s in zip(out["positions"], seqs))


def test_shuffled_background_differs_from_uniform_on_skewed_data():
    rng = np.random.default_rng(0)
    seqs = ["".join(rng.choice(list("ACGT"), 60, p=[0.4, 0.1, 0.1, 0.4])) for _ in range(10)]
    u = run("gibbs", seqs, 6, background="uniform", seed=3)
    s = run("gibbs", seqs, 6, background="shuffled", seed=3)
    assert u["score"] != s["score"]


@pytest.mark.parametrize("name", list(ALGORITHMS))
def test_invalid_input_raises(name):
    f = ALGORITHMS[name]
    with pytest.raises(ValueError):
        f(["ACGTN", "ACGTA"], 3)          # bad character
    with pytest.raises(ValueError):
        f(["ACG", "ACGT"], 4)             # sequence shorter than k
    with pytest.raises(ValueError):
        f(["ACGT"], 3)                    # fewer than 2 sequences
    with pytest.raises(ValueError):
        f(["ACGT", "ACGT"], 3, background="bogus")

def test_textbook_greedy_example():
    """Sample dataset from Compeau & Pevzner (Greedy Motif Search, k=3, t=5).
    Expected motifs are the book's answers: without pseudocounts
    CAG CAG CAA CAA CAA, with pseudocounts TTC ATC TTC ATC TTC.
    (Matches an independent textbook-style implementation with the
    consensus-mismatch score; verify against your copy of the book.)"""
    dna = ["GGCGTTCAGGCA", "AAGAATCAGTCA", "CAAGGAGTTCGC",
           "CACGTCAATCAC", "CAATAATATTCG"]
    expected = {False: ["CAG", "CAG", "CAA", "CAA", "CAA"],
                True: ["TTC", "ATC", "TTC", "ATC", "TTC"]}
    for pc, motifs in expected.items():
        out = ALGORITHMS["greedy"](dna, 3, pseudocount=pc)
        found = [dna[i][p:p + 3] for i, p in enumerate(out["positions"])]
        assert found == motifs