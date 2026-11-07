"""Evaluation metrics for MotifBench.

Conventions (shared with data/ and algorithms/):
  - nucleotide order is A,C,G,T
  - a profile is a 4 x k numpy array, rows = A,C,G,T, columns sum to 1
"""
import numpy as np

BASES = "ACGT"  # same fixed order as data/jaspar_loader.py
_COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}


# ---------------------------------------------------------------------------
# Section 1: core helpers
# ---------------------------------------------------------------------------

def check_profile(P):
    """Return P as a float array after checking it is a valid 4 x k profile."""
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != 4:
        raise ValueError(f"profile must be 4 x k, got shape {P.shape}")
    if not np.allclose(P.sum(axis=0), 1.0, atol=1e-6):
        raise ValueError("profile columns must sum to 1")
    return P


def revcomp_str(s):
    """Reverse complement of a DNA string, e.g. 'AACG' -> 'CGTT'."""
    return "".join(_COMP[b] for b in reversed(s.upper()))


def revcomp_profile(P):
    """Reverse complement of a profile.

    Reversing the columns flips the direction; reversing the rows swaps
    A<->T and C<->G, which works only because the row order is A,C,G,T.
    """
    return check_profile(P)[::-1, ::-1].copy()


def consensus(P):
    """Most likely base in each column (ties go to the earlier base in ACGT)."""
    P = check_profile(P)
    return "".join(BASES[i] for i in P.argmax(axis=0))


def profile_from_sites(sites, pseudocount=0.0):
    """Build a 4 x k profile from equal-length site strings.

    Used to turn an algorithm's found positions (or the planted sites)
    into a matrix we can score.
    """
    k = len(sites[0])
    if any(len(s) != k for s in sites):
        raise ValueError("all sites must have the same length")
    counts = np.full((4, k), float(pseudocount))
    for s in sites:
        for j, b in enumerate(s.upper()):
            counts[BASES.index(b), j] += 1
    return counts / counts.sum(axis=0, keepdims=True)


def column_entropy(P):
    """Shannon entropy (bits) of each column; 0 = fully conserved, 2 = random."""
    p = np.clip(check_profile(P), 1e-12, 1.0)   # avoid log(0)
    return -(p * np.log2(p)).sum(axis=0)


def information_content(P):
    """Total IC in bits = sum over columns of (2 - entropy).

    Same definition as data/jaspar_loader.py, so our numbers match the
    true_ic values Laiba stores in meta.json.
    """
    return float((2.0 - column_entropy(P)).sum())
