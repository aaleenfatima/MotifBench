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


# ---------------------------------------------------------------------------
# Section 2: alignment, Hamming distance, Pearson correlation
# ---------------------------------------------------------------------------
# A found motif can be shifted a few bases, on the opposite strand, or (for
# STREME) a different width from the true motif. So we slide it along the
# true motif on both strands and keep the best-scoring placement.
#
# offset = column of the TRUE motif where column 0 of the found motif lands
# (negative = found motif starts before the true one). On the "-" strand the
# offset refers to the reverse-complemented found motif.

def _default_min_overlap(kf, kt):
    """Need at least 75% of the shorter motif to overlap."""
    return max(1, int(np.ceil(0.75 * min(kf, kt))))


def _placements(kf, kt, min_overlap):
    """Yield (offset, true_slice, found_slice) for every allowed placement."""
    for o in range(-(kf - 1), kt):
        t0, t1 = max(0, o), min(kt, o + kf)
        if t1 - t0 >= min_overlap:
            yield o, slice(t0, t1), slice(t0 - o, t1 - o)


def hamming_aligned(found, true, min_overlap=None):
    """Smallest Hamming distance between two consensus strings over all shifts
    and both strands.

    distance = mismatches in the overlap + true positions left uncovered,
    so it ranges from 0 (perfect) to len(true). A shifted-but-correct motif
    is not punished for the shift itself, only for what it misses.
    """
    found, true = found.upper(), true.upper()
    kf, kt = len(found), len(true)
    if min_overlap is None:
        min_overlap = _default_min_overlap(kf, kt)
    best = None
    for strand, f in (("+", found), ("-", revcomp_str(found))):
        for o, ts, fs in _placements(kf, kt, min_overlap):
            mism = sum(a != b for a, b in zip(true[ts], f[fs]))
            d = mism + (kt - (ts.stop - ts.start))
            if best is None or d < best["distance"]:
                best = dict(distance=d, normalized=d / kt,
                            offset=o, strand=strand)
    return best


def pearson_aligned(found, true, min_overlap=None):
    """Best Pearson correlation between two profiles over all shifts and both
    strands.

    For each placement the overlapping columns of both matrices are
    flattened into vectors and correlated. 1 = identical shape, ~0 = unrelated.
    """
    F, T = check_profile(found), check_profile(true)
    kf, kt = F.shape[1], T.shape[1]
    if min_overlap is None:
        min_overlap = _default_min_overlap(kf, kt)
    best = None
    for strand, G in (("+", F), ("-", revcomp_profile(F))):
        for o, ts, fs in _placements(kf, kt, min_overlap):
            x, y = G[:, fs].ravel(), T[:, ts].ravel()
            if x.std() == 0 or y.std() == 0:   # e.g. all-uniform columns
                continue
            r = float(np.corrcoef(x, y)[0, 1])
            if best is None or r > best["r"]:
                best = dict(r=r, offset=o, strand=strand,
                            overlap=ts.stop - ts.start)
    if best is None:
        best = dict(r=float("nan"), offset=None, strand=None, overlap=0)
    return best


# ---------------------------------------------------------------------------
# Section 3: site recovery
# ---------------------------------------------------------------------------
# "Did the algorithm point at the right place in each sequence?"
# Two definitions, both reported:
#   exact   - found start == planted start
#   overlap - found site covers >= min_frac of the planted site
# A sequence with no prediction (None, e.g. STREME found no site there)
# counts as a miss.

def site_overlap(found_pos, true_pos, k_found, k_true):
    """Fraction of the planted site [true_pos, true_pos+k_true) that the found
    site [found_pos, found_pos+k_found) covers. 0 if found_pos is None."""
    if found_pos is None:
        return 0.0
    start = max(found_pos, true_pos)
    stop = min(found_pos + k_found, true_pos + k_true)
    return max(0, stop - start) / k_true


def site_recovery(found_positions, true_positions, k_found, k_true=None,
                  min_frac=0.5):
    """Share of sequences whose planted site was recovered.

    found_positions / true_positions: one start index per sequence, same order
    as the FASTA. Returns exact_rate, overlap_rate, mean_overlap (all 0..1).
    """
    if k_true is None:
        k_true = k_found
    if len(found_positions) != len(true_positions):
        raise ValueError("need one found position per sequence")
    fracs = np.array([site_overlap(f, t, k_found, k_true)
                      for f, t in zip(found_positions, true_positions)])
    exact = np.array([f is not None and int(f) == int(t)
                      for f, t in zip(found_positions, true_positions)])
    return dict(exact_rate=float(exact.mean()),
                overlap_rate=float((fracs >= min_frac).mean()),
                mean_overlap=float(fracs.mean()),
                n_seqs=len(true_positions))
