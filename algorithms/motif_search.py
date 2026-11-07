"""Classic motif-search algorithms for MotifBench.

Greedy Motif Search, Randomized Motif Search and Gibbs Sampling
(Compeau & Pevzner, Ch. 2), with two switches:

  background : "uniform"  -> bg = 0.25 for every base
               "shuffled" -> bg estimated from a shuffled copy of the input
  pseudocount: True  -> Laplace add-one pseudocounts in the profile
               False -> raw counts (zeros handled explicitly, see below)

All three functions share the agreed interface:

    f(seqs, k, background="uniform", pseudocount=True, seed=0)
        -> dict(positions=[...], profile=np.array (4, k), consensus="ACGT...",
                score=float)

Nucleotide order is A, C, G, T (rows of `profile`). `score` is an extra key
holding the log-odds score of the returned motif set; callers may ignore it.

Scoring (replaces the textbook consensus score):
    score(motifs) = sum_j sum_b count[b, j] * log(P[b, j] / bg[b])
where P is the profile built from the motif set itself. Higher is better.
Terms with count 0 contribute 0 (the limit of x*log x), so the score is
always finite. For picking a k-mer inside one sequence, a window is scored by
sum_j log(P[b_j, j] / bg[b_j]); with pseudocount=False a window containing a
base that has P = 0 gets -inf, which is handled without errors. If every
window of a sequence is -inf, the first window is chosen (Greedy/Randomized)
or a window is sampled uniformly (Gibbs).
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

NUCLEOTIDES = "ACGT"

# Search budgets. These are PLACEHOLDERS until the team fixes the shared
# budget (Step 0, item 5). Greedy is deterministic and has no budget.
RANDOMIZED_RESTARTS = 200
GIBBS_RESTARTS = 20
GIBBS_ITERATIONS = 1000

_LUT = np.full(256, -1, dtype=np.int64)
for _i, _c in enumerate(NUCLEOTIDES):
    _LUT[ord(_c)] = _i
    _LUT[ord(_c.lower())] = _i


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _encode(seq):
    codes = _LUT[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]
    if (codes < 0).any():
        raise ValueError("sequences may only contain A, C, G, T")
    return codes


class _Data:
    """Pre-computed k-mer windows for all sequences.

    flat_idx[i, p, j] = base_code * k + j for window p of sequence i, so that
    log_odds.ravel()[flat_idx[i, p]].sum() is that window's score. Sequences
    may have different lengths; shorter ones are padded and masked by `valid`.
    """

    def __init__(self, codes, k):
        self.k = k
        self.t = len(codes)
        self.n_pos = np.array([len(c) - k + 1 for c in codes])
        n_max = int(self.n_pos.max())
        cols = np.arange(k)
        self.flat_idx = np.zeros((self.t, n_max, k), dtype=np.intp)
        for i, c in enumerate(codes):
            win = sliding_window_view(c, k)
            self.flat_idx[i, : len(win)] = win * k + cols
        self.valid = np.arange(n_max)[None, :] < self.n_pos[:, None]
        self.rows = np.arange(self.t)

    def counts(self, pos):
        """(4, k) base counts of the k-mers starting at `pos` (one per seq)."""
        sel = self.flat_idx[self.rows, pos]
        return np.bincount(sel.ravel(), minlength=4 * self.k).reshape(4, self.k).astype(float)


def _background(codes, mode, rng):
    if mode == "uniform":
        return np.full(4, 0.25)
    if mode == "shuffled":
        shuffled = np.concatenate([rng.permutation(c) for c in codes])
        freq = np.bincount(shuffled, minlength=4) / shuffled.size
        return np.maximum(freq, 1e-12)  # guard against log(0) if a base is absent
    raise ValueError("background must be 'uniform' or 'shuffled'")


def _profile(counts, n, pseudocount):
    if pseudocount:
        return (counts + 1.0) / (n + 4.0)  # Laplace add-one
    return counts / n


def _log_odds(profile, log_bg):
    with np.errstate(divide="ignore"):
        return np.log(profile) - log_bg[:, None]  # -inf where profile == 0


def _set_score(counts, n, pseudocount, log_bg):
    w = _log_odds(_profile(counts, n, pseudocount), log_bg)
    with np.errstate(invalid="ignore"):
        terms = np.where(counts > 0, counts * w, 0.0)
    return float(terms.sum())


def _prepare(seqs, k, background, pseudocount, seed):
    if k < 1:
        raise ValueError("k must be >= 1")
    if len(seqs) < 2:
        raise ValueError("need at least 2 sequences")
    codes = [_encode(s) for s in seqs]
    if min(len(c) for c in codes) < k:
        raise ValueError("every sequence must be at least k long")
    rng = np.random.default_rng(seed)
    log_bg = np.log(_background(codes, background, rng))
    return _Data(codes, k), log_bg, rng


def _result(data, pos, pseudocount, log_bg):
    counts = data.counts(pos)
    profile = _profile(counts, data.t, pseudocount)
    return {
        "positions": [int(p) for p in pos],
        "profile": profile,
        "consensus": "".join(NUCLEOTIDES[b] for b in profile.argmax(axis=0)),
        "score": _set_score(counts, data.t, pseudocount, log_bg),
    }


def _best_windows(data, log_odds):
    """Best-scoring window start in every sequence (first one on ties)."""
    sc = log_odds.ravel()[data.flat_idx].sum(axis=2)
    sc = np.where(data.valid, sc, -np.inf)
    return sc.argmax(axis=1)


# --------------------------------------------------------------------------
# Algorithms
# --------------------------------------------------------------------------

def greedy_motif_search(seqs, k, background="uniform", pseudocount=True, seed=0):
    """Greedy Motif Search: try every k-mer of sequence 1 as the start, extend
    through the other sequences with the profile-best k-mer, keep the best set.
    Deterministic; `seed` only affects the shuffled background."""
    data, log_bg, _ = _prepare(seqs, k, background, pseudocount, seed)
    t = data.t
    best_score, best_pos = -np.inf, None
    for start in range(int(data.n_pos[0])):
        pos = np.zeros(t, dtype=np.intp)
        pos[0] = start
        counts = np.zeros(4 * k)
        counts[data.flat_idx[0, start]] += 1
        for i in range(1, t):
            w = _log_odds(_profile(counts.reshape(4, k), i, pseudocount), log_bg)
            sc = w.ravel()[data.flat_idx[i, : data.n_pos[i]]].sum(axis=1)
            pos[i] = sc.argmax()
            counts[data.flat_idx[i, pos[i]]] += 1
        score = _set_score(counts.reshape(4, k), t, pseudocount, log_bg)
        if score > best_score:
            best_score, best_pos = score, pos
    return _result(data, best_pos, pseudocount, log_bg)


def randomized_motif_search(seqs, k, background="uniform", pseudocount=True, seed=0,
                            n_restarts=RANDOMIZED_RESTARTS):
    """Randomized Motif Search: from random starting k-mers, alternate
    profile -> best k-mers until the score stops improving; repeat for
    `n_restarts` random starts and keep the best result."""
    data, log_bg, rng = _prepare(seqs, k, background, pseudocount, seed)
    t = data.t
    best_score, best_pos = -np.inf, None
    for _ in range(n_restarts):
        pos = rng.integers(0, data.n_pos)
        counts = data.counts(pos)
        score = _set_score(counts, t, pseudocount, log_bg)
        while True:
            w = _log_odds(_profile(counts, t, pseudocount), log_bg)
            new_pos = _best_windows(data, w)
            new_counts = data.counts(new_pos)
            new_score = _set_score(new_counts, t, pseudocount, log_bg)
            if new_score > score:
                pos, counts, score = new_pos, new_counts, new_score
            else:
                break
        if score > best_score:
            best_score, best_pos = score, pos
    return _result(data, best_pos, pseudocount, log_bg)


def gibbs_sampler(seqs, k, background="uniform", pseudocount=True, seed=0,
                  n_restarts=GIBBS_RESTARTS, n_iterations=GIBBS_ITERATIONS):
    """Gibbs Sampling: from random starting k-mers, repeatedly drop one
    sequence's k-mer, build the profile from the rest, and resample that
    sequence's k-mer with probability proportional to P(window)/bg(window).
    Best motif set seen over `n_restarts` x `n_iterations` is returned."""
    data, log_bg, rng = _prepare(seqs, k, background, pseudocount, seed)
    t = data.t
    best_score, best_pos = -np.inf, None
    for _ in range(n_restarts):
        pos = rng.integers(0, data.n_pos)
        counts = data.counts(pos).ravel()
        run_best = _set_score(counts.reshape(4, k), t, pseudocount, log_bg)
        run_pos = pos.copy()
        for _ in range(n_iterations):
            i = int(rng.integers(t))
            counts[data.flat_idx[i, pos[i]]] -= 1
            w = _log_odds(_profile(counts.reshape(4, k), t - 1, pseudocount), log_bg)
            sc = w.ravel()[data.flat_idx[i, : data.n_pos[i]]].sum(axis=1)
            top = sc.max()
            if np.isfinite(top):
                cum = np.cumsum(np.exp(sc - top))
                pos[i] = min(int(np.searchsorted(cum, rng.random() * cum[-1], side="right")),
                             len(sc) - 1)
            else:  # every window has zero probability: sample uniformly
                pos[i] = rng.integers(len(sc))
            counts[data.flat_idx[i, pos[i]]] += 1
            s = _set_score(counts.reshape(4, k), t, pseudocount, log_bg)
            if s > run_best:
                run_best, run_pos = s, pos.copy()
        if run_best > best_score:
            best_score, best_pos = run_best, run_pos
    return _result(data, best_pos, pseudocount, log_bg)


def score_positions(seqs, k, positions, background="uniform", pseudocount=True, seed=0):
    """Log-odds score (same definition as the algorithms use) of a GIVEN set of
    start positions, e.g. the planted ones. Comparing it with the score of a
    found motif tells whether a miss was a search failure (found < planted)
    or a scoring-model failure (found >= planted)."""
    data, log_bg, _ = _prepare(seqs, k, background, pseudocount, seed)
    return _result(data, np.asarray(positions, dtype=np.intp), pseudocount, log_bg)["score"]


ALGORITHMS = {
    "greedy": greedy_motif_search,
    "randomized": randomized_motif_search,
    "gibbs": gibbs_sampler,
}