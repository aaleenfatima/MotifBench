"""Reference "methods" that bracket every real algorithm on each dataset.

    oracle_scan - scan each sequence with the TRUE matrix and take the best
                  window. This is the ceiling: no method can locate the
                  sites better than the true motif itself can. At weak
                  strength background windows outscore the planted site,
                  so even this ceiling is low.
    random      - random positions. This is the chance floor; note it is
                  not 0 for Pearson, because we keep the best of many
                  shifts/strands, and it is higher on AT-rich data for
                  AT-rich motifs.

Both return the same `found` dict as the algorithms, so they go through
evaluate() and land in the results CSV as algorithm="oracle_scan"/"random".
The runner should add one of each per dataset.
"""
import numpy as np
from evaluation.metrics import profile_from_sites, consensus
from evaluation.streme import best_site


def oracle_scan(seqs, meta):
    T = np.asarray(meta["true_matrix"], dtype=float)
    hits = [best_site(s, T) for s in seqs]
    return dict(profile=T, consensus=consensus(T),
                positions=[h[0] for h in hits])


def random_motif(seqs, k, seed=0):
    rng = np.random.default_rng(seed)
    pos = [int(rng.integers(0, len(s) - k + 1)) for s in seqs]
    sites = [s[p:p + k] for s, p in zip(seqs, pos)]
    return dict(profile=profile_from_sites(sites, pseudocount=1), positions=pos)
