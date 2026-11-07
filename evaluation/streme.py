"""Section 5: convert STREME output into the same `found` dict our own
algorithms return, so STREME goes through the exact same evaluate().

STREME writes several files; we read `streme.txt` (MEME text format):

    MEME version 5.5.4
    ALPHABET= ACGT
    strands: + -
    Background letter frequencies
    A 0.303 C 0.197 G 0.197 T 0.303
    MOTIF 1-CCATATATAG STREME-1
    letter-probability matrix: alength= 4 w= 10 nsites= 48 S= 1.2e-010
     0.020000 0.940000 0.020000 0.020000      <- one row per position
     ...

How we make STREME comparable to Greedy / Randomized / Gibbs:
  1. Width   - STREME picks its own width; the metrics already align
               different widths (Section 2), so we keep STREME's width.
  2. Motif   - STREME ranks several motifs; we use its TOP-ranked one, not
               the one closest to the truth (that would be cheating).
  3. Sites   - our algorithms return exactly one site per sequence, so we
               scan each sequence with STREME's matrix (log-odds against
               STREME's own background, both strands) and take the best
               window. That gives one position per sequence, like the others.
"""
import os
import numpy as np
from evaluation.metrics import BASES, revcomp_profile, consensus


def parse_meme(path):
    """Read a MEME-format text file (e.g. streme.txt).

    Returns (motifs, background). Each motif is a dict with id, alt_id,
    width, nsites, score (STREME's p-value 'S=' or MEME's 'E='), and
    profile as a 4 x w array in A,C,G,T order.
    """
    with open(path) as f:
        lines = [ln.strip() for ln in f]

    background = np.full(4, 0.25)
    motifs = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("ALPHABET=") and ln.split("=")[1].strip() != BASES:
            raise ValueError(f"unexpected alphabet: {ln}")
        if ln.startswith("Background letter frequencies"):
            vals = lines[i + 1].split()                 # "A 0.3 C 0.2 ..."
            freqs = dict(zip(vals[0::2], map(float, vals[1::2])))
            background = np.array([freqs[b] for b in BASES])
        if ln.startswith("MOTIF"):
            parts = ln.split()
            motif = dict(id=parts[1], alt_id=parts[2] if len(parts) > 2 else None)
            while not lines[i].startswith("letter-probability matrix"):
                i += 1
            # "letter-probability matrix: alength= 4 w= 10 nsites= 48 S= 1e-10"
            toks = lines[i].split(":", 1)[1].replace("= ", "=").split()
            info = dict(t.split("=") for t in toks if "=" in t)
            w = int(info["w"])
            rows = [list(map(float, lines[i + 1 + r].split())) for r in range(w)]
            P = np.array(rows).T                        # w x 4  ->  4 x w
            P = P / P.sum(axis=0, keepdims=True)        # fix rounding in file
            motif.update(width=w, profile=P,
                         nsites=int(float(info.get("nsites", 0))),
                         score=float(info.get("S", info.get("E", "nan"))))
            motifs.append(motif)
            i += w
        i += 1
    return motifs, background


def _one_hot(seq):
    idx = np.array([BASES.index(b) for b in seq.upper()])
    return np.eye(4)[idx].T                             # 4 x L


def best_site(seq, profile, background=None, both_strands=True, eps=1e-3):
    """Highest-scoring window of `seq` for `profile`.

    Score = sum of log2(P[base, j] / bg[base]). A tiny eps keeps zeros in
    the matrix from becoming -inf. Returns (start, strand, score).
    """
    bg = np.full(4, 0.25) if background is None else np.asarray(background)
    X = _one_hot(seq)
    k, L = profile.shape[1], X.shape[1]
    best = (None, None, -np.inf)
    strands = [("+", profile)]
    if both_strands:
        strands.append(("-", revcomp_profile(profile)))
    for strand, P in strands:
        W = np.log2((P + eps) / (1 + 4 * eps) / bg[:, None])   # 4 x k
        # score of every window: sum over j of W[:, j] . X[:, s + j]
        scores = sum(W[:, j] @ X[:, j:L - k + 1 + j] for j in range(k))
        s = int(scores.argmax())
        if scores[s] > best[2]:
            best = (s, strand, float(scores[s]))
    return best


def streme_to_found(meme_path, seqs, rank=0):
    """STREME output -> `found` dict for evaluate(), or None if STREME
    reported no motif (evaluate() records that as a failed run)."""
    motifs, bg = parse_meme(meme_path)
    if len(motifs) <= rank:
        return None
    m = motifs[rank]
    hits = [best_site(s, m["profile"], bg) for s in seqs]
    return dict(profile=m["profile"], consensus=consensus(m["profile"]),
                positions=[h[0] for h in hits],
                strands=[h[1] for h in hits],
                streme_id=m["id"], streme_score=m["score"],
                n_motifs=len(motifs))


def write_meme(path, motifs, background=None):
    """Write profiles in MEME format (the reverse of parse_meme).

    Handy for tests and the demo, and lets us hand our own motifs to
    MEME Suite tools such as Tomtom if we ever want to.
    """
    bg = np.full(4, 0.25) if background is None else np.asarray(background)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("MEME version 5\n\nALPHABET= ACGT\n\nstrands: + -\n\n")
        f.write("Background letter frequencies\n")
        f.write(" ".join(f"{b} {p:.3f}" for b, p in zip(BASES, bg)) + "\n\n")
        for n, P in enumerate(motifs, start=1):
            P = np.asarray(P)
            f.write(f"MOTIF {n}-{consensus(P)} STREME-{n}\n")
            f.write(f"letter-probability matrix: alength= 4 w= {P.shape[1]} "
                    f"nsites= 0 S= 1.0e+000\n")
            for col in P.T:
                f.write(" " + " ".join(f"{p:.6f}" for p in col) + "\n")
            f.write("\n")
