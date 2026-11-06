"""Generate planted-motif datasets: FASTA + JSON per dataset."""
import os, json, zlib, argparse
import numpy as np
from jaspar_loader import load_motif, get_ppm, JASPAR_DIR, BASES
from strength import ppm_at_fraction, STRENGTH_FRACTIONS
from jaspar_loader import information_content
from selected_motifs import SELECTED

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic")
N_SEQS, SEQ_LEN, N_REPS = 50, 500, 10

def _iid(p):
    return np.tile(np.array(p, dtype=float), (4, 1))

# transition matrices, rows = previous base, columns = next base, order A,C,G,T
BACKGROUNDS = {
    "uniform": _iid([0.25, 0.25, 0.25, 0.25]),
    "at_rich": _iid([0.35, 0.15, 0.15, 0.35]),
    "markov": np.array([[0.30, 0.20, 0.20, 0.30],
                        [0.30, 0.25, 0.10, 0.35],   # C->G depleted (CpG)
                        [0.30, 0.20, 0.25, 0.25],
                        [0.20, 0.20, 0.25, 0.35]]),
}

def stationary(T):
    pi = np.ones(4) / 4
    for _ in range(500):
        pi = pi @ T
    return pi

def make_background(T, length, rng):
    cum = np.cumsum(T, axis=1).tolist()
    start_cum = np.cumsum(stationary(T)).tolist()
    u = rng.random(length).tolist()
    seq = []
    prev = next((i for i, c in enumerate(start_cum) if u[0] <= c), 3)
    seq.append(prev)
    for t in range(1, length):
        row = cum[prev]
        prev = next((i for i, c in enumerate(row) if u[t] <= c), 3)
        seq.append(prev)
    return np.array(seq)

def sample_site(ppm, rng):
    cum = np.cumsum(ppm, axis=0)                  # 4 x k
    u = rng.random(ppm.shape[1])
    return np.minimum((u[None, :] > cum).sum(axis=0), 3)

def make_dataset(mid, fraction, bg_name, rep, outdir=OUT_DIR):
    name = f"{mid}_f{fraction}_{bg_name}_rep{rep}"
    seed = zlib.crc32(name.encode())              # deterministic per dataset
    rng = np.random.default_rng(seed)
    orig = get_ppm(load_motif(os.path.join(JASPAR_DIR, f"{mid}.jaspar")))
    ppm, alpha, ic = ppm_at_fraction(orig, fraction)
    k = ppm.shape[1]
    T = BACKGROUNDS[bg_name]

    seqs, positions, sites = [], [], []
    for _ in range(N_SEQS):
        s = make_background(T, SEQ_LEN, rng)
        site = sample_site(ppm, rng)
        pos = int(rng.integers(0, SEQ_LEN - k + 1))
        s[pos:pos + k] = site                     # replace, so length stays fixed
        seqs.append("".join(BASES[i] for i in s))
        positions.append(pos)
        sites.append("".join(BASES[i] for i in site))

    d = os.path.join(outdir, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "seqs.fasta"), "w") as f:
        for i, s in enumerate(seqs):
            f.write(f">seq{i}\n{s}\n")
    meta = dict(name=name, motif_id=mid, strength_fraction=fraction,
                alpha=alpha, true_ic=ic, width=k, background=bg_name,
                transition_matrix=T.tolist(), n_seqs=N_SEQS, seq_len=SEQ_LEN,
                replicate=rep, seed=seed, nucleotide_order=BASES,
                positions=positions, planted_sites=sites,
                true_matrix=ppm.tolist(), jaspar_matrix=orig.tolist())
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f)
    return d

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true",
                    help="MA0001.3 only, 3 replicates")
    args = ap.parse_args()
    motifs = ["MA0001.3"] if args.pilot else SELECTED
    reps = 3 if args.pilot else N_REPS
    n = 0
    for mid in motifs:
        for fr in STRENGTH_FRACTIONS:
            for bg in BACKGROUNDS:
                for r in range(reps):
                    make_dataset(mid, fr, bg, r)
                    n += 1
    print("generated", n, "datasets in", OUT_DIR)