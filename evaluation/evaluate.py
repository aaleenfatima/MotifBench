"""Section 4: turn one algorithm run into one CSV row.

The experiment runner (experiments/) does, for every run:

    seqs, meta = load_dataset(dataset_dir)
    found = find_motif(seqs, k=meta["width"], ...)       # Aaleen's interface
    row = evaluate(found, meta, algorithm="gibbs", bg_model="uniform",
                   pseudocount=True, seed=0, runtime_s=1.3)
    append_results("results/results.csv", [row])

`found` is the agreed result dict:
    positions : one start index per sequence (None = no site, e.g. STREME)
    profile   : 4 x k array, rows A,C,G,T
    consensus : optional, recomputed from profile if missing

Two different "backgrounds" exist, so the CSV keeps them apart:
    data_bg  - how the sequences were generated (uniform / at_rich / markov)
    bg_model - what the algorithm assumed when scoring (uniform / shuffled)
"""
import csv, json, os
import numpy as np
from evaluation.metrics import (check_profile, consensus, information_content,
                                hamming_aligned, pearson_aligned, site_recovery)

# Fixed column order for every results CSV, so files from different people
# and different days can be concatenated safely.
RESULT_COLUMNS = [
    # what was run
    "algorithm", "bg_model", "pseudocount", "seed",
    # which dataset
    "dataset", "data_type", "motif_id", "strength", "true_ic", "data_bg", "replicate",
    "true_width",
    # what was found
    "found_consensus", "found_width", "found_ic",
    # metrics
    "hamming", "hamming_norm", "pearson_true", "pearson_jaspar",
    "align_offset", "align_strand",
    "exact_rate", "overlap_rate", "mean_overlap",
    "runtime_s",
]


def load_dataset(dataset_dir):
    """Read one dataset folder made by data/generate.py -> (seqs, meta)."""
    with open(os.path.join(dataset_dir, "meta.json")) as f:
        meta = json.load(f)
    seqs = []
    with open(os.path.join(dataset_dir, "seqs.fasta")) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith(">"):
                seqs.append(line)
    return seqs, meta


def _r(x, nd=4):
    """Round floats for the CSV; leave None / NaN / ints / strings alone."""
    if isinstance(x, float) and not np.isnan(x):
        return round(x, nd)
    return x


def evaluate(found, meta, algorithm, bg_model=None, pseudocount=None,
             seed=None, runtime_s=None):
    """Score one run against the truth in meta.json; returns one CSV row.

    found=None means the method reported no motif (e.g. STREME on a very
    weak motif): profile metrics are left empty and recovery is 0, so the
    failure still counts in the averages instead of silently disappearing.
    """
    if found is None:
        return _failed_row(meta, algorithm, bg_model, pseudocount, seed,
                           runtime_s)
    P = check_profile(found["profile"])
    T = np.asarray(meta["true_matrix"], dtype=float)
    J = np.asarray(meta["jaspar_matrix"], dtype=float)
    cons = found.get("consensus") or consensus(P)
    true_cons = consensus(T)

    ham = hamming_aligned(cons, true_cons)
    pt = pearson_aligned(P, T)
    pj = pearson_aligned(P, J)
    rec = site_recovery(found["positions"], meta["positions"],
                        k_found=P.shape[1], k_true=meta["width"])

    row = dict(
        algorithm=algorithm, bg_model=bg_model, pseudocount=pseudocount,
        seed=seed,
        dataset=meta["name"],
        data_type=meta.get("data_type", "synthetic"),   # real data sets this
        motif_id=meta["motif_id"],
        strength=meta["strength_fraction"], true_ic=meta["true_ic"],
        data_bg=meta["background"], replicate=meta["replicate"],
        true_width=meta["width"],
        found_consensus=cons, found_width=P.shape[1],
        found_ic=information_content(P),
        hamming=ham["distance"], hamming_norm=ham["normalized"],
        pearson_true=pt["r"], pearson_jaspar=pj["r"],
        align_offset=pt["offset"], align_strand=pt["strand"],
        exact_rate=rec["exact_rate"], overlap_rate=rec["overlap_rate"],
        mean_overlap=rec["mean_overlap"],
        runtime_s=runtime_s,
    )
    return {k: _r(row[k]) for k in RESULT_COLUMNS}


def _failed_row(meta, algorithm, bg_model, pseudocount, seed, runtime_s):
    row = dict.fromkeys(RESULT_COLUMNS)                 # everything None
    row.update(
        algorithm=algorithm, bg_model=bg_model, pseudocount=pseudocount,
        seed=seed, dataset=meta["name"],
        data_type=meta.get("data_type", "synthetic"),
        motif_id=meta["motif_id"], strength=meta["strength_fraction"],
        true_ic=_r(meta["true_ic"]), data_bg=meta["background"],
        replicate=meta["replicate"], true_width=meta["width"],
        exact_rate=0.0, overlap_rate=0.0, mean_overlap=0.0,
        runtime_s=runtime_s,
    )
    return row


def append_results(path, rows):
    """Append rows to a results CSV, writing the header if the file is new."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if new:
            w.writeheader()
        w.writerows(rows)
