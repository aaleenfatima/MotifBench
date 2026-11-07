"""Run evaluate() on every pilot dataset with three FAKE algorithms, so the
whole metrics -> CSV path can be checked before the real algorithms exist.

    oracle  - returns the planted positions (perfect answer)
    shifted - planted positions + 2 (right motif, slightly off)
    random  - random positions (a motif finder that failed)

Usage (from the repo root):
    python data/generate.py --pilot
    python -m evaluation.demo_evaluate
Writes results/demo_results.csv (git-ignored). This file is also handy
pilot data for building the plots in Section 6.
"""
import glob, os, sys
import numpy as np
from evaluation.metrics import profile_from_sites
from evaluation.evaluate import load_dataset, evaluate, append_results

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "demo_results.csv")


def fake_find_motif(name, seqs, k, true_pos, rng):
    L = len(seqs[0])
    if name == "oracle":
        pos = list(true_pos)
    elif name == "shifted":
        pos = [min(p + 2, L - k) for p in true_pos]
    else:
        pos = rng.integers(0, L - k + 1, size=len(seqs)).tolist()
    sites = [s[p:p + k] for s, p in zip(seqs, pos)]
    return dict(positions=pos, profile=profile_from_sites(sites, pseudocount=1))


def main():
    dirs = sorted(glob.glob(os.path.join(ROOT, "data", "synthetic", "*")))
    if not dirs:
        sys.exit("no datasets - run `python data/generate.py --pilot` first")
    if os.path.exists(OUT):
        os.remove(OUT)
    rows = []
    for d in dirs:
        seqs, meta = load_dataset(d)
        for alg in ("oracle", "shifted", "random"):
            rng = np.random.default_rng(meta["seed"])
            found = fake_find_motif(alg, seqs, meta["width"], meta["positions"], rng)
            rows.append(evaluate(found, meta, algorithm=alg, bg_model="uniform",
                                 pseudocount=True, seed=0))
    append_results(OUT, rows)
    print(f"wrote {len(rows)} rows to {OUT}\n")

    import pandas as pd
    df = pd.read_csv(OUT)
    cols = ["hamming", "pearson_true", "exact_rate", "overlap_rate"]
    print("mean metrics per fake algorithm and motif strength:")
    print(df.groupby(["algorithm", "strength"])[cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()
