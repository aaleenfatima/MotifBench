"""Run evaluate() on every pilot dataset with FAKE algorithms plus the real
baselines, so metrics -> CSV -> plots can be checked before the real
algorithms exist.

    greedy / randomized / gibbs  - FAKE: each sequence gets the planted site
        with a probability that rises with motif strength, otherwise a
        random one. Pseudocounts on/off and scoring bg uniform/shuffled are
        varied too. The numbers mean NOTHING about the real algorithms.
    streme       - FAKE STREME file (true matrix, opposite strand, 8 wide)
    oracle_scan  - real baseline: scan with the true matrix (ceiling)
    random       - real baseline: random positions (chance)

Usage (from the repo root):
    python data/generate.py --pilot
    python -m evaluation.demo_evaluate
    python -m evaluation.plots results/demo_results.csv --out results/demo_figures --note "DEMO - FAKE ALGORITHMS, NOT RESULTS"
Writes results/demo_results.csv (git-ignored).
"""
import glob, os, sys
import numpy as np
import pandas as pd
from evaluation.metrics import profile_from_sites, revcomp_profile
from evaluation.evaluate import load_dataset, evaluate, append_results
from evaluation.baselines import oracle_scan, random_motif
from evaluation.streme import write_meme, streme_to_found

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "demo_results.csv")

# fake skill: P(hit planted site) = base + slope * strength
FAKE_SKILL = {"greedy": (0.0, 0.6), "randomized": (0.05, 0.75), "gibbs": (0.1, 0.85)}


def fake_algorithm(alg, seqs, meta, pseudocount, bg_model, rng):
    k, L = meta["width"], len(seqs[0])
    base, slope = FAKE_SKILL[alg]
    p_hit = base + slope * meta["strength_fraction"]
    p_hit += 0.05 * (pseudocount) + 0.03 * (bg_model == "shuffled")
    pos = [t if rng.random() < p_hit else int(rng.integers(0, L - k + 1))
           for t in meta["positions"]]
    sites = [s[p:p + k] for s, p in zip(seqs, pos)]
    return dict(positions=pos,
                profile=profile_from_sites(sites, pseudocount=1 if pseudocount else 0))


def main():
    dirs = sorted(glob.glob(os.path.join(ROOT, "data", "synthetic", "*")))
    if not dirs:
        sys.exit("no datasets - run `python data/generate.py --pilot` first")
    if os.path.exists(OUT):
        os.remove(OUT)
    rows = []
    for d in dirs:
        seqs, meta = load_dataset(d)
        rng = np.random.default_rng(meta["seed"])
        for alg in FAKE_SKILL:
            for pc in (True, False):
                for bgm in ("uniform", "shuffled"):
                    f = fake_algorithm(alg, seqs, meta, pc, bgm, rng)
                    rows.append(evaluate(f, meta, algorithm=alg, bg_model=bgm,
                                         pseudocount=pc, seed=0))
        meme = os.path.join(ROOT, "results", "demo_streme", meta["name"], "streme.txt")
        write_meme(meme, [revcomp_profile(np.array(meta["true_matrix"]))[:, 1:9]])
        rows.append(evaluate(streme_to_found(meme, seqs), meta, algorithm="streme"))
        rows.append(evaluate(oracle_scan(seqs, meta), meta, algorithm="oracle_scan"))
        rows.append(evaluate(random_motif(seqs, meta["width"], meta["seed"]), meta,
                             algorithm="random"))
    append_results(OUT, rows)
    print(f"wrote {len(rows)} rows to {OUT}\n")

    df = pd.read_csv(OUT)
    cols = ["pearson_true", "hamming_norm", "overlap_rate", "exact_rate"]
    print("mean metrics per method (all strengths) - FAKE algorithms:")
    print(df.groupby("algorithm")[cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()
