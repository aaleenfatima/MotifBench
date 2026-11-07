"""Budget-sensitivity study: how much do Randomized restarts and Gibbs
restarts x iterations matter?

For each dataset it runs Randomized with several restart counts and Gibbs with
several (restarts, iterations) pairs, scores each run with evaluate(), and
also records whether the search found a motif set scoring at least as high as
the PLANTED positions:
    search_ok = found_score >= planted_score
If search_ok is False the search missed a better optimum (more budget could
help); if it is True the scoring model itself prefers the wrong answer (more
budget will not help).

Usage, from the repo root:
    python -m experiments.budget_study --filter MA0001.3_f0.8 MA0001.3_f1.0 --reps 0 1
Output: results/budget_study.csv plus a printed summary.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse, glob, sys, time
import pandas as pd

from algorithms.motif_search import ALGORITHMS, score_positions
from evaluation.evaluate import load_dataset, evaluate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYN = os.path.join(ROOT, "data", "synthetic")
OUT = os.path.join(ROOT, "results", "budget_study.csv")

# (algorithm, keyword arguments). Gibbs budget = restarts x iterations.
CONFIGS = (
    [("randomized", {"n_restarts": n}) for n in (25, 50, 100, 200, 400)]
    + [("gibbs", {"n_restarts": r, "n_iterations": i})
       for r, i in ((5, 250), (10, 500), (20, 1000), (40, 2000))]
)
BG_MODELS = ("uniform", "shuffled")


def budget_of(alg, kw):
    return kw["n_restarts"] * kw.get("n_iterations", 1)


def run_study(dirs, pseudocount=True):
    rows = []
    for n, d in enumerate(dirs, 1):
        seqs, meta = load_dataset(d)
        k, seed = meta["width"], int(meta["seed"]) % (2 ** 31)
        for bg in BG_MODELS:
            planted = score_positions(seqs, k, meta["positions"], bg, pseudocount, seed)
            for alg, kw in CONFIGS:
                t0 = time.perf_counter()
                found = ALGORITHMS[alg](seqs, k, background=bg, pseudocount=pseudocount,
                                        seed=seed, **kw)
                dt = time.perf_counter() - t0
                r = evaluate(found, meta, algorithm=alg, bg_model=bg,
                             pseudocount=pseudocount, seed=seed, runtime_s=dt)
                rows.append(dict(
                    dataset=meta["name"], strength=meta["strength_fraction"],
                    data_bg=meta["background"], algorithm=alg, bg_model=bg,
                    restarts=kw["n_restarts"], iterations=kw.get("n_iterations", ""),
                    budget=budget_of(alg, kw), overlap_rate=r["overlap_rate"],
                    exact_rate=r["exact_rate"], pearson_true=r["pearson_true"],
                    hamming=r["hamming"], found_score=found["score"],
                    planted_score=planted,
                    search_ok=bool(found["score"] >= planted - 1e-9),
                    runtime_s=round(dt, 3)))
        print(f"[{n}/{len(dirs)}] {meta['name']}", flush=True)
    return pd.DataFrame(rows)


def summarize(df):
    df = df.copy()
    df["setting"] = df.apply(
        lambda r: f"{r.restarts}" if r.algorithm == "randomized"
        else f"{r.restarts}x{int(r.iterations)}", axis=1)
    g = df.groupby(["algorithm", "bg_model", "setting"], sort=False)
    out = g.agg(n=("overlap_rate", "size"),
                overlap=("overlap_rate", "mean"),
                search_ok=("search_ok", "mean"),
                runtime_s=("runtime_s", "mean")).round(3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", nargs="+", default=["MA0001.3_f0.8", "MA0001.3_f1.0"],
                    help="dataset names containing any of these")
    ap.add_argument("--reps", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--no-pseudocount", action="store_true")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    dirs = [os.path.dirname(p) for p in sorted(glob.glob(os.path.join(SYN, "*", "meta.json")))]
    dirs = [d for d in dirs
            if any(f in os.path.basename(d) for f in a.filter)
            and int(os.path.basename(d).rsplit("rep", 1)[1]) in a.reps]
    if not dirs:
        sys.exit("No datasets matched. Run data/generate.py first.")
    df = run_study(dirs, pseudocount=not a.no_pseudocount)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"\n{len(df)} runs on {len(dirs)} datasets -> {a.out}\n")
    print(summarize(df).to_string())


if __name__ == "__main__":
    main()