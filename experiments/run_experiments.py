"""Run the whole MotifBench grid and write one CSV row per run.

For every dataset folder it runs
    greedy / randomized / gibbs  x  bg_model {uniform, shuffled}
                                 x  pseudocount {True, False}      = 12 runs
    streme (reads the streme.txt made by data/run_streme.py)       =  1 run
    oracle_scan (ceiling) and random (chance) baselines            =  2 runs
and scores each with evaluation.evaluate.evaluate().

Usage, from the repo root:
    python -m experiments.run_experiments --filter MA0001.3_f0.6 --workers 2
    python -m experiments.run_experiments --workers 4

Resumable: datasets already in the output CSV are skipped, so you can stop
with Ctrl+C and run the same command again.
"""
import os
# one numpy thread per worker process, otherwise workers fight over cores
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse, csv, glob, json, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from algorithms.motif_search import ALGORITHMS
from evaluation.baselines import oracle_scan, random_motif
from evaluation.evaluate import load_dataset, evaluate, append_results
from evaluation.streme import streme_to_found

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ROOTS = [os.path.join(ROOT, "data", "synthetic"),
                 os.path.join(ROOT, "data", "real_datasets")]
DEFAULT_OUT = os.path.join(ROOT, "results", "results.csv")
DEFAULT_STREME = os.path.join(ROOT, "results", "streme_raw")

# The shared search budget (Step 0, item 5). Greedy is deterministic and has
# no budget. Change these here, in ONE place, and they are saved to
# results/run_config.json next to the CSV.
BUDGET = {
    "greedy": {},
    "randomized": {"n_restarts": 200},
    "gibbs": {"n_restarts": 40, "n_iterations": 2000},
}
BG_MODELS = ("uniform", "shuffled")
PSEUDOCOUNTS = (True, False)


def _git_commit():
    try:
        import subprocess
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def streme_row(seqs, meta, streme_dir):
    """Score STREME's saved output for one dataset -> (row or None, problem or None)."""
    txt = os.path.join(streme_dir, meta["name"], "streme.txt")
    info = os.path.join(streme_dir, meta["name"], "run.json")
    runtime = None
    if os.path.exists(info):
        with open(info) as f:
            runtime = json.load(f).get("runtime_sec")
    try:
        if os.path.exists(txt):
            return evaluate(streme_to_found(txt, seqs), meta,
                            algorithm="streme", runtime_s=runtime), None
        if os.path.exists(info):            # STREME ran but crashed
            return evaluate(None, meta, algorithm="streme",
                            runtime_s=runtime), None
        return None, (f"{meta['name']} streme: no output yet "
                      "(run data/run_streme.py first) - row skipped")
    except Exception:
        return None, f"{meta['name']} streme\n" + traceback.format_exc()


def fill_streme(dirs, out, streme_dir):
    """Add the STREME row to datasets that were run before STREME output existed."""
    have, finished = set(), set()
    with open(out, newline="") as f:
        for r in csv.DictReader(f):
            finished.add(r["dataset"])
            if r["algorithm"] == "streme":
                have.add(r["dataset"])
    added = waiting = bad = 0
    for d in dirs:
        name = os.path.basename(d)
        if name not in finished or name in have:
            continue
        seqs, meta = load_dataset(d)
        row, problem = streme_row(seqs, meta, streme_dir)
        if row:
            append_results(out, [row])
            added += 1
        elif problem and "no output yet" in problem:
            waiting += 1
        else:
            bad += 1
            print(problem)
    print(f"STREME rows added: {added}; still without STREME output: {waiting}; "
          f"errors: {bad}")


def run_dataset(task):
    """Run everything on one dataset. Returns (rows, errors)."""
    ds_dir, streme_dir = task
    seqs, meta = load_dataset(ds_dir)
    k = meta["width"]
    # same seed for every setting of a dataset, so pseudocount on/off is a
    # paired comparison. (uniform vs shuffled is only approximately paired:
    # a shuffled background draws random numbers before the search starts.)
    seed = int(meta["seed"]) % (2 ** 31)
    rows, errors = [], []

    for alg, fn in ALGORITHMS.items():
        for bgm in BG_MODELS:
            for pc in PSEUDOCOUNTS:
                try:
                    t0 = time.perf_counter()
                    found = fn(seqs, k, background=bgm, pseudocount=pc,
                               seed=seed, **BUDGET[alg])
                    dt = time.perf_counter() - t0
                    rows.append(evaluate(found, meta, algorithm=alg,
                                         bg_model=bgm, pseudocount=pc,
                                         seed=seed, runtime_s=round(dt, 3)))
                except Exception:
                    errors.append(f"{meta['name']} {alg} {bgm} pc={pc}\n"
                                  + traceback.format_exc())

    # STREME: output made earlier by data/run_streme.py
    row, problem = streme_row(seqs, meta, streme_dir)
    if row:
        rows.append(row)
    if problem:
        errors.append(problem)

    # baselines: ceiling and chance
    try:
        rows.append(evaluate(oracle_scan(seqs, meta), meta,
                             algorithm="oracle_scan"))
        rows.append(evaluate(random_motif(seqs, k, seed), meta,
                             algorithm="random", seed=seed))
    except Exception:
        errors.append(f"{meta['name']} baselines\n" + traceback.format_exc())
    return rows, errors


def finished_datasets(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return set()
    with open(path, newline="") as f:
        return {r["dataset"] for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS)
    ap.add_argument("--filter", default="", help="only dataset names containing this")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reps", nargs="+", type=int,
                    help="only these replicate numbers, e.g. --reps 0 1 2 3 4")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--streme-dir", default=DEFAULT_STREME)
    ap.add_argument("--fill-streme", action="store_true",
                    help="only add missing STREME rows to finished datasets")
    a = ap.parse_args()

    dirs = []
    for root in a.roots:
        dirs += [os.path.dirname(p) for p in
                 sorted(glob.glob(os.path.join(root, "*", "meta.json")))]
    dirs = [d for d in dirs if a.filter in os.path.basename(d)]
    if a.reps:
        dirs = [d for d in dirs
                if int(os.path.basename(d).rsplit("rep", 1)[1]) in a.reps]
    if a.limit:
        dirs = dirs[:a.limit]
    if not dirs:
        sys.exit("No datasets found. Run data/generate.py / generate_real.py first.")
    if a.fill_streme:
        fill_streme(dirs, a.out, a.streme_dir)
        return
    done = finished_datasets(a.out)
    todo = [d for d in dirs if os.path.basename(d) not in done]
    print(f"{len(dirs)} datasets, {len(dirs) - len(todo)} already in "
          f"{os.path.relpath(a.out, ROOT)}, {len(todo)} to run, "
          f"{a.workers} workers")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(os.path.join(os.path.dirname(a.out), "run_config.json"), "w") as f:
        json.dump(dict(budget=BUDGET, bg_models=BG_MODELS,
                       pseudocounts=PSEUDOCOUNTS,
                       seed_rule="meta['seed'] % 2**31, same for every setting",
                       algorithms_know_true_width=True,
                       numpy=np.__version__, python=sys.version.split()[0],
                       git_commit=_git_commit()),
                  f, indent=2)

    err_path = os.path.join(os.path.dirname(a.out), "run_errors.log")
    n_done, n_err, t_start = 0, 0, time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(run_dataset, (d, a.streme_dir)): d for d in todo}
            for fut in as_completed(futs):
                rows, errors = fut.result()
                if rows:
                    append_results(a.out, rows)   # one writer: the main process
                if errors:
                    n_err += len(errors)
                    with open(err_path, "a") as f:
                        f.write("\n".join(errors) + "\n")
                n_done += 1
                el = time.perf_counter() - t_start
                eta = el / n_done * (len(todo) - n_done)
                print(f"[{n_done}/{len(todo)}] {os.path.basename(futs[fut])}  "
                      f"{len(rows)} rows  {len(errors)} problems  "
                      f"ETA {eta / 60:.0f} min")
    except KeyboardInterrupt:
        print("\nInterrupted. Run the same command again to resume.")

    if n_err:
        print(f"\n{n_err} problems, see {os.path.relpath(err_path, ROOT)}")
    if os.path.exists(a.out):
        import pandas as pd
        df = pd.read_csv(a.out)
        print(f"\n{len(df)} rows in {os.path.relpath(a.out, ROOT)}. "
              "Mean overlap_rate by method and strength:")
        print(df.pivot_table(index="algorithm", columns="strength",
                             values="overlap_rate", aggfunc="mean")
                .round(2).to_string())


if __name__ == "__main__":
    main()