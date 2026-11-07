"""Check the STREME conversion on pilot data without needing STREME installed.

For each pilot dataset we write a fake streme.txt that behaves the way real
STREME output often does: the motif is on the OPPOSITE strand and is only
8 of the 10 columns wide. Then we push it through streme_to_found() and
evaluate() exactly like the runner will do with real STREME output.

Usage (from the repo root):
    python data/generate.py --pilot
    python -m evaluation.demo_streme
    python -m evaluation.demo_streme path/to/real/streme.txt DATASET_NAME
"""
import glob, os, sys
import numpy as np
import pandas as pd
from evaluation.metrics import revcomp_profile
from evaluation.streme import write_meme, streme_to_found
from evaluation.evaluate import load_dataset, evaluate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "demo_streme")


def one(meme_path, dataset_dir):
    seqs, meta = load_dataset(dataset_dir)
    found = streme_to_found(meme_path, seqs)
    return evaluate(found, meta, algorithm="streme")


def main():
    if len(sys.argv) == 3:                       # real STREME output
        d = os.path.join(ROOT, "data", "synthetic", sys.argv[2])
        print(pd.Series(one(sys.argv[1], d)).to_string())
        return
    dirs = sorted(glob.glob(os.path.join(ROOT, "data", "synthetic", "*")))
    if not dirs:
        sys.exit("no datasets - run `python data/generate.py --pilot` first")
    rows = []
    for d in dirs:
        _, meta = load_dataset(d)
        T = np.array(meta["true_matrix"])
        fake = revcomp_profile(T)[:, 1:9]        # opposite strand, 8 wide
        path = os.path.join(OUT, meta["name"], "streme.txt")
        write_meme(path, [fake])
        rows.append(one(path, d))
    df = pd.DataFrame(rows)
    print("fake STREME (reverse strand, 8 of 10 columns) on all pilot data:\n")
    print(df[["dataset", "found_consensus", "found_width", "align_strand",
              "align_offset", "pearson_true", "hamming", "overlap_rate",
              "exact_rate"]].head(6).to_string(index=False))
    print("\nmean per strength:")
    cols = ["pearson_true", "hamming", "overlap_rate", "exact_rate"]
    print(df.groupby("strength")[cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()
