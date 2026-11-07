"""Run the metrics on one of Laiba's generated datasets as a quick sanity check.

Usage (from the repo root):
    python data/generate.py --pilot          # makes data/synthetic/ first
    python -m evaluation.sanity_check
    python -m evaluation.sanity_check MA0001.3_f0.2_at_rich_rep1

The planted sites are the "perfect answer", so they should score close to
the true matrix, and a random motif should score badly.
"""
import json, os, sys
import numpy as np
from evaluation.metrics import (profile_from_sites, consensus, revcomp_profile,
                                information_content, hamming_aligned,
                                pearson_aligned)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNTH = os.path.join(ROOT, "data", "synthetic")


def main(name="MA0001.3_f0.6_uniform_rep0"):
    path = os.path.join(SYNTH, name, "meta.json")
    if not os.path.exists(path):
        sys.exit(f"{path} not found - run `python data/generate.py --pilot` first")
    with open(path) as f:
        m = json.load(f)
    T, J = np.array(m["true_matrix"]), np.array(m["jaspar_matrix"])
    F = profile_from_sites(m["planted_sites"])
    R = np.random.default_rng(0).random(T.shape)
    R /= R.sum(axis=0)

    print(f"dataset: {name}   width={m['width']}  background={m['background']}")
    print(f"true IC  stored={m['true_ic']:.2f}  recomputed={information_content(T):.2f}")
    print(f"planted-sites IC = {information_content(F):.2f}")
    print()
    print("consensus  true  :", consensus(T))
    print("consensus  jaspar:", consensus(J))
    print("consensus  sites :", consensus(F))
    print()
    print("Hamming  sites vs jaspar      :", hamming_aligned(consensus(F), consensus(J)))
    print("Pearson  sites vs true        :", pearson_aligned(F, T))
    print("Pearson  sites (rev) vs true  :", pearson_aligned(revcomp_profile(F), T))
    print("Pearson  shifted sites vs true:", pearson_aligned(F[:, 2:], T))
    print("Pearson  RANDOM vs true       :", pearson_aligned(R, T))


if __name__ == "__main__":
    main(*sys.argv[1:])
