import glob, os
import numpy as np
import pandas as pd
from Bio import motifs

JASPAR_DIR = os.path.join(os.path.dirname(__file__), "jaspar")
BASES = "ACGT"  # fixed order for the whole project

def load_motif(path):
    with open(path) as f:
        m = motifs.read(f, "jaspar")
    return m

def get_ppm(m):
    """4 x k probability matrix, rows in A,C,G,T order, columns sum to 1."""
    counts = np.array([[m.counts[b][i] for i in range(m.length)] for b in BASES],
                      dtype=float)
    return counts / counts.sum(axis=0, keepdims=True)

def information_content(ppm):
    """Total IC in bits: sum over columns of (2 - entropy)."""
    p = np.clip(ppm, 1e-12, 1.0)
    entropy = -(p * np.log2(p)).sum(axis=0)
    return float((2.0 - entropy).sum())

def summarize():
    rows = []
    for path in sorted(glob.glob(os.path.join(JASPAR_DIR, "*.jaspar"))):
        m = load_motif(path)
        ppm = get_ppm(m)
        rows.append(dict(
            matrix_id=m.matrix_id, name=m.name, width=m.length,
            total_ic=round(information_content(ppm), 2),
            ic_per_pos=round(information_content(ppm) / m.length, 2),
            n_sites=int(sum(m.counts[b][0] for b in BASES)),
        ))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    print(summarize().to_string(index=False))