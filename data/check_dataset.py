import json, sys, numpy as np
from jaspar_loader import information_content
d = sys.argv[1]                      # e.g. synthetic/MA0001.3_f0.6_markov_rep0
m = json.load(open(f"{d}/meta.json"))
sites = m["planted_sites"]
k = m["width"]
counts = np.zeros((4, k))
for s in sites:
    for j, b in enumerate(s):
        counts[m["nucleotide_order"].index(b), j] += 1
emp = counts / counts.sum(axis=0)
print("target IC:", round(m["true_ic"], 2),
      "| empirical IC of planted sites:", round(information_content(emp), 2))
seq = "".join(l.strip() for l in open(f"{d}/seqs.fasta") if l[0] != ">")
print("base freqs A,C,G,T:",
      [round(seq.count(b) / len(seq), 3) for b in "ACGT"])