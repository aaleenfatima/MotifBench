"""Real-site datasets: real JASPAR binding sites planted in synthetic background.
Flank source is swappable via get_flank() for a later genomic version."""
import os, json, zlib, argparse
import numpy as np
from jaspar_loader import (load_motif, get_ppm, information_content,
                           JASPAR_DIR, BASES)
from generate import BACKGROUNDS, make_background, SEQ_LEN, N_SEQS, N_REPS

HERE = os.path.dirname(os.path.abspath(__file__))
SITES_DIR = os.path.join(HERE, "real")
OUT_DIR = os.path.join(HERE, "real_datasets")
REAL_MOTIFS = ["MA0014.3", "MA0003.4", "MA0018.4"]


def get_flank(length, bg_name, rng):
    """Option 1: synthetic flank. For option 2, replace the body with a
    random window cut from a genome FASTA (and change flank_source)."""
    return make_background(BACKGROUNDS[bg_name], length, rng)


def read_fasta(path):
    seqs, cur = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                    cur = []
            elif line:
                cur.append(line.upper())
    if cur:
        seqs.append("".join(cur))
    return seqs


def load_sites(mid, width):
    raw = read_fasta(os.path.join(SITES_DIR, f"{mid}_sites.fasta"))
    ok = [s for s in raw if len(s) == width and set(s) <= set(BASES)]
    print(f"{mid}: {len(raw)} sites in file, {len(ok)} usable "
          f"(length {width}, only ACGT)")
    if len(ok) < N_SEQS:
        raise SystemExit(f"{mid}: fewer than {N_SEQS} usable sites. "
                         "Check the file format.")
    return ok


def make_real_dataset(mid, pool, orig, bg_name, rep, outdir=OUT_DIR):
    name = f"real_{mid}_{bg_name}_rep{rep}"
    seed = zlib.crc32(name.encode())
    rng = np.random.default_rng(seed)
    k = orig.shape[1]

    idx = rng.choice(len(pool), size=N_SEQS, replace=False)
    seqs, positions, sites = [], [], []
    for i in idx:
        s = get_flank(SEQ_LEN, bg_name, rng)
        site = pool[i]
        pos = int(rng.integers(0, SEQ_LEN - k + 1))
        s[pos:pos + k] = [BASES.index(c) for c in site]
        seqs.append("".join(BASES[j] for j in s))
        positions.append(pos)
        sites.append(site)

    d = os.path.join(outdir, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "seqs.fasta"), "w") as f:
        for i, s in enumerate(seqs):
            f.write(f">seq{i}\n{s}\n")
    ic = information_content(orig)
    meta = dict(name=name, motif_id=mid, strength_fraction=1.0, alpha=0.0,
                true_ic=ic, width=k, background=bg_name,
                transition_matrix=BACKGROUNDS[bg_name].tolist(),
                n_seqs=N_SEQS, seq_len=SEQ_LEN, replicate=rep, seed=seed,
                nucleotide_order=BASES, positions=positions,
                planted_sites=sites, true_matrix=orig.tolist(),
                jaspar_matrix=orig.tolist(),
                data_type="real_sites",
                flank_source=f"synthetic_{bg_name}",
                site_source="JASPAR2026_sites_fasta",
                site_pool_size=len(pool))
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f)
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="1 motif, 2 reps")
    args = ap.parse_args()
    motifs = REAL_MOTIFS[:1] if args.pilot else REAL_MOTIFS
    reps = 2 if args.pilot else N_REPS
    n = 0
    for mid in motifs:
        orig = get_ppm(load_motif(os.path.join(JASPAR_DIR, f"{mid}.jaspar")))
        pool = load_sites(mid, orig.shape[1])
        for bg in BACKGROUNDS:
            for r in range(reps):
                make_real_dataset(mid, pool, orig, bg, r)
                n += 1
    print("generated", n, "real-site datasets in", OUT_DIR)