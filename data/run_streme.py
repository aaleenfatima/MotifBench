"""Run STREME on every dataset folder and collect the top motif of each.
Resumable: finished datasets are skipped. Run on Linux, where streme is installed.

  python run_streme.py --workers 2              # all datasets found
  python run_streme.py --filter MA0001.3_f0.6   # only names containing this text
"""
import os, re, sys, json, csv, glob, shutil, argparse, subprocess, time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
RESULTS = os.path.join(PROJECT, "results")
RAW = os.path.join(RESULTS, "streme_raw")          # per-dataset output (gitignored)
DEFAULT_ROOTS = [os.path.join(HERE, "synthetic"), os.path.join(HERE, "real_datasets")]

MOTIF_RE = re.compile(r"^MOTIF\s+(\S+)")
HDR_RE = re.compile(r"letter-probability matrix:\s*alength=\s*4\s+w=\s*(\d+)"
                    r"\s+nsites=\s*(\d+)\s+E=\s*(\S+)")


def parse_streme(path):
    """Return the motifs in streme.txt, best-ranked first.
    Each matrix is 4 x w, rows in A,C,G,T order."""
    with open(path) as f:
        lines = f.read().splitlines()
    motifs, name, i = [], None, 0
    while i < len(lines):
        m = MOTIF_RE.match(lines[i])
        if m:
            name = m.group(1)
        h = HDR_RE.search(lines[i])
        if h and name:
            w, nsites, ev = int(h.group(1)), int(h.group(2)), float(h.group(3))
            rows = [[float(x) for x in lines[i + 1 + j].split()] for j in range(w)]
            matrix = [[rows[j][b] for j in range(w)] for b in range(4)]
            cons = "".join("ACGT"[max(range(4), key=lambda b: rows[j][b])]
                           for j in range(w))
            motifs.append(dict(name=name, width=w, nsites=nsites, evalue=ev,
                               consensus=cons, matrix=matrix))
            name = None
            i += w
        i += 1
    return motifs


def run_one(task):
    ds_dir, streme_args, keep_all = task
    name = os.path.basename(ds_dir)
    out = os.path.join(RAW, name)
    info_path = os.path.join(out, "run.json")
    if os.path.exists(info_path):                  # already done
        with open(info_path) as f:
            return json.load(f)
    if os.path.isdir(out):
        shutil.rmtree(out)                         # partial run, start clean
    with open(os.path.join(ds_dir, "meta.json")) as f:
        meta = json.load(f)

    cmd = ["streme", "--p", os.path.join(ds_dir, "seqs.fasta"), "--dna",
           "--oc", out, "--verbosity", "1"] + streme_args
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    runtime = time.perf_counter() - t0
    os.makedirs(out, exist_ok=True)

    motifs, status, err = [], "ok", ""
    txt = os.path.join(out, "streme.txt")
    if proc.returncode != 0 or not os.path.exists(txt):
        status, err = "error", proc.stderr[-500:]
    else:
        motifs = parse_streme(txt)
        if not motifs:
            status = "no_motif"                    # a real result, not a crash
    if not keep_all:                               # keep only the small useful files
        for fn in os.listdir(out):
            if fn != "streme.txt" and not fn.endswith(".tsv"):
                p = os.path.join(out, fn)
                if os.path.isfile(p):
                    os.remove(p)

    info = dict(name=name, data_type=meta.get("data_type", "synthetic"),
                motif_id=meta["motif_id"],
                strength_fraction=meta["strength_fraction"],
                background=meta["background"], replicate=meta["replicate"],
                true_width=meta["width"], status=status,
                runtime_sec=round(runtime, 2), n_motifs=len(motifs),
                top=motifs[0] if motifs else None,
                cmd=" ".join(cmd[:2] + ["<fasta>"] + cmd[3:]), error=err)
    with open(info_path, "w") as f:
        json.dump(info, f)
    return info


def aggregate():
    """Rebuild the two small summary files from all finished runs."""
    infos = []
    for p in sorted(glob.glob(os.path.join(RAW, "*", "run.json"))):
        with open(p) as f:
            infos.append(json.load(f))
    os.makedirs(RESULTS, exist_ok=True)
    cols = ["name", "data_type", "motif_id", "strength_fraction", "background",
            "replicate", "status", "runtime_sec", "n_motifs", "top_width",
            "top_nsites", "top_evalue", "top_consensus"]
    with open(os.path.join(RESULTS, "streme_summary.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        for r in infos:
            t = r["top"] or {}
            wr.writerow([r["name"], r["data_type"], r["motif_id"],
                         r["strength_fraction"], r["background"], r["replicate"],
                         r["status"], r["runtime_sec"], r["n_motifs"],
                         t.get("width", ""), t.get("nsites", ""),
                         t.get("evalue", ""), t.get("consensus", "")])
    tops = {r["name"]: r["top"] for r in infos if r["top"]}
    with open(os.path.join(RESULTS, "streme_top_motifs.json"), "w") as f:
        json.dump(tops, f)
    return infos


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS)
    ap.add_argument("--filter", default="", help="only dataset names containing this")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--minw", type=int, default=None)
    ap.add_argument("--maxw", type=int, default=20,
                    help="STREME default is 15; widened so 16-17 bp motifs fit")
    ap.add_argument("--nmotifs", type=int, default=None,
                    help="stop after this many motifs (default: STREME decides)")
    ap.add_argument("--keep-all", action="store_true", help="keep html/xml outputs")
    a = ap.parse_args()

    if shutil.which("streme") is None:
        sys.exit("streme not found in PATH. Activate the conda env that has MEME.")

    streme_args = ["--maxw", str(a.maxw)]
    if a.minw:
        streme_args += ["--minw", str(a.minw)]
    if a.nmotifs:
        streme_args += ["--nmotifs", str(a.nmotifs)]

    dirs = []
    for root in a.roots:
        dirs += [os.path.dirname(p) for p in
                 sorted(glob.glob(os.path.join(root, "*", "meta.json")))]
    dirs = [d for d in dirs if a.filter in os.path.basename(d)]
    if a.limit:
        dirs = dirs[:a.limit]
    if not dirs:
        sys.exit("No datasets found. Generate them first (generate.py / generate_real.py).")
    print(f"{len(dirs)} datasets, {a.workers} workers, STREME args: {' '.join(streme_args)}")

    done = 0
    t_start = time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(run_one, (d, streme_args, a.keep_all)) for d in dirs]
            for fut in as_completed(futs):
                r = fut.result()
                done += 1
                print(f"[{done}/{len(dirs)}] {r['name']}  {r['status']}  "
                      f"{r['runtime_sec']}s  top={r['top']['consensus'] if r['top'] else '-'}")
    except KeyboardInterrupt:
        print("\nInterrupted. Run the same command again to resume.")
    infos = aggregate()
    counts = {}
    for r in infos:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("status counts:", counts)
    if infos:
        print(f"mean runtime per dataset: "
              f"{sum(r['runtime_sec'] for r in infos) / len(infos):.1f}s; "
              f"wall time this session: {time.perf_counter() - t_start:.0f}s")
    print("wrote results/streme_summary.csv and results/streme_top_motifs.json")