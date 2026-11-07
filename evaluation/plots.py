"""Section 6: figures and summary tables from a results CSV.

Usage (from the repo root):
    python -m evaluation.plots results/results.csv
    python -m evaluation.plots results/results.csv --out results/figures

One figure per metric, metric vs motif strength:
    panel columns = data_bg  (how the DNA was generated)
    panel rows    = bg_model (what the algorithm assumed when scoring)
    colour        = algorithm (fixed: greedy blue, randomized orange,
                    gibbs aqua - never re-coloured between figures)
    line style    = pseudocount (solid = on, dashed = off)
    references    = STREME (black dash-dot; not on exact_rate), oracle
                    ceiling (grey dotted, site-recovery metrics only),
                    chance (grey band + centre line)
Points are means over replicates and motifs; error bars are 95% CIs
(1.96 x standard error).
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # write files, never open windows
import matplotlib.pyplot as plt

# Colours: validated with the dataviz palette checker (3 slots, all pairs,
# colour-blind safe). Aqua is low-contrast on white, so every line also has
# its own marker and a direct label, and the tables carry the numbers.
ALGORITHMS = {                            # name -> (colour, marker, label)
    "greedy":     ("#2a78d6", "o", "Greedy"),
    "randomized": ("#eb6834", "s", "Randomized"),
    "gibbs":      ("#1baf7a", "^", "Gibbs"),
}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

REFERENCE = ("streme", "oracle_scan", "random")

METRICS = {   # column -> (y label, y limits, higher_is_better)
    "pearson_true": ("Pearson r with true motif", (-0.1, 1.05), True),
    "hamming_norm": ("Hamming distance (fraction of width)", (-0.03, 1.0), False),
    "overlap_rate": ("Sites recovered (>= 50% overlap)", (-0.03, 1.03), True),
    "exact_rate":   ("Sites recovered (exact start)", (-0.03, 1.03), True),
}
RECOVERY = ("overlap_rate", "exact_rate")   # where the oracle ceiling matters
NO_STREME = ("exact_rate",)                  # STREME compared on overlap only

BG_ORDER = ["uniform", "at_rich", "markov"]
MODEL_ORDER = ["uniform", "shuffled"]
BG_LABEL = {"uniform": "uniform DNA", "at_rich": "AT-rich DNA",
            "markov": "Markov DNA"}


def load_results(*paths):
    """Read and stack one or more results CSVs."""
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["pseudocount"] = df["pseudocount"].map(
        {True: "on", False: "off", "True": "on", "False": "off"})
    df["bg_model"] = df["bg_model"].fillna("-")
    return df


def mean_ci(df, by, metric):
    """Mean, 95% CI half-width and n of `metric` per group."""
    g = df.groupby(by, dropna=False)[metric]
    out = g.agg(mean="mean", sd="std", n="count").reset_index()
    out["ci"] = 1.96 * out["sd"].fillna(0) / np.sqrt(out["n"].clip(lower=1))
    return out


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=8)


def _direct_labels(ax, ends, ylim):
    """Name each algorithm at the right end of its line (pseudocounts-on
    line), pushing labels apart so they never overlap."""
    if not ends:
        return
    gap = 0.075 * (ylim[1] - ylim[0])
    ends.sort(key=lambda e: e[0])
    ys = [e[0] for e in ends]
    for n in range(1, len(ys)):           # push up anything too close below
        ys[n] = max(ys[n], ys[n - 1] + gap)
    shift = max(0, ys[-1] - ylim[1])      # then pull the stack back inside
    for (_, lab), y in zip(ends, ys):
        ax.annotate(lab, (1.0, y - shift), xytext=(10, 0),
                    textcoords="offset points", va="center",
                    fontsize=8, color=INK2, annotation_clip=False)


def plot_metric(df, metric, out_path, title_note=None):
    ylabel, ylim, higher = METRICS[metric]
    bgs = [b for b in BG_ORDER if b in set(df["data_bg"])]
    algs = df[~df["algorithm"].isin(REFERENCE)]
    unknown = sorted(set(algs["algorithm"]) - set(ALGORITHMS))
    if unknown:
        print(f"  note: not plotting unknown algorithms {unknown}")
    algs = algs[algs["algorithm"].isin(list(ALGORITHMS))]
    present = set(algs["bg_model"])
    models = [m for m in MODEL_ORDER if m in present] + \
             sorted(present - set(MODEL_ORDER)) or ["-"]
    n_series = len(ALGORITHMS) * 2
    dodge = {}                            # small x-shift so error bars don't overlap
    for n, key in enumerate((a, pc) for a in ALGORITHMS for pc in ("on", "off")):
        dodge[key] = (n - (n_series - 1) / 2) * 0.008

    fig, axes = plt.subplots(len(models), len(bgs), sharex=True, sharey=True,
                             figsize=(3.6 * len(bgs), 2.9 * len(models) + 0.9),
                             squeeze=False, facecolor=SURFACE)
    ends = []                             # (y, label) for direct labels
    for i, model in enumerate(models):
        for j, bg in enumerate(bgs):
            ax = axes[i, j]
            _style(ax)
            here = df[df["data_bg"] == bg]

            # chance band
            r = mean_ci(here[here["algorithm"] == "random"], "strength", metric)
            if len(r):
                ax.fill_between(r["strength"], r["mean"] - r["ci"],
                                r["mean"] + r["ci"], color=MUTED, alpha=0.22,
                                lw=0, label="chance (random sites)")
                # thin centre line, so the band never disappears when all
                # replicates agree and the CI is zero
                ax.plot(r["strength"], r["mean"], color=MUTED, lw=0.8)
            # oracle ceiling
            o = mean_ci(here[here["algorithm"] == "oracle_scan"], "strength", metric)
            if len(o) and metric in RECOVERY:
                ax.plot(o["strength"], o["mean"], ls=":", color=MUTED, lw=2,
                        label="ceiling (scan with true motif)")
            # STREME (not on exact_rate: its own width/offset means its start
            # never equals the planted start, so 0 there would be misleading)
            s = mean_ci(here[here["algorithm"] == "streme"], "strength", metric)
            if len(s) and metric not in NO_STREME:
                ax.errorbar(s["strength"], s["mean"], yerr=s["ci"], color=INK,
                            ls="-.", lw=1.5, marker="D", ms=5, capsize=2,
                            label="STREME")
            # our algorithms
            mine = here[(here["bg_model"] == model)]
            for alg, (col, mk, lab) in ALGORITHMS.items():
                for pc, ls in (("on", "-"), ("off", "--")):
                    a = mine[(mine["algorithm"] == alg) & (mine["pseudocount"] == pc)]
                    if not len(a):
                        continue
                    m = mean_ci(a, "strength", metric)
                    ax.errorbar(m["strength"] + dodge[(alg, pc)], m["mean"],
                                yerr=m["ci"], color=col, ls=ls, lw=2, marker=mk, ms=6,
                                mec=SURFACE, mew=1.2, capsize=2,
                                label=f"{lab}, pseudocounts {pc}")
                    if j == len(bgs) - 1 and pc == "on":
                        ends.append([m["mean"].iloc[-1], lab])
            _direct_labels(ax, ends, ylim)
            ends.clear()
            ax.set_ylim(*ylim)
            ax.set_xlim(0.12, 1.12)
            if i == 0:
                ax.set_title(BG_LABEL.get(bg, bg), fontsize=9.5, color=INK)
            if j == 0:
                row = f"scoring bg: {model}\n" if model != "-" else ""
                ax.set_ylabel(row + ylabel, fontsize=8.5, color=INK2)
            if i == len(models) - 1:
                ax.set_xlabel("motif strength (fraction of JASPAR IC)",
                              fontsize=8.5, color=INK2)

    handles, labels = {}, []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in handles:
                handles[l] = h
                labels.append(l)
    arrow = "higher is better" if higher else "lower is better"
    head = f"{ylabel} vs motif strength  ({arrow})"
    if title_note:
        head += f"\n{title_note}"
    fig.suptitle(head, fontsize=11, color=INK)
    fig.tight_layout()
    fig.legend([handles[l] for l in labels], labels, loc="upper center",
               ncol=min(4, len(labels)), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.0), labelcolor=INK2)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_path


def summary_table(df, out_csv=None):
    """Mean +- 95% CI of every metric per condition: the table view of the
    figures, and the numbers to quote in the report."""
    by = ["algorithm", "bg_model", "pseudocount", "data_bg", "strength"]
    parts = []
    for metric in METRICS:
        t = mean_ci(df, by, metric)
        t = t.rename(columns={"mean": f"{metric}_mean", "ci": f"{metric}_ci95"})
        parts.append(t.drop(columns=["sd", "n"] if parts else ["sd"]))
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on=by)
    out = out.round(4)
    if out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        out.to_csv(out_csv, index=False)
    return out


def make_all(paths, out_dir, title_note=None):
    df = load_results(*paths)
    written = [plot_metric(df, m, os.path.join(out_dir, f"{m}_vs_strength.png"),
                           title_note) for m in METRICS]
    written.append(os.path.join(out_dir, "summary_table.csv"))
    summary_table(df, written[-1])
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", nargs="+", help="one or more results CSVs")
    ap.add_argument("--out", default=os.path.join("results", "figures"))
    ap.add_argument("--note", default=None, help="extra line under the title")
    a = ap.parse_args()
    for p in make_all(a.csv, a.out, a.note):
        print("wrote", p)
