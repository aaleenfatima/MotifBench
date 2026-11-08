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
(1.96 x standard error). Strength curves use synthetic data only.

Also writes real_data.png: one dot per method on the real-site datasets
(which have no strength levels), and summary_table.csv split by data_type.
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

METRICS = {   # column -> (y label, y limits, higher_is_better; None = neither)
    "pearson_true": ("Pearson r with true motif", (-0.1, 1.05), True),
    "hamming_norm": ("Hamming distance (fraction of width)", (-0.03, 1.0), False),
    "overlap_rate": ("Sites recovered (>= 50% overlap)", (-0.03, 1.03), True),
    "exact_rate":   ("Sites recovered (exact start)", (-0.03, 1.03), True),
    "found_entropy": ("Entropy of found profile (bits per position)", (-0.03, 2.05), None),
}
RECOVERY = ("overlap_rate", "exact_rate")   # where the oracle ceiling matters
TRUTH_LINE = ("found_entropy",)              # oracle row = the true motif itself
NO_STREME = ("exact_rate",)                  # STREME compared on overlap only

BG_ORDER = ["uniform", "at_rich", "markov"]
MODEL_ORDER = ["uniform", "shuffled"]
BG_LABEL = {"uniform": "uniform DNA", "at_rich": "AT-rich DNA",
            "markov": "Markov DNA"}


def load_results(*paths):
    """Read and stack one or more results CSVs.

    Adds found_entropy = mean Shannon entropy per column of the found
    profile (2 - IC/width bits): 0 = every position fully specific,
    2 = pure noise. Per position, so motifs of different widths compare.
    """
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["pseudocount"] = df["pseudocount"].map(
        {True: "on", False: "off", "True": "on", "False": "off"})
    df["bg_model"] = df["bg_model"].fillna("-")
    df["data_type"] = df["data_type"].astype(object).fillna("synthetic")
    df["found_entropy"] = 2 - df["found_ic"] / df["found_width"]
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


def _arrow(higher):
    if higher is None:
        return "closer to the true motif is better"
    return "higher is better" if higher else "lower is better"


def plot_metric(df, metric, out_path, title_note=None):
    ylabel, ylim, higher = METRICS[metric]
    # strength curves are synthetic-only: real-site datasets carry
    # strength 1.0 as a placeholder and get their own figure (plot_real)
    df = df[df["data_type"] == "synthetic"]
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
            elif len(o) and metric in TRUTH_LINE:
                ax.plot(o["strength"], o["mean"], ls=":", color=MUTED, lw=2,
                        label="true motif")
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
    head = f"{ylabel} vs motif strength  ({_arrow(higher)})"
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
    by = ["data_type", "algorithm", "bg_model", "pseudocount", "data_bg", "strength"]
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


REAL_METRICS = ["overlap_rate", "pearson_true", "hamming_norm", "found_entropy"]


def _method_label(r):
    if r["algorithm"] in ALGORITHMS:
        return (f"{ALGORITHMS[r['algorithm']][2]} · pc {r['pseudocount']}"
                f" · bg {r['bg_model']}")
    return {"streme": "STREME", "oracle_scan": "true motif (ceiling)",
            "random": "random sites (chance)"}[r["algorithm"]]


def plot_real(df, out_path, title_note=None):
    """Real JASPAR binding sites in synthetic flanks: one dot per method
    (mean +- 95% CI over all real datasets), one panel per metric.
    Filled marker = pseudocounts on, hollow = off."""
    real = df[df["data_type"] == "real_sites"].copy()
    if real.empty:
        return None
    real["method"] = real.apply(_method_label, axis=1)
    order = (["true motif (ceiling)", "STREME"] +
             [f"{lab} · pc {pc} · bg {bg}"
              for _, (_, _, lab) in ALGORITHMS.items()
              for pc in ("on", "off") for bg in MODEL_ORDER] +
             ["random sites (chance)"])
    order = [m for m in order if m in set(real["method"])]
    y = {m: len(order) - 1 - i for i, m in enumerate(order)}

    fig, axes = plt.subplots(1, len(REAL_METRICS), sharey=True,
                             figsize=(3.3 * len(REAL_METRICS), 0.36 * len(order) + 1.6),
                             facecolor=SURFACE)
    for ax, metric in zip(axes, REAL_METRICS):
        _style(ax)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.grid(axis="y", visible=False)
        s = mean_ci(real, "method", metric)
        for _, row in s.iterrows():
            r0 = real[real["method"] == row["method"]].iloc[0]
            if r0["algorithm"] in ALGORITHMS:
                col, mk, _ = ALGORITHMS[r0["algorithm"]]
                face = col if r0["pseudocount"] == "on" else SURFACE
            else:
                col, mk, face = (INK, "D", INK) if r0["algorithm"] == "streme" \
                    else (MUTED, "o", MUTED)
            ax.errorbar(row["mean"], y[row["method"]], xerr=row["ci"], fmt=mk,
                        color=col, mfc=face, mec=col, mew=1.5, ms=7, capsize=2)
        label, lim, higher = METRICS[metric]
        ax.set_xlim(*lim)
        ax.set_title(f"{label}\n({_arrow(higher)})", fontsize=8.5, color=INK)
    axes[0].set_yticks(list(y.values()))
    axes[0].set_yticklabels(list(y.keys()), fontsize=8.5, color=INK2)
    n = real["dataset"].nunique()
    head = (f"Real JASPAR binding sites ({n} datasets, "
            f"{', '.join(sorted(real['motif_id'].unique()))})")
    if title_note:
        head += f"\n{title_note}"
    fig.suptitle(head, fontsize=11, color=INK)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_path


def make_all(paths, out_dir, title_note=None):
    df = load_results(*paths)
    written = [plot_metric(df, m, os.path.join(out_dir, f"{m}_vs_strength.png"),
                           title_note) for m in METRICS]
    real = plot_real(df, os.path.join(out_dir, "real_data.png"), title_note)
    if real:
        written.append(real)
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
