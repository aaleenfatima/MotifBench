"""Section 7: statistical tests for the project's claims.

Usage (from the repo root):
    python -m evaluation.stats results/results.csv
    python -m evaluation.stats results/results.csv --metric pearson_true --out results/stats

Why PAIRED tests: every configuration runs on the same datasets, so we
compare A and B dataset-by-dataset ("on this exact dataset, did pseudocounts
help?"). That removes the huge dataset-to-dataset variation (strong vs weak
motif) and is far more sensitive than comparing two averages.

Why Wilcoxon signed-rank and not a t-test: the metrics are bounded (0..1),
skewed and often tied, so we avoid assuming normality.

Why Holm correction: we test many claims at once; at p < 0.05 one in twenty
would look "significant" by luck. Holm adjusts the p-values for that while
being less strict than Bonferroni.
"""
import argparse, os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from evaluation.plots import load_results

ALGS = ("greedy", "randomized", "gibbs")
# what identifies "the same run" apart from the factor being compared
PAIR_KEYS = ["dataset", "algorithm", "bg_model", "pseudocount", "seed"]


def paired_test(df, metric, factor, a, b, where=None):
    """Wilcoxon signed-rank test of `metric` for factor == b vs factor == a,
    paired on everything else that identifies a run.

    Returns a dict: n pairs, mean/median of (b - a), wins/losses/ties for b,
    the p-value, and the rank-biserial effect size r (-1..1; > 0 means b is
    higher; |r| ~0.1 small, ~0.3 medium, >= 0.5 large).
    """
    d = df if where is None else df.query(where)
    keys = [k for k in PAIR_KEYS if k != factor]
    wide = d.pivot_table(index=keys, columns=factor, values=metric,
                         aggfunc="mean", dropna=False)
    if a not in wide or b not in wide:
        return dict(n=0, p=np.nan)
    pairs = wide[[a, b]].dropna()
    diff = (pairs[b] - pairs[a]).to_numpy()
    out = dict(n=len(diff),
               mean_a=pairs[a].mean(), mean_b=pairs[b].mean(),
               mean_diff=diff.mean() if len(diff) else np.nan,
               median_diff=np.median(diff) if len(diff) else np.nan,
               wins=int((diff > 0).sum()), losses=int((diff < 0).sum()),
               ties=int((diff == 0).sum()))
    nz = diff[diff != 0]
    if len(nz) == 0:                       # identical everywhere: no evidence
        out.update(stat=np.nan, p=1.0, effect_r=0.0)
        return out
    res = wilcoxon(nz)
    # matched-pairs rank-biserial correlation
    ranks = pd.Series(np.abs(nz)).rank().to_numpy()
    r = (ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()
    out.update(stat=float(res.statistic), p=float(res.pvalue), effect_r=float(r))
    return out


def vs_reference(df, metric, alg, ref="streme", where=None):
    """Paired test of algorithm `alg` vs a reference method (STREME) on the
    same datasets. `where` picks the algorithm's configuration."""
    a = df[df["algorithm"] == alg]
    if where:
        a = a.query(where)
    a = a.groupby("dataset")[metric].mean()
    r = df[df["algorithm"] == ref].groupby("dataset")[metric].mean()
    both = pd.DataFrame({"alg": a, "ref": r}).dropna()
    tmp = both.reset_index().melt(id_vars="dataset", var_name="method",
                                  value_name=metric)
    tmp["seed"] = 0
    tmp["algorithm"] = tmp["bg_model"] = tmp["pseudocount"] = "-"
    return paired_test(tmp, metric, "method", "ref", "alg")


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (NaN stays NaN)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.argsort(p[ok])
    m = ok.sum()
    adj = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[ok][idx]))
    tmp = np.empty(m)
    tmp[idx] = adj
    out[ok] = tmp
    return out


def standard_claims(df, metric="overlap_rate"):
    """Run the project's central comparisons and return one tidy table.

    1. pseudocounts on vs off            - per algorithm
    2. shuffled vs uniform scoring bg    - per algorithm and per data_bg
    3. each algorithm vs STREME          - best config (pc on, shuffled bg)
    """
    rows = []
    for alg in ALGS:
        rows.append(dict(claim=f"{alg}: pseudocounts on > off",
                         **paired_test(df, metric, "pseudocount", "off", "on",
                                       where=f"algorithm == '{alg}'")))
    for alg in ALGS:
        for bg in sorted(df["data_bg"].dropna().unique()):
            rows.append(dict(claim=f"{alg}: shuffled bg > uniform bg  [{bg} DNA]",
                             **paired_test(df, metric, "bg_model", "uniform",
                                           "shuffled",
                                           where=f"algorithm == '{alg}' and data_bg == '{bg}'")))
    if (df["algorithm"] == "streme").any():
        for alg in ALGS:
            rows.append(dict(claim=f"{alg} (pc on, shuffled) vs STREME",
                             **vs_reference(df, metric, alg,
                                            where="pseudocount == 'on' and bg_model == 'shuffled'")))
    t = pd.DataFrame(rows)
    t = t[t["n"] > 0].reset_index(drop=True)
    t["p_holm"] = holm(t["p"])
    t["significant"] = t["p_holm"] < 0.05
    t.insert(1, "metric", metric)
    return t


def streme_gap(df, metric="overlap_rate"):
    """How much of the gap to STREME do pseudocounts + background close?

    gap(config)   = mean(STREME) - mean(algorithm in that config)
    closed        = (gap(plain) - gap(best)) / gap(plain)
    plain = pseudocounts off, uniform bg (textbook); best = on + shuffled.
    """
    s = df.loc[df["algorithm"] == "streme", metric].mean()
    rows = []
    for alg in ALGS:
        a = df[df["algorithm"] == alg]
        if a.empty:
            continue
        cfg = a.groupby(["pseudocount", "bg_model"])[metric].mean()
        plain = cfg.get(("off", "uniform"), np.nan)
        best = cfg.get(("on", "shuffled"), np.nan)
        g0, g1 = s - plain, s - best
        rows.append(dict(algorithm=alg, metric=metric, streme=s, plain=plain,
                         best=best, gap_plain=g0, gap_best=g1,
                         fraction_closed=(g0 - g1) / g0 if g0 > 0 else np.nan))
    return pd.DataFrame(rows)


def reliability(df, metric="overlap_rate", level=0.8):
    """Lowest motif strength at which each method's mean `metric` reaches
    `level`, per data background: "when is each algorithm reliable?"."""
    rows = []
    for (alg, pc, bgm, dbg), g in df.groupby(
            ["algorithm", "pseudocount", "bg_model", "data_bg"], dropna=False):
        m = g.groupby("strength")[metric].mean()
        ok = m[m >= level]
        rows.append(dict(algorithm=alg, pseudocount=pc, bg_model=bgm, data_bg=dbg,
                         reliable_from_strength=ok.index.min() if len(ok) else np.nan))
    return pd.DataFrame(rows)


def _sentence(row):
    p = row["p_holm"]
    verdict = "SIGNIFICANT" if row["significant"] else "not significant"
    return (f"  {row['claim']:<52} diff {row['mean_diff']:+.3f}  "
            f"(wins {row['wins']}/{row['n']})  r={row['effect_r']:+.2f}  "
            f"p_holm={p:.2g}  -> {verdict}")


def main():
    ap = argparse.ArgumentParser(description="Statistical tests for MotifBench")
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--metric", default="overlap_rate")
    ap.add_argument("--out", default=os.path.join("results", "stats"))
    a = ap.parse_args()
    df = load_results(*a.csv)

    claims = standard_claims(df, a.metric)
    gap = streme_gap(df, a.metric)
    rel = reliability(df, a.metric)
    os.makedirs(a.out, exist_ok=True)
    claims.round(4).to_csv(os.path.join(a.out, f"claims_{a.metric}.csv"), index=False)
    gap.round(4).to_csv(os.path.join(a.out, f"streme_gap_{a.metric}.csv"), index=False)
    rel.to_csv(os.path.join(a.out, f"reliability_{a.metric}.csv"), index=False)

    print(f"\nPaired Wilcoxon tests on {a.metric} (Holm-corrected):\n")
    for _, row in claims.iterrows():
        print(_sentence(row))
    if len(gap):
        print("\nGap to STREME closed by pseudocounts + shuffled background:\n")
        for _, g in gap.iterrows():
            what = ("already at or above STREME - no gap to close"
                    if g["gap_plain"] <= 0 else
                    f"{g['fraction_closed']:.0%} of the gap closed")
            print(f"  {g['algorithm']:<11} plain {g['plain']:.3f} -> best {g['best']:.3f}"
                  f"  (STREME {g['streme']:.3f}): {what}")
    print(f"\nwrote CSVs to {a.out}")


if __name__ == "__main__":
    main()
