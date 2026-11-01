"""
Statistical analysis of a results CSV from train_scorer.py / train_scorer_grid.py.
Unit of analysis = SEQUENCE (the 5 pairs of a sequence are averaged first), because pairs of one sequence are not independent.

Gives, for every comparison:  relative gain, 95% bootstrap CI, win rate, paired t-test, Wilcoxon, Holm-corrected p.
Also: does RL reach +50% over BRISK (bootstrap probability), split results, per-fold consistency.

    python analyze_results.py --csv scorer_results.csv
    python analyze_results.py --csv grid_results.csv --rl rl_tuned --sup supervised_tuned
"""
import argparse, math
import numpy as np
import pandas as pd

try:
    from scipy import stats as sps
except Exception:
    sps = None


def holm(ps):
    ps = np.asarray(ps, dtype=float)
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def ttest_p(x, y):
    if sps is not None:
        return float(sps.ttest_rel(x, y).pvalue)
    d = x - y
    se = d.std(ddof=1) / math.sqrt(len(d))
    return math.erfc(abs(d.mean() / (se + 1e-12)) / math.sqrt(2))   # normal approximation


def wilcoxon_p(x, y):
    if sps is None:
        return float("nan")
    try:
        return float(sps.wilcoxon(x - y).pvalue)
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="scorer_results.csv")
    ap.add_argument("--rl", default="rl")
    ap.add_argument("--sup", default="supervised")
    ap.add_argument("--resp", default="response")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out", default="analysis.txt")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    s = df.groupby(["seq", "method"]).n_correct.mean().unstack()      # sequence x method
    kind = df.groupby("seq").kind.first()
    rng = np.random.default_rng(0)
    L = []

    def P(x=""):
        print(x)
        L.append(str(x))

    P(f"{len(s)} sequences, methods: {list(s.columns)}")

    # ---------------- overall table
    rows = []
    for m in s.columns:
        sub = df[df.method == m]
        rows.append(dict(method=m, n_correct=s[m].mean(), n_matches=sub.n_matches.mean(), precision=sub.precision.mean()))
    summ = pd.DataFrame(rows).set_index("method")
    if "BRISK" in summ.index:
        summ["vs_BRISK_%"] = (summ.n_correct / summ.loc["BRISK", "n_correct"] - 1) * 100
    if a.resp in summ.index:
        summ["vs_response_%"] = (summ.n_correct / summ.loc[a.resp, "n_correct"] - 1) * 100
    P("\n=== overall (mean over sequences) ===")
    P(summ.round(3).sort_values("n_correct").to_string())

    # ---------------- paired comparisons
    cand = [(a.rl, a.sup), (a.rl, a.resp), (a.sup, a.resp), (a.rl, "RootSIFT (default)"),
            (a.rl, "BRISK"), (a.sup, "BRISK"), ("rl_scratch", a.rl)]
    comps = [(x, y) for x, y in cand if x in s.columns and y in s.columns]
    res = []
    for x, y in comps:
        sub = s[[x, y]].dropna()
        xv, yv = sub[x].values, sub[y].values
        n = len(xv)
        idx = rng.integers(0, n, size=(a.boot, n))
        rel = xv[idx].mean(1) / yv[idx].mean(1) - 1
        lo, hi = np.percentile(rel, [2.5, 97.5]) * 100
        res.append(dict(A=x, B=y, n_seq=n, mean_A=xv.mean(), mean_B=yv.mean(),
                        gain_pct=(xv.mean() / yv.mean() - 1) * 100, ci_lo=lo, ci_hi=hi,
                        win_rate=float((xv > yv).mean()), p_ttest=ttest_p(xv, yv), p_wilcoxon=wilcoxon_p(xv, yv)))
    if res:
        r = pd.DataFrame(res)
        r["p_holm"] = holm(r.p_ttest.values)
        P(f"\n=== paired comparisons over sequences (95% bootstrap CI of the relative gain; Holm over {len(r)} tests) ===")
        P(r.round(4).to_string(index=False))
        P("\nVerdicts:")
        for _, q in r.iterrows():
            if q.p_holm < 0.05 and q.ci_lo > 0:
                v = "significantly BETTER"
            elif q.p_holm < 0.05 and q.ci_hi < 0:
                v = "significantly WORSE"
            else:
                v = "NO significant difference"
            P(f"  {q.A} vs {q.B}: {q.gain_pct:+.1f}%  [{q.ci_lo:+.1f}, {q.ci_hi:+.1f}]  -> {v}")

    # ---------------- the +50% over BRISK target
    if "BRISK" in s.columns and a.rl in s.columns:
        sub = s[["BRISK", a.rl]].dropna()
        xv, bv = sub[a.rl].values, sub["BRISK"].values
        idx = rng.integers(0, len(xv), size=(a.boot, len(xv)))
        ratio = xv[idx].mean(1) / bv[idx].mean(1)
        P("\n=== target: +50% correct matches over BRISK ===")
        P(f"  BRISK mean = {bv.mean():.1f}, needed for +50% = {1.5 * bv.mean():.1f}, {a.rl} mean = {xv.mean():.1f}")
        P(f"  gain = {(xv.mean() / bv.mean() - 1) * 100:+.1f}%   bootstrap P(gain >= +50%) = {(ratio >= 1.5).mean():.2f}")

    # ---------------- split by kind
    t = s.join(kind)
    P("\n=== mean correct matches by split ===")
    tab = t.groupby("kind").mean(numeric_only=True).T
    if "BRISK" in tab.index:
        for k in tab.columns:
            tab[f"{k}_vs_BRISK_%"] = (tab[k] / tab.loc["BRISK", k] - 1) * 100
    P(tab.round(1).to_string())

    # ---------------- fold consistency
    if "fold" in df.columns and (df.fold >= 0).any():
        fold_of = df[df.fold >= 0].groupby("seq").fold.first()
        tf = s.join(fold_of, how="inner")
        keep = [c for c in [a.resp, a.sup, a.rl, "rl_scratch"] if c in tf.columns]
        pf = tf.groupby("fold")[keep].mean()
        P("\n=== per fold: mean correct matches ===")
        P(pf.round(1).to_string())
        if a.rl in pf and a.sup in pf:
            P(f"  {a.rl} beats {a.sup} in {int((pf[a.rl] > pf[a.sup]).sum())}/{len(pf)} folds")
        if a.rl in pf and a.resp in pf:
            P(f"  {a.rl} beats {a.resp} in {int((pf[a.rl] > pf[a.resp]).sum())}/{len(pf)} folds")

    with open(a.out, "w") as f:
        f.write("\n".join(L))
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
