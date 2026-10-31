"""Final tables + plots from runs/eval/*.json and *_pairs.csv (eval_baselines.py outputs).

  python proj/scripts/make_report.py [--eval runs/eval] [--out runs/report]

Writes tables.md (all variants, fixed RANSAC threshold chosen on vanilla), significance.md (paired bootstrap of
AUC differences: learned selection vs. the strongest same-budget heuristic), and PNG plots.
"""
import argparse, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfeat import metrics  # noqa: E402

AUC_TH = {"hpatches": [1, 3, 5], "megadepth": [5, 10, 20]}
LBL = {"hpatches": ["@1px", "@3px", "@5px"], "megadepth": ["@5°", "@10°", "@20°"]}
ORDER = ["vanilla", "pool_resp", "pool_grid", "pool_rand", "pool_resp2x", "sup", "sup_grid", "rl", "rl_grid", "rl_res", "rl_res_grid"]
NICE = {"vanilla": "vanilla (OpenCV default)", "pool_resp": "pool, top-K response", "pool_grid": "pool, grid (ANMS-like)",
        "pool_rand": "pool, random K", "pool_resp2x": "pool, top-2K (2x budget!)", "sup": "supervised, top-K",
        "sup_grid": "supervised, grid", "rl": "**RIPE-Select (RL), top-K**", "rl_grid": "**RIPE-Select (RL), grid**",
        "rl_res": "RL residual on response, top-K", "rl_res_grid": "RL residual on response, grid"}
COL = {"vanilla": "#4C72B0", "pool_resp": "#8C8C8C", "pool_grid": "#55A868", "pool_rand": "#C44E52",
       "pool_resp2x": "#DD8452", "sup": "#CCB974", "sup_grid": "#B8860B", "rl": "#8172B3", "rl_grid": "#64B5CD",
       "rl_res": "#937860", "rl_res_grid": "#DA8BC3"}


def load(ev):
    R = {}
    for p in sorted(Path(ev).glob("*.json")):
        j = json.load(open(p))
        a = j["args"]
        R[(a["dataset"], a["method"], a.get("tag", ""))] = (j, pd.read_csv(p.with_name(p.stem + "_pairs.csv")))
    return R


def fixed_th(df, ds):
    van = df[df.variant == "vanilla"]
    ths = sorted({float(c.split("@")[1]) for c in df.columns if c.startswith("err@")})
    return max(ths, key=lambda t: np.mean(metrics.error_auc(van[f"err@{t}"].values, AUC_TH[ds])))


def auc(df, v, th, ds):
    return np.array(metrics.error_auc(df[df.variant == v][f"err@{th}"].values, AUC_TH[ds]))


def boot_diff(df, a, b, th, ds, n=2000, seed=0):
    """Paired bootstrap over test pairs of mean-AUC(a) - mean-AUC(b)."""
    ea = df[df.variant == a].sort_values("pair")[f"err@{th}"].values
    eb = df[df.variant == b].sort_values("pair")[f"err@{th}"].values
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(ea), len(ea))
        d.append(np.mean(metrics.error_auc(ea[i], AUC_TH[ds])) - np.mean(metrics.error_auc(eb[i], AUC_TH[ds])))
    obs = np.mean(metrics.error_auc(ea, AUC_TH[ds])) - np.mean(metrics.error_auc(eb, AUC_TH[ds]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return obs, lo, hi


# Published numbers (RIPE, ICCV 2025, Tab. 2: PoseLib, MNN; HPatches AUC @1/3/5 px, MegaDepth-1500 AUC @5/10/20 deg).
# Copied as reported; their protocol uses all 1500 MegaDepth pairs and its own fixed settings -> reference only.
PUBLISHED = {"megadepth": {"SIFT (RIPE paper)": [37.29, 50.61, 61.89], "RIPE (paper)": [55.11, 68.34, 78.03]},
             "hpatches": {"SIFT (RIPE paper)": [29.98, 53.93, 65.01], "RIPE (paper)": [37.93, 58.93, 69.20]}}


def runtime(sub, v, method):
    """ms per image: vanilla extraction | pool extraction (+ policy scoring for learned selection)."""
    if method == "surf" or "t_van" not in sub:
        return np.nan                      # SURF is read from the py3.7 cache -> no meaningful timing
    if v == "vanilla":
        return sub.t_van.mean()
    return sub.t_pool.mean() + (sub.t_pol.mean() if v.startswith(("rl", "sup")) else 0.0)


def matched_metrics(df, ds, th, m):
    """MMA / purity (inlier ratio) / speed / viewpoint robustness, one row per selection variant."""
    has_mma = "mma@1" in df
    unit = "px reprojection" if ds == "hpatches" else "px epipolar"
    out = [f"\n**Matched metrics — {ds} {m.upper()}** (MMA = share of MNN matches with GT error < t {unit}; "
           f"inlier ratio = RANSAC inliers / matches at th = {th}; runtime = ms per image, single thread)\n",
           "| selection | MMA@1 | MMA@3 | MMA@5 | mean MMA 1-10 | inlier ratio | runtime ms |"
           + (" viewpoint AUC @1/3/5 | illum. AUC @1/3/5 |" if ds == "hpatches" else ""),
           "|---|---|---|---|---|---|---|" + ("---|---|" if ds == "hpatches" else "")]
    for v in ORDER:
        sub = df[df.variant == v]
        if sub.empty:
            continue
        if has_mma:
            mm = [sub[f"mma@{t}"].mean() for t in range(1, 11)]
            mma = f"{100 * mm[0]:.1f} | {100 * mm[2]:.1f} | {100 * mm[4]:.1f} | {100 * np.mean(mm):.1f}"
        else:
            mma = "– | – | – | –"
        ir = (sub[f"inl@{th}"] / sub.nm.clip(lower=1)).mean()
        rt = runtime(sub, v, m)
        row = f"| {NICE[v]} | {mma} | {100 * ir:.1f}% | {'–' if np.isnan(rt) else f'{rt:.0f}'} |"
        if ds == "hpatches":
            row += f" {f3(auc(df[df.kind == 'v'], v, th, ds))} | {f3(auc(df[df.kind == 'i'], v, th, ds))} |"
        out.append(row)
    return out


def f3(x):
    return " / ".join(f"{100 * v:.1f}" for v in x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="runs/eval")
    ap.add_argument("--out", default="runs/report")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    R = load(a.eval)
    md = ["# Results (fixed RANSAC threshold chosen on vanilla, applied to every variant)\n",
          "Published reference (copied from the RIPE paper, Tab. 2; full benchmark, their settings — not re-run here):\n",
          "| method | MegaDepth-1500 AUC @5/10/20° | HPatches AUC @1/3/5 px |", "|---|---|---|"]
    md += [f"| {k} | {' / '.join(map(str, PUBLISHED['megadepth'][k]))} | {' / '.join(map(str, PUBLISHED['hpatches'][k]))} |"
           for k in PUBLISHED["megadepth"]]
    sig, summary = \
        ["# Paired bootstrap (95% CI) of mean-AUC difference over test pairs\n",
         "| dataset | method | comparison | Δ mean AUC (pts) | 95% CI | significant |", "|---|---|---|---|---|---|"], []
    for ds in ("hpatches", "megadepth"):
        for m in ("sift", "orb", "surf", "brisk"):
            if (ds, m, "") not in R:
                continue
            j, df = R[(ds, m, "")]
            th = fixed_th(df, ds)
            npairs = df.pair.nunique()
            md += [f"\n## {ds} — {m.upper()}  (K={j['args']['k']}, {npairs} pairs, RANSAC th = {th})\n",
                   f"| selection (same pool, same K, same descriptor, MNN, PoseLib) | AUC {' / '.join(LBL[ds])} | "
                   "match prec | #correct | #inliers |" + (" AUC illum. | AUC viewp. |" if ds == "hpatches" else ""),
                   "|---|---|---|---|---|" + ("---|---|" if ds == "hpatches" else "")]
            seeds = R.get((ds, m, "_seeds"))
            for v in ORDER:
                sub = df[df.variant == v]
                if sub.empty:
                    continue
                A = auc(df, v, th, ds)
                cell = f3(A)
                if v in ("rl", "rl_grid") and seeds is not None:   # mean ± std over seeds 0,1,2
                    sdf = seeds[1]
                    vs = [v.replace("rl", f"rl_s{s}") for s in (1, 2)]
                    allA = np.array([A] + [auc(sdf, x, th, ds) for x in vs if (sdf.variant == x).any()])
                    cell = " / ".join(f"{100 * mu:.1f}±{100 * sd:.1f}" for mu, sd in zip(allA.mean(0), allA.std(0)))
                row = (f"| {NICE[v]} | {cell} | {sub.prec.mean():.3f} | {sub.ncorr.mean():.0f} | "
                       f"{sub[f'inl@{th}'].mean():.0f} |")
                if ds == "hpatches":
                    row += f" {f3(auc(df[df.kind == 'i'], v, th, ds))} | {f3(auc(df[df.kind == 'v'], v, th, ds))} |"
                md.append(row)
                summary.append(dict(dataset=ds, method=m, variant=v, th=th, **{f"auc{t}": x for t, x in zip(AUC_TH[ds], A)},
                                    prec=sub.prec.mean(), ncorr=sub.ncorr.mean()))
            d = j.get("diag") or {}
            if d:
                md.append(f"\nDiagnostics: ROC-AUC(detector response → keypoint has a correct match) = "
                          f"{d['roc_auc_response']:.3f}; share of the pool's correct matches kept by top-K response = "
                          f"{100 * d['frac_correct_in_topK']:.0f}%; correct-match rate per response quartile = "
                          f"{', '.join(f'{100 * x:.1f}%' for x in d['useful_rate_by_quartile'])}.")
            md += matched_metrics(df, ds, th, m)
            have = set(df.variant)
            for x, y in (("rl", "vanilla"), ("rl", "pool_grid"), ("rl", "pool_resp"), ("rl", "sup"), ("rl_grid", "vanilla"),
                         ("rl_grid", "pool_grid"), ("rl_grid", "sup_grid"), ("sup", "vanilla"), ("pool_grid", "vanilla"),
                         ("rl_res", "pool_resp")):
                if x in have and y in have:
                    o, lo, hi = boot_diff(df, x, y, th, ds)
                    sig.append(f"| {ds} | {m.upper()} | {x} − {y} | {100 * o:+.2f} | [{100 * lo:+.2f}, {100 * hi:+.2f}] | "
                               f"{'yes' if lo > 0 or hi < 0 else 'no'} |")
            # bar plot
            vs = [v for v in ORDER if v in have]
            fig, ax = plt.subplots(figsize=(8, 3.8))
            x = np.arange(3); w = 0.85 / len(vs)
            for i, v in enumerate(vs):
                ax.bar(x + i * w, 100 * auc(df, v, th, ds), w, color=COL[v], label=NICE[v].replace("*", ""),
                       hatch="//" if v == "pool_resp2x" else None, edgecolor="white")
            ax.set_xticks(x + w * (len(vs) - 1) / 2, LBL[ds]); ax.set_ylabel("AUC (%)")
            ax.set_title(f"{ds} — {m.upper()} (same K = {j['args']['k']}, th = {th})"); ax.grid(axis="y", alpha=.3)
            ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(out / f"{ds}_{m}_auc.png", dpi=130); plt.close(fig)
    (out / "tables.md").write_text("\n".join(md), encoding="utf-8")
    (out / "significance.md").write_text("\n".join(sig), encoding="utf-8")
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)

    # matchability by response quartile (all methods/datasets)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for (ds, m, tag), (j, _) in R.items():
        d = j.get("diag")
        if tag == "" and d:
            ax.plot(["Q1 (strongest)", "Q2", "Q3", "Q4"], 100 * np.array(d["useful_rate_by_quartile"]), "o-",
                    label=f"{ds} {m.upper()} (ROC-AUC {d['roc_auc_response']:.2f})")
    ax.set_ylabel("% candidates with a correct match"); ax.set_xlabel("detector response quartile within the pool")
    ax.grid(alpha=.3); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(out / "matchability_by_response_quartile.png", dpi=130); plt.close(fig)
    print("\n".join(md)); print(); print("\n".join(sig))


if __name__ == "__main__":
    main()

