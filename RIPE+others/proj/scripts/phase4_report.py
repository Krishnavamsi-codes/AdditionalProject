"""Phase 4 tables + plots from results/<dataset>_<method>_k<K>.json (written by eval_baselines.py).

Usage: python proj/scripts/phase4_report.py [--res results] [--out results/phase4]
"""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VARIANTS = ["vanilla", "pool_resp", "pool_grid", "pool_rand", "pool_resp2x"]
AUC_LBL = {"hpatches": ["@1px", "@3px", "@5px"], "megadepth": ["@5°", "@10°", "@20°"]}
COLORS = {"vanilla": "#4C72B0", "pool_resp": "#8C8C8C", "pool_grid": "#55A868",
          "pool_rand": "#C44E52", "pool_resp2x": "#DD8452"}


def load(res):
    out = {}
    for p in sorted(Path(res).glob("*_k*.json")):
        ds, method = p.stem.split("_")[:2]
        out[(ds, method)] = json.load(open(p))
    return out


def pct(x):
    return " / ".join(f"{100 * v:.2f}" for v in x)


def tables(R):
    lines = []
    for (ds, method), r in R.items():
        S = r["summary"]
        th = S["vanilla"]["th_fixed"]
        a = r["args"]
        n = "all" if not a.get("limit") else a["limit"]
        lines += [f"\n### {ds} / {method.upper()} (K={a['k']}, pairs={n}, RANSAC th fixed on vanilla = {th})\n",
                  f"| variant | AUC {' / '.join(AUC_LBL[ds])} (fixed th) | AUC (best th) | best th | #kpts | #matches | match prec | #correct | RANSAC inl |"
                  + (" AUC i (fixed) | AUC v (fixed) |" if ds == "hpatches" else ""),
                  "|---|---|---|---|---|---|---|---|---|" + ("---|---|" if ds == "hpatches" else "")]
        for v in VARIANTS:
            if v not in S:
                continue
            s = S[v]
            row = (f"| {v} | {pct(s['auc_fixed'])} | {pct(s['auc_best'])} | {s['th_best']} | {s['n_kpts']:.0f} | "
                   f"{s['n_matches']:.0f} | {s['match_prec']:.3f} | {s['n_correct']:.0f} | {s['ransac_inl']:.0f} |")
            if ds == "hpatches":
                row += f" {pct(s['auc_fixed_i'])} | {pct(s['auc_fixed_v'])} |"
            lines.append(row)
        d = r.get("diag") or {}
        if d:
            q = d["useful_rate_by_quartile"]
            lines.append(f"\nDiagnostics (candidate pool, 4K): ROC-AUC(response → correct MNN match) = "
                         f"{d['roc_auc_response']:.3f}, ROC-AUC(size) = {d['roc_auc_size']:.3f}; "
                         f"correct-match rate by response quartile (top→bottom) = "
                         f"{', '.join(f'{100 * x:.1f}%' for x in q)}; share of pool's correct matches kept by "
                         f"top-K response = {100 * d['frac_correct_in_topK']:.1f}%; pool correct matches/pair = "
                         f"{d['pool_correct']:.0f}.")
    return "\n".join(lines)


def bars(R, out):
    for ds in ("hpatches", "megadepth"):
        methods = [m for (d, m) in R if d == ds]
        if not methods:
            continue
        fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 4), squeeze=False)
        for ax, m in zip(axes[0], methods):
            S = R[(ds, m)]["summary"]
            vs = [v for v in VARIANTS if v in S]
            x = np.arange(3); w = 0.8 / len(vs)
            for i, v in enumerate(vs):
                ax.bar(x + i * w, 100 * np.array(S[v]["auc_fixed"]), w, label=v, color=COLORS[v],
                       hatch="//" if v == "pool_resp2x" else None, edgecolor="white")
            ax.set_xticks(x + w * (len(vs) - 1) / 2, AUC_LBL[ds])
            ax.set_title(f"{ds} — {m.upper()} (th={S['vanilla']['th_fixed']})")
            ax.set_ylabel("AUC (%)"); ax.grid(axis="y", alpha=.3)
        axes[0][0].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / f"{ds}_auc_bars.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for (ds, m), r in R.items():
        d = r.get("diag")
        if d:
            ax.plot(["Q1 (top)", "Q2", "Q3", "Q4"], 100 * np.array(d["useful_rate_by_quartile"]), "o-",
                    label=f"{ds} {m.upper()} (ROC-AUC {d['roc_auc_response']:.2f})")
    ax.set_ylabel("% of pool keypoints with a correct MNN match"); ax.set_xlabel("detector-response quartile")
    ax.set_title("Is detector response a good matchability predictor?"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "matchability_by_response_quartile.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results")
    ap.add_argument("--out", default="results/phase4")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    R = load(a.res)
    md = tables(R)
    (out / "phase4_tables.md").write_text(md, encoding="utf-8")
    bars(R, out)
    print(md)
