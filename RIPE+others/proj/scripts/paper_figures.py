"""All figures of the ORB/BRISK paper (vector PDF unless the figure is a photo), one palette, 8 pt fonts.

  python proj/scripts/paper_figures.py [--out paper/figures] [--only teaser,forest,...]

Inputs: runs/eval/*_pairs.csv and *.json (eval_baselines.py), paper/figures/stats.json (paper_stats.py),
paper/figures/timing.json (timing_benchmark.py), runs/report/paper_analysis_*.json (paper_analysis.py),
and the seed-0 policies in runs/train/<method>_rl_s0/best.npz for the qualitative figures.
"""
import argparse, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfeat import datasets, extract, match, metrics, policy  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "runs" / "eval"
MD_ROOT = ROOT / "data" / "megadepth1500" / "megadepth1500"
TH = 1.0                                   # MegaDepth RANSAC threshold used in all tables

# One palette for the whole paper (colour-blind safe; mirrored in paper/main.tex)
C = {"vanilla": "#3A6FB0", "pool_resp": "#A3A3A3", "pool_grid": "#2E9E77", "sup": "#5E5E5E", "rl": "#6F4E9C"}
LABEL = {"vanilla": "Vanilla OpenCV", "pool_resp": "Pool, top-$K$ response", "pool_grid": "Pool, grid",
         "sup": "Supervised", "rl": "RIPE-Select (ours)"}
LS = {"vanilla": "-", "pool_resp": ":", "pool_grid": "--", "sup": "-.", "rl": "-"}
GOOD, BAD, KP = "#1FA35B", "#E05A1B", "#F7D23E"
FULL, COL = 6.875, 3.25                    # CVPR text and column width (in)
MAX_LINES = 70                             # inlier lines drawn per pair panel

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.2, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "legend.frameon": False, "savefig.dpi": 300})


# ------------------------------------------------------------------ helpers

def errs(df, v, th=TH):
    return df[df.variant == v].sort_values("pair")[f"err@{th}"].values


def seed_errs(method, df):
    """Pose errors of RL seeds 0-4 on MegaDepth: seed 0 is in df, 1-2 in _seeds, 3-4 in _seeds34."""
    out = [errs(df, "rl")]
    for tag, ss in (("_seeds", (1, 2)), ("_seeds34", (3, 4))):
        d = pd.read_csv(EVAL / f"megadepth_{method}_k2048{tag}_pairs.csv")
        out += [errs(d, f"rl_s{s}") for s in ss]
    return out


def auc(e, ths):
    return 100 * np.array(metrics.error_auc(e, list(ths)))


def save(fig, out, name):
    fig.savefig(out / name, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print("  ", name)


def md_pairs(need):
    """Only the requested pair indices of the 500-pair subset (+ first pairs of each scene for negatives)."""
    out = {}
    for n, s in enumerate(datasets.megadepth1500_pairs(MD_ROOT, 1200, 500)):
        if n in need or len(out) < 4:
            out[n] = s
        if n >= max(need):
            break
    return out


_FEATS = {}


def features(method, name, img, k=2048):
    """vanilla / pool / pool_grid / rl selections for one image (seed-0 policy), cached."""
    key = (method, name)
    if key not in _FEATS:
        pol = policy.NumpyPolicy(ROOT / "runs" / "train" / f"{method}_rl_s0" / "best.npz")
        van = extract.detect_describe(img, method, k)
        pool = extract.select(extract.detect_describe(img, method, k, pool=True), 4 * k, "response")
        sc = pol(policy.candidate_features(pool, img, method))
        _FEATS[key] = {"vanilla": van, "pool": pool,
                       "pool_grid": extract.select(pool, k, "grid", img_shape=img.shape),
                       "rl": extract.select(pool, k, "score", scores=sc)}
    return _FEATS[key]


def match_pair(method, f0, f1, s):
    m = match.mnn(f0["desc"], f1["desc"], binary=extract.is_binary(method))
    p0, p1 = f0["xy"][m[:, 0]], f1["xy"][m[:, 1]]
    est, inl = metrics.estimate_pose(p0, p1, s["K0"], s["K1"], s["wh0"], s["wh1"], TH)
    err = metrics.pose_error(est, s["R"], s["t"])
    good = metrics.sym_epipolar_px(p0, p1, s["K0"], s["K1"], s["R"], s["t"]) < 2 if len(m) else np.zeros(0, bool)
    return p0[inl], p1[inl], good[inl], err


def stacked(img0, img1, gap=12):
    """img0 above img1, both centred; returns canvas and offsets."""
    h0, w0 = img0.shape; h1, w1 = img1.shape
    W = max(w0, w1)
    cv = np.full((h0 + gap + h1, W), 255, np.uint8)
    o0 = ((W - w0) // 2, 0); o1 = ((W - w1) // 2, h0 + gap)
    cv[:h0, o0[0]:o0[0] + w0] = img0
    cv[o1[1]:, o1[0]:o1[0] + w1] = img1
    return cv, o0, o1


def draw_pair(ax, method, s, variant, title_col):
    F0 = features(method, s["name0"], s["img0"])[variant]
    F1 = features(method, s["name1"], s["img1"])[variant]
    q0, q1, good, err = match_pair(method, F0, F1, s)
    cv, o0, o1 = stacked(s["img0"], s["img1"])
    ax.imshow(cv, cmap="gray", vmin=0, vmax=255, alpha=0.85, interpolation="lanczos")
    for f, o in ((F0, o0), (F1, o1)):
        ax.scatter(f["xy"][:, 0] + o[0], f["xy"][:, 1] + o[1], s=0.6, c=KP, lw=0, alpha=0.8, rasterized=True)
    show = np.random.default_rng(0).permutation(len(q0))[:MAX_LINES]    # readable: a fixed random subset
    for a, b, g in sorted(zip(q0[show], q1[show], good[show]), key=lambda z: z[2]):   # correct lines on top
        ax.plot([a[0] + o0[0], b[0] + o1[0]], [a[1] + o0[1], b[1] + o1[1]], lw=0.55,
                c=GOOD if g else BAD, alpha=0.9, solid_capstyle="round")
    ax.set_axis_off()
    ax.text(0.02, 0.015, f"pose error {err:.1f}°\n{len(q0)} inliers, {int(good.sum())} correct",
            transform=ax.transAxes, fontsize=7, va="bottom", ha="left", color="black",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.88))
    return err, len(q0), int(good.sum())


def summary_bars(ax, method):
    """Benchmark-level mean pose AUC on MegaDepth (RL: 5-seed mean +- std) next to the example pair."""
    df = pd.read_csv(EVAL / f"megadepth_{method}_k2048_pairs.csv")
    vs = ("vanilla", "pool_grid", "rl")
    vals, err = [], []
    for v in vs:
        if v == "rl":
            a = [auc(e, [5, 10, 20]).mean() for e in seed_errs(method, df)]
            vals.append(np.mean(a)); err.append(np.std(a, ddof=1))
        else:
            vals.append(auc(errs(df, v), [5, 10, 20]).mean()); err.append(0)
    y = np.arange(3)[::-1]
    ax.barh(y, vals, color=[C[v] for v in vs], height=0.62)
    ax.errorbar(vals[2], y[2], xerr=err[2], color="black", lw=0.8, capsize=1.5)
    for yi, x in zip(y, vals):
        ax.text(x - 2.2, yi, f"{x:.1f}", va="center", ha="right", fontsize=7.5, color="white", fontweight="bold")
    ax.set_yticks(y, ["Vanilla", "Grid", "Ours"])
    ax.set_xlim(0, max(vals) * 1.12)
    ax.set_xlabel("mean pose AUC (%)", labelpad=1)
    ax.set_title(f"MegaDepth, 500 pairs", fontsize=8.5, fontweight="bold", pad=3)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def pair_figure(out, name, method, pair_idx, pairs, width=FULL, summary=False):
    s = pairs[pair_idx]
    h0, w0 = s["img0"].shape; h1, w1 = s["img1"].shape
    aspect = (h0 + h1 + 12) / max(w0, w1)
    ratios = [1, 1, 1] + ([0.62] if summary else [])
    panel_w = width / sum(ratios)
    fig, axes = plt.subplots(1, len(ratios), figsize=(width, panel_w * aspect + 0.25),
                             gridspec_kw=dict(width_ratios=ratios))
    if summary:
        summary_bars(axes[3], method)
    info = {}
    for ax, v in zip(axes, ("vanilla", "pool_grid", "rl")):
        info[v] = draw_pair(ax, method, s, v, C[v])
        ax.set_title(LABEL[v] if v != "vanilla" else f"Vanilla {method.upper()}", color=C[v], fontsize=8.5,
                     fontweight="bold", pad=3)
    fig.subplots_adjust(wspace=0.03, left=0, right=1, top=0.94, bottom=0)
    if summary:   # the bar panel needs room for its axis labels
        pos = axes[3].get_position()
        axes[3].set_position([pos.x0 + 0.045, pos.y0 + 0.16, pos.width - 0.05, pos.height - 0.32])
    save(fig, out, name)
    return info


# ------------------------------------------------------------------ figures

def fig_teaser(out, stats, pairs):
    p = stats["picks"]["orb"]["representative"]["pair"]
    return {"teaser": pair_figure(out, "teaser.pdf", "orb", p, pairs, summary=True)}


def fig_supp_qualitative(out, stats, pairs):
    return {"supp_brisk": pair_figure(out, "supp_brisk_rep.pdf", "brisk", stats["picks"]["brisk"]["representative"]["pair"], pairs),
            "supp_fail": pair_figure(out, "supp_orb_fail.pdf", "orb", stats["picks"]["orb"]["failure"]["pair"], pairs)}


def fig_auc(out):
    ths = np.arange(1, 31)
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.55))
    for ax, m in zip(axes, ("orb", "brisk")):
        df = pd.read_csv(EVAL / f"megadepth_{m}_k2048_pairs.csv")
        for v in ("pool_resp", "vanilla", "sup", "pool_grid", "rl"):
            if v == "rl":
                cs = np.stack([auc(e, ths) for e in seed_errs(m, df)])
                ax.fill_between(ths, cs.min(0), cs.max(0), color=C[v], alpha=.22, lw=0)
                ax.plot(ths, cs.mean(0), color=C[v], ls=LS[v], label=LABEL[v], lw=1.6, zorder=5)
            else:
                ax.plot(ths, auc(errs(df, v), ths), color=C[v], ls=LS[v], label=LABEL[v])
        ax.set_title(m.upper(), pad=2, fontweight="bold")
        ax.set_xlabel("threshold (°)", labelpad=1)
        ax.set_xlim(1, 30); ax.set_xticks([1, 10, 20, 30])
        ax.grid(alpha=.25, lw=.4)
    axes[0].set_ylabel("pose AUC (%)")
    h, l = axes[0].get_legend_handles_labels()
    order = [4, 1, 3, 2, 0]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.27), handlelength=1.8, columnspacing=0.9)
    fig.subplots_adjust(wspace=0.3)
    save(fig, out, "auc_curves.pdf")


def fig_forest(out, stats):
    comps = [("rl-vanilla", "Ours − vanilla"), ("rl-pool_resp", "Ours − pool, top-$K$ resp."),
             ("rl-sup", "Ours − supervised"), ("rl-pool_grid", "Ours − pool, grid"),
             ("pool_grid-vanilla", "Pool, grid − vanilla")]
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.45), sharey=True)
    y = np.arange(len(comps))[::-1]
    for ax, m in zip(axes, ("orb", "brisk")):
        for ds, mk, off, fill in (("megadepth", "o", 0.14, True), ("hpatches", "s", -0.14, False)):
            B = stats["bootstrap"][f"{ds}_{m}"]
            for yi, (k, _) in zip(y, comps):
                d, lo, hi = B[k]
                sig = lo > 0 or hi < 0
                col = C["pool_grid"] if k.startswith("pool_grid") else C["rl"]
                if not sig:
                    col = "#9A9A9A"
                ax.plot([lo, hi], [yi + off] * 2, color=col, lw=1.1, solid_capstyle="round")
                ax.plot(d, yi + off, marker=mk, ms=3.6, color=col, mfc=col if fill else "white", mew=0.9)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_title(m.upper(), pad=2, fontweight="bold")
        ax.set_xlabel("Δ mean AUC (points)", labelpad=1)
        ax.grid(axis="x", alpha=.25, lw=.4)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, [c[1] for c in comps])
    axes[0].set_ylim(-0.6, len(comps) - 0.4)
    leg = [Line2D([], [], marker="o", color="#444", lw=0, ms=3.6, label="MegaDepth"),
           Line2D([], [], marker="s", color="#444", mfc="white", lw=0, ms=3.6, label="HPatches"),
           Line2D([], [], color="#9A9A9A", lw=1.1, label="CI includes 0")]
    fig.legend(handles=leg, loc="upper center", ncol=3, bbox_to_anchor=(0.62, 1.18), columnspacing=1.0)
    fig.subplots_adjust(wspace=0.08)
    save(fig, out, "forest.pdf")


def fig_pareto(out):
    T = json.loads((ROOT / "paper" / "figures" / "timing.json").read_text())["megadepth"]
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.5))
    for ax, m in zip(axes, ("orb", "brisk")):
        df = pd.read_csv(EVAL / f"megadepth_{m}_k2048_pairs.csv")
        t = T[m]
        rt = {"vanilla": t["vanilla"], "pool_resp": t["pool"], "pool_grid": t["pool"], "sup": t["rl_total"],
              "rl": t["rl_total"]}
        for v in ("vanilla", "pool_resp", "pool_grid", "sup", "rl"):
            if v == "rl":
                a = [auc(e, [5, 10, 20]).mean() for e in seed_errs(m, df)]
                ax.errorbar(rt[v], np.mean(a), yerr=np.std(a, ddof=1), fmt="*", ms=7, color=C[v], capsize=1.5,
                            elinewidth=0.8, zorder=5)
            else:
                ax.plot(rt[v], auc(errs(df, v), [5, 10, 20]).mean(), marker="o", ms=4.2, color=C[v], lw=0,
                        mfc=C[v] if v != "pool_resp" else "white", mew=1.0)
        ax.set_xscale("log")
        ax.set_title(m.upper(), pad=2, fontweight="bold")
        ax.set_xlabel("ms / image (log)", labelpad=1)
        ax.grid(alpha=.25, lw=.4, which="both")
        lo, hi = ax.get_ylim(); ax.set_ylim(lo - 1, hi + 1)
    axes[0].set_ylabel("mean pose AUC (%)")
    leg = [Line2D([], [], marker="*" if v == "rl" else "o", ms=6 if v == "rl" else 4, lw=0, color=C[v],
                  mfc=C[v] if v != "pool_resp" else "white", label=LABEL[v]) for v in C]
    fig.legend(handles=leg, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.27), columnspacing=0.8,
               handletextpad=0.2)
    fig.subplots_adjust(wspace=0.32)
    save(fig, out, "pareto.pdf")


def fig_coverage(out, stats, pairs):
    import cv2
    s = pairs[stats["picks"]["orb"]["representative"]["pair"]]
    img = s["img0"]
    F = features("orb", s["name0"], img)
    A = {m: json.loads((ROOT / "runs" / "report" / f"paper_analysis_{m}.json").read_text())["summary"]
         for m in ("orb", "brisk")}
    h, w = img.shape
    fig, axes = plt.subplots(1, 3, figsize=(COL, COL / 3 * h / w + 0.32))
    from matplotlib.colors import LinearSegmentedColormap
    for ax, v in zip(axes, ("vanilla", "pool_grid", "rl")):
        H = np.zeros((h, w), np.float32)
        xy = np.clip(F[v]["xy"].astype(int), 0, [w - 1, h - 1])
        np.add.at(H, (xy[:, 1], xy[:, 0]), 1)
        H = cv2.GaussianBlur(H, (0, 0), 28)
        H = np.sqrt(H / H.max())
        cmap = LinearSegmentedColormap.from_list(v, [(1, 1, 1, 0), C[v]])
        ax.imshow(img, cmap="gray", alpha=0.55)
        ax.imshow(H, cmap=cmap, alpha=0.9, vmin=0, vmax=1)
        ax.set_axis_off()
        ax.set_title(f"{'Vanilla' if v == 'vanilla' else LABEL[v].replace(' (ours)', '')}\n"
                     f"{100 * A['orb'][v]['inl_cov']:.1f}% cells w/ inlier", color=C[v], fontsize=7, pad=2,
                     linespacing=1.1)
    fig.subplots_adjust(wspace=0.04, left=0, right=1, bottom=0, top=0.8)
    save(fig, out, "coverage.pdf")


def fig_matchability(out):
    fig, ax = plt.subplots(figsize=(COL * 0.62, 1.35))
    for m, mk in (("orb", "o"), ("brisk", "s")):
        for ds, ls in (("megadepth", "-"), ("hpatches", "--")):
            k = 2048 if ds == "megadepth" else 1024
            d = json.loads((EVAL / f"{ds}_{m}_k{k}.json").read_text())["diag"]
            ax.plot(range(1, 5), 100 * np.array(d["useful_rate_by_quartile"]), ls=ls, marker=mk, ms=3,
                    color="#2B2B2B" if m == "orb" else "#9A9A9A", mfc="white" if ds == "hpatches" else None,
                    label=f"{m.upper()} {'MD' if ds == 'megadepth' else 'HP'} ({d['roc_auc_response']:.2f})")
    ax.set_xticks(range(1, 5), ["Q1", "Q2", "Q3", "Q4"])
    ax.set_xlabel("response quartile (Q1 strongest)", labelpad=1)
    ax.set_ylabel("% correct matches")
    ax.grid(alpha=.25, lw=.4)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), handlelength=1.6, title="ROC-AUC", title_fontsize=7)
    save(fig, out, "matchability.pdf")


def thumbnails(out, stats, pairs):
    """Small PNGs used inside the TikZ method diagram."""
    import cv2
    s = pairs[stats["picks"]["orb"]["representative"]["pair"]]
    F0 = features("orb", s["name0"], s["img0"]); F1 = features("orb", s["name1"], s["img1"])
    # negative partner from the other scene
    neg = next(p for p in pairs.values() if p["name1"].split("/")[0] != s["name0"].split("/")[0])
    G1 = features("orb", neg["name1"], neg["img1"])

    def crop(img):
        h, w = img.shape
        c = min(h, w)
        return (h - c) // 2, (w - c) // 2, c

    def thumb(img, pts=None, name="x", ms=1.0, color=KP):
        y0, x0, c = crop(img)
        fig = plt.figure(figsize=(1, 1)); ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img[y0:y0 + c, x0:x0 + c], cmap="gray", alpha=0.9)
        if pts is not None:
            p = pts - [x0, y0]
            p = p[(p[:, 0] >= 0) & (p[:, 0] < c) & (p[:, 1] >= 0) & (p[:, 1] < c)]
            ax.scatter(p[:, 0], p[:, 1], s=ms, c=color, lw=0)
        ax.set_axis_off()
        fig.savefig(out / f"thumb_{name}.png", dpi=220); plt.close(fig)

    thumb(s["img0"], name="image")
    thumb(s["img0"], F0["pool"]["xy"], "pool", ms=0.35)
    thumb(s["img0"], F0["rl"]["xy"], "kept", ms=0.9, color=C["rl"])

    def pair_thumb(sa, fa, imgb, fb, Kb, whb, name, positive):
        y0a, x0a, ca = crop(sa["img0"]); y0b, x0b, cb = crop(imgb)
        a = cv2.resize(sa["img0"][y0a:y0a + ca, x0a:x0a + ca], (300, 300))
        b = cv2.resize(imgb[y0b:y0b + cb, x0b:x0b + cb], (300, 300))
        fig = plt.figure(figsize=(2, 1)); ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(np.hstack([a, np.full((300, 16), 255, np.uint8), b]), cmap="gray", alpha=0.9)
        m = match.mnn(fa["desc"], fb["desc"], binary=True)
        p0, p1 = fa["xy"][m[:, 0]], fb["xy"][m[:, 1]]
        _, inl = metrics.estimate_pose(p0, p1, sa["K0"], Kb, sa["wh0"], whb, TH)
        p0, p1 = p0[inl], p1[inl]
        sa_, sb_ = 300 / ca, 300 / cb
        q0 = (p0 - [x0a, y0a]) * sa_; q1 = (p1 - [x0b, y0b]) * sb_
        keep = (q0.min(1) >= 0) & (q0.max(1) < 300) & (q1.min(1) >= 0) & (q1.max(1) < 300)
        for u, w_ in list(zip(q0[keep], q1[keep]))[:25]:
            ax.plot([u[0], w_[0] + 316], [u[1], w_[1]], lw=0.6, c=GOOD if positive else BAD)
        ax.set_axis_off()
        fig.savefig(out / f"thumb_{name}.png", dpi=220); plt.close(fig)

    pair_thumb(s, F0["rl"], s["img1"], F1["rl"], s["K1"], s["wh1"], "pos", True)
    pair_thumb(s, F0["rl"], neg["img1"], G1["rl"], neg["K1"], neg["wh1"], "neg", False)
    print("   thumbnails")


def fig_landscape(out):
    """Vanilla OpenCV -> RIPE-Select (five-seed mean) on MegaDepth-1500, per AUC threshold and mean."""
    labels = ["AUC@5°", "AUC@10°", "AUC@20°", "mean"]
    fig, ax = plt.subplots(figsize=(COL, 1.95))
    yt, yl = [], []
    for g, m in enumerate(("brisk", "orb")):          # ORB drawn on top
        df = pd.read_csv(EVAL / f"megadepth_{m}_k2048_pairs.csv")
        v = auc(errs(df, "vanilla"), [5, 10, 20])
        r = np.mean([auc(e, [5, 10, 20]) for e in seed_errs(m, df)], 0)
        v, r = np.append(v, v.mean()), np.append(r, r.mean())
        for i in range(4):
            y = g * 5.2 + (3 - i)
            bold = i == 3
            ax.plot([v[i], r[i]], [y, y], color=C["rl"], lw=2.4 if bold else 1.6, alpha=0.35, solid_capstyle="round")
            ax.plot(v[i], y, "o", ms=4.5 if bold else 3.8, color=C["vanilla"])
            ax.plot(r[i], y, "*", ms=8 if bold else 6.5, color=C["rl"])
            ax.text(r[i] + 1.2, y, f"+{r[i] - v[i]:.1f}", va="center", fontsize=6.8, color=C["rl"],
                    fontweight="bold")
            yt.append(y); yl.append(labels[i])
        ax.text(9.5, g * 5.2 + 3.85, m.upper(), fontsize=8.5, fontweight="bold", va="bottom")
    ax.set_yticks(yt, yl); ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.7, 9.6)
    ax.set_xlim(10, 66); ax.set_xlabel("pose AUC (%), MegaDepth-1500", labelpad=1)
    ax.grid(axis="x", alpha=.25, lw=.4)
    ax.spines["left"].set_visible(False)
    leg = [Line2D([], [], marker="o", lw=0, ms=4, color=C["vanilla"], label="OpenCV (vanilla)"),
           Line2D([], [], marker="*", lw=0, ms=7, color=C["rl"], label="+ RIPE-Select (5 seeds)")]
    fig.legend(handles=leg, loc="upper center", ncol=2, bbox_to_anchor=(0.55, 1.06), fontsize=6.8,
               handletextpad=0.2, columnspacing=1.2)
    save(fig, out, "landscape.pdf")


def pair_scores(e):
    """Per-pair contribution to mean AUC@5/10/20 (the mean of these over pairs is the mean AUC)."""
    e = np.nan_to_num(e, nan=np.inf)
    return 100 * np.mean([np.clip(1 - e / t, 0, 1) for t in (5, 10, 20)], 0)


def fig_gains(out):
    """(a) gain of every seed over vanilla with the across-seed 95% CI; (b) sorted per-pair gains."""
    from scipy import stats as st
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL, 1.55), gridspec_kw=dict(width_ratios=[0.8, 1.4]))
    info = {}
    for i, m in enumerate(("orb", "brisk")):
        df = pd.read_csv(EVAL / f"megadepth_{m}_k2048_pairs.csv")
        van = auc(errs(df, "vanilla"), [5, 10, 20]).mean()
        E = seed_errs(m, df)
        g = np.array([auc(e, [5, 10, 20]).mean() - van for e in E])
        ci = st.t.ppf(0.975, len(g) - 1) * g.std(ddof=1) / np.sqrt(len(g))
        jit = np.linspace(-0.12, 0.12, len(g))
        a1.scatter(i + jit, g, s=11, color=C["rl"], alpha=0.75, lw=0, zorder=3)
        a1.errorbar(i + 0.3, g.mean(), yerr=ci, fmt="D", ms=3.5, color="black", capsize=2, elinewidth=0.9)
        d = np.mean([pair_scores(e) for e in E], 0) - pair_scores(errs(df, "vanilla"))
        d = np.sort(d)[::-1]
        x = np.linspace(0, 100, len(d))
        ls = "-" if m == "orb" else "--"
        a2.plot(x, d, color="black", lw=0.9, ls=ls, label=m.upper())
        if m == "orb":
            a2.fill_between(x, 0, d, where=d > 0, color=GOOD, alpha=0.25, lw=0)
            a2.fill_between(x, 0, d, where=d < 0, color=BAD, alpha=0.25, lw=0)
        info[m] = {"seed_gains": g.round(2).tolist(), "ci95": round(float(ci), 2),
                   "pairs_better": round(float((d > 0.5).mean() * 100), 1),
                   "pairs_worse": round(float((d < -0.5).mean() * 100), 1)}
    a1.axhline(0, color="black", lw=0.6)
    a1.set_xticks([0, 1], ["ORB", "BRISK"]); a1.set_xlim(-0.4, 1.55)
    a1.set_ylim(0, 11); a1.set_ylabel("gain over vanilla\n(mean AUC points)", labelpad=1)
    a1.set_title("(a) five seeds, 95% CI", pad=2)
    a1.grid(axis="y", alpha=.25, lw=.4)
    a2.axhline(0, color="black", lw=0.6)
    a2.set_xlabel("test pairs, sorted (%)", labelpad=1); a2.set_ylabel("per-pair gain", labelpad=1)
    a2.set_title("(b) 500 MegaDepth pairs, seed mean", pad=2)
    a2.set_xlim(0, 100); a2.legend(loc="upper right", handlelength=1.6)
    a2.grid(alpha=.25, lw=.4)
    fig.subplots_adjust(wspace=0.42)
    save(fig, out, "gains.pdf")
    return info


def fig_learned(out, stats, pairs):
    """What the policy changes: keypoints it adds / drops against top-K response of the same pool, and the
    cues its score correlates with (runs/report/policy_cues.json from paper_policy_cues.py)."""
    s = pairs[stats["picks"]["orb"]["representative"]["pair"]]
    img = s["img0"]
    h, w = img.shape
    R = json.loads((ROOT / "runs" / "report" / "policy_cues.json").read_text())
    sets = {}
    for m in ("orb", "brisk"):
        F = features(m, s["name0"], img)
        key = lambda f: {tuple(p) for p in np.round(f["xy"], 1)}
        kr, ks = key(F["rl"]), key(extract.select(F["pool"], 2048, "response"))
        sets[m] = np.array(sorted(kr - ks)), np.array(sorted(ks - kr))
    allp = np.concatenate([p for a in sets.values() for p in a])           # zoom on where the keypoints are
    (x0, y0), (x1, y1) = np.percentile(allp, 1, 0) - 15, np.percentile(allp, 99, 0) + 15
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    fig = plt.figure(figsize=(FULL, FULL * 0.27))
    gs = fig.add_gridspec(1, 3, width_ratios=[(x1 - x0) / (y1 - y0)] * 2 + [1.25], wspace=0.06)
    for j, m in enumerate(("orb", "brisk")):
        ax = fig.add_subplot(gs[0, j])
        add, drop = sets[m]
        ax.imshow(img, cmap="gray", alpha=0.55)
        ax.scatter(drop[:, 0], drop[:, 1], s=3.0, c=C["vanilla"], lw=0, alpha=0.8, rasterized=True)
        ax.scatter(add[:, 0], add[:, 1], s=3.0, c=C["rl"], lw=0, alpha=0.9, rasterized=True)
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
        ax.set_axis_off()
        ax.set_title(f"{m.upper()}: {100 * len(add) / 2048:.0f}% swapped", pad=2,
                     fontweight="bold")
    ax.legend(handles=[Line2D([], [], marker="o", lw=0, ms=3, color=C["rl"], label="added by the policy"),
                       Line2D([], [], marker="o", lw=0, ms=3, color=C["vanilla"], label="dropped (top-K response)")],
              loc="lower right", fontsize=6.3, framealpha=0.85, frameon=True, borderpad=0.3, handletextpad=0.1)
    ax = fig.add_subplot(gs[0, 2])
    cues = ["2nd-NN desc. distance", "NN desc. distance", "patch mean", "patch std", "patch gradient",
            "local density", "log response"]
    nice = {"2nd-NN desc. distance": "2nd-NN desc. dist.", "NN desc. distance": "NN desc. dist.",
            "patch mean": "patch brightness", "patch std": "patch contrast", "patch gradient": "patch gradient",
            "local density": "local density", "log response": "detector response"}
    y = np.arange(len(cues))[::-1]
    for off, m, col in ((0.18, "orb", C["rl"]), (-0.18, "brisk", "#B39DDB")):
        v = [R[m]["spearman"][c][0] for c in cues]
        ax.barh(y + off, v, height=0.34, color=col, label=m.upper())
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(y, [nice[c] for c in cues]); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Spearman ρ with policy score", labelpad=1); ax.set_xlim(-0.7, 0.55)
    ax.yaxis.tick_right(); ax.spines["left"].set_visible(False); ax.spines["right"].set_visible(True)
    ax.legend(loc="lower left", fontsize=6.5, handlelength=1)
    ax.grid(axis="x", alpha=.25, lw=.4)
    ax.set_title("cues the score follows (40 images)", pad=2, fontweight="bold")
    save(fig, out, "learned.pdf")


def fig_training(out):
    """Validation mean AUC (held-out Reichstag) over training for all five seeds, against plain response ranking."""
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.45))
    info = {}
    for ax, m in zip(axes, ("orb", "brisk")):
        best = []
        for s in range(5):
            L = json.loads((ROOT / "runs" / "train" / f"{m}_rl_s{s}" / "log.json").read_text())["log"]
            st = [e["step"] for e in L]
            v = [100 * np.mean(e["score"]["auc"]) for e in L]
            ax.plot(st, v, color=C["rl"], lw=0.9, alpha=0.75, label="RIPE-Select (5 seeds)" if s == 0 else None)
            i = int(np.argmax(v)); best.append(v[i])
            ax.plot(st[i], v[i], "*", ms=5, color=C["rl"])
        r = 100 * np.mean(L[0]["response"]["auc"])
        ax.axhline(r, color=C["vanilla"], ls="--", lw=1.0, label="response ranking")
        ax.set_title(m.upper(), pad=2, fontweight="bold")
        ax.set_xlabel("training step", labelpad=1); ax.set_xlim(0, 1200); ax.set_xticks([0, 400, 800, 1200])
        ax.grid(alpha=.25, lw=.4)
        info[m] = {"response": round(r, 1), "best": [round(b, 1) for b in best]}
    axes[0].set_ylabel("val. mean AUC (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.13), handlelength=1.8)
    fig.subplots_adjust(wspace=0.3)
    save(fig, out, "training.pdf")
    return info


def md_meta():
    """Scene and ground-truth relative rotation angle (deg) of the 500 evaluated MegaDepth pairs, without images."""
    views = {}
    for line in (MD_ROOT / "views.txt").read_text().strip().split("\n"):
        tok = line.strip().split(" ")
        views[tok[0]] = datasets._parse_view(tok[1:])
    pairs = [l.split(" ")[:2] for l in (MD_ROOT / "pairs.txt").read_text().strip().split("\n")]
    pairs = pairs[::max(1, len(pairs) // 500)][:500]          # same evenly spaced subset as the evaluation
    scene, ang = [], []
    for n0, n1 in pairs:
        R = views[n1][0] @ views[n0][0].T
        ang.append(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
        scene.append(n0.split("/")[0])
    return np.array(scene), np.array(ang)


def fig_difficulty(out):
    """Mean pose AUC by viewpoint change (quartiles of the GT rotation angle) and by scene."""
    scene, ang = md_meta()
    q = np.quantile(ang, [0, .25, .5, .75, 1])
    b = np.clip(np.searchsorted(q, ang, side="right") - 1, 0, 3)
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.55), gridspec_kw=dict(width_ratios=[1.6, 1]))
    info = {"rot_quartiles": q.round(1).tolist()}
    groups = [("rot", [b == i for i in range(4)], [f"{q[i]:.0f}-{q[i + 1]:.0f}°" for i in range(4)]),
              ("scene", [scene == s for s in ("0015", "0022")], ["0015", "0022"])]
    for ax, (key, masks, names) in zip(axes, groups):
        x = np.arange(len(masks))
        for m in ("orb", "brisk"):
            df = pd.read_csv(EVAL / f"megadepth_{m}_k2048_pairs.csv")
            van = pair_scores(errs(df, "vanilla"))
            rl = np.mean([pair_scores(e) for e in seed_errs(m, df)], 0)
            off = -0.2 if m == "orb" else 0.2
            hv = [van[k].mean() for k in masks]; hr = [rl[k].mean() for k in masks]
            ax.bar(x + off - 0.09, hv, 0.18, color=C["vanilla"], alpha=1 if m == "orb" else 0.55)
            ax.bar(x + off + 0.09, hr, 0.18, color=C["rl"], alpha=1 if m == "orb" else 0.55)
            info[f"{key}_{m}"] = [[round(a, 1), round(c, 1)] for a, c in zip(hv, hr)]
        ax.set_xticks(x, names); ax.grid(axis="y", alpha=.25, lw=.4)
        ax.set_xlabel("GT rotation (quartiles)" if key == "rot" else "scene", labelpad=1)
    axes[0].set_ylabel("mean pose AUC (%)")
    leg = [plt.Rectangle((0, 0), 1, 1, color=C["vanilla"], label="ORB vanilla"),
           plt.Rectangle((0, 0), 1, 1, color=C["rl"], label="ORB + ours"),
           plt.Rectangle((0, 0), 1, 1, color=C["vanilla"], alpha=0.55, label="BRISK vanilla"),
           plt.Rectangle((0, 0), 1, 1, color=C["rl"], alpha=0.55, label="BRISK + ours")]
    fig.legend(handles=leg, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.12), handlelength=1.0,
               columnspacing=0.7, handletextpad=0.3, fontsize=6.5)
    fig.subplots_adjust(wspace=0.25)
    save(fig, out, "difficulty.pdf")
    return info


def gallery_picks(m="orb", qs=(0.95, 0.8, 0.6)):
    """Pairs at fixed quantiles of the seed-0 per-pair gain: chosen by rule, not by eye."""
    df = pd.read_csv(EVAL / f"megadepth_{m}_k2048_pairs.csv")
    d = pair_scores(errs(df, "rl")) - pair_scores(errs(df, "vanilla"))
    order = np.argsort(d)
    idx = [int(order[int(round(qq * (len(d) - 1)))]) for qq in qs]
    return [(i, qq, float(d[i])) for i, qq in zip(idx, qs)]


def draw_side(ax, method, s, variant):
    """img0 | img1 side by side with a subset of the RANSAC inliers (green correct, orange wrong)."""
    F0 = features(method, s["name0"], s["img0"])[variant]
    F1 = features(method, s["name1"], s["img1"])[variant]
    q0, q1, good, err = match_pair(method, F0, F1, s)
    import cv2
    H = 480                                                    # both images at the same height
    fa, fb = H / s["img0"].shape[0], H / s["img1"].shape[0]
    a = cv2.resize(s["img0"], (round(s["img0"].shape[1] * fa), H), interpolation=cv2.INTER_AREA)
    b = cv2.resize(s["img1"], (round(s["img1"].shape[1] * fb), H), interpolation=cv2.INTER_AREA)
    W = 2 * H * 4 // 3 + 12                                    # fixed 4:3 + 4:3 canvas, images centred
    cv = np.full((H, max(W, a.shape[1] + 12 + b.shape[1])), 255, np.uint8)
    xa = max(0, H * 4 // 3 - a.shape[1]) // 2; xb = cv.shape[1] - b.shape[1] - max(0, H * 4 // 3 - b.shape[1]) // 2
    cv[:, xa:xa + a.shape[1]] = a; cv[:, xb:xb + b.shape[1]] = b
    ax.imshow(cv, cmap="gray", vmin=0, vmax=255, alpha=0.85, interpolation="lanczos")
    show = np.random.default_rng(0).permutation(len(q0))[:50]
    for u, v, g in sorted(zip(q0[show], q1[show], good[show]), key=lambda z: z[2]):
        ax.plot([u[0] * fa + xa, v[0] * fb + xb], [u[1] * fa, v[1] * fb], lw=0.45, c=GOOD if g else BAD, alpha=0.9)
    ax.set_axis_off()
    ax.text(0.01, 0.03, f"error {err:.1f}°, {int(good.sum())}/{len(q0)} inliers correct", transform=ax.transAxes,
            fontsize=6.3, va="bottom", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.88))
    return float(err)


def fig_gallery(out, picks, pairs):
    """Three ORB pairs at fixed percentiles of the per-pair gain: vanilla (top) vs ours (bottom)."""
    fig, axes = plt.subplots(2, 3, figsize=(FULL, 2.05))
    info = []
    for j, (i, qq, d) in enumerate(picks):
        s = pairs[i]
        e = [draw_side(axes[r, j], "orb", s, v) for r, v in enumerate(("vanilla", "rl"))]
        axes[0, j].set_title(f"{int(qq * 100)}th percentile of the per-pair gain", pad=2, fontsize=7.5)
        info.append({"pair": i, "quantile": qq, "gain": round(d, 1), "err": [round(x, 1) for x in e]})
    for r, v in enumerate(("vanilla", "rl")):
        axes[r, 0].text(-0.03, 0.5, "Vanilla ORB" if v == "vanilla" else "RIPE-Select", rotation=90, color=C[v],
                        transform=axes[r, 0].transAxes, va="center", ha="right", fontweight="bold", fontsize=8)
    fig.subplots_adjust(wspace=0.03, hspace=0.06, left=0.02, right=1, top=0.93, bottom=0)
    save(fig, out, "gallery.pdf")
    return info


def fig_scoremap(out, stats, pairs):
    """Detector response vs learned score over the candidate pool of one image, as smoothed rank maps."""
    import cv2
    from matplotlib.colors import LinearSegmentedColormap
    s = pairs[stats["picks"]["orb"]["representative"]["pair"]]
    img = s["img0"]; h, w = img.shape
    pool = features("orb", s["name0"], img)["pool"]
    pol = policy.NumpyPolicy(ROOT / "runs" / "train" / "orb_rl_s0" / "best.npz")
    sc = pol(policy.candidate_features(pool, img, "orb"))
    fig, axes = plt.subplots(1, 2, figsize=(COL, COL / 2 * h / w + 0.2))
    xy = np.clip(pool["xy"].astype(int), 0, [w - 1, h - 1])
    for ax, (name, v, col) in zip(axes, (("detector response", pool["response"], C["vanilla"]),
                                         ("learned score", sc, C["rl"]))):
        r = np.argsort(np.argsort(v)) / (len(v) - 1)            # rank in [0, 1]
        num = np.zeros((h, w), np.float32); den = np.zeros((h, w), np.float32)
        np.add.at(num, (xy[:, 1], xy[:, 0]), r); np.add.at(den, (xy[:, 1], xy[:, 0]), 1)
        num = cv2.GaussianBlur(num, (0, 0), 14); den = cv2.GaussianBlur(den, (0, 0), 14)
        M = np.where(den > 1e-4, num / np.maximum(den, 1e-9), np.nan)
        cmap = LinearSegmentedColormap.from_list(name, ["#F2F2F2", col])
        ax.imshow(img, cmap="gray", alpha=0.6)
        ax.imshow(M, cmap=cmap, alpha=0.75, vmin=0.2, vmax=0.8)
        ax.set_axis_off(); ax.set_title(name, color=col, fontweight="bold", pad=2)
    fig.subplots_adjust(wspace=0.04, left=0, right=1, bottom=0, top=0.86)
    save(fig, out, "scoremap.pdf")


FIGS = ["teaser", "supp", "auc", "forest", "pareto", "coverage", "matchability", "thumbs", "landscape", "gains",
        "training", "difficulty", "gallery", "scoremap",
        "learned"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "paper" / "figures"))
    ap.add_argument("--only", default=",".join(FIGS))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    only = set(a.only.split(","))
    stats = json.loads((out / "stats.json").read_text())
    need_pairs = only & {"teaser", "supp", "coverage", "thumbs", "learned", "gallery", "scoremap"}
    pk = stats["picks"]
    picks = gallery_picks()
    pairs = md_pairs({pk["orb"]["representative"]["pair"], pk["orb"]["failure"]["pair"],
                      pk["brisk"]["representative"]["pair"]} | {p[0] for p in picks}) if need_pairs else None
    info = {}
    if "teaser" in only: info.update(fig_teaser(out, stats, pairs))
    if "supp" in only: info.update(fig_supp_qualitative(out, stats, pairs))
    if "auc" in only: fig_auc(out)
    if "forest" in only: fig_forest(out, stats)
    if "pareto" in only: fig_pareto(out)
    if "coverage" in only: fig_coverage(out, stats, pairs)
    if "matchability" in only: fig_matchability(out)
    if "thumbs" in only: thumbnails(out, stats, pairs)
    if "landscape" in only: fig_landscape(out)
    if "gains" in only: print(json.dumps(fig_gains(out)))
    if "learned" in only: fig_learned(out, stats, pairs)
    if "training" in only: print(json.dumps(fig_training(out)))
    if "difficulty" in only: print(json.dumps(fig_difficulty(out)))
    if "gallery" in only: print(json.dumps(fig_gallery(out, picks, pairs)))
    if "scoremap" in only: fig_scoremap(out, stats, pairs)
    if info:
        (out / "qualitative_info.json").write_text(json.dumps(info, indent=1))
        print(json.dumps(info))
