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

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.2, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "legend.frameon": False, "savefig.dpi": 300})


# ------------------------------------------------------------------ helpers

def errs(df, v, th=TH):
    return df[df.variant == v].sort_values("pair")[f"err@{th}"].values


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
    for a, b, g in sorted(zip(q0, q1, good), key=lambda z: z[2]):          # correct lines on top
        ax.plot([a[0] + o0[0], b[0] + o1[0]], [a[1] + o0[1], b[1] + o1[1]], lw=0.45,
                c=GOOD if g else BAD, alpha=0.9, solid_capstyle="round")
    ax.set_axis_off()
    ax.text(0.02, 0.015, f"pose error {err:.1f}°\n{len(q0)} inliers, {int(good.sum())} correct",
            transform=ax.transAxes, fontsize=7, va="bottom", ha="left", color="black",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.88))
    return err, len(q0), int(good.sum())


def summary_bars(ax, method):
    """Benchmark-level mean pose AUC on MegaDepth (RL: 3-seed mean +- std) next to the example pair."""
    df = pd.read_csv(EVAL / f"megadepth_{method}_k2048_pairs.csv")
    ds = pd.read_csv(EVAL / f"megadepth_{method}_k2048_seeds_pairs.csv")
    vs = ("vanilla", "pool_grid", "rl")
    vals, err = [], []
    for v in vs:
        if v == "rl":
            a = [auc(errs(df, "rl"), [5, 10, 20]).mean()] + [auc(errs(ds, f"rl_s{s}"), [5, 10, 20]).mean() for s in (1, 2)]
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
        ds = pd.read_csv(EVAL / f"megadepth_{m}_k2048_seeds_pairs.csv")
        for v in ("pool_resp", "vanilla", "sup", "pool_grid", "rl"):
            if v == "rl":
                cs = np.stack([auc(errs(df, "rl"), ths)] + [auc(errs(ds, f"rl_s{s}"), ths) for s in (1, 2)])
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
        ds = pd.read_csv(EVAL / f"megadepth_{m}_k2048_seeds_pairs.csv")
        t = T[m]
        rt = {"vanilla": t["vanilla"], "pool_resp": t["pool"], "pool_grid": t["pool"], "sup": t["rl_total"],
              "rl": t["rl_total"]}
        for v in ("vanilla", "pool_resp", "pool_grid", "sup", "rl"):
            if v == "rl":
                a = [auc(errs(df, "rl"), [5, 10, 20]).mean()] + [auc(errs(ds, f"rl_s{s}"), [5, 10, 20]).mean() for s in (1, 2)]
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


FIGS = ["teaser", "supp", "auc", "forest", "pareto", "coverage", "matchability", "thumbs"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "paper" / "figures"))
    ap.add_argument("--only", default=",".join(FIGS))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    only = set(a.only.split(","))
    stats = json.loads((out / "stats.json").read_text())
    need_pairs = only & {"teaser", "supp", "coverage", "thumbs"}
    pk = stats["picks"]
    pairs = md_pairs({pk["orb"]["representative"]["pair"], pk["orb"]["failure"]["pair"],
                      pk["brisk"]["representative"]["pair"]}) if need_pairs else None
    info = {}
    if "teaser" in only: info.update(fig_teaser(out, stats, pairs))
    if "supp" in only: info.update(fig_supp_qualitative(out, stats, pairs))
    if "auc" in only: fig_auc(out)
    if "forest" in only: fig_forest(out, stats)
    if "pareto" in only: fig_pareto(out)
    if "coverage" in only: fig_coverage(out, stats, pairs)
    if "matchability" in only: fig_matchability(out)
    if "thumbs" in only: thumbnails(out, stats, pairs)
    if info:
        (out / "qualitative_info.json").write_text(json.dumps(info, indent=1))
        print(json.dumps(info))
