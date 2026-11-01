"""Qualitative figure: kept keypoints + RANSAC-inlier matches for vanilla / pool_grid / RL selection.

  python proj/scripts/qualitative.py --method sift --pair 300 [--out runs/report]
"""
import argparse, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfeat import datasets, extract, match, metrics, policy  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
NAMES = {"vanilla": "Vanilla (OpenCV)", "pool_grid": "Pool, grid", "RL (RIPE-Select), top-K": "RIPE-Select"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="sift")
    ap.add_argument("--pair", type=int, default=300, help="index into the 500-pair MegaDepth subset")
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--th", type=float, default=1.5)
    ap.add_argument("--fontsize", type=float, default=10)
    ap.add_argument("--out", default=str(ROOT / "runs" / "report"))
    a = ap.parse_args()
    pol = policy.NumpyPolicy(ROOT / "runs" / "train" / f"{a.method}_rl_s0" / "best.npz")
    s = next(x for n, x in enumerate(datasets.megadepth1500_pairs(ROOT / "data/megadepth1500/megadepth1500", 1200, 500))
             if n == a.pair)
    feats = {}
    for i in (0, 1):
        img = s[f"img{i}"]
        van = extract.detect_describe(img, a.method, a.k)
        pool = extract.select(extract.detect_describe(img, a.method, a.k, pool=True), 4 * a.k, "response")
        sc = pol(policy.candidate_features(pool, img, a.method))
        feats[i] = {"vanilla": van, "pool_grid": extract.select(pool, a.k, "grid", img_shape=img.shape),
                    "RL (RIPE-Select), top-K": extract.select(pool, a.k, "score", scores=sc)}
    fig, axes = plt.subplots(3, 1, figsize=(12, 13))
    h0, w0 = s["img0"].shape; h1, w1 = s["img1"].shape
    canvas = np.zeros((max(h0, h1), w0 + w1), np.uint8)
    canvas[:h0, :w0] = s["img0"]; canvas[:h1, w0:] = s["img1"]
    for ax, v in zip(axes, feats[0]):
        f0, f1 = feats[0][v], feats[1][v]
        m = match.mnn(f0["desc"], f1["desc"], binary=extract.is_binary(a.method))
        p0, p1 = f0["xy"][m[:, 0]], f1["xy"][m[:, 1]]
        est, inl = metrics.estimate_pose(p0, p1, s["K0"], s["K1"], s["wh0"], s["wh1"], a.th)
        err = metrics.pose_error(est, s["R"], s["t"])
        good = metrics.sym_epipolar_px(p0, p1, s["K0"], s["K1"], s["R"], s["t"]) < 2
        ax.imshow(canvas, cmap="gray"); ax.axis("off")
        ax.scatter(f0["xy"][:, 0], f0["xy"][:, 1], s=1, c="yellow", alpha=.5)
        ax.scatter(f1["xy"][:, 0] + w0, f1["xy"][:, 1], s=1, c="yellow", alpha=.5)
        for q0, q1, g in zip(p0[inl], p1[inl], good[inl]):
            ax.plot([q0[0], q1[0] + w0], [q0[1], q1[1]], lw=.4, c="lime" if g else "red", alpha=.7)
        ax.set_title(f"{NAMES.get(v, v)}: {inl.sum()} RANSAC inliers "
                     f"({good[inl].sum()} correct), pose error {err:.1f}°", fontsize=a.fontsize)
    fig.tight_layout()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(a.out) / f"qualitative_{a.method}_pair{a.pair}.png", dpi=110)
    print("saved")
