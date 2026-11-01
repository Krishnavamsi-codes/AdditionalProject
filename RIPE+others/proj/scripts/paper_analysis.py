"""Paper analysis on a MegaDepth subset: spatial coverage of RANSAC inliers, and RANSAC-surviving matches on
negative (cross-scene) pairs, for vanilla / pool_grid / RL selection.

  python proj/scripts/paper_analysis.py --method orb [--pairs 100] [--out runs/report]
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfeat import datasets, extract, match, metrics, policy  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("vanilla", "pool_grid", "rl")


def coverage(xy, shape, g=16):
    """Share of g x g image cells that contain at least one point."""
    if len(xy) == 0:
        return 0.0
    h, w = shape
    cx = np.clip((xy[:, 0] * g / w).astype(int), 0, g - 1)
    cy = np.clip((xy[:, 1] * g / h).astype(int), 0, g - 1)
    return len(np.unique(cy * g + cx)) / (g * g)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["orb", "brisk", "sift"])
    ap.add_argument("--pairs", type=int, default=100, help="evenly spaced pairs of the 500-pair subset")
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--th", type=float, default=1.0)
    ap.add_argument("--out", default=str(ROOT / "runs" / "report"))
    a = ap.parse_args()
    import cv2
    cv2.setNumThreads(1)
    pol = policy.NumpyPolicy(ROOT / "runs" / "train" / f"{a.method}_rl_s0" / "best.npz")
    binary = extract.is_binary(a.method)
    step = 500 // a.pairs
    pairs = [s for n, s in enumerate(datasets.megadepth1500_pairs(ROOT / "data/megadepth1500/megadepth1500", 1200, 500))
             if n % step == 0][:a.pairs]

    feats = {}

    def get(name, img):
        if name not in feats:
            van = extract.detect_describe(img, a.method, a.k)
            pool = extract.select(extract.detect_describe(img, a.method, a.k, pool=True), 4 * a.k, "response")
            sc = pol(policy.candidate_features(pool, img, a.method))
            feats[name] = {"vanilla": van, "pool_grid": extract.select(pool, a.k, "grid", img_shape=img.shape),
                           "rl": extract.select(pool, a.k, "score", scores=sc)}
        return feats[name]

    def ransac(f0, f1, K0, K1, wh0, wh1):
        m = match.mnn(f0["desc"], f1["desc"], binary=binary)
        p0, p1 = f0["xy"][m[:, 0]], f1["xy"][m[:, 1]]
        est, inl = metrics.estimate_pose(p0, p1, K0, K1, wh0, wh1, a.th)
        return p0, p1, est, inl

    res = {v: {"kp_cov": [], "inl_cov": [], "pos_inl": [], "neg_inl": [], "neg_ok": []} for v in VARIANTS}
    t0 = time.time()
    for i, s in enumerate(pairs):
        F0, F1 = get(s["name0"], s["img0"]), get(s["name1"], s["img1"])
        # negative partner: image 1 of the next pair from the other scene
        j = next(j for j in list(range(i + 1, len(pairs))) + list(range(i)) if
                 pairs[j]["name1"].split("/")[0] != s["name0"].split("/")[0])
        n = pairs[j]
        G1 = get(n["name1"], n["img1"])
        for v in VARIANTS:
            p0, p1, est, inl = ransac(F0[v], F1[v], s["K0"], s["K1"], s["wh0"], s["wh1"])
            res[v]["kp_cov"].append(coverage(F0[v]["xy"], s["img0"].shape))
            res[v]["inl_cov"].append(coverage(p0[inl], s["img0"].shape))
            res[v]["pos_inl"].append(int(inl.sum()))
            _, _, est_n, inl_n = ransac(F0[v], G1[v], s["K0"], n["K1"], s["wh0"], n["wh1"])
            res[v]["neg_inl"].append(int(inl_n.sum()))
            res[v]["neg_ok"].append(bool(est_n is not None and inl_n.sum() >= 15))
        if i % 10 == 0:
            print(f"[{a.method}] pair {i}/{len(pairs)}  {time.time() - t0:.0f}s", flush=True)

    summary = {v: {k: float(np.mean(x)) for k, x in d.items()} | {"neg_inl_median": float(np.median(d["neg_inl"]))}
               for v, d in res.items()}
    out = Path(a.out) / f"paper_analysis_{a.method}.json"
    out.write_text(json.dumps({"args": vars(a), "summary": summary, "per_pair": res}, indent=1))
    print(json.dumps(summary, indent=1))
