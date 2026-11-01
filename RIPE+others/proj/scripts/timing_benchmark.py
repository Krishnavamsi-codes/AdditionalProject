"""Clean runtime per image (ms), single thread, warm-up pass, median over 20 images.

  python proj/scripts/timing_benchmark.py megadepth|hpatches

Stages: vanilla extraction | pool extraction | cues | policy MLP | top-K select. Run with nothing else on the
machine; the paper's runtime numbers (paper/figures/timing.json) come from this script.
"""
import sys, time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfeat import datasets, extract, policy  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    cv2.setNumThreads(1)
    ds = sys.argv[1]
    N = 20
    if ds == "megadepth":
        it = datasets.megadepth1500_pairs(ROOT / "data/megadepth1500/megadepth1500", 1200, limit=N)
        K = 2048
    else:
        it = datasets.hpatches_pairs(ROOT / "data/hpatches-sequences-release/hpatches-sequences-release", 480)
        K = 1024
    imgs = []
    for p in it:
        imgs.append(p["img0"])
        if len(imgs) >= N:
            break
    for m in ("sift", "orb", "brisk"):
        pol = policy.NumpyPolicy(ROOT / "runs" / "train" / f"{m}_rl_s0" / "best.npz")
        T = {k: [] for k in ("vanilla", "pool", "cues", "mlp", "select")}
        for r in range(2):                     # pass 0 = warm-up
            for img in imgs:
                t0 = time.perf_counter(); extract.detect_describe(img, m, K, pool=False)
                t1 = time.perf_counter(); pool = extract.select(extract.detect_describe(img, m, K, pool=True), 4 * K, "response")
                t2 = time.perf_counter(); F = policy.candidate_features(pool, img, m)
                t3 = time.perf_counter(); s = pol(F)
                t4 = time.perf_counter(); extract.select(pool, K, "score", scores=s)
                t5 = time.perf_counter()
                if r:
                    for k, a, b in (("vanilla", t0, t1), ("pool", t1, t2), ("cues", t2, t3), ("mlp", t3, t4), ("select", t4, t5)):
                        T[k].append(1e3 * (b - a))
        med = {k: np.median(v) for k, v in T.items()}
        rl = med["pool"] + med["cues"] + med["mlp"] + med["select"]
        print(f"{ds} {m}: vanilla {med['vanilla']:.0f} | pool {med['pool']:.0f} | cues {med['cues']:.0f} | "
              f"mlp {med['mlp']:.1f} | select {med['select']:.1f} | RL total {rl:.0f} ms ({rl / med['vanilla']:.1f}x)")
