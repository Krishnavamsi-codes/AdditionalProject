import sys, time; sys.path.insert(0, "proj")
import numpy as np
from cfeat import datasets, extract, match, metrics
for method in ["sift", "orb"]:
    for s in datasets.megadepth1500_pairs("data/megadepth1500/megadepth1500", 1200, limit=2):
        t = time.time(); van = extract.detect_describe(s["img0"], method, 2048); t1 = time.time() - t
        t = time.time(); pool = extract.detect_describe(s["img0"], method, 2048, pool=True); tp = time.time() - t
        v1 = extract.detect_describe(s["img1"], method, 2048)
        t = time.time(); m = match.mnn(van["desc"], v1["desc"], binary=method == "orb"); t2 = time.time() - t
        pp = extract.select(pool, 8192, "response")
        t = time.time(); match.mnn(pp["desc"], pp["desc"], binary=method == "orb"); t3 = time.time() - t
        print(method, "extract van %.2fs pool %.2fs (%d) match %.2fs poolmatch %.2fs" % (t1, tp, len(pool["xy"]), t2, t3))
        p0, p1 = van["xy"][m[:, 0]], v1["xy"][m[:, 1]]
        for th in [0.5, 1, 2, 3]:
            t = time.time(); est, inl = metrics.estimate_pose(p0, p1, s["K0"], s["K1"], s["wh0"], s["wh1"], th)
            print(method, "th", th, "ransac %.2fs" % (time.time() - t), "nm", len(m), "inl", inl.sum())
