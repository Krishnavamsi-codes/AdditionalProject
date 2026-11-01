"""Vanilla SIFT / SURF / ORB baselines on HPatches and MegaDepth-1500 + diagnostics.

Protocol (follows RIPE Tab.2 / glue-factory):
  HPatches   : short side 480, 108 seqs / 540 pairs, MNN, PoseLib homography RANSAC,
               AUC of mean corner error @1/3/5 px, K=1024
  MegaDepth  : long side 1200, 1500 pairs, MNN, PoseLib relative-pose RANSAC,
               AUC of max(R,t) angular error @5/10/20 deg, K=2048
RANSAC threshold: a sweep is evaluated; we report the best threshold per variant (glue-factory
'ransac_th=-1') *and* the fixed threshold chosen on the vanilla baseline.

Selection variants (same descriptor, same K, same matcher, same RANSAC):
  vanilla     OpenCV default thresholds, top-K by detector response
  pool_resp   lowered thresholds -> pool of 4K candidates, top-K by response
  pool_grid   same pool, spatially bucketed top-K (ANMS-like heuristic)
  pool_rand   same pool, random K (floor)
  pool_resp2x top-2K by response (budget headroom, NOT a fair comparison - diagnostic only)
"""
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfeat import datasets, extract, match, metrics, policy  # noqa: E402

TH = {"hpatches": [1.0, 2.0, 3.0, 4.0, 6.0], "megadepth": [0.5, 1.0, 1.5, 2.0, 3.0]}
AUC_TH = {"hpatches": [1, 3, 5], "megadepth": [5, 10, 20]}


def roc_auc(score, label):
    label = label.astype(bool)
    if label.all() or (~label).all():
        return np.nan
    r = np.argsort(np.argsort(score)) + 1
    npos = label.sum(); nneg = len(label) - npos
    return float((r[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))


_POLICIES = {}


def load_policies(specs):
    """--policy name=path.npz ... -> {name: NumpyPolicy} (cached per process)."""
    for spec in specs or []:
        name, path = spec.split("=", 1)
        if name not in _POLICIES:
            _POLICIES[name] = policy.NumpyPolicy(path)
    return {s.split("=", 1)[0]: _POLICIES[s.split("=", 1)[0]] for s in specs or []}


class FeatCache:
    def __init__(self, method, k, pool_mult=4, policies=None):
        self.method, self.k, self.P = method, k, pool_mult * k
        self.policies = policies or {}
        self.mem = {}
        self.times = {}            # per image, ms: vanilla extraction, pool extraction, policy scoring

    def get(self, key, img):
        if key not in self.mem:
            if len(self.mem) > 64:
                self.mem.pop(next(iter(self.mem)))
            t0 = time.perf_counter()
            van = extract.detect_describe(img, self.method, self.k, pool=False)
            t1 = time.perf_counter()
            pool = extract.detect_describe(img, self.method, self.k, pool=True)
            pool = extract.select(pool, self.P, "response")
            t2 = time.perf_counter()
            if self.policies:      # policy scores over the same candidate pool
                F = policy.candidate_features(pool, img, self.method)
                pool["scores"] = {name: pol(F) for name, pol in self.policies.items()}
            t3 = time.perf_counter()
            n_pol = max(len(self.policies), 1)
            self.times[key] = dict(t_van=1e3 * (t1 - t0), t_pool=1e3 * (t2 - t1), t_pol=1e3 * (t3 - t2) / n_pol)
            self.mem[key] = (van, pool)
        return self.mem[key]


def variants(van, pool, k, img_shape, keep=None):
    sc = pool.get("scores", {})
    pool = {key: v for key, v in pool.items() if key != "scores"}
    V = {
        "vanilla": lambda: van,
        "pool_resp": lambda: extract.select(pool, k, "response"),
        "pool_grid": lambda: extract.select(pool, k, "grid", img_shape=img_shape),
        "pool_rand": lambda: extract.select(pool, k, "random", rng=np.random.default_rng(0)),
        "pool_resp2x": lambda: extract.select(pool, 2 * k, "response"),
    }
    for name, s in sc.items():   # learned selection: same pool, same K, same descriptor
        V[name] = lambda s=s: extract.select(pool, k, "score", scores=s)
        V[name + "_grid"] = lambda s=s: extract.select(pool, k, "grid", img_shape=img_shape, scores=s)
    return {v: f() for v, f in V.items() if keep is None or v in keep}


def _init_worker():
    # one thread per worker: PoseLib RANSAC is single-threaded, parallelism comes from pairs
    import cv2
    cv2.setNumThreads(1)


def _pair_job(job):
    n, s, args = job
    fc = FeatCache(args.method, args.k, policies=load_policies(args.policy))
    rows, diag = [], defaultdict(list)
    process_pair(n, s, args, fc, rows, diag)
    return rows, diag


def run(args):
    fc = FeatCache(args.method, args.k, policies=load_policies(args.policy))
    rows = []
    diag = defaultdict(list)
    t0 = time.time()
    if args.dataset == "hpatches":
        it = datasets.hpatches_pairs(args.root, 480)
    else:
        it = datasets.megadepth1500_pairs(args.root, 1200, limit=args.limit)
    if args.dataset == "hpatches" and args.limit:
        it = (s for n, s in enumerate(it) if n < args.limit)
    if args.workers <= 1:
        for n, s in enumerate(it):
            process_pair(n, s, args, fc, rows, diag)
            if n % 50 == 0:
                print(f"[{args.method}/{args.dataset}] pair {n}  {time.time()-t0:.0f}s", flush=True)
        return rows, diag
    import multiprocessing as mp, os
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"          # inherited by spawned workers before they import numpy
    with mp.Pool(args.workers, initializer=_init_worker) as pool:
        jobs = ((n, s, args) for n, s in enumerate(it))
        for n, (r, d) in enumerate(pool.imap(_pair_job, jobs, chunksize=1)):
            rows.extend(r)
            for key, v in d.items():
                diag[key].extend(v)
            if n % 50 == 0:
                print(f"[{args.method}/{args.dataset}] pair {n}  {time.time()-t0:.0f}s", flush=True)
    return rows, diag


def process_pair(n, s, args, fc, rows, diag):
    binary = extract.is_binary(args.method)
    k0 = (s["seq"], 1) if args.dataset == "hpatches" else s["name0"]
    k1 = (s["seq"], s["idx"]) if args.dataset == "hpatches" else s["name1"]
    van0, pool0 = fc.get(k0, s["img0"])
    van1, pool1 = fc.get(k1, s["img1"])
    V0 = variants(van0, pool0, args.k, s["img0"].shape, args.variants)
    V1 = variants(van1, pool1, args.k, s["img1"].shape, args.variants)
    for v in V0:
        f0, f1 = V0[v], V1[v]
        m = match.mnn(f0["desc"], f1["desc"], binary=binary, ratio=args.ratio)
        p0, p1 = f0["xy"][m[:, 0]], f1["xy"][m[:, 1]]
        row = dict(pair=n, variant=v, kind=s.get("kind", "md"),
                   nk0=len(f0["xy"]), nk1=len(f1["xy"]), nm=len(m))
        for tk in ("t_van", "t_pool", "t_pol"):         # ms per image (mean of the two images)
            row[tk] = (fc.times.get(k0, {}).get(tk, np.nan) + fc.times.get(k1, {}).get(tk, np.nan)) / 2
        # GT precision of putative matches
        if args.dataset == "hpatches":
            gt_err = np.linalg.norm(metrics.warp(p0, s["H"]) - p1, axis=1) if len(m) else np.zeros(0)
            row["prec"] = float((gt_err < 3).mean()) if len(m) else 0.0
            row["ncorr"] = int((gt_err < 3).sum())
            for t in range(1, 11):                       # MMA (D2-Net / HPatches): reprojection error < t px
                row[f"mma@{t}"] = float((gt_err < t).mean()) if len(m) else 0.0
        else:
            gt_err = metrics.sym_epipolar_px(p0, p1, s["K0"], s["K1"], s["R"], s["t"]) if len(m) else np.zeros(0)
            row["prec"] = float((gt_err < 2).mean()) if len(m) else 0.0
            row["ncorr"] = int((gt_err < 2).sum())
            for t in range(1, 11):                       # MMA analogue: symmetric epipolar error < t px
                row[f"mma@{t}"] = float((gt_err < t).mean()) if len(m) else 0.0
        for th in (args.ths or TH[args.dataset]):
            if args.dataset == "hpatches":
                Hest, inl = metrics.estimate_homography(p0, p1, th)
                err = metrics.corner_error(Hest, s["H"], s["wh0"]) if Hest is not None else np.inf
            else:
                est, inl = metrics.estimate_pose(p0, p1, s["K0"], s["K1"], s["wh0"], s["wh1"], th)
                err = metrics.pose_error(est, s["R"], s["t"])
            row[f"err@{th}"] = err
            row[f"inl@{th}"] = int(inl.sum())
        rows.append(row)
    # ---- diagnostic: does the detector's own response predict which keypoints match correctly?
    if args.diag:
        m = match.mnn(pool0["desc"], pool1["desc"], binary=binary, ratio=args.ratio)
        p0, p1 = pool0["xy"][m[:, 0]], pool1["xy"][m[:, 1]]
        if args.dataset == "hpatches":
            ok = np.linalg.norm(metrics.warp(p0, s["H"]) - p1, axis=1) < 3
        else:
            ok = metrics.sym_epipolar_px(p0, p1, s["K0"], s["K1"], s["R"], s["t"]) < 2
        lab0 = np.zeros(len(pool0["xy"]), bool); lab0[m[ok, 0]] = True
        diag["roc_auc_response"].append(roc_auc(pool0["response"], lab0))
        diag["roc_auc_size"].append(roc_auc(pool0["size"], lab0))
        # correct-match rate per response quartile
        q = np.argsort(np.argsort(-pool0["response"])) * 4 // max(len(lab0), 1)
        diag["useful_rate_by_quartile"].append([float(lab0[q == i].mean()) if (q == i).any() else np.nan
                                                for i in range(4)])
        # share of the pool's correct matches whose image-0 keypoint survives the top-K response cut
        topk = np.zeros(len(lab0), bool); topk[np.argsort(-pool0["response"])[:args.k]] = True
        diag["frac_correct_in_topK"].append(float(topk[lab0].mean()) if lab0.any() else np.nan)
        diag["pool_correct"].append(int(ok.sum()))


def summarize(rows, dataset):
    import pandas as pd
    df = pd.DataFrame(rows)
    out = {}
    ths = TH[dataset]
    # threshold picked on vanilla (fair fixed setting for everybody)
    def mauc(sub, th):
        return float(np.mean(metrics.error_auc(sub[f"err@{th}"].values, AUC_TH[dataset])))
    van = df[df.variant == "vanilla"]
    th_fixed = max(ths, key=lambda th: mauc(van, th))
    for v, sub in df.groupby("variant", sort=False):
        th_best = max(ths, key=lambda th: mauc(sub, th))
        r = {"th_fixed": th_fixed, "th_best": th_best,
             "auc_fixed": metrics.error_auc(sub[f"err@{th_fixed}"].values, AUC_TH[dataset]),
             "auc_best": metrics.error_auc(sub[f"err@{th_best}"].values, AUC_TH[dataset]),
             "n_kpts": float((sub.nk0 + sub.nk1).mean() / 2), "n_matches": float(sub.nm.mean()),
             "match_prec": float(sub.prec.mean()), "n_correct": float(sub.ncorr.mean()),
             "ransac_inl": float(sub[f"inl@{th_fixed}"].mean())}
        if dataset == "hpatches":
            for kind in ("i", "v"):
                s2 = sub[sub.kind == kind]
                r[f"auc_fixed_{kind}"] = metrics.error_auc(s2[f"err@{th_fixed}"].values, AUC_TH[dataset])
                r[f"match_prec_{kind}"] = float(s2.prec.mean())
        out[v] = r
    return out, df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["hpatches", "megadepth"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--method", choices=["sift", "surf", "orb", "brisk"], required=True)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--ratio", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--policy", nargs="*", default=None, help="name=path.npz learned selection policies")
    ap.add_argument("--variants", nargs="*", default=None, help="restrict to these variants")
    ap.add_argument("--tag", default="", help="suffix for output files")
    ap.add_argument("--ths", type=float, nargs="+", default=None, help="override RANSAC threshold sweep")
    ap.add_argument("--workers", type=int, default=1, help="parallel pairs (PoseLib RANSAC is single-threaded)")
    args = ap.parse_args()
    args.k = args.k or (1024 if args.dataset == "hpatches" else 2048)
    if args.ths:
        TH[args.dataset] = args.ths
    rows, diag = run(args)
    summ, df = summarize(rows, args.dataset)
    tag = f"{args.dataset}_{args.method}_k{args.k}" + (f"_r{args.ratio}" if args.ratio else "") + args.tag
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{tag}_pairs.csv", index=False)
    d = {k: (np.nanmean(np.array(v, float), 0).tolist()) for k, v in diag.items()}
    json.dump({"summary": summ, "diag": d, "args": vars(args)}, open(out / f"{tag}.json", "w"), indent=1)
    for v, r in summ.items():
        print(f"{v:12s} AUC(fixed th={r['th_fixed']}) {np.round(np.array(r['auc_fixed'])*100,2)}  "
              f"best(th={r['th_best']}) {np.round(np.array(r['auc_best'])*100,2)}  kpts {r['n_kpts']:.0f} "
              f"matches {r['n_matches']:.0f} prec {r['match_prec']:.3f}")
    if d:
        print("diag:", json.dumps(d))
