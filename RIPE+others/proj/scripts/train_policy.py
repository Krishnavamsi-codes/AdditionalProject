"""Train RIPE-Select (RL keypoint selection) for SIFT / SURF / ORB on IMC-2020 scenes (never test scenes).

RIPE ingredients kept (Künzel et al. ICCV'25, ripe/train.py + utils.get_rewards):
  * only a binary pair label is used: positive = covisible pair of one scene, negative = two scenes (50/50)
  * reward = MNN on the (unchanged) descriptors of the kept keypoints -> fundamental-matrix RANSAC (PoseLib);
    every kept keypoint in an MNN & inlier match gets +1 on a positive pair and -1 on a negative pair
  * REINFORCE over per-keypoint stochastic decisions (a contextual bandit, like RIPE's per-cell sampling)
Adapted for a classical detector:
  * action = keep / drop each pool candidate, p_i = sigmoid(s_i + b) with b solved per image so that
    sum(p) = K (budget-normalised; replaces RIPE's keypoint-count regulariser); inference = top-K by s
  * baseline = per-image mean reward (variance reduction)
Supervised ablation (--mode sup): BCE on the same RANSAC pseudo-labels, computed once on the full pool.

  python proj/scripts/train_policy.py --method sift --out runs/train/sift_rl_s0 [--mode rl|sup] [--seed 0]
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfeat import extract, imc, match, metrics, policy  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------------------- feature cache

def _cache_one(job):
    import cv2
    cv2.setNumThreads(1)
    data_root, scene, name, method, k, path = job
    if Path(path).exists():
        return path
    S = _SCENES.get(scene) or imc.Scene(data_root, scene)
    img, K = S.load(name)
    f = extract.select(extract.detect_describe(img, method, k, pool=True), 4 * k, "response")
    F = policy.candidate_features(f, img, method)[:, :policy.N_SCALAR]
    desc = f["desc"] if extract.is_binary(method) else f["desc"].astype(np.float16)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, xy=f["xy"], desc=desc, F=F, K=K, response=f["response"], wh=np.array(img.shape[::-1]))
    return path


_SCENES = {}


def build_cache(args, scenes):
    jobs = []
    for sc, S in scenes.items():
        for n in S.names:
            p = ROOT / "cache" / "train" / f"{args.method}_k{args.k}" / sc / (Path(n).stem + ".npz")
            jobs.append((args.data, sc, n, args.method, args.k, str(p)))
    todo = [j for j in jobs if not Path(j[-1]).exists()]
    if todo:
        print(f"caching {len(todo)} images ...", flush=True)
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            for i, _ in enumerate(pool.imap_unordered(_cache_one, todo, chunksize=4)):
                if i % 100 == 0:
                    print(f"  cached {i}/{len(todo)}", flush=True)
    cache = {}
    for sc, S in scenes.items():
        for n in S.names:
            z = np.load(ROOT / "cache" / "train" / f"{args.method}_k{args.k}" / sc / (Path(n).stem + ".npz"))
            # descriptors stay compact (uint8 ORB / float16 SIFT,SURF); policy inputs are built per use (RAM)
            cache[(sc, n)] = dict(xy=z["xy"], desc=z["desc"], F=z["F"], K=z["K"], response=z["response"],
                                  wh=tuple(z["wh"]))
    return cache


def full_desc(c):
    return c["desc"] if c["desc"].dtype == np.uint8 else c["desc"].astype(np.float32)


def inputs(c, method):
    return np.concatenate([c["F"], policy.desc_block(full_desc(c), method)], 1)




# ----------------------------------------------------------------------------- reward (workers)

def reward_job(job):
    """MNN + F-RANSAC on kept keypoints -> per-keypoint reward vectors (RIPE utils.get_rewards)."""
    xy0, d0, xy1, d1, label, binary, th, iters = job
    r0 = np.zeros(len(xy0), np.float32); r1 = np.zeros(len(xy1), np.float32)
    m = match.mnn(d0, d1, binary=binary)
    if len(m) >= 8:
        import poselib
        _, info = poselib.estimate_fundamental(xy0[m[:, 0]].astype(float), xy1[m[:, 1]].astype(float),
                                               {"max_epipolar_error": th, "max_iterations": iters, "seed": 0})
        inl = np.asarray(info["inliers"], bool)
        r0[m[inl, 0]] = label; r1[m[inl, 1]] = label
    return r0, r1


def val_job(job):
    """Held-out validation (reichstag): top-K by score -> MNN -> relative pose (same as MegaDepth eval)."""
    xy0, d0, xy1, d1, K0, K1, wh0, wh1, R, t, binary, th = job
    m = match.mnn(d0, d1, binary=binary)
    est, inl = metrics.estimate_pose(xy0[m[:, 0]], xy1[m[:, 1]], K0, K1, wh0, wh1, th)
    return metrics.pose_error(est, R, t), int(inl.sum())


# ----------------------------------------------------------------------------- training

def solve_bias(s, k, iters=40):
    """b such that sum(sigmoid(s + b)) = k (bisection, no gradient)."""
    lo, hi = -50.0 - s.max(), 50.0 - s.min()
    for _ in range(iters):
        b = (lo + hi) / 2
        if (1 / (1 + np.exp(-(s + b)))).sum() > k:
            hi = b
        else:
            lo = b
    return (lo + hi) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["sift", "surf", "orb", "brisk"], required=True)
    ap.add_argument("--mode", choices=["rl", "sup"], default="rl")
    ap.add_argument("--data", default=str(ROOT / "data" / "imc2020"))
    ap.add_argument("--train", nargs="+", default=["palace_of_westminster", "grand_place_brussels"])
    ap.add_argument("--val", default="reichstag")
    ap.add_argument("--max-images", type=int, default=350)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--th", type=float, default=1.5, help="F-RANSAC epipolar threshold (px @1200)")
    ap.add_argument("--iters", type=int, default=2000, help="F-RANSAC iterations in the reward")
    ap.add_argument("--neg", type=float, default=0.5, help="share of negative (cross-scene) pairs")
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--val-pairs", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--residual", action="store_true",
                    help="score = alpha * z(log response) + MLP (last layer zero-init): starts at the detector ranking")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    torch.manual_seed(args.seed); torch.set_num_threads(4)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    binary = extract.is_binary(args.method)

    scenes = {sc: imc.Scene(args.data, sc, args.max_images) for sc in args.train + [args.val]}
    _SCENES.update(scenes)
    cache = build_cache(args, scenes)
    pos = {sc: scenes[sc].positive_pairs() for sc in args.train}
    print({sc: (len(scenes[sc].names), len(p)) for sc, p in pos.items()}, flush=True)
    vrng = np.random.default_rng(123)
    vp = scenes[args.val].positive_pairs(min_shared=100, min_ratio=0.15)
    vp = [vp[i] for i in vrng.choice(len(vp), min(args.val_pairs, len(vp)), replace=False)]

    allX = np.concatenate([inputs(cache[key], args.method)[::16] for key in list(cache)[::4]])
    D = allX.shape[1]
    mu, sd = allX.mean(0), allX.std(0) + 1e-6
    net = torch.nn.Sequential(torch.nn.Linear(D, 128), torch.nn.ReLU(), torch.nn.Linear(128, 64),
                              torch.nn.ReLU(), torch.nn.Linear(64, 1))
    alpha = torch.nn.Parameter(torch.tensor(3.0 if args.residual else 0.0), requires_grad=args.residual)
    if args.residual:
        torch.nn.init.zeros_(net[-1].weight); torch.nn.init.zeros_(net[-1].bias)
    opt = torch.optim.Adam(list(net.parameters()) + ([alpha] if args.residual else []), lr=args.lr)
    MU, SD = torch.tensor(mu), torch.tensor(sd)

    def scores(key, grad=True):
        X = torch.from_numpy(inputs(cache[key], args.method))
        with torch.set_grad_enabled(grad):
            return net((X - MU) / SD)[:, 0] + alpha * X[:, 1]     # X[:, 1] = z-scored log response

    def sample_pair():
        if rng.random() < args.neg:
            a, b = rng.choice(args.train, 2, replace=False)
            return (a, rng.choice(scenes[a].names)), (b, rng.choice(scenes[b].names)), -1.0
        sc = args.train[rng.integers(len(args.train))]
        n0, n1 = pos[sc][rng.integers(len(pos[sc]))]
        if rng.random() < 0.5:
            n0, n1 = n1, n0
        return (sc, n0), (sc, n1), 1.0

    def save(path):
        W = [m for m in net if isinstance(m, torch.nn.Linear)]
        np.savez(path, mu=mu, sd=sd, alpha=float(alpha.detach()), **{f"W{i}": l.weight.detach().numpy() for i, l in enumerate(W)},
                 **{f"b{i}": l.bias.detach().numpy() for i, l in enumerate(W)})

    import multiprocessing as mp
    pool = mp.Pool(args.workers)

    def validate():
        res = {}
        for rule in ("response", "score", "grid_resp", "grid_score"):
            jobs = []
            for n0, n1 in vp:
                ff = []
                for n in (n0, n1):
                    c = cache[(args.val, n)]
                    f = dict(xy=c["xy"], desc=full_desc(c), response=c["response"])
                    s = scores((args.val, n), grad=False).numpy() if "score" in rule else None
                    img_shape = (c["wh"][1], c["wh"][0])
                    if rule == "response":
                        g = extract.select(f, args.k, "response")
                    elif rule == "score":
                        g = extract.select(f, args.k, "score", scores=s)
                    else:
                        g = extract.select(f, args.k, "grid", img_shape=img_shape, scores=s)
                    ff.append((g, c["K"], (img_shape[1], img_shape[0])))
                R, t = scenes[args.val].rel_pose(n0, n1)
                jobs.append((ff[0][0]["xy"], ff[0][0]["desc"], ff[1][0]["xy"], ff[1][0]["desc"], ff[0][1], ff[1][1],
                             ff[0][2], ff[1][2], R, t, binary, 1.5))
            r = pool.map(val_job, jobs)
            err = np.array([e for e, _ in r])
            res[rule] = dict(auc=metrics.error_auc(err, [5, 10, 20]), inl=float(np.mean([i for _, i in r])))
        return res

    # ---- supervised ablation: pseudo-labels from the full pool, computed once (same reward machinery)
    if args.mode == "sup":
        lab_pairs = [sample_pair() for _ in range(args.steps * args.batch // 8)]
        lab_pairs = [p for p in lab_pairs if p[2] > 0]
        jobs = ((cache[a]["xy"], full_desc(cache[a]), cache[b]["xy"], full_desc(cache[b]), 1.0, binary, args.th, args.iters)
                for a, b, _ in lab_pairs)          # generator: never hold all descriptor copies at once
        labels = list(pool.imap(reward_job, jobs, chunksize=1))
        print(f"sup labels: {len(labels)} pairs, positive rate {np.mean([r.mean() for r, _ in labels]):.3f}", flush=True)

    log, best, t0 = [], -1.0, time.time()
    v = validate(); v["step"] = 0; log.append(v)
    best = float(np.mean(v["score"]["auc"])); save(out / "best.npz")      # the initial policy is a candidate too
    print(f"step 0 val {json.dumps({k: [round(100 * a, 2) for a in x['auc']] for k, x in v.items() if k != 'step'})}", flush=True)
    for step in range(1, args.steps + 1):
        if args.mode == "rl":
            batch = [sample_pair() for _ in range(args.batch)]
            side, jobs = [], []          # per (pair, image) samples; one image may appear in several pairs
            for a, b, lab in batch:
                kept = []
                for key in (a, b):
                    s = scores(key)
                    sn = s.detach().numpy().astype(np.float64)
                    bias = solve_bias(sn, min(args.k, len(sn) - 1))
                    p = torch.sigmoid(s + float(bias)).clamp(1e-6, 1 - 1e-6)
                    m = rng.random(len(sn)) < p.detach().numpy()
                    side.append((p, m))
                    kept.append(np.flatnonzero(m))
                c0, c1 = cache[a], cache[b]
                jobs.append((c0["xy"][kept[0]], full_desc(c0)[kept[0]], c1["xy"][kept[1]], full_desc(c1)[kept[1]], lab,
                             binary, args.th, args.iters))
            rewards = pool.map(reward_job, jobs)
            loss, rsum, n_inl = 0.0, 0.0, 0
            for j, ((a, b, lab), (r0, r1)) in enumerate(zip(batch, rewards)):
                for (p, m), r in ((side[2 * j], r0), (side[2 * j + 1], r1)):
                    R = np.zeros(len(m), np.float32); R[np.flatnonzero(m)] = r
                    A = torch.from_numpy(R - R.mean())
                    mt = torch.from_numpy(m.astype(np.float32))
                    logp = mt * torch.log(p) + (1 - mt) * torch.log(1 - p)
                    loss = loss - (A * logp).sum() / args.k
                    rsum += float(r.sum()) * (1 if lab > 0 else 0); n_inl += int((r != 0).sum())
            loss = loss / (2 * len(batch))
        else:
            idx = rng.choice(len(lab_pairs), args.batch)
            loss = 0.0
            for i in idx:
                a, b, _ = lab_pairs[i]
                for key, y in ((a, labels[i][0]), (b, labels[i][1])):
                    loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
                        scores(key), torch.from_numpy((y > 0).astype(np.float32)))
            loss = loss / (2 * len(idx)); rsum = n_inl = 0
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 25 == 0:
            print(f"step {step} loss {float(loss):.4f} pos-inlier-kp/pair {rsum / max(args.batch, 1):.1f} "
                  f"{time.time() - t0:.0f}s", flush=True)
        if step % args.val_every == 0 or step == args.steps:
            v = validate(); v["step"] = step; log.append(v)
            sc = float(np.mean(v["score"]["auc"]))
            print(f"step {step} val {json.dumps({k: [round(100 * a, 2) for a in x['auc']] for k, x in v.items() if k != 'step'})}",
                  flush=True)
            save(out / "last.npz")
            if sc > best:
                best = sc; save(out / "best.npz")
            json.dump(dict(args=vars(args), log=log, best_val_mean_auc=best), open(out / "log.json", "w"), indent=1)
    pool.close()
    print("done; best val mean AUC (score rule)", round(100 * best, 2))


if __name__ == "__main__":
    main()
