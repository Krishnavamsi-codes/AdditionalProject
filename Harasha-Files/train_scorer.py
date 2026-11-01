"""
Step 3: learn WHICH SIFT keypoints to keep (fixed budget K), instead of "top-K by response".

Methods compared at the same budget K (default 1000), with 5-fold cross-validation BY SEQUENCE:
  response   : top-K by SIFT response            (what OpenCV / your baseline does)
  supervised : MLP scorer trained with BCE on "this keypoint yields a correct match" labels
  rl         : the same MLP, fine-tuned with REINFORCE to maximise the TRUE objective
               (number of correct matches after ratio test + mutual NN, using the selected subsets)
  rl_scratch : (optional, --rl_scratch) RL from random init, to show what supervised init buys you
and your earlier BRISK / SIFT / RootSIFT numbers from baseline_results.csv on the same pairs.

Why RL can beat supervised here: supervised labels were made against ALL ~2900 candidates as distractors,
but at test time only K keypoints compete in the ratio test. Selecting K keypoints changes which matches
survive, so the reward depends on the whole selected SET. REINFORCE optimises that directly.

Policy: Plackett-Luce sampling of K keypoints without replacement, probability ~ exp(beta * score).
Test time: deterministic top-K by score.

Run (CPU is fine):
    python train_scorer.py --cache cache --baseline baseline_results.csv --K 1000 --folds 5
Quick test:
    python train_scorer.py --cache cache --max_folds 1 --rl_steps 200 --epochs 3
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows/Anaconda: torch + numpy-MKL both ship libiomp5md.dll
import argparse, glob, time
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ------------------------------------------------------------------ data
def load_cache(cache_dir, limit=0):
    files = sorted(glob.glob(os.path.join(cache_dir, "*.npz")))
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit(f"No .npz files in '{cache_dir}'. Run build_cache.py first.")
    seqs = {}
    for f in files:
        z = np.load(f)
        name = os.path.basename(f)[:-4]
        d = {"name": name, "kind": "illumination" if name.startswith("i_") else "viewpoint"}
        for k in range(1, 7):
            for key in ("xy", "feat", "desc", "resp"):
                d[f"{key}{k}"] = z[f"{key}{k}"]
        for j in range(2, 7):
            H = z[f"H{j}"]
            d[f"proj{j}"] = cv2.perspectiveTransform(
                d["xy1"].reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)
            for side in "AB":
                for lab in ("rep", "nn", "good"):
                    d[f"{side}_{lab}{j}"] = z[f"{side}_{lab}{j}"]
        seqs[name] = d
    return seqs


def make_folds(names, nfold, seed=0):
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(nfold)]
    for g in sorted(set(n[:2] for n in names)):      # stratify: i_ and v_ spread evenly over folds
        ns = sorted(n for n in names if n[:2] == g)
        rng.shuffle(ns)
        for i, n in enumerate(ns):
            folds[i % nfold].append(n)
    return folds


def standardize(seqs, mean, std):
    for d in seqs:
        for k in range(1, 7):
            d[f"x{k}"] = torch.from_numpy(np.clip((d[f"feat{k}"] - mean) / std, -5, 5).astype(np.float32))


# ------------------------------------------------------------------ matching / reward
def match_count(d, j, ia, ib, ratio=0.8, thr=3.0):
    """Match image-1 subset ia against image-j subset ib (ratio test + mutual NN).
    Returns (n_matches, n_correct). Same protocol as baseline_eval.py."""
    Da = d["desc1"][ia].astype(np.float32)
    Db = d[f"desc{j}"][ib].astype(np.float32)
    S = (torch.from_numpy(Da) @ torch.from_numpy(Db).T).numpy()   # torch matmul: avoids a 2nd OpenMP runtime on Windows
    ar = np.arange(len(ia))
    nn_ = S.argmax(1)
    s1 = S[ar, nn_]
    S[ar, nn_] = -1.0
    s2 = S.max(1)
    S[ar, nn_] = s1
    back = S.argmax(0)
    d1 = np.sqrt(np.maximum(2 - 2 * s1, 0))
    d2 = np.sqrt(np.maximum(2 - 2 * s2, 0))
    ok = (d1 < ratio * d2) & (back[nn_] == ar)
    if not ok.any():
        return 0, 0
    err = np.linalg.norm(d[f"proj{j}"][ia][ok] - d[f"xy{j}"][ib][nn_[ok]], axis=1)
    return int(ok.sum()), int((err <= thr).sum())


def topk(score, K):
    K = min(K, len(score))
    return np.argsort(-score)[:K]


# ------------------------------------------------------------------ model
class Scorer(nn.Module):
    def __init__(self, d_in=15, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_supervised(model, train_seqs, label, epochs, lr=1e-3, bs=4096, seed=0):
    X, Y = [], []
    for d in train_seqs:
        X.append(d["x1"])
        Y.append(np.mean([d[f"A_{label}{j}"] for j in range(2, 7)], axis=0).astype(np.float32))
        for j in range(2, 7):
            X.append(d[f"x{j}"])
            Y.append(d[f"B_{label}{j}"].astype(np.float32))
    X = torch.cat(X)
    Y = torch.from_numpy(np.concatenate(Y))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        tot = 0.0
        for i in range(0, len(X), bs):
            b = perm[i:i + bs]
            loss = lossf(model(X[b]), Y[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        print(f"    supervised epoch {ep + 1}/{epochs}  loss {tot / len(X):.4f}")
    return model


# ------------------------------------------------------------------ RL (REINFORCE, Plackett-Luce)
def sample_pl(s, K, gen):
    """Sample K items without replacement with prob ~ exp(s); return (indices, log-prob). s: float64 tensor."""
    N = s.shape[0]
    K = min(K, N)
    u = torch.rand(N, generator=gen, dtype=torch.float64).clamp_(1e-12, 1 - 1e-12)
    g = -torch.log(-torch.log(u))
    idx = torch.topk(s.detach() + g, K).indices          # Gumbel-top-k == Plackett-Luce sample, in order
    c = s.detach().max()
    w = torch.exp(s - c)
    ws = w[idx]
    remaining = w.sum() - (torch.cumsum(ws, 0) - ws)     # mass still available at each step
    logp = ((s[idx] - c) - torch.log(remaining.clamp_min(1e-300))).sum()
    return idx, logp


def train_rl(model, train_seqs, K, steps, pps, M, lr, beta0, seed):
    log_beta = nn.Parameter(torch.tensor(float(np.log(beta0))))
    opt = torch.optim.Adam(list(model.parameters()) + [log_beta], lr=lr)
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    window = []
    for step in range(1, steps + 1):
        loss = 0.0
        for _ in range(pps):
            d = train_seqs[rng.integers(len(train_seqs))]
            j = int(rng.integers(2, 7))
            beta = log_beta.exp()
            sA = (model(d["x1"]) * beta).double()
            sB = (model(d[f"x{j}"]) * beta).double()
            R, LP = [], []
            for _m in range(M):
                ia, la = sample_pl(sA, K, gen)
                ib, lb = sample_pl(sB, K, gen)
                _, nc = match_count(d, j, ia.numpy(), ib.numpy())
                R.append(float(nc)); LP.append(la + lb)
            R = np.array(R)
            baseline = (R.sum() - R) / (M - 1)              # leave-one-out baseline
            adv = torch.from_numpy(R - baseline)
            loss = loss - (adv * torch.stack(LP)).mean()
            window.append(R.mean())
        opt.zero_grad()
        (loss / pps).backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [log_beta], 5.0)
        opt.step()
        if step % 100 == 0 or step == steps:
            print(f"    rl step {step}/{steps}  mean sampled reward (last 100) {np.mean(window[-100 * pps:]):.1f}"
                  f"  beta {log_beta.exp().item():.2f}")
    return model


# ------------------------------------------------------------------ evaluation
def evaluate(score_fns, test_seqs, K, fold):
    rows = []
    for d in test_seqs:
        scores = {m: {k: fn(d, k) for k in range(1, 7)} for m, fn in score_fns.items()}
        for j in range(2, 7):
            for m in score_fns:
                ia = topk(scores[m][1], K)
                ib = topk(scores[m][j], K)
                nm, nc = match_count(d, j, ia, ib)
                rows.append(dict(method=m, fold=fold, seq=d["name"], kind=d["kind"], pair=f"1-{j}",
                                 n_matches=nm, n_correct=nc, precision=nc / max(nm, 1)))
    return rows


def model_scorer(model):
    def fn(d, k):
        with torch.no_grad():
            return model(d[f"x{k}"]).numpy()
    return fn


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--baseline", default="baseline_results.csv")
    ap.add_argument("--K", type=int, default=1000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max_folds", type=int, default=0, help="run only the first N folds (quick test)")
    ap.add_argument("--label", default="good", choices=["good", "nn", "rep"])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--rl_steps", type=int, default=1000)
    ap.add_argument("--pps", type=int, default=2, help="image pairs per RL step")
    ap.add_argument("--M", type=int, default=4, help="sampled selections per pair")
    ap.add_argument("--rl_lr", type=float, default=1e-3)
    ap.add_argument("--beta0", type=float, default=2.0, help="initial policy sharpness")
    ap.add_argument("--rl_scratch", action="store_true", help="also train RL from random init (ablation)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="scorer_results.csv")
    ap.add_argument("--models", default="models")
    a = ap.parse_args()

    os.makedirs(a.models, exist_ok=True)
    torch.manual_seed(a.seed)
    seqs = load_cache(a.cache, a.limit)
    names = sorted(seqs)
    print(f"{len(names)} sequences, {a.folds}-fold CV by sequence, K={a.K}")
    folds = make_folds(names, a.folds, a.seed)
    run_folds = range(a.max_folds if a.max_folds else a.folds)

    all_rows = []
    for f in run_folds:
        t0 = time.time()
        test_names = set(folds[f])
        train = [seqs[n] for n in names if n not in test_names]
        test = [seqs[n] for n in names if n in test_names]
        print(f"\n=== fold {f + 1}/{a.folds}: train {len(train)} seq, test {len(test)} seq ===")

        feats = np.concatenate([np.concatenate([d[f"feat{k}"] for k in range(1, 7)]) for d in train])
        mean, std = feats.mean(0), feats.std(0) + 1e-6
        standardize(train + test, mean, std)

        fns = {"response": lambda d, k: d[f"resp{k}"]}

        print("  supervised ...")
        sup = train_supervised(Scorer(), train, a.label, a.epochs, seed=a.seed)
        fns["supervised"] = model_scorer(sup)

        print("  RL (initialised from supervised) ...")
        rl = Scorer(); rl.load_state_dict(sup.state_dict())
        rl = train_rl(rl, train, a.K, a.rl_steps, a.pps, a.M, a.rl_lr, a.beta0, a.seed)
        fns["rl"] = model_scorer(rl)

        if a.rl_scratch:
            print("  RL from scratch ...")
            rs = train_rl(Scorer(), train, a.K, a.rl_steps, a.pps, a.M, a.rl_lr, a.beta0, a.seed)
            fns["rl_scratch"] = model_scorer(rs)

        torch.save({"sup": sup.state_dict(), "rl": rl.state_dict(), "mean": mean, "std": std},
                   os.path.join(a.models, f"fold{f}.pt"))
        rows = evaluate(fns, test, a.K, f)
        all_rows += rows
        part = pd.DataFrame(rows).groupby("method").n_correct.mean().round(1).to_dict()
        print(f"  fold {f + 1} mean correct matches: {part}   ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(all_rows)

    # add the earlier baselines on exactly the same (sequence, pair) set
    if a.baseline and os.path.exists(a.baseline):
        b = pd.read_csv(a.baseline)
        b = b[b.seq.isin(df.seq.unique())].copy()
        b["method"] = b.method.map({"brisk": "BRISK", "sift": "SIFT (default)", "rootsift": "RootSIFT (default)"})
        b["fold"] = -1
        df = pd.concat([df, b[["method", "fold", "seq", "kind", "pair", "n_matches", "n_correct", "precision"]]])
    df.to_csv(a.out, index=False)

    pd.set_option("display.width", 200)
    cols = ["n_matches", "n_correct", "precision"]
    summ = df.groupby("method")[cols].mean()
    if "BRISK" in summ.index:
        summ["vs_BRISK_%"] = (summ.n_correct / summ.loc["BRISK", "n_correct"] - 1) * 100
    if "response" in summ.index:
        summ["vs_response_%"] = (summ.n_correct / summ.loc["response", "n_correct"] - 1) * 100
    print("\n=== overall (same test pairs for every method) ===")
    print(summ.round(3).sort_values("n_correct"))
    print("\n=== by split: mean correct matches ===")
    print(df.groupby(["method", "kind"]).n_correct.mean().unstack().round(1))
    print("\n=== per fold: mean correct matches ===")
    print(df[df.fold >= 0].groupby(["method", "fold"]).n_correct.mean().unstack().round(1))
    print(f"\nSaved {a.out}; models in {a.models}/")


if __name__ == "__main__":
    main()
