"""
Descriptor compression: attack SIFT's biggest drawback (512 bytes per descriptor vs BRISK's 64) and use RL for it.

Keypoints are selected first (the RL scorer from models/foldN.pt if present, else top-K by response),
then the 128-dim RootSIFT descriptor of each selected keypoint is compressed in several ways and matched:

  float32_128      512 B   reference (no compression)
  uint8_128        128 B   8-bit quantisation
  pca{d}_uint8       d B   PCA to d dims (fit on training sequences) + 8-bit quantisation          [non-RL baseline]
  var{d}_uint8       d B   keep the d highest-variance dimensions                                   [non-RL baseline]
  rand{d}_uint8      d B   keep d random dimensions                                                 [control]
  rl{d}_uint8        d B   RL learns WHICH d of the 128 dimensions to keep                          [RL]
BRISK is 64 bytes, so d=64 is the same-size comparison.

RL for dimension selection: a policy over the 128 dimensions (Plackett-Luce, exactly d chosen); reward = number of
correct matches obtained with only the chosen dimensions (ratio test + mutual NN, same protocol as before).

    python compress_eval.py --cache cache --models models --baseline baseline_results.csv --K 1000
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from train_scorer import load_cache, make_folds, standardize, Scorer, topk, sample_pl


# ------------------------------------------------------------------ helpers (torch matmul: avoids 2nd OpenMP runtime on Windows)
def mm(A, B):
    return (torch.from_numpy(np.ascontiguousarray(A, dtype=np.float32)) @
            torch.from_numpy(np.ascontiguousarray(B, dtype=np.float32))).numpy()


def match_generic(Da, Db, proj, xyb, ratio=0.8, thr=3.0):
    """L2 matching with ratio test + mutual NN on arbitrary (compressed) descriptors. Returns (n_matches, n_correct)."""
    Ta = torch.from_numpy(np.ascontiguousarray(Da, dtype=np.float32))
    Tb = torch.from_numpy(np.ascontiguousarray(Db, dtype=np.float32))
    D2 = (Ta * Ta).sum(1, keepdim=True) + (Tb * Tb).sum(1)[None, :] - 2.0 * (Ta @ Tb.T)
    D = D2.clamp_min(0).sqrt().numpy()
    ar = np.arange(len(D))
    nn_ = D.argmin(1)
    d1 = D[ar, nn_].copy()
    D[ar, nn_] = np.inf
    d2 = D.min(1)
    D[ar, nn_] = d1
    back = D.argmin(0)
    ok = (d1 < ratio * d2) & (back[nn_] == ar)
    if not ok.any():
        return 0, 0
    err = np.linalg.norm(proj[ok] - xyb[nn_[ok]], axis=1)
    return int(ok.sum()), int((err <= thr).sum())


def fit_pca(sample):
    mean = sample.mean(0)
    X = torch.from_numpy(np.ascontiguousarray(sample - mean, dtype=np.float32)).double()
    C = X.T @ X / (len(X) - 1)
    _, evecs = torch.linalg.eigh(C)
    V = torch.flip(evecs, dims=[1]).float().numpy()          # columns sorted by decreasing variance
    return mean.astype(np.float32), V


def load_ckpt(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


# ------------------------------------------------------------------ RL for dimension selection
def train_dims(train_seqs, sel, dq, d, steps, pps, M, lr, seed):
    theta = nn.Parameter(torch.zeros(128, dtype=torch.float64))
    opt = torch.optim.Adam([theta], lr=lr)
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    window = []
    for step in range(1, steps + 1):
        loss = 0.0
        for _ in range(pps):
            s = train_seqs[rng.integers(len(train_seqs))]
            j = int(rng.integers(2, 7))
            n = s["name"]
            proj = s[f"proj{j}"][sel[n][1]]
            xyb = s[f"xy{j}"][sel[n][j]]
            R, LP = [], []
            for _m in range(M):
                idx, lp = sample_pl(theta, d, gen)
                ii = idx.numpy()
                _, nc = match_generic(dq[n][1][:, ii], dq[n][j][:, ii], proj, xyb)
                R.append(float(nc)); LP.append(lp)
            R = np.array(R)
            adv = torch.from_numpy(R - (R.sum() - R) / (M - 1))
            loss = loss - (adv * torch.stack(LP)).mean()
            window.append(R.mean())
        opt.zero_grad()
        (loss / pps).backward()
        opt.step()
        if step % 100 == 0 or step == steps:
            print(f"      dims d={d} step {step}/{steps}  mean sampled reward (last 100) {np.mean(window[-100 * pps:]):.1f}")
    return np.sort(np.argsort(-theta.detach().numpy())[:d])


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--models", default="models")
    ap.add_argument("--baseline", default="baseline_results.csv")
    ap.add_argument("--selector", default="rl", choices=["rl", "response"])
    ap.add_argument("--K", type=int, default=1000)
    ap.add_argument("--dims", default="32,64")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max_folds", type=int, default=0)
    ap.add_argument("--rl_dim_steps", type=int, default=400)
    ap.add_argument("--rl_dim_lr", type=float, default=0.1)
    ap.add_argument("--pps", type=int, default=2)
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--sample_per_img", type=int, default=500)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="compress_results.csv")
    a = ap.parse_args()

    dims = [int(x) for x in a.dims.split(",")]
    torch.manual_seed(a.seed)
    seqs = load_cache(a.cache, a.limit)
    names = sorted(seqs)
    folds = make_folds(names, a.folds, a.seed)
    run_folds = range(a.max_folds if a.max_folds else a.folds)
    rng = np.random.default_rng(a.seed)

    bytes_of = {"float32_128": 512, "uint8_128": 128}
    for d in dims:
        for p in ("pca", "var", "rand", "rl"):
            bytes_of[f"{p}{d}_uint8"] = d

    all_rows = []
    for f in run_folds:
        t0 = time.time()
        test_names = set(folds[f])
        train = [seqs[n] for n in names if n not in test_names]
        test = [seqs[n] for n in names if n in test_names]
        everyone = train + test
        print(f"\n=== fold {f + 1}/{a.folds}: train {len(train)}, test {len(test)} ===")

        # ---- 1) keypoint selection (fixed budget K)
        ck_path = os.path.join(a.models, f"fold{f}.pt")
        use_rl = a.selector == "rl" and os.path.exists(ck_path)
        if a.selector == "rl" and not use_rl:
            print(f"  (no {ck_path}; selecting keypoints by response instead)")
        if use_rl:
            ck = load_ckpt(ck_path)
            standardize(everyone, ck["mean"], ck["std"])
            model = Scorer(); model.load_state_dict(ck["rl"]); model.eval()
        sel = {}
        for d_ in everyone:
            sel[d_["name"]] = {}
            for k in range(1, 7):
                if use_rl:
                    with torch.no_grad():
                        sc = model(d_[f"x{k}"]).numpy()
                else:
                    sc = d_[f"resp{k}"]
                sel[d_["name"]][k] = topk(sc, a.K)
        print(f"  keypoint selection: {'RL scorer' if use_rl else 'response'}")

        # ---- 2) fit compressors on TRAIN descriptors only
        sample = np.concatenate([
            d_[f"desc{k}"][rng.choice(len(d_[f"desc{k}"]), min(a.sample_per_img, len(d_[f"desc{k}"])), replace=False)]
            for d_ in train for k in range(1, 7)]).astype(np.float32)
        scale = float(np.percentile(sample, 99.9))
        q8 = lambda X: np.round(np.clip(X / scale, 0, 1) * 255).astype(np.float32)
        mean, V = fit_pca(sample)
        var_order = np.argsort(-sample.var(0))

        sel_desc = {d_["name"]: {k: d_[f"desc{k}"][sel[d_["name"]][k]].astype(np.float32) for k in range(1, 7)}
                    for d_ in everyone}
        dq = {n: {k: q8(v) for k, v in dd.items()} for n, dd in sel_desc.items() if n in {t["name"] for t in train}}

        variants = {"float32_128": lambda X: X, "uint8_128": q8}
        for d in dims:
            Vd = np.ascontiguousarray(V[:, :d])
            Z = mm(sample - mean, Vd)
            sc_d = np.percentile(np.abs(Z), 99.9, axis=0) / 127.0 + 1e-9
            variants[f"pca{d}_uint8"] = (lambda X, Vd=Vd, sc_d=sc_d:
                                         (np.round(np.clip(mm(X - mean, Vd) / sc_d, -127, 127)) * sc_d).astype(np.float32))
            vi = np.sort(var_order[:d])
            variants[f"var{d}_uint8"] = lambda X, vi=vi: q8(X)[:, vi]
            ri = np.sort(rng.choice(128, d, replace=False))
            variants[f"rand{d}_uint8"] = lambda X, ri=ri: q8(X)[:, ri]
            print(f"  RL dimension selection d={d} ...")
            li = train_dims(train, sel, dq, d, a.rl_dim_steps, a.pps, a.M, a.rl_dim_lr, a.seed)
            variants[f"rl{d}_uint8"] = lambda X, li=li: q8(X)[:, li]
            ov = len(set(li.tolist()) & set(vi.tolist()))
            print(f"      RL dims overlap with top-variance dims: {ov}/{d}")

        # ---- 3) evaluate on held-out sequences
        for d_ in test:
            n = d_["name"]
            for vname, enc in variants.items():
                E = {k: enc(sel_desc[n][k]) for k in range(1, 7)}
                for j in range(2, 7):
                    proj = d_[f"proj{j}"][sel[n][1]]
                    xyb = d_[f"xy{j}"][sel[n][j]]
                    nm, nc = match_generic(E[1], E[j], proj, xyb)
                    all_rows.append(dict(method=vname, bytes=bytes_of[vname], fold=f, seq=n, kind=d_["kind"],
                                         pair=f"1-{j}", n_matches=nm, n_correct=nc, precision=nc / max(nm, 1)))
        part = pd.DataFrame([r for r in all_rows if r["fold"] == f]).groupby("method").n_correct.mean().round(1)
        print(f"  fold {f + 1} mean correct matches:\n{part.to_string()}\n  ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(all_rows)
    if a.baseline and os.path.exists(a.baseline):
        b = pd.read_csv(a.baseline)
        b = b[b.seq.isin(df.seq.unique())].copy()
        b["bytes"] = b.method.map({"brisk": 64, "sift": 512, "rootsift": 512})
        b["method"] = b.method.map({"brisk": "BRISK", "sift": "SIFT (default)", "rootsift": "RootSIFT (default)"})
        b["fold"] = -1
        df = pd.concat([df, b[["method", "bytes", "fold", "seq", "kind", "pair", "n_matches", "n_correct", "precision"]]])
    df.to_csv(a.out, index=False)

    pd.set_option("display.width", 200)
    summ = df.groupby("method").agg(bytes=("bytes", "first"), n_matches=("n_matches", "mean"),
                                    n_correct=("n_correct", "mean"), precision=("precision", "mean"))
    if "BRISK" in summ.index:
        summ["vs_BRISK_%"] = (summ.n_correct / summ.loc["BRISK", "n_correct"] - 1) * 100
    summ["vs_float32_%"] = (summ.n_correct / summ.loc["float32_128", "n_correct"] - 1) * 100
    print(f"\n=== descriptor compression (keypoints chosen by: {'RL scorer' if a.selector == 'rl' else 'response'}) ===")
    print(summ.sort_values(["bytes", "n_correct"], ascending=[True, False]).round(3).to_string())
    print("\nRead it like this: compare rows with the SAME bytes. BRISK = 64 B. "
          "rl64_uint8 vs pca64_uint8 vs var64_uint8 vs rand64_uint8 shows whether RL's choice of dimensions adds anything.")
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
