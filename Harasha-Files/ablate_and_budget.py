"""
Two things reviewers always ask:
  (1) FEATURE ABLATION: which of the 15 per-keypoint features does the scorer actually need?
      In particular: are the descriptor-based features (self_d1, self_ratio, desc_max, desc_entropy) necessary?
      (If yes, selection needs descriptors for ALL candidates first, so it does not save description time.)
  (2) BUDGET CURVE: does the gain hold for other keypoint budgets K (250, 500, 1000, 2000)?
      Supervised scorers do not depend on K, so the same trained model is evaluated at every K.
      The RL scorer from models/foldN.pt was trained at K=1000; at other K it is a transfer test.

Same 5-fold CV by sequence as train_scorer.py (use the same --seed, default 0).
    python ablate_and_budget.py --cache cache --models models --baseline baseline_results.csv --K 1000
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse, time
import numpy as np
import pandas as pd
import torch

from train_scorer import (load_cache, make_folds, standardize, Scorer, train_supervised,
                          model_scorer, evaluate)

ALL = list(range(15))
GROUPS = {
    "sup_full": ALL,
    "sup_no_response": [i for i in ALL if i not in (0, 1)],           # log_resp, log_resp_rel
    "sup_no_scale": [i for i in ALL if i not in (2, 3, 4)],           # size, octave, layer
    "sup_no_image_stats": [i for i in ALL if i not in (5, 6, 7, 8)],  # local_std, grad_mag, mineig, edge_ratio
    "sup_no_context": [i for i in ALL if i not in (9, 10)],           # border_dist, density
    "sup_no_descriptor": [i for i in ALL if i not in (11, 12, 13, 14)],  # self_d1, self_ratio, desc_max, desc_entropy
    "sup_only_response": [0, 1],
    "sup_only_descriptor": [11, 12, 13, 14],
}


def slice_x(seqs, keep):
    for d in seqs:
        for k in range(1, 7):
            d[f"x{k}"] = d[f"x{k}"][:, keep]


def tag(rows, K, main_K):
    if K != main_K:
        for r in rows:
            r["method"] = f'{r["method"]}@{K}'
    return rows


def load_ckpt(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--models", default="models")
    ap.add_argument("--baseline", default="baseline_results.csv")
    ap.add_argument("--K", type=int, default=1000)
    ap.add_argument("--budgets", default="250,500,2000")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max_folds", type=int, default=0)
    ap.add_argument("--label", default="good")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ablation_results.csv")
    a = ap.parse_args()

    Ks = sorted(set([a.K] + [int(x) for x in a.budgets.split(",") if x]))
    torch.manual_seed(a.seed)
    seqs = load_cache(a.cache, a.limit)
    names = sorted(seqs)
    folds = make_folds(names, a.folds, a.seed)
    run_folds = range(a.max_folds if a.max_folds else a.folds)
    resp_fn = {"response": lambda d, k: d[f"resp{k}"]}

    rows = []
    for f in run_folds:
        t0 = time.time()
        test_names = set(folds[f])
        train = [seqs[n] for n in names if n not in test_names]
        test = [seqs[n] for n in names if n in test_names]
        print(f"\n=== fold {f + 1}/{a.folds}: train {len(train)}, test {len(test)} ===")

        for K in Ks:
            rows += tag(evaluate(resp_fn, test, K, f), K, a.K)

        ck_path = os.path.join(a.models, f"fold{f}.pt")
        if os.path.exists(ck_path):                       # RL scorer from the main run, evaluated at every K
            ck = load_ckpt(ck_path)
            standardize(test, ck["mean"], ck["std"])
            rl = Scorer(); rl.load_state_dict(ck["rl"]); rl.eval()
            for K in Ks:
                rows += tag(evaluate({"rl": model_scorer(rl)}, test, K, f), K, a.K)
        else:
            print(f"  (no {ck_path}: skipping the RL model)")

        feats = np.concatenate([np.concatenate([d[f"feat{k}"] for k in range(1, 7)]) for d in train])
        mean, std = feats.mean(0), feats.std(0) + 1e-6
        for gname, keep in GROUPS.items():
            print(f"  [{gname}] {len(keep)} features")
            standardize(train + test, mean, std)
            slice_x(train + test, keep)
            model = train_supervised(Scorer(d_in=len(keep)), train, a.label, a.epochs, seed=a.seed)
            for K in (Ks if gname == "sup_full" else [a.K]):
                rows += tag(evaluate({gname: model_scorer(model)}, test, K, f), K, a.K)
        part = pd.DataFrame([r for r in rows if r["fold"] == f and "@" not in r["method"]]
                            ).groupby("method").n_correct.mean().round(1)
        print(f"  fold {f + 1} mean correct matches (K={a.K}):\n{part.to_string()}\n  ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    if a.baseline and os.path.exists(a.baseline):
        b = pd.read_csv(a.baseline)
        b = b[b.seq.isin(df.seq.unique())].copy()
        b["method"] = b.method.map({"brisk": "BRISK", "sift": "SIFT (default)", "rootsift": "RootSIFT (default)"})
        b["fold"] = -1
        df = pd.concat([df, b[["method", "fold", "seq", "kind", "pair", "n_matches", "n_correct", "precision"]]])
    df.to_csv(a.out, index=False)

    pd.set_option("display.width", 200)
    s = df.groupby(["seq", "method"]).n_correct.mean().unstack()
    rng = np.random.default_rng(0)

    print(f"\n=== feature ablation (supervised scorer, K={a.K}); change relative to sup_full, 95% bootstrap CI over sequences ===")
    full = s["sup_full"].values
    out = []
    for g in GROUPS:
        x = s[g].values
        idx = rng.integers(0, len(x), size=(5000, len(x)))
        rel = x[idx].mean(1) / full[idx].mean(1) - 1
        lo, hi = np.percentile(rel, [2.5, 97.5]) * 100
        out.append(dict(variant=g, n_features=len(GROUPS[g]), n_correct=x.mean(),
                        change_pct=(x.mean() / full.mean() - 1) * 100, ci_lo=lo, ci_hi=hi))
    ab = pd.DataFrame(out)
    if "response" in s.columns:
        print(f"(reference: response = {s['response'].mean():.1f})")
    print(ab.round(2).to_string(index=False))

    print("\n=== budget curve: mean correct matches vs number of kept keypoints K ===")
    rows_b = []
    for K in Ks:
        sfx = "" if K == a.K else f"@{K}"
        r = {"K": K}
        for m in ("response", "sup_full", "rl"):
            if f"{m}{sfx}" in s.columns:
                r[m] = s[f"{m}{sfx}"].mean()
        if "response" in r and "sup_full" in r:
            r["sup_vs_resp_%"] = (r["sup_full"] / r["response"] - 1) * 100
        if "response" in r and "rl" in r:
            r["rl_vs_resp_%"] = (r["rl"] / r["response"] - 1) * 100
        rows_b.append(r)
    print(pd.DataFrame(rows_b).round(1).to_string(index=False))
    print("(rl was trained at K=%d; other K values are a transfer test)" % a.K)
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
