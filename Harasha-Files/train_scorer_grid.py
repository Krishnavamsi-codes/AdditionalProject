"""
Hyperparameter grid for the keypoint scorer, done WITHOUT touching the test folds (nested validation).

For every outer fold:
  1. split the outer-train sequences into inner-train / inner-validation (by sequence, stratified)
  2. try every config on inner-validation:  supervised(label)  and  RL(label, beta0, lr)
  3. pick the best supervised config and the best RL config BY VALIDATION SCORE
  4. retrain both on the full outer-train set and evaluate once on the outer-test sequences
So 'supervised_tuned' gets the same tuning effort as 'rl_tuned' (a fair comparison).

    python train_scorer_grid.py --cache cache --baseline baseline_results.csv --K 1000
Then:
    python analyze_results.py --csv grid_results.csv --rl rl_tuned --sup supervised_tuned
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse, itertools, time
import numpy as np
import pandas as pd
import torch

from train_scorer import (load_cache, make_folds, standardize, Scorer, train_supervised,
                          train_rl, model_scorer, evaluate)


def train_stats(seqs):
    feats = np.concatenate([np.concatenate([d[f"feat{k}"] for k in range(1, 7)]) for d in seqs])
    return feats.mean(0), feats.std(0) + 1e-6


def val_score(fn, val, K):
    rows = evaluate({"m": fn}, val, K, -1)
    return float(np.mean([r["n_correct"] for r in rows]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--baseline", default="baseline_results.csv")
    ap.add_argument("--K", type=int, default=1000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max_folds", type=int, default=0)
    ap.add_argument("--labels", default="good", help="comma list from: good,nn,rep")
    ap.add_argument("--betas", default="1,2,4")
    ap.add_argument("--lrs", default="0.0005,0.002")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--rl_steps", type=int, default=1000)
    ap.add_argument("--pps", type=int, default=2)
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="grid_results.csv")
    a = ap.parse_args()

    labels = a.labels.split(",")
    betas = [float(x) for x in a.betas.split(",")]
    lrs = [float(x) for x in a.lrs.split(",")]
    torch.manual_seed(a.seed)
    seqs = load_cache(a.cache, a.limit)
    names = sorted(seqs)
    folds = make_folds(names, a.folds, a.seed)
    run_folds = range(a.max_folds if a.max_folds else a.folds)
    print(f"{len(names)} sequences; grid: labels={labels} betas={betas} lrs={lrs} "
          f"({len(labels) * (1 + len(betas) * len(lrs))} configs per fold)")

    all_rows, choices = [], []
    for f in run_folds:
        t0 = time.time()
        test_names = set(folds[f])
        train_names = [n for n in names if n not in test_names]
        nf = max(2, round(1 / a.val_frac))
        vf = make_folds(train_names, nf, a.seed + f)
        val_names = set(next(x for x in vf if x))
        inner = [seqs[n] for n in train_names if n not in val_names]
        val = [seqs[n] for n in train_names if n in val_names]
        test = [seqs[n] for n in names if n in test_names]
        print(f"\n=== outer fold {f + 1}/{a.folds}: inner-train {len(inner)}, inner-val {len(val)}, test {len(test)} ===")

        # ---- grid on inner validation
        mean_i, std_i = train_stats(inner)
        standardize(inner + val, mean_i, std_i)
        results = []
        for lab in labels:
            print(f"  [grid] supervised label={lab}")
            sup = train_supervised(Scorer(), inner, lab, a.epochs, seed=a.seed)
            v = val_score(model_scorer(sup), val, a.K)
            results.append(dict(kind="sup", label=lab, beta=np.nan, lr=np.nan, val=v))
            print(f"  [grid] -> val {v:.1f}")
            for beta, lr in itertools.product(betas, lrs):
                rl = Scorer(); rl.load_state_dict(sup.state_dict())
                print(f"  [grid] RL label={lab} beta0={beta} lr={lr}")
                train_rl(rl, inner, a.K, a.rl_steps, a.pps, a.M, lr, beta, a.seed)
                v = val_score(model_scorer(rl), val, a.K)
                results.append(dict(kind="rl", label=lab, beta=beta, lr=lr, val=v))
                print(f"  [grid] -> val {v:.1f}")
        R = pd.DataFrame(results)
        best_sup = R[R.kind == "sup"].sort_values("val").iloc[-1]
        best_rl = R[R.kind == "rl"].sort_values("val").iloc[-1]
        print("\n  validation results:")
        print(R.round(3).to_string(index=False))
        print(f"  chosen supervised: label={best_sup.label} (val {best_sup.val:.1f})")
        print(f"  chosen RL: label={best_rl.label} beta0={best_rl.beta} lr={best_rl.lr} (val {best_rl.val:.1f})")
        choices.append(dict(fold=f, sup_label=best_sup.label, rl_label=best_rl.label, rl_beta=best_rl.beta,
                            rl_lr=best_rl.lr, sup_val=best_sup.val, rl_val=best_rl.val))

        # ---- retrain on the full outer-train set with the chosen configs, test once
        train = inner + val
        mean, std = train_stats(train)
        standardize(train + test, mean, std)
        print("  [final] supervised (tuned)")
        sup_f = train_supervised(Scorer(), train, best_sup.label, a.epochs, seed=a.seed)
        if best_rl.label == best_sup.label:
            init = sup_f
        else:
            print("  [final] supervised for RL init")
            init = train_supervised(Scorer(), train, best_rl.label, a.epochs, seed=a.seed)
        rl_f = Scorer(); rl_f.load_state_dict(init.state_dict())
        print("  [final] RL (tuned)")
        train_rl(rl_f, train, a.K, a.rl_steps, a.pps, a.M, float(best_rl.lr), float(best_rl.beta), a.seed)

        fns = {"response": lambda d, k: d[f"resp{k}"],
               "supervised_tuned": model_scorer(sup_f), "rl_tuned": model_scorer(rl_f)}
        rows = evaluate(fns, test, a.K, f)
        all_rows += rows
        part = pd.DataFrame(rows).groupby("method").n_correct.mean().round(1).to_dict()
        print(f"  fold {f + 1} TEST mean correct matches: {part}   ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(all_rows)
    pd.DataFrame(choices).to_csv("grid_choices.csv", index=False)
    if a.baseline and os.path.exists(a.baseline):
        b = pd.read_csv(a.baseline)
        b = b[b.seq.isin(df.seq.unique())].copy()
        b["method"] = b.method.map({"brisk": "BRISK", "sift": "SIFT (default)", "rootsift": "RootSIFT (default)"})
        b["fold"] = -1
        df = pd.concat([df, b[["method", "fold", "seq", "kind", "pair", "n_matches", "n_correct", "precision"]]])
    df.to_csv(a.out, index=False)

    pd.set_option("display.width", 200)
    summ = df.groupby("method")[["n_matches", "n_correct", "precision"]].mean()
    if "BRISK" in summ.index:
        summ["vs_BRISK_%"] = (summ.n_correct / summ.loc["BRISK", "n_correct"] - 1) * 100
    summ["vs_response_%"] = (summ.n_correct / summ.loc["response", "n_correct"] - 1) * 100
    print("\n=== overall TEST results (tuned on validation only) ===")
    print(summ.round(3).sort_values("n_correct"))
    print("\nchosen configs per fold:")
    print(pd.DataFrame(choices).round(4).to_string(index=False))
    print(f"\nSaved {a.out} and grid_choices.csv")


if __name__ == "__main__":
    main()
