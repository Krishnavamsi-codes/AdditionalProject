"""
Average several runs of train_scorer.py that used different --seed (different CV splits AND different network inits).
Per (method, sequence, pair) the n_correct is averaged over seeds; only methods present in ALL files are kept.
The merged CSV works with analyze_results.py (the per-fold table is skipped because fold = -1).

    python merge_seeds.py --csvs scorer_results.csv scorer_results_s1.csv scorer_results_s2.csv --out merged_seeds.csv
"""
import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--csvs", nargs="+", required=True)
ap.add_argument("--out", default="merged_seeds.csv")
a = ap.parse_args()

dfs = [pd.read_csv(p) for p in a.csvs]
common = set.intersection(*[set(d.method.unique()) for d in dfs])
dropped = set.union(*[set(d.method.unique()) for d in dfs]) - common
al = pd.concat([d[d.method.isin(common)] for d in dfs])
g = al.groupby(["method", "seq", "kind", "pair"], as_index=False)[["n_matches", "n_correct", "precision"]].mean()
g["fold"] = -1
g.to_csv(a.out, index=False)
print(f"merged {len(dfs)} runs -> {a.out}; methods kept: {sorted(common)}; dropped (not in every run): {sorted(dropped)}")
