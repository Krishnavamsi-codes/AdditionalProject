"""Statistics behind the paper figures: paired bootstrap CIs and the qualitative pair picks.

  python proj/scripts/paper_stats.py [--out paper/figures/stats.json]

Per-pair score = mean over the AUC thresholds of max(0, 1 - err / t); its mean over pairs equals the AUC, so
per-pair score differences decompose the AUC differences reported in the tables.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "runs" / "eval"
AUC_TH = {"hpatches": [1, 3, 5], "megadepth": [5, 10, 20]}
TH = {("megadepth", "orb"): 1.0, ("megadepth", "brisk"): 1.0, ("hpatches", "orb"): 2.0, ("hpatches", "brisk"): 4.0}
K = {"megadepth": 2048, "hpatches": 1024}
COMPARISONS = [("rl", "vanilla"), ("rl", "pool_resp"), ("rl", "sup"), ("rl", "pool_grid"), ("pool_grid", "vanilla")]


def pair_scores(df, v, ds, th):
    e = df[df.variant == v].sort_values("pair")[f"err@{th}"].values.astype(float)
    e = np.where(np.isfinite(e), e, np.inf)
    return np.mean([np.clip(1 - e / t, 0, 1) for t in AUC_TH[ds]], axis=0)


def bootstrap(d, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    b = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "paper" / "figures" / "stats.json"))
    a = ap.parse_args()
    out = {"bootstrap": {}, "picks": {}}
    for ds in ("megadepth", "hpatches"):
        for m in ("orb", "brisk"):
            df = pd.read_csv(EVAL / f"{ds}_{m}_k{K[ds]}_pairs.csv")
            th = TH[(ds, m)]
            S = {v: pair_scores(df, v, ds, th) for v in ("vanilla", "pool_resp", "pool_grid", "sup", "rl")}
            out["bootstrap"][f"{ds}_{m}"] = {f"{x}-{y}": [100 * v for v in bootstrap(S[x] - S[y])]
                                            for x, y in COMPARISONS}
            if ds == "megadepth":
                d = S["rl"] - S["vanilla"]
                mean_gain = d.mean()
                # representative success: per-pair gain closest to the mean gain; failure: closest to -mean gain
                ok = np.where(d > 0)[0]
                rep = int(ok[np.argmin(np.abs(d[ok] - mean_gain))])
                bad = np.where(d < 0)[0]
                fail = int(bad[np.argmin(np.abs(d[bad] + mean_gain))])
                pairs = sorted(df.pair.unique())
                out["picks"][m] = {"mean_gain": 100 * float(mean_gain), "median_gain": 100 * float(np.median(d)),
                                   "share_improved": float((d > 0).mean()), "share_worse": float((d < 0).mean()),
                                   "representative": {"pair": int(pairs[rep]), "gain": 100 * float(d[rep])},
                                   "failure": {"pair": int(pairs[fail]), "gain": 100 * float(d[fail])}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
