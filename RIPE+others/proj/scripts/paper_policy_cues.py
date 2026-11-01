"""What the selection policy learned: rank correlation of its score with each scalar cue, and how the keypoints
it adds (kept by the policy, not by top-K response of the same pool) differ from the ones it drops.

  python proj/scripts/paper_policy_cues.py [--images 40]

Uses the seed-0 policies on MegaDepth-1500 test images (analysis only, nothing is trained or selected here).
Writes runs/report/policy_cues.json.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfeat import datasets, extract, match, metrics, policy  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MD_ROOT = ROOT / "data" / "megadepth1500" / "megadepth1500"
CUES = ["log response", "z(log response)", "response rank", "log size", "octave", "layer", "duplicates",
        "local density", "local response rank", "NN desc. distance", "2nd-NN desc. distance", "NN ratio",
        "patch mean", "patch std", "patch gradient", "rel. patch mean", "rel. patch std", "log pool size",
        "image mean", "image std"]
PER_IMAGE = {17, 18, 19}          # constant within an image: no within-image correlation


def correct_rates(m, pol, pairs, k, n=50, px=2.0):
    """Share of keypoints in image 0 that end up in a correct MNN match (symmetric epipolar error < px),
    for the keypoints both rules keep, the ones only the policy keeps, and the ones only top-K response keeps.
    Each rule is matched with its own selection in both images, as at test time."""
    hit = {"both": [0, 0], "added": [0, 0], "dropped": [0, 0]}
    for i in np.linspace(0, len(pairs) - 1, n).astype(int):
        s = pairs[i]
        sel = {}
        for j in (0, 1):
            img = s[f"img{j}"]
            pool = extract.select(extract.detect_describe(img, m, k, pool=True), 4 * k, "response")
            sc = pol(policy.candidate_features(pool, img, m))
            sel[j] = {"pool": pool, "rl": np.argsort(-sc)[:k], "resp": np.argsort(-pool["response"])[:k]}
        ok = {}
        for rule in ("rl", "resp"):
            f0, f1 = (sel[j]["pool"] for j in (0, 1))
            i0, i1 = sel[0][rule], sel[1][rule]
            mm = match.mnn(f0["desc"][i0], f1["desc"][i1], binary=True)
            e = metrics.sym_epipolar_px(f0["xy"][i0][mm[:, 0]], f1["xy"][i1][mm[:, 1]], s["K0"], s["K1"], s["R"], s["t"])
            ok[rule] = set(i0[mm[e < px, 0]].tolist())
        rl, rs = set(sel[0]["rl"].tolist()), set(sel[0]["resp"].tolist())
        for name, ids, rule in (("both", rl & rs, "rl"), ("added", rl - rs, "rl"), ("dropped", rs - rl, "resp")):
            hit[name][0] += len(ids & ok[rule]); hit[name][1] += len(ids)
    return {k_: 100 * a / max(b, 1) for k_, (a, b) in hit.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=40)
    ap.add_argument("--k", type=int, default=2048)
    a = ap.parse_args()
    pairs = list(datasets.megadepth1500_pairs(MD_ROOT, 1200, 500))
    pick = np.linspace(0, len(pairs) - 1, a.images // 2).astype(int)
    imgs = [(pairs[i][f"name{j}"], pairs[i][f"img{j}"]) for i in pick for j in (0, 1)]
    out = {}
    for m in ("orb", "brisk"):
        pol = policy.NumpyPolicy(ROOT / "runs" / "train" / f"{m}_rl_s0" / "best.npz")
        rho = {c: [] for i, c in enumerate(CUES) if i not in PER_IMAGE}
        added, dropped = [], []
        for name, img in imgs:
            pool = extract.select(extract.detect_describe(img, m, a.k, pool=True), 4 * a.k, "response")
            F = policy.candidate_features(pool, img, m)
            s = pol(F)
            for i, c in enumerate(CUES):
                if i not in PER_IMAGE:
                    rho[c].append(stats.spearmanr(s, F[:, i]).statistic)
            by_score = set(np.argsort(-s)[:a.k]); by_resp = set(np.argsort(-pool["response"])[:a.k])
            added.append(F[sorted(by_score - by_resp), :20]); dropped.append(F[sorted(by_resp - by_score), :20])
            print(m, name, len(by_score - by_resp), flush=True)
        A, D = np.concatenate(added), np.concatenate(dropped)
        out[m] = {"spearman": {c: [float(np.nanmean(v)), float(np.nanstd(v))] for c, v in rho.items()},
                  "swapped_share": len(A) / (len(imgs) * a.k),
                  "added_median": dict(zip(CUES, map(float, np.median(A, 0)))),
                  "dropped_median": dict(zip(CUES, map(float, np.median(D, 0))))}
        out[m]["correct_rate"] = correct_rates(m, pol, pairs, a.k)
    p = ROOT / "runs" / "report" / "policy_cues.json"
    p.write_text(json.dumps(out, indent=1))
    for m, r in out.items():
        print(f"\n{m}: {100 * r['swapped_share']:.0f}% of kept keypoints differ from top-K response; "
              f"correct-match rate (%) {r['correct_rate']}")
        for c, (mu, sd) in sorted(r["spearman"].items(), key=lambda z: -abs(z[1][0])):
            print(f"  {c:24s} rho {mu:+.2f} ± {sd:.2f}   added {r['added_median'][c]:8.3f}  dropped {r['dropped_median'][c]:8.3f}")


if __name__ == "__main__":
    main()
