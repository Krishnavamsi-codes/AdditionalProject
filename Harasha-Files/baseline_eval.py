"""
Step 1: baseline evaluation harness on HPatches (BRISK vs SIFT vs RootSIFT).

Setup:
    pip install opencv-python numpy pandas tqdm
    Download + extract HPatches sequences (~1.3 GB):
      http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz
    You should get a folder hpatches-sequences-release/ containing i_* (illumination)
    and v_* (viewpoint) folders, each with 1.ppm..6.ppm and H_1_2..H_1_6.

Run:
    python baseline_eval.py --root hpatches-sequences-release --max_kp 1000
    python baseline_eval.py --root hpatches-sequences-release --max_kp 1000 --limit 20   # quick test
"""
import argparse, glob, os, time
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------- feature extractors ----------------
def top_k(kps, des, k):
    if des is None or len(kps) == 0:
        return [], None
    order = np.argsort([-kp.response for kp in kps])[:k]
    return [kps[i] for i in order], des[order]


def make_extractor(name):
    if name == "brisk":
        det = cv2.BRISK_create()
        return det, cv2.NORM_HAMMING, (lambda d: d)
    if name == "sift":
        det = cv2.SIFT_create(nfeatures=0)
        return det, cv2.NORM_L2, (lambda d: d)
    if name == "rootsift":
        det = cv2.SIFT_create(nfeatures=0)

        def root(d):
            d = d / (np.abs(d).sum(axis=1, keepdims=True) + 1e-7)
            return np.sqrt(d).astype(np.float32)

        return det, cv2.NORM_L2, root
    raise ValueError(name)


# ---------------- matching ----------------
def match(des1, des2, norm, ratio=0.8):
    """Ratio test + mutual nearest neighbour. Returns list of (i, j)."""
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return []
    bf = cv2.BFMatcher(norm)
    m12 = bf.knnMatch(des1, des2, k=2)
    m21 = bf.knnMatch(des2, des1, k=1)
    back = {m[0].queryIdx: m[0].trainIdx for m in m21 if len(m) > 0}
    out = []
    for pair in m12:
        if len(pair) < 2:
            continue
        a, b = pair
        if a.distance < ratio * b.distance and back.get(a.trainIdx) == a.queryIdx:
            out.append((a.queryIdx, a.trainIdx))
    return out


def correct_mask(kps1, kps2, matches, H, thr=3.0):
    if len(matches) == 0:
        return np.zeros(0, dtype=bool)
    p1 = np.float32([kps1[i].pt for i, _ in matches]).reshape(-1, 1, 2)
    p2 = np.float32([kps2[j].pt for _, j in matches])
    proj = cv2.perspectiveTransform(p1, H).reshape(-1, 2)
    return np.linalg.norm(proj - p2, axis=1) <= thr


def repeatability(kps1, kps2, H, shape2, thr=3.0):
    """Fraction of image-1 keypoints that land within thr px of some image-2 keypoint."""
    if len(kps1) == 0 or len(kps2) == 0:
        return 0.0
    p1 = np.float32([k.pt for k in kps1]).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(p1, H).reshape(-1, 2)
    h, w = shape2[:2]
    inside = (proj[:, 0] >= 0) & (proj[:, 0] < w) & (proj[:, 1] >= 0) & (proj[:, 1] < h)
    if inside.sum() == 0:
        return 0.0
    p2 = np.float32([k.pt for k in kps2])
    d = np.linalg.norm(proj[inside][:, None, :] - p2[None, :, :], axis=2).min(axis=1)
    return float((d <= thr).mean())


# ---------------- main ----------------
def evaluate(root, methods, max_kp, limit):
    seqs = sorted(glob.glob(os.path.join(root, "*")))
    seqs = [s for s in seqs if os.path.isdir(s)]
    if limit:
        seqs = seqs[:limit]
    rows = []
    for method in methods:
        det, norm, post = make_extractor(method)
        for s in tqdm(seqs, desc=method):
            name = os.path.basename(s)
            kind = "illumination" if name.startswith("i_") else "viewpoint"
            img1 = cv2.imread(os.path.join(s, "1.ppm"), cv2.IMREAD_GRAYSCALE)
            t0 = time.time()
            k1, d1 = det.detectAndCompute(img1, None)
            k1, d1 = top_k(k1, d1, max_kp)
            t_ext = time.time() - t0
            if d1 is not None:
                d1 = post(d1)
            for j in range(2, 7):
                img2 = cv2.imread(os.path.join(s, f"{j}.ppm"), cv2.IMREAD_GRAYSCALE)
                H = np.loadtxt(os.path.join(s, f"H_1_{j}"))
                k2, d2 = det.detectAndCompute(img2, None)
                k2, d2 = top_k(k2, d2, max_kp)
                if d2 is not None:
                    d2 = post(d2)
                t0 = time.time()
                m = match(d1, d2, norm)
                t_match = time.time() - t0
                ok = correct_mask(k1, k2, m, H)
                n_corr = int(ok.sum())
                rows.append(dict(
                    method=method, seq=name, kind=kind, pair=f"1-{j}",
                    n_kp1=len(k1), n_kp2=len(k2), n_matches=len(m), n_correct=n_corr,
                    precision=n_corr / max(len(m), 1),
                    mma=n_corr / max(min(len(k1), len(k2)), 1),
                    repeat=repeatability(k1, k2, H, img2.shape),
                    desc_bytes=(d1.shape[1] * d1.dtype.itemsize) if d1 is not None else 0,
                    t_extract=t_ext, t_match=t_match,
                ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--max_kp", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0, help="only first N sequences (quick test)")
    ap.add_argument("--methods", default="brisk,sift,rootsift")
    ap.add_argument("--out", default="baseline_results.csv")
    a = ap.parse_args()

    df = evaluate(a.root, a.methods.split(","), a.max_kp, a.limit)
    df.to_csv(a.out, index=False)

    cols = ["n_matches", "n_correct", "precision", "mma", "repeat", "desc_bytes", "t_match"]
    pd.set_option("display.width", 200)
    print("\n=== overall ===")
    print(df.groupby("method")[cols].mean().round(4))
    print("\n=== by split ===")
    print(df.groupby(["method", "kind"])[cols].mean().round(4))

    b = df[df.method == "brisk"]["n_correct"].mean()
    for m in df.method.unique():
        v = df[df.method == m]["n_correct"].mean()
        print(f"{m}: correct matches = {v:.1f}  ({(v / max(b, 1e-9) - 1) * 100:+.0f}% vs BRISK)")
    print(f"\nSaved to {a.out}")
