"""
Step 2: build a cache for training the keypoint scorer.

For every HPatches sequence and every image it stores:
  - up to --max_cand candidate SIFT keypoints (loose thresholds, so many candidates)
  - RootSIFT descriptors (float16)
  - 15 per-keypoint features (what a scorer is allowed to look at)
and for every pair (image 1 -> image j), in BOTH directions, three labels per keypoint:
  rep  : a keypoint exists within 3 px of the projected location (geometric repeatability)
  nn   : the nearest descriptor neighbour in the other image is the correct one
  good : nn AND mutual nearest neighbour AND ratio test (<0.8)  -> "this keypoint yields a correct match"

Run:
    python build_cache.py --root hpatches-sequences-release --out cache --max_cand 3000
    python build_cache.py --root hpatches-sequences-release --out cache --limit 5     # quick test
"""
import argparse, glob, os, time
import cv2
import numpy as np
from tqdm import tqdm

FEATS = ["log_resp", "log_resp_rel", "log_size", "octave", "layer", "local_std", "grad_mag",
         "log_mineig", "log_edge_ratio", "border_dist", "density", "self_d1", "self_ratio",
         "desc_max", "desc_entropy"]


def root_sift(d):
    d = d / (np.abs(d).sum(axis=1, keepdims=True) + 1e-7)
    return np.sqrt(d).astype(np.float32)


def unpack_octave(o):
    octave = o & 255
    layer = (o >> 8) & 255
    if octave >= 128:
        octave -= 256
    return octave, layer


def extract(gray, sift, max_cand):
    kps, des = sift.detectAndCompute(gray, None)
    if des is None or len(kps) < 20:
        return None
    order = np.argsort([-k.response for k in kps])[:max_cand]
    kps = [kps[i] for i in order]
    return kps, root_sift(des[order])


def compute_features(gray, kps, des):
    g = gray.astype(np.float32) / 255.0
    h, w = g.shape
    n = len(kps)
    xy = np.float32([k.pt for k in kps])
    resp = np.float32([k.response for k in kps])
    size = np.float32([k.size for k in kps])
    ol = np.float32([unpack_octave(k.octave) for k in kps])

    xi = np.clip(np.round(xy[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(xy[:, 1]).astype(int), 0, h - 1)

    mean = cv2.blur(g, (9, 9))
    sq = cv2.blur(g * g, (9, 9))
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    gm = cv2.blur(np.sqrt(gx * gx + gy * gy), (9, 9))
    mineig = cv2.cornerMinEigenVal(g, 5, 3)
    gs = cv2.GaussianBlur(g, (0, 0), 2.0)
    Ixx = cv2.Sobel(gs, cv2.CV_32F, 2, 0, ksize=3)
    Iyy = cv2.Sobel(gs, cv2.CV_32F, 0, 2, ksize=3)
    Ixy = cv2.Sobel(gs, cv2.CV_32F, 1, 1, ksize=3)
    tr, det = Ixx + Iyy, Ixx * Iyy - Ixy * Ixy
    edge_ratio = np.clip(np.where(det > 1e-12, tr * tr / np.maximum(det, 1e-12), 100.0), 0, 100)

    border = np.minimum.reduce([xy[:, 0], xy[:, 1], w - 1 - xy[:, 0], h - 1 - xy[:, 1]]) / min(h, w)

    d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    density = (d2 < 25.0 ** 2).sum(1) - 1

    # self-distinctiveness: how similar is this descriptor to another one in the SAME image
    S = des @ des.T
    ar = np.arange(n)
    S[ar, ar] = -1
    idx = S.argmax(1)
    s1 = S[ar, idx]
    S[ar, idx] = -1
    s2 = S.max(1)
    d1 = np.sqrt(np.maximum(2 - 2 * s1, 0))
    d2n = np.sqrt(np.maximum(2 - 2 * s2, 0))
    self_ratio = d1 / (d2n + 1e-6)

    p = des ** 2
    ent = -(p * np.log(p + 1e-12)).sum(1)

    f = np.stack([
        np.log(resp + 1e-6),
        np.log((resp + 1e-6) / (np.median(resp) + 1e-6)),
        np.log(size + 1e-6),
        ol[:, 0], ol[:, 1],
        std[yi, xi], gm[yi, xi],
        np.log(mineig[yi, xi] + 1e-8),
        np.log1p(edge_ratio[yi, xi]),
        np.clip(border, 0, 1),
        density.astype(np.float32),
        d1, self_ratio,
        des.max(1), ent,
    ], axis=1).astype(np.float32)
    assert f.shape[1] == len(FEATS)
    return xy, resp, f


def labels(S, xy_a, xy_b, H, shape_b, thr=3.0):
    """Labels for keypoints of image a, given similarity matrix S (Na x Nb) and H: a -> b."""
    na = S.shape[0]
    ar = np.arange(na)
    p = cv2.perspectiveTransform(xy_a.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)
    hb, wb = shape_b[:2]
    inside = (p[:, 0] >= 0) & (p[:, 0] < wb) & (p[:, 1] >= 0) & (p[:, 1] < hb)

    dist = np.sqrt(((p[:, None, :] - xy_b[None, :, :]) ** 2).sum(-1))
    rep = inside & (dist.min(1) <= thr)

    nn = S.argmax(1)
    s1 = S[ar, nn]
    S2 = S.copy()
    S2[ar, nn] = -1
    s2 = S2.max(1)
    nn_ok = inside & (np.linalg.norm(p - xy_b[nn], axis=1) <= thr)
    mutual = S.argmax(0)[nn] == ar
    d1 = np.sqrt(np.maximum(2 - 2 * s1, 0))
    d2 = np.sqrt(np.maximum(2 - 2 * s2, 0))
    good = nn_ok & mutual & (d1 < 0.8 * d2)
    return rep.astype(np.uint8), nn_ok.astype(np.uint8), good.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="cache")
    ap.add_argument("--max_cand", type=int, default=3000)
    ap.add_argument("--contrast", type=float, default=0.01)   # default SIFT is 0.04 -> we are looser
    ap.add_argument("--edge", type=float, default=20)         # default SIFT is 10   -> we are looser
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    seqs = sorted(s for s in glob.glob(os.path.join(a.root, "*"))
                  if os.path.exists(os.path.join(s, "1.ppm")))
    if not seqs:
        raise SystemExit(f"No sequences (1.ppm + H_1_2...) found in '{a.root}'")
    if a.limit:
        seqs = seqs[:a.limit]

    sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=a.contrast, edgeThreshold=a.edge)
    stats = {"cand": [], "rep": [], "nn": [], "good": []}
    t0 = time.time()
    for s in tqdm(seqs):
        name = os.path.basename(s)
        out_path = os.path.join(a.out, name + ".npz")
        if os.path.exists(out_path):
            continue
        data, shapes, ok = {}, [], True
        descs = {}
        for k in range(1, 7):
            gray = cv2.imread(os.path.join(s, f"{k}.ppm"), cv2.IMREAD_GRAYSCALE)
            r = extract(gray, sift, a.max_cand)
            if r is None:
                ok = False
                break
            kps, des = r
            xy, resp, feat = compute_features(gray, kps, des)
            data[f"xy{k}"], data[f"resp{k}"], data[f"feat{k}"] = xy, resp, feat
            data[f"desc{k}"] = des.astype(np.float16)
            descs[k] = des
            shapes.append(gray.shape)
            stats["cand"].append(len(kps))
        if not ok:
            print(f"skip {name}: too few keypoints")
            continue
        for j in range(2, 7):
            H = np.loadtxt(os.path.join(s, f"H_1_{j}"))
            S = descs[1] @ descs[j].T
            ra, na, ga = labels(S, data["xy1"], data[f"xy{j}"], H, shapes[j - 1])
            rb, nb, gb = labels(S.T, data[f"xy{j}"], data["xy1"], np.linalg.inv(H), shapes[0])
            data[f"H{j}"] = H
            data[f"A_rep{j}"], data[f"A_nn{j}"], data[f"A_good{j}"] = ra, na, ga
            data[f"B_rep{j}"], data[f"B_nn{j}"], data[f"B_good{j}"] = rb, nb, gb
            stats["rep"].append(ra.mean()); stats["nn"].append(na.mean()); stats["good"].append(ga.mean())
        np.savez_compressed(out_path, names=np.array(FEATS), shapes=np.array(shapes), **data)

    print(f"\nDone in {time.time() - t0:.0f}s. Cache in '{a.out}/'")
    if stats["cand"]:
        print(f"avg candidates per image : {np.mean(stats['cand']):.0f}")
        print(f"label rate  rep={np.mean(stats['rep']):.3f}  nn={np.mean(stats['nn']):.3f}  good={np.mean(stats['good']):.3f}")


if __name__ == "__main__":
    main()
