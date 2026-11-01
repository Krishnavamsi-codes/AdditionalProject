"""RIPE-Select: a learned keypoint-selection policy over the classical candidate pool.

The detector and descriptor are untouched; the policy only decides WHICH K of the pool's candidates are
kept (RIPE decides where to put keypoints on a dense heatmap; with a classical detector the action space is
the candidate pool). Per-candidate inputs never contain absolute x, y (dataset bias).

Inference: score = MLP(features); keep top-K (rule 'score') or round-robin over a 16x16 grid by score
(rule 'grid' with scores) - the same selection rules the baselines use with the detector response.
The forward pass is plain numpy so evaluation workers do not need torch.
"""
import cv2
import numpy as np
from scipy.spatial import cKDTree

from .extract import is_binary
from .match import _dist


def _sift_layer(octave):
    o = octave & 255
    o = np.where(o >= 128, o - 256, o)              # signed octave (-1 = upsampled base)
    layer = (octave >> 8) & 255
    return o.astype(np.float32), layer.astype(np.float32)


def candidate_features(f, img, method):
    """[N, D] float32 per-candidate features (+ descriptor block)."""
    n = len(f["xy"])
    if n == 0:
        return np.zeros((0, feat_dim(method)), np.float32)
    xy, resp, size = f["xy"], f["response"].astype(np.float64), f["size"].astype(np.float64)
    lr = np.log(resp + 1e-12)
    rank = np.argsort(np.argsort(-resp)) / max(n - 1, 1)            # 0 = strongest
    if method == "sift":
        o, layer = _sift_layer(f["octave"])
    else:
        o, layer = f["octave"].astype(np.float32), np.zeros(n, np.float32)
    # orientation duplicates / spatial context
    tree = cKDTree(xy)
    dup = np.array([len(x) for x in tree.query_ball_point(xy, 0.5)], np.float32) - 1
    nb = tree.query_ball_point(xy, 16.0)
    dens = np.array([len(x) for x in nb], np.float32) - 1
    loc_rank = np.array([(resp[x] > resp[i]).sum() / max(len(x) - 1, 1) for i, x in enumerate(nb)], np.float32)
    # intra-image distinctiveness: distance to the 2 nearest *other* descriptors (excluding same-location dups)
    binary = is_binary(method)
    pairs = tree.query_pairs(1.0, output_type="ndarray")
    part = np.ones((n, 2))
    if n > 2:                                       # row-chunked: never an N x N matrix (RAM)
        for s in range(0, n, 1024):
            D = _dist(f["desc"][s:s + 1024], f["desc"], binary)
            r = np.arange(len(D))
            D[r, r + s] = np.inf
            if len(pairs):
                for a, b in ((0, 1), (1, 0)):
                    sel = (pairs[:, a] >= s) & (pairs[:, a] < s + len(D))
                    D[pairs[sel, a] - s, pairs[sel, b]] = np.inf
            part[s:s + len(D)] = np.partition(D, 1, axis=1)[:, :2]
    part[~np.isfinite(part)] = np.nanmax(part[np.isfinite(part)]) if np.isfinite(part).any() else 1.0
    med = np.median(part[:, 0]) + 1e-6
    d1, d2 = part[:, 0] / med, part[:, 1] / med
    ratio = part[:, 0] / (part[:, 1] + 1e-6)
    # local appearance at the keypoint scale
    g = img.astype(np.float32) / 255.0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    gm = np.sqrt(gx * gx + gy * gy)
    ii, ii2, iig = (cv2.integral(g, sdepth=cv2.CV_64F), cv2.integral(g * g, sdepth=cv2.CV_64F),
                    cv2.integral(gm, sdepth=cv2.CV_64F))
    h, w = g.shape
    r = np.clip(size * 0.75, 3, 48)
    x0 = np.clip((xy[:, 0] - r).astype(int), 0, w - 1); x1 = np.clip((xy[:, 0] + r).astype(int) + 1, 1, w)
    y0 = np.clip((xy[:, 1] - r).astype(int), 0, h - 1); y1 = np.clip((xy[:, 1] + r).astype(int) + 1, 1, h)
    area = np.maximum((x1 - x0) * (y1 - y0), 1)

    def box(I):
        return (I[y1, x1] - I[y0, x1] - I[y1, x0] + I[y0, x0]) / area
    mean = box(ii); std = np.sqrt(np.maximum(box(ii2) - mean ** 2, 0)); grad = box(iig)
    gmean, gstd = g.mean(), g.std() + 1e-6
    scal = np.stack([
        lr, (lr - lr.mean()) / (lr.std() + 1e-6), rank, np.log(size), o, layer, np.log1p(dup),
        np.log1p(dens), loc_rank, d1, d2, ratio, mean, std, grad, (mean - gmean) / gstd, std / gstd,
        np.full(n, np.log(n)), np.full(n, gmean), np.full(n, gstd),
    ], 1).astype(np.float32)
    return np.concatenate([scal, desc_block(f["desc"], method)], 1)


def desc_block(desc, method):
    if is_binary(method):
        return np.unpackbits(desc, axis=1).astype(np.float32) * 2 - 1
    d = desc.astype(np.float32)
    if method == "sift":                                 # RootSIFT
        d = np.sqrt(d / (np.abs(d).sum(1, keepdims=True) + 1e-8))
    return d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)


N_SCALAR = 20


def feat_dim(method):
    return N_SCALAR + {"sift": 128, "surf": 64, "orb": 256, "brisk": 512}[method]


class NumpyPolicy:
    """Inference-only MLP: weights from train_policy.py (npz with mu, sd, W0, b0, W1, b1, W2, b2)."""

    def __init__(self, path):
        z = np.load(path)
        self.mu, self.sd = z["mu"], z["sd"]
        self.layers = [(z[f"W{i}"], z[f"b{i}"]) for i in range(3)]
        self.alpha = float(z["alpha"]) if "alpha" in z.files else 0.0   # residual on z(log response)

    def __call__(self, F):
        x = (F - self.mu) / self.sd
        prior = self.alpha * F[:, 1]
        for i, (W, b) in enumerate(self.layers):
            x = x @ W.T + b
            if i < 2:
                x = np.maximum(x, 0)
        return x[:, 0] + prior
