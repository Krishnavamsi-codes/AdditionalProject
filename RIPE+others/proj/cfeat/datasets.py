"""Loaders for HPatches-sequences and MegaDepth-1500 (glue-factory package layout)."""
from pathlib import Path
import cv2
import numpy as np

# Same 8 high-resolution sequences excluded by D2-Net / glue-factory -> 108 sequences, 540 pairs
HPATCHES_IGNORED = ("i_contruction", "i_crownnight", "i_dc", "i_pencils",
                    "i_whitebuilding", "v_artisans", "v_astronautis", "v_talent")


def load_resized_gray(path, size, side):
    """Read image as grayscale and resize so that the `side` ('short'|'long') has `size` px.
    Returns image and per-axis scale (sx, sy) such that x_new = x_old * s (corner convention)."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape
    if size is None:
        return img, (1.0, 1.0)
    if (side == "short") ^ (h / w < 1.0):     # same rule as glue-factory ImagePreprocessor
        nh, nw = int(size * h / w), size
    else:
        nh, nw = size, int(size * w / h)
    if (side == "short" and min(h, w) == size) or (side == "long" and max(h, w) == size):
        return img, (1.0, 1.0)
    interp = cv2.INTER_AREA if nh < h else cv2.INTER_LINEAR
    out = cv2.resize(img, (nw, nh), interpolation=interp)
    return out, (nw / w, nh / h)


def _scale_mat(s):
    return np.diag([s[0], s[1], 1.0])


# ------------------------------------------------------------------------------- HPatches

def hpatches_pairs(root, resize=480):
    """Yield dicts: seq, idx, kind('i'|'v'), img0, img1, H (resized coords, OpenCV pixel-centre)."""
    root = Path(root)
    seqs = sorted(p.name for p in root.iterdir() if p.is_dir())
    for seq in seqs:
        if seq in HPATCHES_IGNORED:
            continue
        img0, s0 = load_resized_gray(root / seq / "1.ppm", resize, "short")
        for i in range(2, 7):
            img1, s1 = load_resized_gray(root / seq / f"{i}.ppm", resize, "short")
            H = np.loadtxt(root / seq / f"H_1_{i}")
            # GT H is in the original pixel frame; rescale (corner convention, then shift to centre conv.)
            Hs = _scale_mat(s1) @ H @ np.linalg.inv(_scale_mat(s0))
            T = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1.0]])
            Hc = np.linalg.inv(T) @ Hs @ T           # pixel-centre convention used by OpenCV keypoints
            yield dict(seq=seq, idx=i, kind=seq[0], img0=img0, img1=img1, H=Hc,
                       wh0=(img0.shape[1], img0.shape[0]))


# --------------------------------------------------------------------------- MegaDepth-1500

def _parse_view(tok):
    R = np.array(tok[:9], float).reshape(3, 3)
    t = np.array(tok[9:12], float)
    model, w, h = tok[12], int(tok[13]), int(tok[14])
    p = np.array(tok[15:], float)
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
        fx = fy = p[0]; cx, cy = p[1], p[2]
    else:                                             # PINHOLE / OPENCV ...
        fx, fy, cx, cy = p[:4]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])   # COLMAP corner convention
    return R, t, K, (w, h)


def megadepth1500_pairs(root, resize=1200, limit=None, scenes_out=None):
    """Glue-factory layout: <root>/images/*, <root>/views.txt, <root>/pairs.txt (and depths/).
    Yields dicts with grayscale images, intrinsics (OpenCV pixel-centre convention, resized)
    and GT relative pose T_0to1 = (R, t)."""
    root = Path(root)
    views = {}
    for line in (root / "views.txt").read_text().strip().split("\n"):
        tok = line.strip().split(" ")
        views[tok[0]] = _parse_view(tok[1:])
    pairs = [l.split(" ") for l in (root / "pairs.txt").read_text().strip().split("\n")]
    if limit:  # evenly spaced subset -> covers both scenes
        step = max(1, len(pairs) // limit)
        pairs = pairs[::step][:limit]
    cache = {}

    def load(name):
        if name not in cache:
            img, s = load_resized_gray(root / "images" / name, resize, "long")
            R, t, K, _ = views[name]
            Ks = _scale_mat(s) @ K
            Ks[0, 2] -= 0.5; Ks[1, 2] -= 0.5           # corner -> pixel-centre convention
            cache.clear()
            cache[name] = (img, Ks, R, t)
        return cache[name]

    for n0, n1 in (p[:2] for p in pairs):
        img0, K0, R0, t0 = load(n0)
        img1, K1, R1, t1 = load(n1)
        R = R1 @ R0.T
        t = t1 - R @ t0
        yield dict(name0=n0, name1=n1, img0=img0, img1=img1, K0=K0, K1=K1, R=R, t=t,
                   wh0=(img0.shape[1], img0.shape[0]), wh1=(img1.shape[1], img1.shape[0]))
