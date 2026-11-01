"""IMC-2020 PhotoTourism training scenes (COLMAP layout <scene>/dense/{images,sparse}).

Used ONLY for training / validation of the selection policy. Never contains MegaDepth-1500 test scenes
(brandenburg_gate = 0022, st_peters_square = 0015 are refused below) and never HPatches.
Positive pairs = covisible images (shared COLMAP 3D points), negatives = images from different scenes.
"""
import struct
from pathlib import Path

import numpy as np

from .datasets import load_resized_gray, _scale_mat

FORBIDDEN = ("brandenburg_gate", "st_peters_square")
CAM_NPARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}


def _rd(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def read_cameras(path):
    cams = {}
    with open(path, "rb") as f:
        for _ in range(_rd(f, "Q")[0]):
            cid, model, w, h = _rd(f, "iiQQ")
            p = np.array(_rd(f, "d" * CAM_NPARAMS[model]))
            if model in (0, 2, 3):          # SIMPLE_*: f, cx, cy
                fx = fy = p[0]; cx, cy = p[1], p[2]
            else:
                fx, fy, cx, cy = p[:4]
            cams[cid] = (np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), (w, h))
    return cams


def qvec2rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def read_images(path):
    ims = {}
    with open(path, "rb") as f:
        for _ in range(_rd(f, "Q")[0]):
            iid = _rd(f, "i")[0]
            q = _rd(f, "dddd"); t = _rd(f, "ddd"); cid = _rd(f, "i")[0]
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            n2d = _rd(f, "Q")[0]
            data = np.frombuffer(f.read(24 * n2d), dtype=[("x", "<f8"), ("y", "<f8"), ("p", "<i8")])
            pts = data["p"][data["p"] >= 0]
            ims[name.decode()] = dict(R=qvec2rot(q), t=np.array(t), cam=cid, p3d=np.unique(pts))
    return ims


class Scene:
    def __init__(self, root, name, max_images=None, seed=0):
        assert name not in FORBIDDEN, f"{name} is a MegaDepth-1500 test scene"
        self.name, self.dir = name, Path(root) / name / "dense"
        cams = read_cameras(self.dir / "sparse" / "cameras.bin")
        ims = read_images(self.dir / "sparse" / "images.bin")
        names = sorted(n for n in ims if (self.dir / "images" / n).exists() and len(ims[n]["p3d"]) >= 100)
        if max_images and len(names) > max_images:
            names = sorted(np.random.default_rng(seed).choice(names, max_images, replace=False).tolist())
        self.names = names
        self.info = {n: dict(K=cams[ims[n]["cam"]][0], R=ims[n]["R"], t=ims[n]["t"], p3d=ims[n]["p3d"])
                     for n in names}

    def path(self, n):
        return self.dir / "images" / n

    def positive_pairs(self, min_shared=50, min_ratio=0.1):
        """Covisible pairs: shared 3D points >= min_shared and >= min_ratio of the smaller image's points."""
        sets = [set(self.info[n]["p3d"].tolist()) for n in self.names]
        out = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                s = len(sets[i] & sets[j])
                if s >= min_shared and s >= min_ratio * min(len(sets[i]), len(sets[j])):
                    out.append((self.names[i], self.names[j]))
        return out

    def load(self, n, resize=1200):
        """Grayscale image (long side `resize`) + intrinsics in OpenCV pixel-centre convention."""
        img, s = load_resized_gray(self.path(n), resize, "long")
        K = _scale_mat(s) @ self.info[n]["K"]
        K[0, 2] -= 0.5; K[1, 2] -= 0.5
        return img, K

    def rel_pose(self, n0, n1):
        a, b = self.info[n0], self.info[n1]
        R = b["R"] @ a["R"].T
        return R, b["t"] - R @ a["t"]


def iter_images(root, scenes, max_images=None, resize=1200):
    for sc in scenes:
        S = Scene(root, sc, max_images)
        for n in S.names:
            yield S.load(n, resize)[0]
