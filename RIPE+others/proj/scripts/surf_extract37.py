"""SURF extraction in the Python 3.7 env (opencv-contrib-python 3.4.2.17 still ships nonfree SURF).

The main env cannot run SURF (pip OpenCV >= 3.4.3 has nonfree disabled), so the main env writes the exact
resized grayscale images as lossless PNG named <md5>.png (see cfeat/surf_bridge.py) and this script writes
<md5>_van.npz (hessianThreshold=100, OpenCV default) and <md5>_pool.npz (hessianThreshold=20) next to them.
Keypoints are stored sorted by response; the main env applies the same top-K `select` as for SIFT/ORB.

Usage (py3.7 env):  python surf_extract37.py <cache_dir> [max_pool]
"""
import sys
from pathlib import Path

import cv2
import numpy as np


def run(img, thr, cap):
    det = cv2.xfeatures2d.SURF_create(hessianThreshold=thr)   # nOctaves=4, nOctaveLayers=3, 64-D, not upright
    kps, desc = det.detectAndCompute(img, None)
    if desc is None or len(kps) == 0:
        return dict(xy=np.zeros((0, 2), np.float32), desc=np.zeros((0, 64), np.float32),
                    response=np.zeros(0, np.float32), size=np.zeros(0, np.float32),
                    angle=np.zeros(0, np.float32), octave=np.zeros(0, np.int32))
    resp = np.abs(np.array([k.response for k in kps], np.float32))
    o = np.argsort(-resp, kind="stable")[:cap]
    return dict(xy=np.array([k.pt for k in kps], np.float32)[o], desc=desc[o].astype(np.float32),
                response=resp[o], size=np.array([k.size for k in kps], np.float32)[o],
                angle=np.array([k.angle for k in kps], np.float32)[o],
                octave=np.array([k.octave for k in kps], np.int32)[o])


def main():
    d = Path(sys.argv[1])
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
    todo = [p for p in sorted(d.glob("*.png")) if not (d / (p.stem + "_pool.npz")).exists()]
    for i, p in enumerate(todo):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        np.savez(str(d / (p.stem + "_van.npz")), **run(img, 100, cap))
        np.savez(str(d / (p.stem + "_pool.npz")), **run(img, 20, cap))
        if i % 100 == 0:
            print("surf", i, "/", len(todo), flush=True)
    print("surf done", len(todo))


if __name__ == "__main__":
    main()
