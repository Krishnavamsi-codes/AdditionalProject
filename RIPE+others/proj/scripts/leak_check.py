"""Data-leak check: are any training / validation images also MegaDepth-1500 test images (or near-duplicates)?

1) exact: image file stems (both datasets name images by Flickr photo id)
2) near-duplicate: 64-bit difference hash (dHash) of the grayscale image, Hamming distance <= --max-ham
3) HPatches: same dHash check against the 696 HPatches images

  python proj/scripts/leak_check.py [--max-ham 6]
"""
import argparse, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfeat import imc  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def dhash(path):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    s = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    return np.packbits((s[:, 1:] > s[:, :-1]).ravel())


def ham(A, B):
    a = np.unpackbits(A, axis=1).astype(np.int16); b = np.unpackbits(B, axis=1).astype(np.int16)
    return a.sum(1)[:, None] + b.sum(1)[None] - 2 * a @ b.T


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-ham", type=int, default=6)
    a = ap.parse_args()
    md = sorted((ROOT / "data/megadepth1500/megadepth1500/images").rglob("*.jpg"))
    hp = sorted((ROOT / "data/hpatches-sequences-release/hpatches-sequences-release").rglob("*.ppm"))
    train = {}
    for sc in ("palace_of_westminster", "grand_place_brussels", "reichstag"):
        S = imc.Scene(ROOT / "data/imc2020", sc, 350)          # exactly the images used for training / validation
        train[sc] = [S.path(n) for n in S.names]
    tr = [p for v in train.values() for p in v]
    print(f"test: {len(md)} MegaDepth-1500 images, {len(hp)} HPatches images; train/val: "
          + ", ".join(f"{k} {len(v)}" for k, v in train.items()))

    stems_md = {p.stem for p in md}
    exact = [p for p in tr if p.stem in stems_md]
    print(f"[1] identical Flickr ids train/val vs MegaDepth-1500: {len(exact)}", [str(p) for p in exact[:5]])

    Hmd = np.stack([dhash(p) for p in md]); Hhp = np.stack([dhash(p) for p in hp]); Htr = np.stack([dhash(p) for p in tr])
    for name, H, files in (("MegaDepth-1500", Hmd, md), ("HPatches", Hhp, hp)):
        D = ham(Htr, H)
        i, j = np.nonzero(D <= a.max_ham)
        print(f"[2] near-duplicates (dHash Hamming <= {a.max_ham}) train/val vs {name}: {len(i)}; "
              f"min distance over all {D.size} pairs = {D.min()}")
        for x, y in list(zip(i, j))[:10]:
            print(f"    {tr[x]}  ~  {files[y]}  (d={D[x, y]})")
    # control: the check does find duplicates when they exist (an image vs. itself after JPEG re-encoding)
    p = md[0]; g = cv2.imread(str(p)); ok, buf = cv2.imencode(".jpg", cv2.resize(g, None, fx=0.5, fy=0.5), [cv2.IMWRITE_JPEG_QUALITY, 70])
    tmp = ROOT / "runs" / "archive" / "_leak_control.jpg"; tmp.write_bytes(buf.tobytes())
    print(f"[control] re-encoded half-size copy of a test image -> distance {int(ham(dhash(tmp)[None], dhash(p)[None])[0, 0])}")
    tmp.unlink()
