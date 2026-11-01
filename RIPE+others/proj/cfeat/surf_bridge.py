"""SURF through a cache filled by the Python 3.7 env (see scripts/surf_extract37.py).

Images are keyed by the md5 of the exact grayscale array, so the SURF keypoints belong to exactly the
image the evaluation/training code sees (PNG is lossless)."""
import hashlib
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("SURF_CACHE", ROOT / "cache" / "surf"))
PY37 = ROOT / ".venv_surf37" / "python.exe"
SCRIPT = ROOT / "proj" / "scripts" / "surf_extract37.py"


def key(img):
    return hashlib.md5(np.ascontiguousarray(img).tobytes() + str(img.shape).encode()).hexdigest()


def dump(img):
    """Write the image for the py3.7 extractor (no-op if already cached)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    k = key(img)
    if not (CACHE / f"{k}_pool.npz").exists() and not (CACHE / f"{k}.png").exists():
        cv2.imwrite(str(CACHE / f"{k}.png"), img)
    return k


def extract_pending():
    env = dict(os.environ)
    e = str(PY37.parent)
    env["PATH"] = f"{e};{e}\\Library\\bin;{e}\\Scripts;" + env.get("PATH", "")
    subprocess.run([str(PY37), str(SCRIPT), str(CACHE)], check=True, env=env)
    for p in CACHE.glob("*.png"):
        p.unlink()


def load(img, pool):
    p = CACHE / f"{key(img)}_{'pool' if pool else 'van'}.npz"
    if not p.exists():
        raise FileNotFoundError(f"SURF cache miss {p.name}: run proj/scripts/prepare_surf.py for this dataset first")
    z = np.load(p)
    return {k: z[k] for k in ("xy", "desc", "response", "size", "angle", "octave")}
