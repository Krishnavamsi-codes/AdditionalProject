"""Fill the SURF cache for a dataset: dump the exact resized images, then run the py3.7 SURF extractor.

  python proj/scripts/prepare_surf.py --dataset hpatches|megadepth|imc --root R [--limit N]
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfeat import datasets, surf_bridge  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["hpatches", "megadepth", "imc"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--max-images", type=int, default=350)
    a = ap.parse_args()
    n = 0
    if a.dataset == "hpatches":
        for s in datasets.hpatches_pairs(a.root, 480):
            surf_bridge.dump(s["img0"]); surf_bridge.dump(s["img1"]); n += 1
    elif a.dataset == "megadepth":
        for s in datasets.megadepth1500_pairs(a.root, 1200, limit=a.limit):
            surf_bridge.dump(s["img0"]); surf_bridge.dump(s["img1"]); n += 1
    else:
        from cfeat import imc
        for img in imc.iter_images(a.root, a.scenes, a.max_images):
            surf_bridge.dump(img); n += 1
    print("dumped", n, "items; extracting with py3.7 SURF ...", flush=True)
    surf_bridge.extract_pending()
