"""Classical feature extraction (SIFT / SURF / ORB) with a *candidate pool* + *selection* split.

The split matters for the RIPE-style enhancement later: the detector produces a pool of
candidates, and a selection rule picks the K keypoints that are actually used.
  * vanilla  : OpenCV defaults, top-K by the detector's own response (what everybody reports)
  * pool     : lowered thresholds -> large candidate pool, still selected by response
Both use exactly the same descriptor and the same budget K.
"""
import cv2
import numpy as np

# ---------------------------------------------------------------- detector factories

def _sift(pool):
    # OpenCV defaults: contrastThreshold=0.04, edgeThreshold=10, nOctaveLayers=3, sigma=1.6
    if pool:
        return cv2.SIFT_create(nfeatures=0, contrastThreshold=0.01, edgeThreshold=20)
    return cv2.SIFT_create(nfeatures=0)


def _surf(pool):
    # OpenCV defaults: hessianThreshold=100, nOctaves=4, nOctaveLayers=3, extended=False (64-D), upright=False
    if not hasattr(cv2, "xfeatures2d"):
        raise RuntimeError("SURF needs opencv-contrib built with OPENCV_ENABLE_NONFREE")
    return cv2.xfeatures2d.SURF_create(hessianThreshold=20 if pool else 100)


_SURF_OK = None


def _has_surf():
    global _SURF_OK
    if _SURF_OK is None:
        try:
            cv2.xfeatures2d.SURF_create()
            _SURF_OK = True
        except Exception:
            _SURF_OK = False
    return _SURF_OK


BINARY = ("orb", "brisk")          # Hamming-matched descriptors


def is_binary(method):
    return method.lower() in BINARY


def _brisk(pool):
    # OpenCV defaults: thresh=30 (AGAST), octaves=3, patternScale=1.0; 512-bit descriptor. No nfeatures: top-K by response.
    create = cv2.BRISK_create if hasattr(cv2, "BRISK_create") else cv2.xfeatures2d.BRISK_create   # moved to contrib in 5.0
    return create(thresh=10 if pool else 30)


def _orb(pool, k):
    # OpenCV defaults: scaleFactor=1.2, nlevels=8, edgeThreshold=31, fastThreshold=20, HARRIS_SCORE
    if pool:
        return cv2.ORB_create(nfeatures=8 * k, fastThreshold=7)
    return cv2.ORB_create(nfeatures=k)


def detect_describe(img_gray, method, k, pool=False):
    """Return dict with xy [N,2], desc [N,D], response, size, angle, octave (all numpy).

    N <= K for vanilla (top-K by response); for pool=True, N can be >> K and selection
    happens later with `select`.
    """
    method = method.lower()
    if method == "surf" and not _has_surf():
        from . import surf_bridge          # SURF from the py3.7 cache (pip OpenCV has nonfree disabled)
        out = surf_bridge.load(img_gray, pool)
        return out if pool else select(out, k, "response")
    if method == "sift":
        det = _sift(pool)
    elif method == "surf":
        det = _surf(pool)
    elif method == "orb":
        det = _orb(pool, k)
    elif method == "brisk":
        det = _brisk(pool)
    else:
        raise ValueError(method)

    kps, desc = det.detectAndCompute(img_gray, None)
    if desc is None or len(kps) == 0:
        D = {"orb": 32, "brisk": 64, "surf": 64}.get(method, 128)
        dt = np.uint8 if is_binary(method) else np.float32
        return dict(xy=np.zeros((0, 2), np.float32), desc=np.zeros((0, D), dt),
                    response=np.zeros(0, np.float32), size=np.zeros(0, np.float32),
                    angle=np.zeros(0, np.float32), octave=np.zeros(0, np.int32))
    out = dict(
        xy=np.array([kp.pt for kp in kps], np.float32),
        desc=desc,
        response=np.abs(np.array([kp.response for kp in kps], np.float32)),
        size=np.array([kp.size for kp in kps], np.float32),
        angle=np.array([kp.angle for kp in kps], np.float32),
        octave=np.array([kp.octave for kp in kps], np.int32),
    )
    if not pool:
        out = select(out, k, "response")
    return out


# ---------------------------------------------------------------- selection rules

def _take(f, idx):
    return {key: v[idx] for key, v in f.items()}


def select(f, k, rule="response", img_shape=None, scores=None, grid=None, rng=None):
    """Pick at most k keypoints from a pool.

    rule = 'response' : detector's own ranking (vanilla behaviour)
           'grid'     : response ranking, round-robin over a spatial grid (ANMS-like, ORB-SLAM style)
           'random'   : uniform random (sanity floor)
           'score'    : external per-keypoint score (e.g. a learned policy) -> top-k
    """
    n = len(f["xy"])
    if n <= k:
        return f
    if rule == "response":
        idx = np.argsort(-f["response"], kind="stable")[:k]
    elif rule == "score":
        idx = np.argsort(-scores, kind="stable")[:k]
    elif rule == "random":
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(n, k, replace=False)
    elif rule == "grid":
        assert img_shape is not None
        h, w = img_shape[:2]
        g = grid or 16
        cell = (np.clip(f["xy"][:, 1] * g / h, 0, g - 1).astype(int) * g
                + np.clip(f["xy"][:, 0] * g / w, 0, g - 1).astype(int))
        key = f["response"] if scores is None else scores     # grid over the policy score when given
        order = np.argsort(-key, kind="stable")
        # rank of each kp inside its cell
        rank = np.zeros(n, int)
        seen = {}
        for i in order:
            c = cell[i]
            rank[i] = seen.get(c, 0)
            seen[c] = rank[i] + 1
        # sort by (rank within cell, -response)
        idx = np.lexsort((-key, rank))[:k]
    else:
        raise ValueError(rule)
    return _take(f, np.sort(idx))
