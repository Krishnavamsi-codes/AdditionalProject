"""Metrics, re-implemented to follow glue-factory (used by RIPE / LightGlue papers)."""
import numpy as np
import poselib


def error_auc(errors, thresholds):
    """Identical to gluefactory.utils.tools.cal_error_auc."""
    errors = np.sort(np.asarray(errors, float))
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last = np.searchsorted(errors, t)
        r = np.r_[recall[:last], recall[last - 1]]
        e = np.r_[errors[:last], t]
        aucs.append(float(np.trapezoid(r, x=e) / t))
    return aucs


# ------------------------------------------------------------------ homography (HPatches)

def warp(pts, H):
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


def corner_error(H_est, H_gt, wh):
    w, h = wh
    c = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    return float(np.linalg.norm(warp(c, H_est) - warp(c, H_gt), axis=1).mean())


def estimate_homography(p0, p1, th, seed=0):
    if len(p0) < 4:
        return None, np.zeros(len(p0), bool)
    H, info = poselib.estimate_homography(p0.astype(float), p1.astype(float),
                                          {"max_reproj_error": th, "seed": seed})
    if H is None or not np.isfinite(H).all():
        return None, np.zeros(len(p0), bool)
    return H, np.asarray(info["inliers"], bool)


# ------------------------------------------------------------------ relative pose (MegaDepth)

def camdict(K, w, h):
    return {"model": "PINHOLE", "width": int(w), "height": int(h),
            "params": [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]}


def angle_R(R1, R2):
    c = np.clip((np.trace(R1.T @ R2) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(c)))


def angle_t(t1, t2):
    n = max(np.linalg.norm(t1) * np.linalg.norm(t2), 1e-10)
    a = float(np.degrees(np.arccos(np.clip(t1 @ t2 / n, -1, 1))))
    return min(a, 180 - a)


def estimate_pose(p0, p1, K0, K1, wh0, wh1, th, seed=0):
    if len(p0) < 5:
        return None, np.zeros(len(p0), bool)
    pose, info = poselib.estimate_relative_pose(
        p0.astype(float), p1.astype(float), camdict(K0, *wh0), camdict(K1, *wh1),
        {"max_epipolar_error": th, "seed": seed})
    if pose is None:
        return None, np.zeros(len(p0), bool)
    return (pose.R, pose.t), np.asarray(info["inliers"], bool)


def pose_error(est, R_gt, t_gt):
    if est is None:
        return float("inf")
    R, t = est
    return max(angle_R(R, R_gt), angle_t(t, t_gt))


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def sym_epipolar_px(p0, p1, K0, K1, R, t):
    """Symmetric point-to-epipolar-line distance in pixels (GT geometry)."""
    F = np.linalg.inv(K1).T @ skew(t) @ R @ np.linalg.inv(K0)
    x0 = np.c_[p0, np.ones(len(p0))]
    x1 = np.c_[p1, np.ones(len(p1))]
    l1 = x0 @ F.T          # lines in image 1
    l0 = x1 @ F            # lines in image 0
    num = np.abs(np.sum(x1 * l1, 1))
    d1 = num / np.maximum(np.linalg.norm(l1[:, :2], axis=1), 1e-12)
    d0 = num / np.maximum(np.linalg.norm(l0[:, :2], axis=1), 1e-12)
    return np.maximum(d0, d1)
