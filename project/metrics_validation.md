# Phase 1: Scientific Audit of Evaluation Metrics

This document provides a brutally honest, rigorous audit of the benchmarking methodology implemented in `metrics/evaluator.py`. The objective is to verify that the metrics align with state-of-the-art academic practices (e.g., HPatches, D2-Net, SuperPoint) and to identify any underlying assumptions, simplifications, or biases.

---

## 1. Repeatability

### Mathematical Definition
Repeatability measures the geometric stability of a feature detector. It is defined as the number of keypoints detected in both images (within a distance threshold after homography projection) divided by the minimum number of keypoints detected in the co-visible regions of either image.

### Implementation Audit
- **Algorithm**: Keypoints from image 1 (`pts1`) are projected into image 2 using the ground-truth homography $H$. A KD-Tree is used to query the nearest neighbor in `pts2` for each projected point. If the Euclidean distance is $\le \epsilon$ (3 pixels by default), it counts as repeatable.
- **Co-visibility Filtering**: Strictly enforced. Keypoints that project outside the bounds of the opposing image are discarded prior to evaluation.
- **Denominator**: `min(n1_covis, n2_covis)` where $n$ is the number of co-visible keypoints.
- **Assumptions**: One-to-one matching is implicitly enforced by nearest neighbor, but KD-Tree query `k=1` means multiple `pts1` can technically map to the same `pts2` if they are densely clustered.
- **Literature Match**: Yes, this aligns perfectly with the standard Oxford/HPatches protocol.
- **Bias**: KD-Tree `k=1` without mutual exclusion favors methods that detect extremely dense, overlapping keypoints (like FAST variants), slightly inflating their repeatability compared to a strict Hungarian matching algorithm.

---

## 2. Mean Matching Accuracy (MMA)

### Mathematical Definition
MMA measures the fraction of correctly established matches at a specific pixel threshold, evaluating both the detector's localization accuracy and the descriptor's distinctiveness.

### Implementation Audit
- **Algorithm**: Computes the Euclidean distance between the homography-projected query point and the actual train point for all established matches. Evaluated at thresholds $\epsilon \in \{1, 2, 3, 5, 10\}$.
- **Co-visibility Filtering**: The pipeline validates that the query point from image 1 projects inside image 2.
- **Literature Match**: Exact match with the **D2-Net (2019)** evaluation protocol.
- **Bias**: Because MMA is a *fraction* of established matches (not total keypoints), a method that returns only 5 highly-confident, perfectly accurate matches will score an MMA of 1.0 (100%), even if it fails to cover the image. It does not penalize low match counts.

---

## 3. Matching Score

### Mathematical Definition
The number of correct matches (inliers based on threshold $\epsilon$) divided by the total number of features detected in the shared viewpoint (co-visible region of image 1).

### Implementation Audit
- **Algorithm**: `correct_matches / n1_covis`.
- **Literature Match**: Exact match with the **SuperPoint (2018)** and **SuperGlue (2020)** protocols.
- **Bias**: Strongly penalizes methods that detect many non-repeatable or non-distinctive keypoints, balancing the weakness of MMA.

---

## 4. Average Precision (AP), Precision, and Recall

### Mathematical Definition
Evaluates the descriptor ranking. Matches are sorted by descriptor distance. Precision and Recall are computed cumulatively. AP is the area under the PR curve.

### Implementation Audit
- **Algorithm**: Matches are sorted by raw descriptor distance (Hamming or L2). Cumulative True Positives (TP) and False Positives (FP) are tracked. Recall denominator is `min(n1_covis, n2_covis)`. 
- **Assumptions**: Uses `cv2.BFMatcher` without cross-check to allow the PR curve to evaluate the raw nearest-neighbor ranking without mutual exclusion artificially truncating the curve.
- **Literature Match**: Matches the standard HPatches descriptor evaluation protocol.
- **Simplification**: The denominator for maximum possible matches is assumed to be `min(n1_covis, n2_covis)`. In reality, the maximum possible matches is the true number of repeatable points, not the total detected points. This implies the calculated Recall is technically a "Pseudo-Recall" bound by detector limits, which is standard but worth noting.

---

## 5. Homography Estimation Metrics

### Reprojection Error (Corner Error)
- **Algorithm**: The 4 corners of the $H \times W$ image are projected using $H_{gt}$ and $H_{est}$. The error is the mean Euclidean distance between the 4 projected corners.
- **Status**: FAITHFUL. This is the gold-standard metric for homography evaluation (used in SuperPoint). It translates abstract matrix error into physically meaningful pixel error.

### Homography Error (Matrix Norm)
- **Algorithm**: `np.linalg.norm(H_gt_norm - H_est_norm)` where matrices are normalized by $H_{3,3}$.
- **Status**: FLAWED/SIMPLIFIED. The $L_2$ norm of a $3 \times 3$ matrix difference is mathematically unstable because the scale, translation, and perspective terms operate in fundamentally different metric spaces. Normalizing by $H_{3,3}$ can lead to extreme sensitivity if $H_{3,3} \approx 0$ (which happens in extreme perspective warps). 
- **Recommendation**: Reprojection error is the only reliable metric for geometric transformations. Matrix norm error should be ignored in the final analysis.

---

## Conclusion
The benchmark is scientifically rigorous and matches modern evaluation protocols (D2-Net, SuperPoint). The only theoretical weakness is the Homography Matrix Norm error, which is an inherently flawed metric in projective geometry (though standard in simplistic evaluations). Reprojection error serves as the robust alternative. All other metrics are mathematically sound and properly localized via co-visible inversion.
