# Metrics Audit

This document reviews the metrics currently implemented in `metrics/evaluator.py`, evaluating them against standard literature protocols (specifically HPatches full-image matching evaluation).

## 1. Repeatability

* **Status**: C — simplified implementation
* **Mathematical Definition**: $Rep = \frac{| \{ x_i \in K_1 \mid \exists y_j \in K_2, \| H(x_i) - y_j \|_2 \le \epsilon \} |}{\min(|K_1|, |K_2|)}$
* **Implementation**: We project all keypoints from image 1 to image 2 using $H$, find the nearest neighbor, and check if the distance is $\le \epsilon$. We normalize by $\min(|K_1|, |K_2|)$.
* **Assumptions**: We assume the bounding box of image 1 completely maps within image 2.
* **Deviations from Literature**: In rigorous evaluation, repeatability is only computed for keypoints that lie in the **co-visible (overlapping) region**. Keypoints projected outside the target image boundaries should not penalize the detector. Furthermore, the normalization term should be the number of keypoints strictly inside the overlapping region, not the total keypoints.
* **Possible Improvements**: Filter $K_1$ such that $H(x_i)$ is within image 2 boundaries. Filter $K_2$ such that $H^{-1}(y_j)$ is within image 1 boundaries. Use the counts of these filtered sets for calculation and normalization.

## 2. Average Precision (AP) (Previously NN mAP)

* **Status**: B — adapted implementation
* **Mathematical Definition**: $AP = \int_0^1 P(R) dR$. Discrete: $\sum_k P(k) \Delta R(k)$ for ranked matches.
* **Implementation**: Matches are sorted by descriptor distance. We compute cumulative TP and FP, yielding Precision and Recall at each threshold, and aggregate AP.
* **Assumptions**: Total possible true matches is $\min(|K_1|, |K_2|)$.
* **Deviations from Literature**: True *mAP* is the mean of Average Precisions over many queries (e.g., in patch retrieval). For pair-wise image matching, we are computing AP per pair.
* **Possible Improvements**: Rename to AP to accurately reflect the computation. Normalize recall by the number of keypoints in the co-visible region rather than $\min(|K_1|, |K_2|)$.

## 3. Matching Score

* **Status**: B — adapted implementation
* **Mathematical Definition**: $MS = \frac{\text{Correct Matches}}{\text{Total Keypoints in Reference}}$
* **Implementation**: Computes ratio of correct matches (geometric distance $\le \epsilon$) over total extracted keypoints $|K_1|$.
* **Assumptions**: Uses all $K_1$.
* **Deviations from Literature**: Like Repeatability, standard Matching Score normalizes only over keypoints in the shared visible region.
* **Possible Improvements**: Restrict the denominator to $|K_{1, shared}|$.

## 4. Precision & 5. Recall

* **Status**: A — faithful implementation (for overall match set)
* **Mathematical Definition**: $Precision = \frac{TP}{TP + FP}$, $Recall = \frac{TP}{\text{Possible Matches}}$
* **Implementation**: Evaluated on the final match set proposed by the matcher (after cross-check / ratio test).
* **Assumptions**: Same normalization assumptions as above.
* **Possible Improvements**: Same as above (co-visible region filtering).

## 6. PR Curve

* **Status**: B — adapted implementation
* **Mathematical Definition**: Plot of Precision vs. Recall as the match acceptance threshold varies.
* **Implementation**: Sweeps through matches sorted by descriptor distance.
* **Deviations from Literature**: Strict HPatches PR curves are patch-based. Full-image PR curves are common but fundamentally different from the 2017 paper's verification task.
* **Possible Improvements**: Provide AP alongside MMA.

## 7. Homography Error

* **Status**: C — simplified implementation
* **Mathematical Definition**: $\| H_{gt_{norm}} - H_{est_{norm}} \|_F$
* **Implementation**: Normalizes $H[2,2]$ to 1, computes Frobenius norm.
* **Assumptions**: Assumes $H[2,2]$ is non-zero and stable.
* **Deviations from Literature**: Matrix norms for homographies are scale-dependent and often uninformative.
* **Possible Improvements**: Reprojection error is the gold standard; Homography Error can be omitted or kept as secondary.

## 8. Reprojection Error

* **Status**: A — faithful implementation
* **Mathematical Definition**: $\frac{1}{4} \sum \| H_{gt}(c) - H_{est}(c) \|_2$
* **Implementation**: Projects the 4 corners of image 1 using $H_{gt}$ and $H_{est}$ and calculates the mean Euclidean distance.
* **Assumptions**: Assumes the image plane is defined by $(0,0)$ to $(w,h)$.
* **Possible Improvements**: None required. It is standard and exact.
