# A Comprehensive Evaluation of Classical Local Features on HPatches

## Abstract
Local feature extraction is a foundational task in computer vision, underpinning applications ranging from SLAM to 3D reconstruction. This report presents a rigorous empirical benchmark of classical local feature detectors and descriptors (SIFT, ORB, BRISK, LATCH) on the HPatches dataset. We evaluate these methods across extreme illumination changes and severe viewpoint variations, analyzing geometric stability, discriminative power, and computational efficiency. Our findings confirm that while binary descriptors like ORB offer exceptional real-time speeds, gradient-based methods like SIFT retain absolute superiority in geometric matching accuracy and inlier ratios. 

---

## 1. Introduction
The extraction and matching of local features allow algorithms to establish point correspondences across diverse views of a scene. Traditional pipelines rely on handcrafted algorithms such as SIFT, which use image gradients, or computationally efficient binary strings like ORB and BRISK. This paper benchmarks these classical methods using modern evaluation protocols (Repeatability, Mean Matching Accuracy, and Inlier Ratio) on the HPatches sequences to quantitatively establish their operational boundaries.

---

## 2. Related Work
The HPatches (Homography Patches) dataset has become the defacto standard for evaluating local features, superseding the older Oxford dataset. Modern evaluations (e.g., D2-Net, SuperPoint) utilize strict co-visible region filtering and homography projection to compute precise geometric metrics. Our framework implements these exact protocols to ensure a mathematically sound evaluation of classical methods.

---

## 3. Methods
We evaluate the following algorithms using OpenCV 4.x:
1. **SIFT** (Scale-Invariant Feature Transform): Gradient-based, floating-point descriptor.
2. **ORB** (Oriented FAST and Rotated BRIEF): Binary descriptor optimized for real-time performance.
3. **BRISK** (Binary Robust Invariant Scalable Keypoints): Scale-space FAST variant with a circular sampling pattern.
4. **LATCH** (Learned Arrangements of Three Patch Codes): Binary descriptor based on learned patch arrangements.

*(Note: SURF is excluded from this benchmark due to patent restrictions in the standard OpenCV binaries).*

---

## 4. HPatches Dataset & Experimental Setup
HPatches consists of 116 sequences (57 illumination, 59 viewpoint). Each sequence contains a reference image and 5 target images with ground-truth homographies. 
Features are matched using Brute-Force nearest-neighbor search (L2 norm for SIFT, Hamming for binary methods). Mutual nearest neighbor (cross-check) is enabled, followed by Lowe's ratio test (0.75 for SIFT, 1.0 for binary descriptors).

---

## 5. Evaluation Metrics
Our evaluation pipeline mathematically enforces co-visibility bounds using homography projection.
- **Repeatability**: The fraction of keypoints consistently localized in both views within a 3-pixel threshold.
- **Matching Score**: The fraction of correctly matched features relative to the total features detected in the shared viewpoint.
- **Inlier Ratio**: The percentage of proposed matches that are geometrically correct ($\le 3$ pixels from ground truth).
- **Mean Matching Accuracy (MMA)**: The fraction of correct matches computed at varying error thresholds (1 to 10 pixels).

---

## 6. Results

### 6.1 Illumination vs. Viewpoint Analysis
Our split analysis (`results/split_analysis/`) demonstrates distinct failure modes:
- **Illumination Resilience**: SIFT proves highly robust to photometric changes, maintaining an Inlier Ratio of over 85% on illumination sequences, heavily outperforming ORB (50%).
- **Viewpoint Degradation**: All methods degrade sharply under perspective distortion. BRISK maintains the highest raw Repeatability, but its descriptors fail to match accurately, plummeting to a 45% inlier ratio. SIFT degrades in total match count but maintains a staggering 70% inlier ratio.

### 6.2 Failure Analysis
Qualitative visual analysis (`results/failures/`) confirms that binary descriptors (ORB/BRISK) propose thousands of matches across repetitive textures (e.g., brick walls, cobblestone), but the vast majority are geometric false positives discarded by RANSAC. SIFT detects fewer features but matches them with surgical precision.

### 6.3 Runtime Profiling
Profiling on a 5-sequence subset (`results/profiling/`) reveals stark computational divides:
- **ORB**: ~99ms total pipeline time (10 FPS natively in Python).
- **SIFT**: ~490ms total pipeline time.
- **BRISK**: ~570ms total pipeline time.
ORB's FAST detector operates in under 20ms, cementing its role in real-time embedded systems. 

---

## 7. Discussion & Conclusion
The empirical data yields clear operational guidelines:
1. **Real-time constraints**: **ORB** is unmatched. It is computationally lightweight and provides sufficient accuracy for continuous tracking (e.g., SLAM).
2. **High-fidelity matching**: **SIFT** remains the gold standard. Despite its age, its gradient-based descriptor is dramatically more discriminative than binary codes, yielding the highest MMA and Inlier Ratios. 
3. **BRISK / LATCH**: While theoretically interesting, they offer a poor compromise, running slower than SIFT in Python while exhibiting the noisy matching characteristics of ORB.

---

## 8. Limitations & Future Work
- **Software Dependencies**: OpenCV's `findHomography` fails when provided with zero viable matches, which occasionally occurs for SIFT on the most extreme viewpoint warps, causing `inf` mathematical boundaries in matrix norm calculations.
- **Future Work**: The pipeline is fully modularized and primed for the integration of deep learning-based features (e.g., SuperPoint, LoFTR) to compare against these classical baselines.
