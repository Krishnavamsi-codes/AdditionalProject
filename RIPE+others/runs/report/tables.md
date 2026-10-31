# Results (fixed RANSAC threshold chosen on vanilla, applied to every variant)

Published reference (copied from the RIPE paper, Tab. 2; full benchmark, their settings — not re-run here):

| method | MegaDepth-1500 AUC @5/10/20° | HPatches AUC @1/3/5 px |
|---|---|---|
| SIFT (RIPE paper) | 37.29 / 50.61 / 61.89 | 29.98 / 53.93 / 65.01 |
| RIPE (paper) | 55.11 / 68.34 / 78.03 | 37.93 / 58.93 / 69.2 |

## hpatches — SIFT  (K=1024, 540 pairs, RANSAC th = 4.0)

| selection (same pool, same K, same descriptor, MNN, PoseLib) | AUC @1px / @3px / @5px | match prec | #correct | #inliers | AUC illum. | AUC viewp. |
|---|---|---|---|---|---|---|
| vanilla (OpenCV default) | 31.8 / 63.5 / 74.1 | 0.569 | 260 | 265 | 40.5 / 71.1 / 80.3 | 24.0 / 56.5 / 68.5 |
| pool, top-K response | 32.3 / 64.1 / 75.4 | 0.574 | 282 | 288 | 41.6 / 73.3 / 82.8 | 23.8 / 55.8 / 68.6 |
| pool, grid (ANMS-like) | 33.9 / 64.6 / 75.3 | 0.491 | 216 | 222 | 42.3 / 72.2 / 81.6 | 26.2 / 57.6 / 69.6 |
| pool, random K | 27.4 / 57.4 / 68.3 | 0.239 | 88 | 91 | 33.6 / 63.6 / 72.8 | 21.9 / 51.7 / 64.3 |
| pool, top-2K (2x budget!) | 36.1 / 66.8 / 77.0 | 0.560 | 526 | 537 | 46.3 / 75.4 / 83.9 | 26.9 / 59.0 / 70.7 |
| supervised, top-K | 29.6 / 61.0 / 72.4 | 0.497 | 234 | 240 | 36.8 / 67.9 / 77.9 | 23.1 / 54.8 / 67.5 |
| supervised, grid | 31.3 / 62.7 / 73.6 | 0.456 | 199 | 205 | 39.5 / 70.6 / 80.3 | 23.9 / 55.5 / 67.5 |
| **RIPE-Select (RL), top-K** | 30.5 / 61.7 / 73.1 | 0.513 | 227 | 233 | 38.3 / 68.6 / 78.6 | 23.4 / 55.5 / 68.2 |
| **RIPE-Select (RL), grid** | 31.8 / 63.0 / 73.9 | 0.450 | 185 | 191 | 40.1 / 71.1 / 80.4 | 24.1 / 55.7 / 68.1 |

Diagnostics: ROC-AUC(detector response → keypoint has a correct match) = 0.614; share of the pool's correct matches kept by top-K response = 42%; correct-match rate per response quartile = 30.9%, 26.7%, 22.3%, 16.0%.

**Matched metrics — hpatches SIFT** (MMA = share of MNN matches with GT error < t px reprojection; inlier ratio = RANSAC inliers / matches at th = 4.0; runtime = ms per image, single thread)

| selection | MMA@1 | MMA@3 | MMA@5 | mean MMA 1-10 | inlier ratio | runtime ms | viewpoint AUC @1/3/5 | illum. AUC @1/3/5 |
|---|---|---|---|---|---|---|---|---|
| vanilla (OpenCV default) | – | – | – | – | 58.4% | – | 24.0 / 56.5 / 68.5 | 40.5 / 71.1 / 80.3 |
| pool, top-K response | – | – | – | – | 58.8% | – | 23.8 / 55.8 / 68.6 | 41.6 / 73.3 / 82.8 |
| pool, grid (ANMS-like) | – | – | – | – | 50.7% | – | 26.2 / 57.6 / 69.6 | 42.3 / 72.2 / 81.6 |
| pool, random K | – | – | – | – | 24.7% | – | 21.9 / 51.7 / 64.3 | 33.6 / 63.6 / 72.8 |
| pool, top-2K (2x budget!) | – | – | – | – | 57.5% | – | 26.9 / 59.0 / 70.7 | 46.3 / 75.4 / 83.9 |
| supervised, top-K | – | – | – | – | 51.3% | – | 23.1 / 54.8 / 67.5 | 36.8 / 67.9 / 77.9 |
| supervised, grid | – | – | – | – | 47.2% | – | 23.9 / 55.5 / 67.5 | 39.5 / 70.6 / 80.3 |
| **RIPE-Select (RL), top-K** | – | – | – | – | 53.1% | – | 23.4 / 55.5 / 68.2 | 38.3 / 68.6 / 78.6 |
| **RIPE-Select (RL), grid** | – | – | – | – | 46.6% | – | 24.1 / 55.7 / 68.1 | 40.1 / 71.1 / 80.4 |

## hpatches — ORB  (K=1024, 540 pairs, RANSAC th = 2.0)

| selection (same pool, same K, same descriptor, MNN, PoseLib) | AUC @1px / @3px / @5px | match prec | #correct | #inliers | AUC illum. | AUC viewp. |
|---|---|---|---|---|---|---|
| vanilla (OpenCV default) | 21.0 / 45.0 / 56.9 | 0.516 | 231 | 210 | 33.0 / 58.2 / 68.1 | 10.0 / 33.0 / 46.6 |
| pool, top-K response | 18.4 / 42.9 / 55.1 | 0.502 | 219 | 202 | 29.1 / 54.9 / 65.5 | 8.6 / 31.9 / 45.5 |
| pool, grid (ANMS-like) | 24.5 / 51.0 / 62.3 | 0.456 | 183 | 168 | 37.9 / 63.7 / 73.3 | 12.2 / 39.4 / 52.2 |
| pool, random K | 14.0 / 37.5 / 49.5 | 0.279 | 103 | 87 | 21.1 / 44.5 / 55.9 | 7.7 / 31.1 / 43.8 |
| pool, top-2K (2x budget!) | 24.0 / 49.7 / 61.4 | 0.509 | 440 | 401 | 36.9 / 62.5 / 72.3 | 12.2 / 37.9 / 51.5 |
| supervised, top-K | 18.1 / 42.5 / 54.4 | 0.459 | 187 | 169 | 28.6 / 53.6 / 63.5 | 8.5 / 32.3 / 46.1 |
| supervised, grid | 23.6 / 49.4 / 60.8 | 0.404 | 155 | 140 | 35.5 / 61.0 / 70.8 | 12.6 / 38.7 / 51.6 |
| **RIPE-Select (RL), top-K** | 20.3 / 44.8 / 56.3 | 0.478 | 194 | 176 | 31.5 / 55.2 / 64.7 | 10.1 / 35.4 / 48.6 |
| **RIPE-Select (RL), grid** | 23.7 / 48.6 / 60.2 | 0.419 | 157 | 142 | 35.6 / 60.0 / 69.7 | 12.8 / 38.2 / 51.4 |

Diagnostics: ROC-AUC(detector response → keypoint has a correct match) = 0.564; share of the pool's correct matches kept by top-K response = 31%; correct-match rate per response quartile = 25.2%, 22.7%, 20.5%, 16.8%.

**Matched metrics — hpatches ORB** (MMA = share of MNN matches with GT error < t px reprojection; inlier ratio = RANSAC inliers / matches at th = 2.0; runtime = ms per image, single thread)

| selection | MMA@1 | MMA@3 | MMA@5 | mean MMA 1-10 | inlier ratio | runtime ms | viewpoint AUC @1/3/5 | illum. AUC @1/3/5 |
|---|---|---|---|---|---|---|---|---|
| vanilla (OpenCV default) | – | – | – | – | 46.7% | – | 10.0 / 33.0 / 46.6 | 33.0 / 58.2 / 68.1 |
| pool, top-K response | – | – | – | – | 46.1% | – | 8.6 / 31.9 / 45.5 | 29.1 / 54.9 / 65.5 |
| pool, grid (ANMS-like) | – | – | – | – | 41.6% | – | 12.2 / 39.4 / 52.2 | 37.9 / 63.7 / 73.3 |
| pool, random K | – | – | – | – | 23.3% | – | 7.7 / 31.1 / 43.8 | 21.1 / 44.5 / 55.9 |
| pool, top-2K (2x budget!) | – | – | – | – | 46.1% | – | 12.2 / 37.9 / 51.5 | 36.9 / 62.5 / 72.3 |
| supervised, top-K | – | – | – | – | 41.2% | – | 8.5 / 32.3 / 46.1 | 28.6 / 53.6 / 63.5 |
| supervised, grid | – | – | – | – | 36.2% | – | 12.6 / 38.7 / 51.6 | 35.5 / 61.0 / 70.8 |
| **RIPE-Select (RL), top-K** | – | – | – | – | 43.3% | – | 10.1 / 35.4 / 48.6 | 31.5 / 55.2 / 64.7 |
| **RIPE-Select (RL), grid** | – | – | – | – | 37.7% | – | 12.8 / 38.2 / 51.4 | 35.6 / 60.0 / 69.7 |

## hpatches — SURF  (K=1024, 540 pairs, RANSAC th = 4.0)

| selection (same pool, same K, same descriptor, MNN, PoseLib) | AUC @1px / @3px / @5px | match prec | #correct | #inliers | AUC illum. | AUC viewp. |
|---|---|---|---|---|---|---|
| vanilla (OpenCV default) | 26.0 / 56.8 / 68.8 | 0.513 | 230 | 241 | 31.0 / 64.3 / 75.4 | 21.6 / 50.0 / 62.8 |
| pool, top-K response | 26.2 / 57.0 / 68.9 | 0.512 | 232 | 243 | 31.4 / 64.8 / 75.9 | 21.6 / 49.9 / 62.7 |
| pool, grid (ANMS-like) | 28.8 / 59.2 / 70.3 | 0.455 | 190 | 201 | 34.8 / 68.1 / 78.4 | 23.5 / 51.1 / 63.0 |
| pool, random K | 18.5 / 48.6 / 60.0 | 0.247 | 89 | 98 | 23.9 / 57.0 / 68.1 | 13.6 / 41.0 / 52.7 |
| pool, top-2K (2x budget!) | 30.3 / 60.2 / 71.3 | 0.486 | 405 | 427 | 37.3 / 69.2 / 79.1 | 24.0 / 52.0 / 64.1 |
| supervised, top-K | 23.4 / 53.4 / 65.6 | 0.442 | 190 | 202 | 29.3 / 61.9 / 73.4 | 18.1 / 45.7 / 58.5 |
| supervised, grid | 26.3 / 56.2 / 67.8 | 0.400 | 163 | 174 | 33.6 / 66.3 / 76.8 | 19.7 / 46.9 / 59.7 |
| **RIPE-Select (RL), top-K** | 23.6 / 54.6 / 66.5 | 0.463 | 181 | 192 | 30.1 / 64.1 / 75.3 | 17.8 / 46.0 / 58.5 |
| **RIPE-Select (RL), grid** | 26.6 / 56.9 / 68.8 | 0.422 | 164 | 174 | 34.6 / 67.2 / 77.9 | 19.4 / 47.5 / 60.5 |
| RL residual on response, top-K | 23.9 / 54.6 / 66.6 | 0.469 | 184 | 196 | 30.0 / 63.8 / 75.2 | 18.5 / 46.1 / 58.9 |
| RL residual on response, grid | 27.0 / 57.1 / 68.8 | 0.423 | 165 | 176 | 34.3 / 67.0 / 77.9 | 20.4 / 48.2 / 60.5 |

Diagnostics: ROC-AUC(detector response → keypoint has a correct match) = 0.631; share of the pool's correct matches kept by top-K response = 50%; correct-match rate per response quartile = 25.4%, 20.5%, 16.2%, 10.8%.

**Matched metrics — hpatches SURF** (MMA = share of MNN matches with GT error < t px reprojection; inlier ratio = RANSAC inliers / matches at th = 4.0; runtime = ms per image, single thread)

| selection | MMA@1 | MMA@3 | MMA@5 | mean MMA 1-10 | inlier ratio | runtime ms | viewpoint AUC @1/3/5 | illum. AUC @1/3/5 |
|---|---|---|---|---|---|---|---|---|
| vanilla (OpenCV default) | – | – | – | – | 54.1% | – | 21.6 / 50.0 / 62.8 | 31.0 / 64.3 / 75.4 |
| pool, top-K response | – | – | – | – | 54.0% | – | 21.6 / 49.9 / 62.7 | 31.4 / 64.8 / 75.9 |
| pool, grid (ANMS-like) | – | – | – | – | 48.3% | – | 23.5 / 51.1 / 63.0 | 34.8 / 68.1 / 78.4 |
| pool, random K | – | – | – | – | 27.4% | – | 13.6 / 41.0 / 52.7 | 23.9 / 57.0 / 68.1 |
| pool, top-2K (2x budget!) | – | – | – | – | 51.6% | – | 24.0 / 52.0 / 64.1 | 37.3 / 69.2 / 79.1 |
| supervised, top-K | – | – | – | – | 47.3% | – | 18.1 / 45.7 / 58.5 | 29.3 / 61.9 / 73.4 |
| supervised, grid | – | – | – | – | 43.1% | – | 19.7 / 46.9 / 59.7 | 33.6 / 66.3 / 76.8 |
| **RIPE-Select (RL), top-K** | – | – | – | – | 49.8% | – | 17.8 / 46.0 / 58.5 | 30.1 / 64.1 / 75.3 |
| **RIPE-Select (RL), grid** | – | – | – | – | 45.2% | – | 19.4 / 47.5 / 60.5 | 34.6 / 67.2 / 77.9 |
| RL residual on response, top-K | – | – | – | – | 50.4% | – | 18.5 / 46.1 / 58.9 | 30.0 / 63.8 / 75.2 |
| RL residual on response, grid | – | – | – | – | 45.3% | – | 20.4 / 48.2 / 60.5 | 34.3 / 67.0 / 77.9 |

## megadepth — SIFT  (K=2048, 500 pairs, RANSAC th = 1.0)

| selection (same pool, same K, same descriptor, MNN, PoseLib) | AUC @5° / @10° / @20° | match prec | #correct | #inliers |
|---|---|---|---|---|
| vanilla (OpenCV default) | 38.7 / 51.8 / 63.2 | 0.248 | 144 | 144 |
| pool, top-K response | 40.6 / 54.1 / 66.2 | 0.246 | 168 | 168 |
| pool, grid (ANMS-like) | 43.3 / 56.9 / 68.7 | 0.227 | 146 | 146 |
| supervised, top-K | 42.5 / 57.1 / 69.9 | 0.298 | 213 | 215 |
| supervised, grid | 43.2 / 57.0 / 69.0 | 0.224 | 147 | 148 |
| **RIPE-Select (RL), top-K** | 43.5 / 58.6 / 70.8 | 0.273 | 167 | 170 |
| **RIPE-Select (RL), grid** | 41.4 / 55.6 / 68.1 | 0.215 | 129 | 129 |

Diagnostics: ROC-AUC(detector response → keypoint has a correct match) = 0.583; share of the pool's correct matches kept by top-K response = 39%; correct-match rate per response quartile = 10.4%, 10.0%, 8.7%, 6.0%.

**Matched metrics — megadepth SIFT** (MMA = share of MNN matches with GT error < t px epipolar; inlier ratio = RANSAC inliers / matches at th = 1.0; runtime = ms per image, single thread)

| selection | MMA@1 | MMA@3 | MMA@5 | mean MMA 1-10 | inlier ratio | runtime ms |
|---|---|---|---|---|---|---|
| vanilla (OpenCV default) | – | – | – | – | 24.9% | – |
| pool, top-K response | – | – | – | – | 24.7% | – |
| pool, grid (ANMS-like) | – | – | – | – | 22.8% | – |
| supervised, top-K | – | – | – | – | 30.2% | – |
| supervised, grid | – | – | – | – | 22.6% | – |
| **RIPE-Select (RL), top-K** | – | – | – | – | 27.8% | – |
| **RIPE-Select (RL), grid** | – | – | – | – | 21.7% | – |

## megadepth — ORB  (K=2048, 500 pairs, RANSAC th = 1.0)

| selection (same pool, same K, same descriptor, MNN, PoseLib) | AUC @5° / @10° / @20° | match prec | #correct | #inliers |
|---|---|---|---|---|
| vanilla (OpenCV default) | 18.8 / 29.3 / 40.6 | 0.243 | 150 | 146 |
| pool, top-K response | 15.4 / 25.2 / 36.5 | 0.239 | 149 | 146 |
| pool, grid (ANMS-like) | 25.3 / 37.0 / 49.1 | 0.177 | 100 | 98 |
| supervised, top-K | 20.1 / 32.3 / 45.4 | 0.239 | 150 | 145 |
| supervised, grid | 21.2 / 32.8 / 45.3 | 0.159 | 90 | 89 |
| **RIPE-Select (RL), top-K** | 25.3 / 38.2 / 51.3 | 0.244 | 134 | 131 |
| **RIPE-Select (RL), grid** | 21.7 / 33.8 / 46.0 | 0.163 | 83 | 82 |

Diagnostics: ROC-AUC(detector response → keypoint has a correct match) = 0.598; share of the pool's correct matches kept by top-K response = 40%; correct-match rate per response quartile = 8.4%, 5.6%, 4.9%, 4.2%.

**Matched metrics — megadepth ORB** (MMA = share of MNN matches with GT error < t px epipolar; inlier ratio = RANSAC inliers / matches at th = 1.0; runtime = ms per image, single thread)

| selection | MMA@1 | MMA@3 | MMA@5 | mean MMA 1-10 | inlier ratio | runtime ms |
|---|---|---|---|---|---|---|
| vanilla (OpenCV default) | – | – | – | – | 24.0% | – |
| pool, top-K response | – | – | – | – | 23.9% | – |
| pool, grid (ANMS-like) | – | – | – | – | 17.6% | – |
| supervised, top-K | – | – | – | – | 23.3% | – |
| supervised, grid | – | – | – | – | 15.8% | – |
| **RIPE-Select (RL), top-K** | – | – | – | – | 24.0% | – |
| **RIPE-Select (RL), grid** | – | – | – | – | 16.4% | – |

## megadepth — SURF  (K=2048, 500 pairs, RANSAC th = 1.0)

| selection (same pool, same K, same descriptor, MNN, PoseLib) | AUC @5° / @10° / @20° | match prec | #correct | #inliers |
|---|---|---|---|---|
| vanilla (OpenCV default) | 33.7 / 47.0 / 60.2 | 0.215 | 137 | 134 |
| pool, top-K response | 33.0 / 46.7 / 60.2 | 0.216 | 140 | 137 |
| pool, grid (ANMS-like) | 31.3 / 44.7 / 57.8 | 0.191 | 118 | 116 |
| supervised, top-K | 28.6 / 42.9 / 58.0 | 0.252 | 166 | 165 |
| supervised, grid | 29.3 / 43.2 / 56.2 | 0.180 | 109 | 108 |
| **RIPE-Select (RL), top-K** | 27.0 / 41.8 / 56.1 | 0.217 | 121 | 120 |
| **RIPE-Select (RL), grid** | 26.6 / 39.4 / 53.0 | 0.177 | 98 | 97 |
| RL residual on response, top-K | 29.0 / 43.8 / 57.3 | 0.216 | 121 | 120 |
| RL residual on response, grid | 27.1 / 40.9 / 54.1 | 0.179 | 100 | 99 |

Diagnostics: ROC-AUC(detector response → keypoint has a correct match) = 0.611; share of the pool's correct matches kept by top-K response = 47%; correct-match rate per response quartile = 8.0%, 7.0%, 5.7%, 3.6%.

**Matched metrics — megadepth SURF** (MMA = share of MNN matches with GT error < t px epipolar; inlier ratio = RANSAC inliers / matches at th = 1.0; runtime = ms per image, single thread)

| selection | MMA@1 | MMA@3 | MMA@5 | mean MMA 1-10 | inlier ratio | runtime ms |
|---|---|---|---|---|---|---|
| vanilla (OpenCV default) | – | – | – | – | 21.2% | – |
| pool, top-K response | – | – | – | – | 21.2% | – |
| pool, grid (ANMS-like) | – | – | – | – | 19.0% | – |
| supervised, top-K | – | – | – | – | 25.4% | – |
| supervised, grid | – | – | – | – | 18.0% | – |
| **RIPE-Select (RL), top-K** | – | – | – | – | 21.9% | – |
| **RIPE-Select (RL), grid** | – | – | – | – | 17.8% | – |
| RL residual on response, top-K | – | – | – | – | 21.7% | – |
| RL residual on response, grid | – | – | – | – | 17.9% | – |