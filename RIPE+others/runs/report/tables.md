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
| **RIPE-Select (RL), top-K** | 30.7±0.8 / 61.6±0.3 / 72.8±0.4 | 0.513 | 227 | 233 | 38.3 / 68.6 / 78.6 | 23.4 / 55.5 / 68.2 |
| **RIPE-Select (RL), grid** | 31.9±0.1 / 62.8±0.2 / 73.7±0.2 | 0.450 | 185 | 191 | 40.1 / 71.1 / 80.4 | 24.1 / 55.7 / 68.1 |

Diagnostics: ROC-AUC(detector response → keypoint has a correct match) = 0.614; share of the pool's correct matches kept by top-K response = 42%; correct-match rate per response quartile = 30.9%, 26.7%, 22.3%, 16.0%.

**Matched metrics — hpatches SIFT** (MMA = share of MNN matches with GT error < t px reprojection; inlier ratio = RANSAC inliers / matches at th = 4.0; runtime = ms per image, single thread)

| selection | MMA@1 | MMA@3 | MMA@5 | mean MMA 1-10 | inlier ratio | runtime ms | viewpoint AUC @1/3/5 | illum. AUC @1/3/5 |
|---|---|---|---|---|---|---|---|---|
| vanilla (OpenCV default) | 42.8 | 56.9 | 58.7 | 56.8 | 58.4% | 380 | 24.0 / 56.5 / 68.5 | 40.5 / 71.1 / 80.3 |
| pool, top-K response | 42.7 | 57.4 | 59.3 | 57.3 | 58.8% | 607 | 23.8 / 55.8 / 68.6 | 41.6 / 73.3 / 82.8 |
| pool, grid (ANMS-like) | 35.4 | 49.1 | 51.1 | 49.2 | 50.7% | 607 | 26.2 / 57.6 / 69.6 | 42.3 / 72.2 / 81.6 |
| pool, random K | 17.2 | 23.9 | 25.1 | 24.4 | 24.7% | 607 | 21.9 / 51.7 / 64.3 | 33.6 / 63.6 / 72.8 |
| pool, top-2K (2x budget!) | 41.3 | 56.0 | 57.9 | 55.9 | 57.5% | 607 | 26.9 / 59.0 / 70.7 | 46.3 / 75.4 / 83.9 |
| supervised, top-K | 35.2 | 49.7 | 51.7 | 49.7 | 51.3% | 991 | 23.1 / 54.8 / 67.5 | 36.8 / 67.9 / 77.9 |
| supervised, grid | 31.6 | 45.6 | 47.7 | 45.8 | 47.2% | 991 | 23.9 / 55.5 / 67.5 | 39.5 / 70.6 / 80.3 |
| **RIPE-Select (RL), top-K** | 37.3 | 51.3 | 53.5 | 51.6 | 53.1% | 991 | 23.4 / 55.5 / 68.2 | 38.3 / 68.6 / 78.6 |
| **RIPE-Select (RL), grid** | 31.8 | 45.0 | 47.1 | 45.4 | 46.6% | 991 | 24.1 / 55.7 / 68.1 | 40.1 / 71.1 / 80.4 |

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
| **RIPE-Select (RL), top-K** | 20.4±0.2 / 45.2±0.4 / 56.9±0.6 | 0.478 | 194 | 176 | 31.5 / 55.2 / 64.7 | 10.1 / 35.4 / 48.6 |
| **RIPE-Select (RL), grid** | 23.3±0.4 / 48.5±0.1 / 60.0±0.2 | 0.419 | 157 | 142 | 35.6 / 60.0 / 69.7 | 12.8 / 38.2 / 51.4 |

Diagnostics: ROC-AUC(detector response → keypoint has a correct match) = 0.564; share of the pool's correct matches kept by top-K response = 31%; correct-match rate per response quartile = 25.2%, 22.7%, 20.5%, 16.8%.

**Matched metrics — hpatches ORB** (MMA = share of MNN matches with GT error < t px reprojection; inlier ratio = RANSAC inliers / matches at th = 2.0; runtime = ms per image, single thread)

| selection | MMA@1 | MMA@3 | MMA@5 | mean MMA 1-10 | inlier ratio | runtime ms | viewpoint AUC @1/3/5 | illum. AUC @1/3/5 |
|---|---|---|---|---|---|---|---|---|
| vanilla (OpenCV default) | 27.4 | 51.6 | 55.6 | 51.9 | 46.7% | 18 | 10.0 / 33.0 / 46.6 | 33.0 / 58.2 / 68.1 |
| pool, top-K response | 26.7 | 50.2 | 54.0 | 50.4 | 46.1% | 47 | 8.6 / 31.9 / 45.5 | 29.1 / 54.9 / 65.5 |
| pool, grid (ANMS-like) | 24.2 | 45.6 | 49.2 | 45.8 | 41.6% | 47 | 12.2 / 39.4 / 52.2 | 37.9 / 63.7 / 73.3 |
| pool, random K | 12.4 | 27.9 | 33.5 | 30.7 | 23.3% | 47 | 7.7 / 31.1 / 43.8 | 21.1 / 44.5 / 55.9 |
| pool, top-2K (2x budget!) | 26.7 | 50.9 | 54.9 | 51.2 | 46.1% | 47 | 12.2 / 37.9 / 51.5 | 36.9 / 62.5 / 72.3 |
| supervised, top-K | 23.3 | 45.9 | 50.4 | 46.9 | 41.2% | 237 | 8.5 / 32.3 / 46.1 | 28.6 / 53.6 / 63.5 |
| supervised, grid | 20.6 | 40.4 | 44.7 | 41.5 | 36.2% | 237 | 12.6 / 38.7 / 51.6 | 35.5 / 61.0 / 70.8 |
| **RIPE-Select (RL), top-K** | 24.8 | 47.8 | 52.1 | 48.5 | 43.3% | 237 | 10.1 / 35.4 / 48.6 | 31.5 / 55.2 / 64.7 |
| **RIPE-Select (RL), grid** | 21.6 | 41.9 | 45.9 | 42.7 | 37.7% | 237 | 12.8 / 38.2 / 51.4 | 35.6 / 60.0 / 69.7 |

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

## hpatches — BRISK  (K=1024, 540 pairs, RANSAC th = 4.0)

| selection (same pool, same K, same descriptor, MNN, PoseLib) | AUC @1px / @3px / @5px | match prec | #correct | #inliers | AUC illum. | AUC viewp. |
|---|---|---|---|---|---|---|
| vanilla (OpenCV default) | 27.5 / 58.0 / 69.4 | 0.535 | 216 | 223 | 35.1 / 65.3 / 75.8 | 20.6 / 51.5 / 63.6 |
| pool, top-K response | 28.2 / 58.6 / 70.1 | 0.535 | 239 | 246 | 35.9 / 65.7 / 76.2 | 21.2 / 52.1 / 64.6 |
| pool, grid (ANMS-like) | 31.9 / 61.9 / 72.6 | 0.481 | 198 | 205 | 41.1 / 70.4 / 79.6 | 23.5 / 54.2 / 66.3 |
| pool, random K | 19.7 / 50.6 / 62.6 | 0.259 | 91 | 98 | 23.6 / 54.8 / 65.6 | 16.2 / 46.8 / 60.0 |
| pool, top-2K (2x budget!) | 31.4 / 62.1 / 72.8 | 0.534 | 459 | 474 | 39.3 / 69.5 / 78.7 | 24.3 / 55.4 / 67.5 |
| supervised, top-K | 27.1 / 57.2 / 69.0 | 0.491 | 197 | 205 | 34.4 / 64.4 / 74.5 | 20.5 / 50.8 / 64.0 |
| supervised, grid | 31.4 / 61.5 / 72.0 | 0.440 | 168 | 175 | 40.3 / 69.6 / 78.6 | 23.3 / 54.0 / 66.0 |
| **RIPE-Select (RL), top-K** | 28.1±0.3 / 58.0±0.2 / 69.6±0.5 | 0.537 | 214 | 223 | 36.1 / 65.1 / 75.2 | 21.3 / 51.4 / 63.8 |
| **RIPE-Select (RL), grid** | 30.7±0.3 / 60.1±0.3 / 71.3±0.3 | 0.470 | 173 | 181 | 39.6 / 68.5 / 78.5 | 22.6 / 52.4 / 64.8 |

Diagnostics: ROC-AUC(detector response → keypoint has a correct match) = 0.604; share of the pool's correct matches kept by top-K response = 38%; correct-match rate per response quartile = 28.1%, 23.7%, 19.3%, 14.3%.

**Matched metrics — hpatches BRISK** (MMA = share of MNN matches with GT error < t px reprojection; inlier ratio = RANSAC inliers / matches at th = 4.0; runtime = ms per image, single thread)

| selection | MMA@1 | MMA@3 | MMA@5 | mean MMA 1-10 | inlier ratio | runtime ms | viewpoint AUC @1/3/5 | illum. AUC @1/3/5 |
|---|---|---|---|---|---|---|---|---|
| vanilla (OpenCV default) | 36.3 | 53.5 | 56.0 | 53.6 | 55.7% | 136 | 20.6 / 51.5 / 63.6 | 35.1 / 65.3 / 75.8 |
| pool, top-K response | 36.3 | 53.5 | 56.1 | 53.6 | 55.5% | 378 | 21.2 / 52.1 / 64.6 | 35.9 / 65.7 / 76.2 |
| pool, grid (ANMS-like) | 32.6 | 48.1 | 50.6 | 48.4 | 50.1% | 378 | 23.5 / 54.2 / 66.3 | 41.1 / 70.4 / 79.6 |
| pool, random K | 15.1 | 25.9 | 29.0 | 27.4 | 28.0% | 378 | 16.2 / 46.8 / 60.0 | 23.6 / 54.8 / 65.6 |
| pool, top-2K (2x budget!) | 35.3 | 53.4 | 56.2 | 53.6 | 55.5% | 378 | 24.3 / 55.4 / 67.5 | 39.3 / 69.5 / 78.7 |
| supervised, top-K | 32.1 | 49.1 | 51.9 | 49.4 | 51.3% | 591 | 20.5 / 50.8 / 64.0 | 34.4 / 64.4 / 74.5 |
| supervised, grid | 29.0 | 44.0 | 46.7 | 44.5 | 46.1% | 591 | 23.3 / 54.0 / 66.0 | 40.3 / 69.6 / 78.6 |
| **RIPE-Select (RL), top-K** | 34.5 | 53.7 | 57.1 | 54.2 | 56.2% | 591 | 21.3 / 51.4 / 63.8 | 36.1 / 65.1 / 75.2 |
| **RIPE-Select (RL), grid** | 30.6 | 47.0 | 50.1 | 47.7 | 49.3% | 591 | 22.6 / 52.4 / 64.8 | 39.6 / 68.5 / 78.5 |

## megadepth — SIFT  (K=2048, 500 pairs, RANSAC th = 1.0)

| selection (same pool, same K, same descriptor, MNN, PoseLib) | AUC @5° / @10° / @20° | match prec | #correct | #inliers |
|---|---|---|---|---|
| vanilla (OpenCV default) | 38.7 / 51.8 / 63.2 | 0.248 | 144 | 144 |
| pool, top-K response | 40.6 / 54.1 / 66.2 | 0.246 | 168 | 168 |
| pool, grid (ANMS-like) | 43.3 / 56.9 / 68.7 | 0.227 | 146 | 146 |
| supervised, top-K | 42.5 / 57.1 / 69.9 | 0.298 | 213 | 215 |
| supervised, grid | 43.2 / 57.0 / 69.0 | 0.224 | 147 | 148 |
| **RIPE-Select (RL), top-K** | 41.0±2.4 / 55.3±3.3 / 67.3±3.7 | 0.273 | 167 | 170 |
| **RIPE-Select (RL), grid** | 41.3±0.8 / 55.3±1.0 / 67.4±0.9 | 0.215 | 129 | 129 |

Diagnostics: ROC-AUC(detector response → keypoint has a correct match) = 0.583; share of the pool's correct matches kept by top-K response = 39%; correct-match rate per response quartile = 10.4%, 10.0%, 8.7%, 6.0%.

**Matched metrics — megadepth SIFT** (MMA = share of MNN matches with GT error < t px epipolar; inlier ratio = RANSAC inliers / matches at th = 1.0; runtime = ms per image, single thread)

| selection | MMA@1 | MMA@3 | MMA@5 | mean MMA 1-10 | inlier ratio | runtime ms |
|---|---|---|---|---|---|---|
| vanilla (OpenCV default) | 20.6 | 26.4 | 28.0 | 27.3 | 24.9% | 384 |
| pool, top-K response | 20.1 | 26.3 | 27.9 | 27.2 | 24.7% | 731 |
| pool, grid (ANMS-like) | 18.1 | 24.5 | 26.1 | 25.3 | 22.8% | 731 |
| supervised, top-K | 24.2 | 31.7 | 33.2 | 32.3 | 30.2% | 1044 |
| supervised, grid | 17.9 | 24.1 | 25.6 | 24.8 | 22.6% | 1044 |
| **RIPE-Select (RL), top-K** | 22.0 | 29.3 | 31.0 | 30.1 | 27.8% | 1044 |
| **RIPE-Select (RL), grid** | 17.1 | 23.3 | 24.9 | 24.2 | 21.7% | 1044 |

## megadepth — ORB  (K=2048, 500 pairs, RANSAC th = 1.0)

| selection (same pool, same K, same descriptor, MNN, PoseLib) | AUC @5° / @10° / @20° | match prec | #correct | #inliers |
|---|---|---|---|---|
| vanilla (OpenCV default) | 18.8 / 29.3 / 40.6 | 0.243 | 150 | 146 |
| pool, top-K response | 15.4 / 25.2 / 36.5 | 0.239 | 149 | 146 |
| pool, grid (ANMS-like) | 25.3 / 37.0 / 49.1 | 0.177 | 100 | 98 |
| supervised, top-K | 20.1 / 32.3 / 45.4 | 0.239 | 150 | 145 |
| supervised, grid | 21.2 / 32.8 / 45.3 | 0.159 | 90 | 89 |
| **RIPE-Select (RL), top-K** | 24.9±0.4 / 38.4±0.2 / 51.8±0.4 | 0.244 | 134 | 131 |
| **RIPE-Select (RL), grid** | 22.0±0.7 / 33.8±0.6 / 45.7±0.3 | 0.163 | 83 | 82 |

Diagnostics: ROC-AUC(detector response → keypoint has a correct match) = 0.598; share of the pool's correct matches kept by top-K response = 40%; correct-match rate per response quartile = 8.4%, 5.6%, 4.9%, 4.2%.

**Matched metrics — megadepth ORB** (MMA = share of MNN matches with GT error < t px epipolar; inlier ratio = RANSAC inliers / matches at th = 1.0; runtime = ms per image, single thread)

| selection | MMA@1 | MMA@3 | MMA@5 | mean MMA 1-10 | inlier ratio | runtime ms |
|---|---|---|---|---|---|---|
| vanilla (OpenCV default) | 16.9 | 27.4 | 30.0 | 28.6 | 24.0% | 37 |
| pool, top-K response | 16.5 | 27.1 | 29.9 | 28.5 | 23.9% | 77 |
| pool, grid (ANMS-like) | 12.2 | 20.1 | 22.2 | 21.1 | 17.6% | 77 |
| supervised, top-K | 16.3 | 27.2 | 30.1 | 28.6 | 23.3% | 481 |
| supervised, grid | 10.8 | 18.2 | 20.3 | 19.3 | 15.8% | 481 |
| **RIPE-Select (RL), top-K** | 17.0 | 27.5 | 30.1 | 28.7 | 24.0% | 481 |
| **RIPE-Select (RL), grid** | 11.1 | 18.6 | 20.7 | 19.7 | 16.4% | 481 |

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

## megadepth — BRISK  (K=2048, 500 pairs, RANSAC th = 1.0)

| selection (same pool, same K, same descriptor, MNN, PoseLib) | AUC @5° / @10° / @20° | match prec | #correct | #inliers |
|---|---|---|---|---|
| vanilla (OpenCV default) | 29.0 / 41.6 / 54.0 | 0.223 | 102 | 101 |
| pool, top-K response | 30.8 / 43.9 / 56.3 | 0.217 | 126 | 124 |
| pool, grid (ANMS-like) | 33.8 / 48.7 / 62.1 | 0.201 | 112 | 112 |
| supervised, top-K | 33.4 / 48.3 / 61.7 | 0.237 | 136 | 135 |
| supervised, grid | 33.6 / 47.1 / 59.7 | 0.193 | 105 | 104 |
| **RIPE-Select (RL), top-K** | 32.6±1.3 / 46.6±1.3 / 59.4±1.1 | 0.243 | 127 | 127 |
| **RIPE-Select (RL), grid** | 30.3±1.1 / 43.3±1.0 / 56.4±0.8 | 0.195 | 95 | 95 |

Diagnostics: ROC-AUC(detector response → keypoint has a correct match) = 0.564; share of the pool's correct matches kept by top-K response = 44%; correct-match rate per response quartile = 8.0%, 7.0%, 6.3%, 5.6%.

**Matched metrics — megadepth BRISK** (MMA = share of MNN matches with GT error < t px epipolar; inlier ratio = RANSAC inliers / matches at th = 1.0; runtime = ms per image, single thread)

| selection | MMA@1 | MMA@3 | MMA@5 | mean MMA 1-10 | inlier ratio | runtime ms |
|---|---|---|---|---|---|---|
| vanilla (OpenCV default) | 16.7 | 24.7 | 27.1 | 26.1 | 22.2% | 142 |
| pool, top-K response | 16.1 | 24.1 | 26.3 | 25.3 | 21.5% | 439 |
| pool, grid (ANMS-like) | 15.0 | 22.3 | 24.3 | 23.3 | 20.2% | 439 |
| supervised, top-K | 17.9 | 26.0 | 28.1 | 27.0 | 23.7% | 894 |
| supervised, grid | 14.4 | 21.2 | 23.0 | 22.1 | 19.3% | 894 |
| **RIPE-Select (RL), top-K** | 18.1 | 26.9 | 29.2 | 28.0 | 24.5% | 894 |
| **RIPE-Select (RL), grid** | 14.3 | 21.6 | 23.7 | 22.7 | 19.7% | 894 |