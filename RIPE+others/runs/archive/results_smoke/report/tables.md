# Results (fixed RANSAC threshold chosen on vanilla, applied to every variant)


## hpatches — SIFT  (K=1024, 540 pairs, RANSAC th = 4.0)

| selection (same pool, same K, same descriptor, MNN, PoseLib) | AUC @1px / @3px / @5px | match prec | #correct | #inliers | AUC illum. | AUC viewp. |
|---|---|---|---|---|---|---|
| vanilla (OpenCV default) | 31.8 / 63.5 / 74.1 | 0.569 | 260 | 265 | 40.5 / 71.1 / 80.3 | 24.0 / 56.5 / 68.5 |
| pool, top-K response | 32.3 / 64.1 / 75.4 | 0.574 | 282 | 288 | 41.6 / 73.3 / 82.8 | 23.8 / 55.8 / 68.6 |
| pool, grid (ANMS-like) | 33.9 / 64.6 / 75.3 | 0.491 | 216 | 222 | 42.3 / 72.2 / 81.6 | 26.2 / 57.6 / 69.6 |
| pool, random K | 27.4 / 57.4 / 68.3 | 0.239 | 88 | 91 | 33.6 / 63.6 / 72.8 | 21.9 / 51.7 / 64.3 |
| pool, top-2K (2x budget!) | 36.1 / 66.8 / 77.0 | 0.560 | 526 | 537 | 46.3 / 75.4 / 83.9 | 26.9 / 59.0 / 70.7 |

Diagnostics: ROC-AUC(detector response → keypoint has a correct match) = 0.614; share of the pool's correct matches kept by top-K response = 42%; correct-match rate per response quartile = 30.9%, 26.7%, 22.3%, 16.0%.

## hpatches — ORB  (K=1024, 540 pairs, RANSAC th = 2.0)

| selection (same pool, same K, same descriptor, MNN, PoseLib) | AUC @1px / @3px / @5px | match prec | #correct | #inliers | AUC illum. | AUC viewp. |
|---|---|---|---|---|---|---|
| vanilla (OpenCV default) | 21.0 / 45.0 / 56.9 | 0.516 | 231 | 210 | 33.0 / 58.2 / 68.1 | 10.0 / 33.0 / 46.6 |
| pool, top-K response | 18.4 / 42.9 / 55.1 | 0.502 | 219 | 202 | 29.1 / 54.9 / 65.5 | 8.6 / 31.9 / 45.5 |
| pool, grid (ANMS-like) | 24.5 / 51.0 / 62.3 | 0.456 | 183 | 168 | 37.9 / 63.7 / 73.3 | 12.2 / 39.4 / 52.2 |
| pool, random K | 14.0 / 37.5 / 49.5 | 0.279 | 103 | 87 | 21.1 / 44.5 / 55.9 | 7.7 / 31.1 / 43.8 |
| pool, top-2K (2x budget!) | 24.0 / 49.7 / 61.4 | 0.509 | 440 | 401 | 36.9 / 62.5 / 72.3 | 12.2 / 37.9 / 51.5 |

Diagnostics: ROC-AUC(detector response → keypoint has a correct match) = 0.564; share of the pool's correct matches kept by top-K response = 31%; correct-match rate per response quartile = 25.2%, 22.7%, 20.5%, 16.8%.