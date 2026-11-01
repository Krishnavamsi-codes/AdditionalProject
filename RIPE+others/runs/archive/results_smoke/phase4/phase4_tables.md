
### hpatches / SIFT (K=1024, pairs=3, RANSAC th fixed on vanilla = 3.0)

| variant | AUC @1px / @3px / @5px (fixed th) | AUC (best th) | best th | #kpts | #matches | match prec | #correct | RANSAC inl | AUC i (fixed) | AUC v (fixed) |
|---|---|---|---|---|---|---|---|---|---|---|
| vanilla | 26.31 / 53.33 / 58.67 | 26.31 / 53.33 / 58.67 | 3.0 | 670 | 235 | 0.339 | 120 | 120 | 26.31 / 53.33 / 58.67 | 0.00 / 0.00 / 0.00 |
| pool_resp | 26.35 / 51.48 / 57.56 | 26.33 / 53.73 / 58.91 | 6.0 | 1024 | 336 | 0.337 | 151 | 150 | 26.35 / 51.48 / 57.56 | 0.00 / 0.00 / 0.00 |
| pool_grid | 26.13 / 46.17 / 54.37 | 25.42 / 53.89 / 59.00 | 6.0 | 1024 | 330 | 0.287 | 116 | 115 | 26.13 / 46.17 / 54.37 | 0.00 / 0.00 / 0.00 |
| pool_rand | 21.08 / 29.25 / 30.88 | 16.75 / 45.77 / 54.13 | 4.0 | 1024 | 297 | 0.157 | 53 | 51 | 21.08 / 29.25 / 30.88 | 0.00 / 0.00 / 0.00 |
| pool_resp2x | 26.51 / 54.34 / 59.27 | 26.52 / 55.63 / 74.34 | 6.0 | 1996 | 627 | 0.298 | 247 | 246 | 26.51 / 54.34 / 59.27 | 0.00 / 0.00 / 0.00 |

Diagnostics (candidate pool, 4K): ROC-AUC(response → correct MNN match) = 0.725, ROC-AUC(size) = 0.673; correct-match rate by response quartile (top→bottom) = 17.1%, 11.1%, 7.1%, 3.4%; share of pool's correct matches kept by top-K response = 63.4%; pool correct matches/pair = 288.

### megadepth / ORB (K=2048, pairs=3, RANSAC th fixed on vanilla = 1.0)

| variant | AUC @5° / @10° / @20° (fixed th) | AUC (best th) | best th | #kpts | #matches | match prec | #correct | RANSAC inl |
|---|---|---|---|---|---|---|---|---|
| vanilla | 0.00 / 36.99 / 51.83 | 0.00 / 36.99 / 51.83 | 1.0 | 2048 | 519 | 0.161 | 88 | 81 |
| pool_resp | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 17.80 | 0.5 | 2048 | 524 | 0.128 | 73 | 72 |
| pool_grid | 24.20 / 28.77 / 54.07 | 24.75 / 47.09 / 69.49 | 2.0 | 2048 | 501 | 0.108 | 56 | 51 |
| pool_rand | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.5 | 2048 | 559 | 0.056 | 31 | 35 |
| pool_resp2x | 27.58 / 46.10 / 56.38 | 31.76 / 50.67 / 58.67 | 1.5 | 4096 | 979 | 0.133 | 138 | 129 |

Diagnostics (candidate pool, 4K): ROC-AUC(response → correct MNN match) = 0.613, ROC-AUC(size) = 0.557; correct-match rate by response quartile (top→bottom) = 5.6%, 2.2%, 1.7%, 1.5%; share of pool's correct matches kept by top-K response = 44.2%; pool correct matches/pair = 224.

### megadepth / SIFT (K=2048, pairs=1, RANSAC th fixed on vanilla = 1.5)

| variant | AUC @5° / @10° / @20° (fixed th) | AUC (best th) | best th | #kpts | #matches | match prec | #correct | RANSAC inl |
|---|---|---|---|---|---|---|---|---|
| vanilla | 0.00 / 74.67 / 87.33 | 0.00 / 74.67 / 87.33 | 1.5 | 2048 | 604 | 0.051 | 31 | 38 |
| pool_resp | 0.00 / 0.00 / 52.35 | 0.00 / 0.00 / 53.64 | 0.5 | 2048 | 567 | 0.053 | 30 | 36 |
| pool_grid | 70.12 / 85.06 / 92.53 | 81.57 / 90.79 / 95.39 | 0.5 | 2048 | 618 | 0.181 | 112 | 125 |
| pool_rand | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 67.86 | 2.0 | 2048 | 615 | 0.047 | 29 | 36 |
| pool_resp2x | 0.00 / 60.16 / 80.08 | 0.00 / 73.11 / 86.55 | 3.0 | 4096 | 1076 | 0.066 | 71 | 70 |

Diagnostics (candidate pool, 4K): ROC-AUC(response → correct MNN match) = 0.313, ROC-AUC(size) = 0.635; correct-match rate by response quartile (top→bottom) = 1.4%, 1.9%, 4.1%, 7.6%; share of pool's correct matches kept by top-K response = 9.5%; pool correct matches/pair = 306.