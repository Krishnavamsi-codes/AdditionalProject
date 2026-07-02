# Phase 3: Split Analysis (Illumination vs Viewpoint)

## Illumination Sequences
| Method   |   Repeatability |       AP |   Matching Score |   Inlier Ratio |   MMA@3px |
|:---------|----------------:|---------:|-----------------:|---------------:|----------:|
| BRISK    |        0.753057 | 0.236857 |         0.215468 |       0.487513 |  0.487513 |
| LATCH    |        0.692303 | 0.27354  |         0.195202 |       0.526941 |  0.526941 |
| ORB      |        0.692303 | 0.224162 |         0.210459 |       0.477224 |  0.477224 |
| SIFT     |        0.502658 | 0.194154 |         0.17767  |       0.79016  |  0.79016  |

## Viewpoint Sequences
| Method   |   Repeatability |       AP |   Matching Score |   Inlier Ratio |   MMA@3px |
|:---------|----------------:|---------:|-----------------:|---------------:|----------:|
| BRISK    |        1.09706  | 0.26033  |         0.188084 |       0.497584 |  0.497584 |
| LATCH    |        0.928065 | 0.157335 |         0.109464 |       0.344335 |  0.344335 |
| ORB      |        0.928065 | 0.262186 |         0.204812 |       0.500801 |  0.500801 |
| SIFT     |        0.696977 | 0.242564 |         0.188209 |       0.786799 |  0.786799 |

## Discussion
- **Illumination**: Focuses on photometric robustness. Methods relying on pure pixel intensities or simple gradients without robust normalization often degrade.
- **Viewpoint**: Focuses on geometric robustness (scale, rotation, perspective). Methods without true affine/scale invariance (e.g. basic FAST/BRIEF) degrade significantly here.
