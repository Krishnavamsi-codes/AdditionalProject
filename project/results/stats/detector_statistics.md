# Phase 5: Detector Statistics

This document provides statistical distributions (mean, median, bounds) across all 580 sequence pairs.

## Summary Statistics
| Method   |   ('Num Keypoints 1', 'mean') |   ('Num Keypoints 1', 'median') |   ('Num Keypoints 1', 'min') |   ('Num Keypoints 1', 'max') |   ('Num Matches', 'mean') |   ('Num Matches', 'median') |   ('Num Matches', 'min') |   ('Num Matches', 'max') |   ('Inlier Ratio', 'mean') |   ('Inlier Ratio', 'median') |   ('Inlier Ratio', 'min') |   ('Inlier Ratio', 'max') |   ('Total Time (ms)', 'mean') |   ('Total Time (ms)', 'median') |   ('Total Time (ms)', 'min') |   ('Total Time (ms)', 'max') |
|:---------|------------------------------:|--------------------------------:|-----------------------------:|-----------------------------:|--------------------------:|----------------------------:|-------------------------:|-------------------------:|---------------------------:|-----------------------------:|--------------------------:|--------------------------:|------------------------------:|--------------------------------:|-----------------------------:|-----------------------------:|
| BRISK    |                       7842.21 |                            4516 |                          235 |                        68859 |                  2459.71  |                      1572   |                       79 |                    16991 |                   0.492635 |                     0.520484 |                         0 |                  0.970645 |                      573.222  |                         229.883 |                      12.8794 |                    14670.1   |
| LATCH    |                       4480.06 |                            5000 |                         1028 |                         5000 |                  1287.34  |                      1233.5 |                      229 |                     3638 |                   0.434064 |                     0.420591 |                         0 |                  0.98983  |                      337.341  |                         368.07  |                      73.993  |                      683.737 |
| ORB      |                       4480.06 |                            5000 |                         1028 |                         5000 |                  1605.29  |                      1549   |                      242 |                     3869 |                   0.489216 |                     0.516068 |                         0 |                  0.981132 |                       99.2238 |                         101.308 |                      13.3886 |                      622.04  |
| SIFT     |                       3279.23 |                            3591 |                          264 |                         5001 |                   620.776 |                       411   |                        4 |                     3715 |                   0.78845  |                     0.895121 |                         0 |                  0.999211 |                      494.525  |                         474.495 |                      77.3301 |                     2568.39  |

## Visual Distributions
![Keypoints](stats/dist_keypoints.png)

![Matches](stats/dist_matches.png)

![Inliers](stats/dist_inliers.png)

