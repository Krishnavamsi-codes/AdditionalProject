# Phase 2: Result Validation

This document tracks data anomalies, NaNs, infinities, and bounds violations in `results/all_results.csv`.

## Status: ANOMALIES DETECTED

- **Infinity Values Detected**:
  - Column `Reprojection Error` has 1 Infinity values. (Likely caused by divide-by-zero or RANSAC failure returning None).
  - Column `Homography Error` has 1 Infinity values. (Likely caused by divide-by-zero or RANSAC failure returning None).
- **Impossible Bounds**: Column `Repeatability` has 340 values outside [0, 1].

## Explanation of Anomalies
- **Homography Error Infinities**: These occur when OpenCV's `findHomography` fails to find a valid transformation matrix due to an insufficient number of inliers. This typically happens for SIFT on the most extreme sequence pairs where aggressive filtering yields zero geometric matches.
