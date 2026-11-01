## Summary

Adds RIPE-Select: a learned keypoint selection step for classical local features (SIFT, ORB, SURF, BRISK), trained with the reward from RIPE (ICCV 2025).

The detector and descriptor stay unchanged. From a larger candidate pool, a small MLP scores each keypoint, and the top K are kept. Baseline and enhanced versions use the same keypoint budget, matcher (mutual nearest neighbour) and RANSAC, so the comparison is fair.

## What's included

- **`proj/cfeat`**: policy features and numpy inference, IMC-2020 scene reader, SURF bridge (OpenCV 3.4.2, py3.7), chunked MNN matching, BRISK support.
- **`proj/scripts`**: RL (REINFORCE) and supervised training, evaluation on HPatches and MegaDepth-1500, report with paired bootstrap CIs, qualitative figures, train/test leak check.
- **`runs/`**: trained policies, per-pair results, logs, metric tables and `REPORT.md`.
- **Metrics**: AUC (HPatches corner error, MegaDepth pose), MMA@1–10 px, inlier ratio, runtime, and viewpoint vs illumination splits, next to the published RIPE numbers.

## Data hygiene

- Training uses IMC-2020 phototourism scenes only.
- The MegaDepth-1500 test scenes (St. Peter's and Brandenburger Tor) are refused by the scene loader.
- HPatches is never used for training.
- A dHash leak check found no shared images with either test set.

## Status

Still in progress, so this is not ready to merge:

- BRISK supervised training.
- SIFT, ORB and BRISK evals with the new metric columns.
- Seed 1/2 runs for SIFT, ORB and BRISK.

These will be pushed to this branch as they finish.
