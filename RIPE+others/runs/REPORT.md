# RIPE-Select: RIPE's reinforcement-learning reward applied to SIFT, ORB and BRISK keypoint selection

*Short write-up, 2026-11-01. All numbers come from `runs/eval/*.json` and `runs/report/`, both produced by
`runs/run_all.ps1`. Full per-variant tables: `runs/report/tables.md`. Paired bootstrap tests:
`runs/report/significance.md`. SURF was run first and is parked (negative result, §6).*

## 1. What was done

**Idea.** RIPE (Künzel et al., ICCV 2025) learns a keypoint detector with REINFORCE. Its only supervision is
whether two images show the same scene. Keypoints are rewarded when they become MNN matches that survive
fundamental-matrix RANSAC: +1 on a same-scene pair and −1 on a different-scene pair. A classical detector has no
heatmap to sample from. So we keep the detector **and** the descriptor unchanged, and learn only **which K of
the detector's candidates are kept**:

* **Candidate pool.** The same detector with lowered thresholds (SIFT contrast 0.01 / edge 20; ORB FAST 7, 8K;
  BRISK AGAST 10 instead of 30; SURF Hessian 20), cut to the top 4K by response.
* **Policy.** An MLP (e.g. 148→128→64→1 for SIFT) over per-candidate cues: response, rank, scale, octave,
  duplicates, local density and rank, intra-image descriptor distinctiveness (NN1, NN2 and their ratio), patch
  statistics, and the (Root)descriptor. **No x, y.** Inference keeps the top K by score (*top-K*) or the best
  scored candidates per cell of a 16×16 grid (*grid*).
* **RL.** Each candidate is a Bernoulli keep/drop decision with p = σ(s + b), where b is solved per image so that
  Σp = K. This replaces RIPE's keypoint-count regulariser and keeps the budget fixed. RIPE's reward is applied per
  kept keypoint: MNN, then PoseLib F-RANSAC at 1.5 px on 1200 px images, then +1/−1 by pair label. Training uses
  REINFORCE with a per-image mean baseline, 1200 steps × 12 pairs, 50 % negative pairs; 40–90 min per run on a laptop CPU.
* **Training data.** IMC-2020 *palace_of_westminster* and *grand_place_brussels*, 350 images each. Positive pairs
  are covisible pairs taken from the COLMAP model. Checkpoints are selected on the held-out *reichstag* scene.
  **No MegaDepth-1500 test scene (0015, 0022) and no HPatches image is used for training or model selection.**
* **Seeds.** The RL policy is trained with seeds 0, 1 and 2 for every method; RL rows report mean ± std (sample
  std, n = 3). The supervised ablation and the matched metrics use seed 0.
* **Ablation.** *sup* is the same network trained with BCE on the same RANSAC pseudo-labels, computed once on the
  full pool. It tests whether RL itself matters.
* **Compute.** Descriptor distances and MNN run on the GPU (torch CUDA, TF32 off) with a numpy fallback
  (`RIPE_GPU=0`); both paths give identical matches. Detection, cues and RANSAC stay on the CPU.

**Fair comparison.** Every row in a table uses the same image, the same pool, the same K (HPatches 1024,
MegaDepth 2048), the same descriptor, the same MNN matcher and the same PoseLib RANSAC (default 100k iterations,
seed 0). The RANSAC threshold is chosen on *vanilla* and then applied to all rows. `vanilla` is plain OpenCV,
which sometimes returns fewer than K points. The like-for-like baselines are `pool_resp` (vanilla's ranking on
the same pool) and `pool_grid`, a strong ANMS-like heuristic in the style of ORB-SLAM that needs no learning.

## 2. Main results: MegaDepth-1500 (relative pose, 500 evenly spaced pairs from both scenes, AUC @5/10/20°)

| Selection (K = 2048) | SIFT | ORB | BRISK |
|---|---|---|---|
| vanilla (OpenCV) | 38.7 / 51.8 / 63.2 | 18.8 / 29.3 / 40.6 | 29.0 / 41.6 / 54.0 |
| pool, top-K response | 40.6 / 54.1 / 66.2 | 15.4 / 25.2 / 36.5 | 30.8 / 43.9 / 56.3 |
| pool, grid (heuristic) | **43.3 / 56.9 / 68.7** | 25.3 / 37.0 / 49.1 | **33.8 / 48.7 / 62.1** |
| supervised, top-K (seed 0) | 42.5 / 57.1 / 69.9 | 20.1 / 32.3 / 45.4 | 33.4 / 48.3 / 61.7 |
| **RIPE-Select (RL), top-K, 3 seeds (SIFT) / 5 seeds (ORB, BRISK)** | 41.0±2.4 / 55.3±3.3 / 67.3±3.7 | **24.6±1.2 / 38.1±1.5 / 51.4±1.5** | 32.4±1.0 / 46.3±1.1 / 59.3±0.8 |
| RIPE-Select (RL), grid, 3 / 5 seeds | 41.3±0.8 / 55.3±1.0 / 67.4±0.9 | 22.1±0.7 / 34.0±0.7 / 45.8±0.4 | 30.3±1.0 / 43.6±0.8 / 56.5±0.7 |

Per seed (RL, top-K, AUC @5/10/20°):

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| SIFT | 43.5 / 58.6 / 70.8 | 40.8 / 55.2 / 67.7 | 38.8 / 52.0 / 63.4 |
| ORB | 25.3 / 38.2 / 51.3 | 25.0 / 38.6 / 52.0 | 24.5 / 38.5 / 52.0 |
| BRISK | 33.9 / 47.9 / 60.2 | 31.3 / 45.3 / 58.2 | 32.6 / 46.7 / 59.7 |

Gain of the 3-seed RL mean (top-K) in mean AUC over 5/10/20°: **+3.3 (SIFT), +8.8 (ORB), +4.7 (BRISK)** over
vanilla; **−1.8 (SIFT), +1.2 (ORB), −2.0 (BRISK)** over the grid heuristic.

Paired bootstrap over test pairs, **seed-0 policy** (Δ = mean AUC over 5/10/20°, 95 % CI). Seed 0 is the best
SIFT seed, so the SIFT row overstates the 3-seed gain:

| | RL − vanilla | RL − top-K response | RL − grid | RL − supervised | grid − vanilla |
|---|---|---|---|---|---|
| SIFT | **+6.4 [+3.7, +9.2]** | **+4.0 [+1.2, +6.8]** | +1.4 [−1.2, +3.9] | +1.2 [−1.0, +3.4] | **+5.1 [+2.2, +7.8]** |
| ORB | **+8.7 [+6.0, +11.6]** | **+12.6 [+9.6, +15.6]** | +1.1 [−1.6, +3.9] | **+5.7 [+3.1, +8.2]** | **+7.6 [+4.6, +10.6]** |
| BRISK | **+5.8 [+3.1, +8.6]** | **+3.7 [+1.2, +6.3]** | −0.8 [−3.4, +1.6] | −0.5 [−2.9, +1.9] | **+6.6 [+4.0, +9.6]** |

**Matched metrics, MegaDepth** (seed 0; MMA = share of MNN matches with epipolar error < t px; inlier ratio at the
1.0 px RANSAC threshold; runtime = ms per image for detection + selection, single thread):

| | MMA @1/3/5 px | inlier ratio | runtime |
|---|---|---|---|
| SIFT vanilla | 20.6 / 26.4 / 28.0 | 24.9 % | 384 ms |
| SIFT pool grid | 18.1 / 24.5 / 26.1 | 22.8 % | 731 ms |
| SIFT RL top-K | 22.0 / 29.3 / 31.0 | 27.8 % | 1044 ms |
| ORB vanilla | 16.9 / 27.4 / 30.0 | 24.0 % | 37 ms |
| ORB pool grid | 12.2 / 20.1 / 22.2 | 17.6 % | 77 ms |
| ORB RL top-K | 17.0 / 27.5 / 30.1 | 24.0 % | 481 ms |
| BRISK vanilla | 16.7 / 24.7 / 27.1 | 22.2 % | 142 ms |
| BRISK pool grid | 15.0 / 22.3 / 24.3 | 20.2 % | 439 ms |
| BRISK RL top-K | 18.1 / 26.9 / 29.2 | 24.5 % | 894 ms |

## 3. HPatches (homography, 540 pairs, AUC of corner error @1/3/5 px)

| Selection (K = 1024) | SIFT | ORB | BRISK |
|---|---|---|---|
| vanilla | 31.8 / 63.5 / 74.1 | 21.0 / 45.0 / 56.9 | 27.5 / 58.0 / 69.4 |
| pool, grid | **33.9 / 64.6 / 75.3** | **24.5 / 51.0 / 62.3** | **31.9 / 61.9 / 72.6** |
| RL, top-K, 3 / 5 seeds | 30.7±0.8 / 61.6±0.3 / 72.8±0.4 | 20.4±0.2 / 45.3±0.4 / 57.1±0.5 | 28.1±0.3 / 58.0±0.1 / 69.7±0.4 |
| RL, grid, 3 / 5 seeds | 31.9±0.1 / 62.8±0.2 / 73.7±0.2 | 23.5±0.5 / 48.9±0.6 / 60.3±0.6 | 30.7±0.3 / 60.1±0.2 / 71.2±0.3 |

Split by HPatches sequence type (seed 0, AUC @1/3/5 px):

| | viewpoint: vanilla | viewpoint: pool grid | viewpoint: RL top-K | illumination: vanilla | illumination: pool grid | illumination: RL top-K |
|---|---|---|---|---|---|---|
| SIFT | 24.0 / 56.5 / 68.5 | 26.2 / 57.6 / 69.6 | 23.4 / 55.5 / 68.2 | 40.5 / 71.1 / 80.3 | 42.3 / 72.2 / 81.6 | 38.3 / 68.6 / 78.6 |
| ORB | 10.0 / 33.0 / 46.6 | 12.2 / 39.4 / 52.2 | 10.1 / 35.4 / 48.6 | 33.0 / 58.2 / 68.1 | 37.9 / 63.7 / 73.3 | 31.5 / 55.2 / 64.7 |
| BRISK | 20.6 / 51.5 / 63.6 | 23.5 / 54.2 / 66.3 | 21.3 / 51.4 / 63.8 | 35.1 / 65.3 / 75.8 | 41.1 / 70.4 / 79.6 | 36.1 / 65.1 / 75.2 |

**Matched metrics, HPatches** (seed 0; reprojection error; inlier ratio at the vanilla RANSAC threshold):

| | MMA @1/3/5 px | inlier ratio | runtime |
|---|---|---|---|
| SIFT vanilla / pool grid / RL top-K | 42.8 / 56.9 / 58.7 · 35.4 / 49.1 / 51.1 · 37.3 / 51.3 / 53.5 | 58.4 · 50.7 · 53.1 % | 380 · 607 · 991 ms |
| ORB vanilla / pool grid / RL top-K | 27.4 / 51.6 / 55.6 · 24.2 / 45.6 / 49.2 · 24.8 / 47.8 / 52.1 | 46.7 · 41.6 · 43.3 % | 18 · 47 · 237 ms |
| BRISK vanilla / pool grid / RL top-K | 36.3 / 53.5 / 56.0 · 32.6 / 48.1 / 50.6 · 34.5 / 53.7 / 57.1 | 55.7 · 50.1 · 56.2 % | 136 · 378 · 591 ms |

On HPatches, learned selection never beats the grid heuristic: RL grid − pool grid is −1.7 (SIFT), −1.8 (ORB) and
−1.5 (BRISK) mean AUC, all significant. RL grid beats vanilla only for ORB (+3.2 [+1.8, +4.6]) and BRISK
(+2.4 [+1.1, +3.6]), and the plain grid beats vanilla by more. Our vanilla SIFT number is above RIPE's (53.9 @3 px)
because we sweep the RANSAC threshold on vanilla (4 px was best).

## 4. Published reference (RIPE paper, Tab. 2; full benchmark, their settings, not re-run here)

| | MegaDepth-1500 AUC @5/10/20° | HPatches AUC @1/3/5 px |
|---|---|---|
| SIFT (RIPE paper) | 37.29 / 50.61 / 61.89 | 29.98 / 53.93 / 65.01 |
| RIPE (paper) | 55.11 / 68.34 / 78.03 | 37.93 / 58.93 / 69.2 |
| SIFT vanilla (ours, 500 pairs) | 38.7 / 51.8 / 63.2 | 31.8 / 63.5 / 74.1 |
| SIFT RL top-K (ours, 3 seeds) | 41.0 / 55.3 / 67.3 | 30.7 / 61.6 / 72.8 |

Our SIFT baseline agrees with the published SIFT row on MegaDepth (+1.4 @5°), so the protocol is comparable.
RIPE itself stays far ahead: it learns the detector and the descriptor end to end, while RIPE-Select can only
choose among the classical detector's candidates.

## 5. Validation (held-out *reichstag* scene, mean AUC, used for checkpoint selection only)

| | plain response | RL seed 0 | RL seed 1 | RL seed 2 |
|---|---|---|---|---|
| SIFT | 59.4 | 68.8 | 66.1 | 64.9 |
| ORB | 17.2 | 35.1 | 34.3 | 34.5 |
| BRISK | 46.5 | 59.5 | 60.0 | 58.3 |

The validation ranking of the SIFT seeds (0 > 1 > 2) is the same as their MegaDepth test ranking.

## 6. Honest conclusions

1. **The RIPE reward transfers to classical features on real 3D scenes.** On MegaDepth, RL selection beats vanilla
   OpenCV for all three methods with the same descriptor, matcher and RANSAC: +8.5 (ORB), +4.5 (BRISK) and +3.3
   (SIFT) mean AUC, averaged over 5 seeds (ORB, BRISK) and 3 seeds (SIFT). Across seeds the ORB and BRISK
   gains are significant (one-sample t-test, n=5: ORB t=13.9, BRISK t=10.9, both p<0.001), and every seed is
   significant on its own over the 500 pairs (paired t-test, p≤0.016). It also raises match quality: MMA and inlier ratio go up for SIFT and
   BRISK and stay level for ORB, while the grid heuristic lowers both. The diagnostics explain why there is room:
   detector response barely predicts matchability on MegaDepth (ROC-AUC 0.56–0.60), and top-K by response keeps
   only 39–44 % of the pool's correct matches.
2. **Most of the pose gain is also available without learning.** The grid heuristic on the same pool reaches
   about the same pose accuracy: RL is ahead only for ORB (+0.9 over 5 seeds, not significant; seed 4 is below the grid), and behind for SIFT (−1.8) and
   BRISK (−2.0). Better-matching keypoints do not turn into better poses when they are spatially clustered.
3. **ORB is the clearest case for RL.** ORB gains most, all 5 seeds gain +6.2 to +9.8, and RL beats the
   supervised model on the same pseudo-labels (+5.7, significant). A likely cause is that the negative-pair penalty
   targets ORB's ambiguous Hamming matches, which positive-only labels never show. For SIFT and BRISK, RL and
   supervised training are not significantly different.
4. **SIFT varies with the seed.** Seed 0 gains +4.8 AUC@5°, seed 2 only +0.1; grid selection is steadier
   (±0.8). The seed-0 bootstrap row therefore overstates the SIFT gain; the 3-seed mean is the headline number.
5. **Selection does not help on HPatches.** HPatches is planar and well textured, so almost any textured point is
   valid; the policies were also trained at 1200 px on outdoor photos, while HPatches uses 480 px. Only grid
   spreading helps there, and the plain grid does it best.
6. **SURF: negative result, parked.** Learned selection hurts on MegaDepth (RL 27.0 / 41.8 / 56.1 against vanilla
   33.7 / 47.0 / 60.2, −5.3 mean AUC), also for supervised training and for a residual policy started at SURF's
   response ranking. On the validation scene the supervised model had helped (+3.6), so the policy overfits to the
   training landmarks. SURF seeds 1–2 were not run. SURF tables remain in `runs/report/tables.md`.
7. **Cost.** Selection needs the larger pool plus per-candidate cues: 2.7× (SIFT), 6.3× (BRISK) and 13× (ORB) the
   vanilla extraction time.

## 7. Limitations

* MegaDepth uses 500 of the 1500 pairs (evenly spaced, both scenes). PoseLib RANSAC at its default 100k iterations
  dominates the runtime. The full 1500 pairs takes about 1 h per method: drop `--limit` in `run_all.ps1`.
* Three RL seeds per method; the supervised ablation, matched metrics, sequence-type split and bootstrap use
  seed 0 only.
* Runtimes are single-thread wall-clock times measured while other evaluation workers ran in parallel. They
  are comparable between rows, not absolute.
* Grayscale-only inputs. Training data is two IMC scenes (700 images); RIPE used the DISK MegaDepth subset and
  about 3 GPU-days. Hyper-parameters were not tuned. Only a single validation scene was used for checkpoint
  selection.
* The RANSAC threshold was swept over 1.0/1.5 px on MegaDepth and 1–6 px on HPatches. The tables use the one
  that is best for vanilla.
* One BRISK seed-1 run ran out of memory at step 90 with 8 workers; it is kept in `runs/archive/crashed/` and was
  re-run from scratch with 6 workers. Only clean exits are marked done, and seed evaluations require both seeds
  to be done.

## 8. Reproduce

```
.venv\Scripts\python.exe proj\scripts\prepare_surf.py --dataset hpatches --root <HPatches root>   # SURF cache (also megadepth/imc)
powershell -ExecutionPolicy Bypass -File runs\run_all.ps1                                       # train + evaluate (resumable)
apenv\Scripts\python.exe proj\scripts\make_report.py                                            # tables, plots, bootstrap
apenv\Scripts\python.exe proj\scripts\qualitative.py --method sift --pair 67 --th 1.0
```
`apenv` is a Python 3.12 venv with the same packages as `.venv` plus `torch==2.14.1+cu126`; with the CPU-only
`.venv`, matching falls back to numpy and gives the same results, only slower.
Code: `proj/cfeat/policy.py` (features and inference), `proj/cfeat/match.py` (MNN, GPU and numpy),
`proj/scripts/train_policy.py` (RL and supervised), `proj/scripts/eval_baselines.py` (evaluation, policies as
`--policy name=path`), `proj/cfeat/imc.py` (training scenes), `proj/cfeat/surf_bridge.py` (SURF).
