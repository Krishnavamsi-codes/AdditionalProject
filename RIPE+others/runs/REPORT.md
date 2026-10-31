# RIPE-Select: RIPE's reinforcement-learning reward applied to SIFT, ORB and SURF keypoint selection

*Short write-up, 2026-10-31. All numbers come from `runs/eval/*.json` and `runs/report/`, both produced by `runs/run_all.ps1`.
Full per-variant tables: `runs/report/tables.md`. Paired bootstrap tests: `runs/report/significance.md`.*

## 1. What was done

**Idea.** RIPE (Künzel et al., ICCV 2025) learns a keypoint detector with REINFORCE. Its only supervision is
whether two images show the same scene. Keypoints are rewarded when they become MNN matches that survive
fundamental-matrix RANSAC: +1 on a same-scene pair and −1 on a different-scene pair. A classical detector has no
heatmap to sample from. So we keep the detector **and** the descriptor unchanged, and learn only **which K of
the detector's candidates are kept**:

* **Candidate pool.** The same detector with lowered thresholds (SIFT contrast 0.01 / edge 20; ORB FAST 7, 8K;
  SURF Hessian 20), cut to the top 4K by response.
* **Policy.** An MLP (148→128→64→1 for SIFT) over per-candidate cues: response, rank, scale, octave, duplicates,
  local density and rank, intra-image descriptor distinctiveness (NN1, NN2 and their ratio), patch statistics,
  and the (Root)descriptor. **No x, y.** Inference keeps the top K by score.
* **RL.** Each candidate is a Bernoulli keep/drop decision with p = σ(s + b), where b is solved per image so that
  Σp = K. This replaces RIPE's keypoint-count regulariser and keeps the budget fixed. RIPE's reward is applied per
  kept keypoint: MNN, then PoseLib F-RANSAC at 1.5 px on 1200 px images, then +1/−1 by pair label. Training uses
  REINFORCE with a per-image mean baseline, 1200 steps × 12 pairs, 50 % negative pairs, on CPU in about 40 min.
* **Training data.** IMC-2020 *palace_of_westminster* and *grand_place_brussels*, 350 images each. Positive pairs
  are covisible pairs taken from the COLMAP model. Checkpoints are selected on the held-out *reichstag* scene.
  **No MegaDepth-1500 test scene (0015, 0022) and no HPatches image is used for training or model selection.**
* **Ablation.** *sup* is the same network trained with BCE on the same RANSAC pseudo-labels, computed once on the
  full pool. It tests whether RL itself matters.
* **SURF.** pip OpenCV has nonfree disabled. SURF therefore runs in OpenCV 3.4.2 (Python 3.7 env, cached by image
  hash); the same algorithm with default settings.

**Fair comparison.** Every row in a table uses the same image, the same pool, the same K (HPatches 1024,
MegaDepth 2048), the same descriptor, the same MNN matcher and the same PoseLib RANSAC (default 100k iterations,
seed 0). The RANSAC threshold is chosen on *vanilla* and then applied to all rows. `vanilla` is plain OpenCV,
which sometimes returns fewer than K points. The like-for-like baselines are `pool_resp` (vanilla's ranking on
the same pool) and `pool_grid`, a strong ANMS-like heuristic in the style of ORB-SLAM.

## 2. Main results: MegaDepth-1500 (relative pose, 500 evenly spaced pairs from both scenes, AUC @5/10/20°)

| Selection (K = 2048) | SIFT | ORB | SURF |
|---|---|---|---|
| vanilla (OpenCV) | 38.7 / 51.8 / 63.2 | 18.8 / 29.3 / 40.6 | **33.7 / 47.0 / 60.2** |
| pool, top-K response | 40.6 / 54.1 / 66.2 | 15.4 / 25.2 / 36.5 | 33.0 / 46.7 / 60.2 |
| pool, grid (heuristic) | 43.3 / 56.9 / 68.7 | 25.3 / 37.0 / 49.1 | 31.3 / 44.7 / 57.8 |
| supervised, same labels | 42.5 / 57.1 / 69.9 | 20.1 / 32.4 / 45.4 | 28.6 / 42.9 / 58.0 |
| **RIPE-Select (RL)** | **43.6 / 58.6 / 70.8** | **25.3 / 38.2 / 51.3** | 27.0 / 41.8 / 56.1 |

Sanity check: our vanilla SIFT (38.7 / 51.8 / 63.2) matches RIPE's published SIFT figure (37.3 / 50.6 / 61.9).

Paired bootstrap over test pairs (Δ = mean AUC over 5/10/20°, 95 % CI):

| | RL − vanilla | RL − top-K response | RL − grid | RL − supervised |
|---|---|---|---|---|
| SIFT | **+6.4 [+3.7, +9.2]** | **+4.0 [+1.2, +6.8]** | +1.4 [−1.2, +3.9] | +1.2 [−1.0, +3.4] |
| ORB | **+8.7 [+6.0, +11.6]** | **+12.6 [+9.6, +15.6]** | +1.1 [−1.6, +3.9] | **+5.7 [+3.1, +8.2]** |
| SURF | **−5.3 [−7.8, −3.0]** | −5.0 [−7.4, −2.7] | −3.0 [−5.2, −0.7] | −1.5 [−3.9, +0.8] |

## 3. HPatches (homography, 540 pairs, AUC of corner error @1/3/5 px)

| Selection (K = 1024) | SIFT | ORB | SURF |
|---|---|---|---|
| vanilla | 31.8 / 63.5 / 74.1 | 21.0 / 45.1 / 56.9 | 26.0 / 56.8 / 68.8 |
| pool, grid | **33.9 / 64.6 / 75.3** | **24.5 / 51.0 / 62.3** | **28.8 / 59.2 / 70.3** |
| RL, top-K | 30.5 / 61.7 / 73.1 | 20.3 / 44.8 / 56.3 | 23.6 / 54.6 / 66.5 |
| RL, grid | 31.8 / 63.0 / 73.9 | 23.7 / 48.6 / 60.2 | 26.6 / 56.9 / 68.8 |

On HPatches, learned selection gives **no gain** over the grid heuristic for any method (significantly worse
by 1.7–2.0 points). It only beats vanilla for ORB (+3.2 [+1.8, +4.6]). Our vanilla SIFT number is above RIPE's
(53.9 @3 px) because we sweep the RANSAC threshold on vanilla (4 px was best).

## 4. Honest conclusions

1. **The RIPE reward does transfer to classical features where keypoint selection is the bottleneck.** On
   MegaDepth it improves SIFT by +6.9 AUC@10° and ORB by +8.9 AUC@10° over vanilla, with the same descriptor,
   matcher and RANSAC. The Phase 4 diagnostics explain why. Detector response barely predicts matchability on
   MegaDepth (ROC-AUC 0.58 for SIFT, 0.60 for ORB, 0.61 for SURF), and top-K by response discards 53–61 % of the pool's
   correct matches. The figure `qualitative_sift_pair67.png` is a hand-picked success case: vanilla spends its
   budget on people and street clutter, while the policy moves it onto the facade. `qualitative_sift_pair250.png`
   is an unselected pair.
2. **A cheap heuristic captures most of this gain.** Against grid/ANMS spreading on the same pool, RL is ahead
   on MegaDepth for SIFT and ORB, by about +1 to +2 AUC@10°, but the difference is **not statistically significant**.
   On HPatches the grid heuristic wins.
3. **RL versus supervised learning on the same pseudo-labels is mixed.** RL clearly wins for ORB (+5.7, significant).
   The most likely cause is that the negative-pair penalty targets ORB's ambiguous Hamming matches, which the
   positive-only labels never show. For SIFT the gap is not significant, and on validation the supervised model
   was even slightly better.
4. **SURF: negative result.** Learned selection hurts on MegaDepth (−5.3). This holds for supervised training
   too, and for a residual policy started exactly at SURF's response ranking. On the validation scene the
   supervised model had helped (+3.6), so the policy overfits to the training landmarks. SURF's Hessian response
   is already the best ranking we found for SURF.
5. **Selection does not help on HPatches.** HPatches is planar and well textured, so almost any textured point is
   valid. The policies were also trained at 1200 px on outdoor photos, while HPatches uses 480 px, a different
   scale distribution.

## 5. Limitations

* MegaDepth uses 500 of the 1500 pairs (evenly spaced, both scenes). PoseLib RANSAC at its default 100k iterations
  dominates the runtime. The full 1500 pairs takes about 1 h per method: `--limit` is dropped in `run_all.ps1`.
* Seed variance: seeds 1 and 2 of the RL policy are evaluated separately, and the table in `runs/report/tables.md`
  shows mean ± std once they finish. The headline numbers above are seed 0.
* Grayscale-only inputs. Training data is two IMC scenes (700 images); RIPE used the DISK MegaDepth subset and
  about 3 GPU-days. Hyper-parameters were not tuned. Only a single validation scene was used for checkpoint
  selection.
* The RANSAC threshold was swept over 1.0/1.5 px on MegaDepth and 1–6 px on HPatches. The tables use the one
  that is best for vanilla.

## 6. Reproduce

```
.venv\Scripts\python.exe proj\scripts\prepare_surf.py --dataset hpatches --root <HPatches root>   # SURF cache (also megadepth/imc)
powershell -ExecutionPolicy Bypass -File runs\run_all.ps1                                       # train + evaluate (resumable)
.venv\Scripts\python.exe proj\scripts\make_report.py                                            # tables, plots, bootstrap
.venv\Scripts\python.exe proj\scripts\qualitative.py --method sift --pair 67 --th 1.0
```
Code: `proj/cfeat/policy.py` (features and inference), `proj/scripts/train_policy.py` (RL and supervised),
`proj/scripts/eval_baselines.py` (evaluation, policies as `--policy name=path`), `proj/cfeat/imc.py`
(training scenes), `proj/cfeat/surf_bridge.py` (SURF).
