# RIPE-Select on SIFT: a critical assessment

*Separate note, 2026-11-01. The paper covers ORB and BRISK; this note explains why SIFT stays out of it for now
and lists what to try next. Sources: `runs/eval/megadepth_sift_k2048*.csv|json`, `runs/eval/hpatches_sift_k1024*`,
`runs/report/tables.md`, `runs/report/significance.md`, the training logs in `runs/logs/train_sift_*.log`, and a
clean single-thread timing run. The per-seed tests below were recomputed from the per-pair CSVs (paired t-test
over the 500 MegaDepth pairs, per-pair score = mean over 5/10/20° of clip(1 − err/t, 0, 1)).*

## 1. Verdict

The SIFT result does not support a claim that RL selection helps. The headline number in `REPORT.md` (+3.3 mean
AUC over vanilla, 3 seeds) is:

* **not significant across seeds** (p = 0.21, one-sample t-test on the three seed gains);
* **smaller than a heuristic that needs no learning** (the grid heuristic on the same pool gives +5.1, significant);
* **mostly explained by the larger candidate pool**, not by the learned ranking;
* **negative on HPatches** (significantly worse than vanilla).

For ORB, by contrast, all three seeds give +8.7 to +9.0 (each p < 1e-7), and BRISK is significant across seeds
(p = 0.022). This is why the paper is limited to ORB and BRISK.

## 2. The evidence

### 2.1 The gain depends on the seed

MegaDepth-1500, 500 pairs, K = 2048, RL top-K. Δ = mean AUC over 5/10/20° (points).

| RL seed | AUC @5/10/20° | Δ vs vanilla | Δ vs pool top-K resp. | Δ vs pool grid | Δ vs supervised (s0) |
|---|---|---|---|---|---|
| 0 | 43.5 / 58.6 / 70.8 | **+6.4** (p < 1e-5) | **+4.0** (p = 0.005) | +1.4 (p = 0.27) | +1.2 (p = 0.31) |
| 1 | 40.8 / 55.2 / 67.7 | **+3.3** (p = 0.03) | +0.9 (p = 0.55) | −1.7 (p = 0.21) | −2.0 (p = 0.11) |
| 2 | 38.8 / 52.0 / 63.4 | +0.1 (p = 0.94) | −2.3 (p = 0.16) | **−4.9** (p < 0.001) | **−5.1** (p < 0.001) |
| across seeds | 41.0±2.4 / 55.3±3.3 / 67.3±3.7 | +3.3 (p = 0.21) | +0.9 (p = 0.68) | −1.8 (p = 0.44) | −2.0 (p = 0.39) |

* Seed 2 is no better than vanilla and significantly worse than both the grid heuristic and the supervised model.
* The seed-0 bootstrap row in `REPORT.md` / `significance.md` (+6.4 [+3.7, +9.2]) comes from the best seed. It
  must not be quoted as the SIFT result.
* The spread comes from training, not evaluation: best validation AUC on *reichstag* is 68.8 / 66.1 / 64.9 for
  seeds 0 / 1 / 2. The validation ranking matches the test ranking, so "pick the best seed on validation" would
  select seed 0. With one validation scene and three seeds, that would be a weak protocol to rely on in a paper.
* n = 3 seeds makes the across-seed p-values imprecise. That cuts both ways: the result is not shown to be
  zero, but it is not shown to be positive either.

### 2.2 A no-learning heuristic does better

| Selection (same pool, K, descriptor, MNN, RANSAC) | AUC @5/10/20° | Δ vs vanilla |
|---|---|---|
| vanilla (OpenCV) | 38.7 / 51.8 / 63.2 | — |
| pool, top-K response | 40.6 / 54.1 / 66.2 | +2.4 |
| **pool, grid (no learning)** | **43.3 / 56.9 / 68.7** | **+5.1 [+2.2, +7.8]** |
| supervised, top-K (seed 0) | 42.5 / 57.1 / 69.9 | +5.3 [+2.4, +8.0] |
| RL top-K, 3-seed mean | 41.0 / 55.3 / 67.3 | +3.3 (n.s. across seeds) |
| RL grid, 3-seed mean | 41.3 / 55.3 / 67.4 | ≈ +3.4 |

* About 2.4 of RL's 3.3 points are already reached by re-ranking the larger pool by plain response. What
  remains for the learned ranking is about +0.9 points, and that is not significant (p = 0.68).
* Combining the policy with the grid (RL grid) does not help either: RL grid − pool grid = −1.2 (seed 0).
* The supervised model trained on the same RANSAC pseudo-labels is at least as good as RL. So, for SIFT, the
  RL-specific part of RIPE (the negative-pair penalty) adds nothing measurable. For ORB it does (+5.7, significant).

### 2.3 Better matches, not better poses

MegaDepth matched metrics (seed 0):

| | MMA @1/3/5 px | inlier ratio | AUC mean |
|---|---|---|---|
| vanilla | 20.6 / 26.4 / 28.0 | 24.9 % | 51.2 |
| pool grid | 18.1 / 24.5 / 26.1 | 22.8 % | 56.3 |
| supervised top-K | **24.2 / 31.7 / 33.2** | **30.2 %** | 56.5 |
| RL top-K | 22.0 / 29.3 / 31.0 | 27.8 % | 57.6 |

The policies do learn to pick keypoints that match more reliably (supervised more than RL). But the grid has the
*lowest* match precision and still reaches about the same pose accuracy. For SIFT, pose accuracy on MegaDepth is
limited by how well the matches are spread over the image, not by their precision. The per-keypoint reward does
not see spread, so it optimises the wrong thing for SIFT.

### 2.4 HPatches: learned selection hurts

K = 1024, 540 pairs, Δ = mean AUC over 1/3/5 px, bootstrap 95 % CI (seed 0):

| | Δ |
|---|---|
| RL top-K − vanilla | **−1.4 [−2.4, −0.3]** |
| RL top-K − pool grid | **−2.8 [−3.8, −1.8]** |
| supervised top-K − vanilla | **−2.1 [−3.4, −0.9]** |
| RL grid − vanilla | −0.2 [−1.3, +0.8] |
| pool grid − vanilla | **+1.4 [+0.5, +2.4]** |

RL is worse on both the viewpoint and the illumination splits (illumination AUC@1: 38.3 vs 40.5). MMA@1 falls
from 42.8 (vanilla) to 37.3. Part of this is domain shift: the policy was trained on 1200 px outdoor photos, and
HPatches uses 480 px planar scenes. But the grid heuristic transfers, and the policy does not.

### 2.5 Why SIFT has little headroom

* SIFT's response already carries signal: the correct-match rate falls steadily from 10.4 % to 6.0 % over the
  response quartiles on MegaDepth (ORB: 8.4 % → 4.2 %, with a sharp drop after the first quartile).
  ROC-AUC(response → correct match) is 0.58 (MegaDepth) and 0.61 (HPatches).
* The SIFT descriptor is distinctive enough that ambiguous matches, which the −1 negative-pair reward is meant to
  penalise, are rarer than with ORB's binary descriptor. This fits RL ≈ supervised for SIFT but RL > supervised
  for ORB. It is a hypothesis; we have not tested it directly.

### 2.6 Cost

In a clean single-thread benchmark (warm-up, median of 20 images), the RL pipeline for SIFT takes **1360 ms
against 307 ms for vanilla on MegaDepth (4.4×)** and 347 ms against 94 ms on HPatches (3.7×). Computing the cues
over the 4K pool (674 ms) is the main cost; the MLP takes 18 ms. The 2.7× in `REPORT.md` comes from timings taken
while other workers ran in parallel and understates the cost. The grid heuristic costs only the pool extraction
(666 ms, 2.2×).

### 2.7 Missing baselines

These baselines would be needed before any SIFT claim, even a positive one:

* **Lowe's ratio test.** All rows use plain MNN. A SIFT paper baseline without the ratio test is easy for a
  reviewer to reject, and the ratio test removes many of the same ambiguous matches that selection tries to avoid.
* **RootSIFT** for vanilla (the policy cues use RootSIFT; the descriptor for matching should be stated the same
  way for all rows).
* **The full 1500 pairs**, instead of 500.

## 3. What to try next

Ordered by expected value per hour of compute.

1. **Fix the baselines first (cheap, decisive).** Add MNN + ratio test (0.8–0.9) and RootSIFT to vanilla, pool
   grid and RL in `eval_baselines.py`. If RL still does not beat pool grid, method changes are needed. If the
   ratio test closes the gap to vanilla, selection is not the bottleneck for SIFT.
2. **Make the reward see spread.** Use a set-level reward instead of a per-keypoint one: e.g. pose error of the
   pair, or the number of inliers weighted by how much new image area they cover (occupancy of a coarse grid).
   Alternatively, train under grid inference (`grid_score`), so that the training selection matches the
   one used at test time.
3. **Reduce seed variance.** Leave-one-out baseline instead of the per-image mean, more pairs per step
   (12 → 32), lower LR or gradient clipping, an entropy term, and 5 seeds. A cheap check first: average the
   scores of the 3 existing seeds (ensemble). If the ensemble is stable and ≥ seed 0, the problem is variance,
   not capacity.
4. **Residual policy on SIFT.** Score = response rank + learned correction, started at the response ranking, so
   the worst case is pool top-K. For SURF this did not help (−3.3 vs pool top-K), but for SIFT, unlike SURF,
   the RL policies do beat plain response on validation (64.9–68.8 vs 59.4).
5. **More training scenes.** Two IMC scenes (700 images) is small, and seed spread together with a
   validation–test gap is typical of overfitting. Add 3–5 more IMC/MegaDepth scenes that are disjoint from 0015/0022.
6. **Change the target.** If selection keeps failing for SIFT, learn something it is not already good at,
   for example a learned match filter or a confidence score on top of the SIFT matches, and keep the detector's
   ranking.

**Stop rule:** if items 1–3 do not give a 3-seed mean that is significantly above *pool grid* on MegaDepth
(paired test per seed and across ≥ 5 seeds), report SIFT in the ORB/BRISK paper only as a limitation in one
paragraph: "selection gains are seed-dependent for SIFT and do not exceed spatial heuristics".
