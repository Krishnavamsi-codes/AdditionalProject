# HANDOFF — RIPE ideas → SIFT / SURF / ORB (context for Claude Code)

Last updated: 2026-10-31 02:25 IST. Written by the Claude chat session that did the research phase.
Detailed analysis (Phases 1–3, with tables + citations): `proj/docs/analysis_draft.md`.

---

## 1. The task (user's own words, condensed)

Improve one or more classical local features (SIFT, SURF, ORB) using reinforcement-learning ideas from
**RIPE** (ICCV 2025, Künzel/Hilsmann/Eisert, arXiv 2507.04839, code github.com/fraunhoferhhi/RIPE).
The improvement must be **clearly visible on HPatches and MegaDepth-1500** with standard metrics and a
**fair** comparison to vanilla. Tight deadline; needs working code, results, plots, tables, short write-up.

Requested workflow — **do not skip**:
1. Phase 1 understand RIPE ✅ 2. Phase 2 analyse SIFT/SURF/ORB ✅ 3. Phase 3 map RIPE onto weaknesses ✅
4. **Phase 4 run vanilla baselines ⏳ (in progress — see §5)**
5. Phase 5 ranked recommendation (2–3 options, design, plan, eval plan) → **then STOP and wait for the
   user to choose. Do NOT implement the enhancement before the user approves.**

### Ground rules (from the user — keep them)
- Never train on MegaDepth-1500 test scenes: **0015 (St. Peter's Square) and 0022 (Brandenburger Tor)**.
  Also never train on HPatches.
- Baseline and enhanced use the **same keypoint budget, matcher and RANSAC settings**.
- Be honest: if an idea doesn't help, say so and why. Cite sources for weakness claims.

### Still unknown (ask the user)
- GPU / VRAM / disk (laptop "tuf", Windows, ASUS TUF — GPU model not given), and the **deadline date**.
- Whether to include **SURF** (see §4).

---

## 2. Data on disk (verified complete)

Note: Windows "Extract All" nested each dataset one level deeper.

| Dataset | Path (root to pass to scripts) | Contents |
|---|---|---|
| HPatches sequences | `D:\AdditionalProject\RIPE+others\data\hpatches-sequences-release\hpatches-sequences-release` | 116 seqs, 696 .ppm, 580 `H_1_i` files |
| MegaDepth-1500 (glue-factory package) | `D:\AdditionalProject\RIPE+others\data\megadepth1500\megadepth1500` | `images/0015,0022/*.jpg` (806), `depths/` (806 h5), `views.txt`, `pairs.txt` (1500), `pairs_calibrated.txt` |

`views.txt` line format: `name R(9) t(3) MODEL W H params` (PINHOLE fx fy cx cy, COLMAP corner convention, T_w2cam).
Relative pose: `R = R1 R0ᵀ`, `t = t1 − R t0`.

Leftovers that can be deleted: `data\_chunks\` (~1.2 GB partial copies), `ripe_classical_code.tgz`.

Training data (not downloaded yet, proposed): IMC-2020 PhotoTourism scenes that are NOT test scenes, e.g.
temple_nara_japan (3.0 GB), palace_of_westminster (3.2 GB), grand_place_brussels (3.6 GB), sacre_coeur (3.9 GB);
links at https://www.cs.ubc.ca/~kmyi/imw2020/data.html — **exclude brandenburg_gate and st_peters_square**.
Positives = same-scene pairs with covisibility > ~0.1 (RIPE's DISK subset also used covisibility to form pairs —
state this honestly); negatives = cross-scene random pairs (50/50).

---

## 3. Code already written (`D:\AdditionalProject\RIPE+others\proj\`)

```
proj/cfeat/extract.py   detect_describe(img, 'sift'|'surf'|'orb', K, pool=False|True); select(f, K, rule)
                         vanilla  = OpenCV defaults, top-K by response
                         pool     = lowered thresholds (SIFT contrast 0.01/edge 20, SURF hessian 20,
                                    ORB 8K feats fastThreshold 7), later cut to top-4K by response
                         rules: response | grid (16x16 bucketed, ANMS-like) | random | score (external, e.g. policy)
proj/cfeat/match.py     mnn(d0, d1, binary, ratio=None)  numpy L2 / Hamming mutual NN (+ optional ratio)
proj/cfeat/metrics.py   glue-factory-identical AUC; PoseLib homography / relative pose (seed=0);
                         corner error; max(R,t) angular pose error; GT symmetric epipolar distance
proj/cfeat/datasets.py  hpatches_pairs(root, 480 short side, 8 standard seqs excluded → 540 pairs)
                         megadepth1500_pairs(root, 1200 long side, limit=N → evenly spaced subset)
                         (coordinates converted to OpenCV pixel-centre convention; K/H rescaled)
proj/scripts/eval_baselines.py  --dataset hpatches|megadepth --root R --method sift|surf|orb
                         [--k] [--ratio] [--limit] [--diag] [--out results]
run_baselines.bat       pip installs deps, opens 2 windows (SIFT, ORB): HPatches (all, --diag) then MegaDepth --limit 500
```

### Evaluation protocol implemented (follows RIPE Tab. 2 / glue-factory)
- HPatches: short side 480, 540 pairs, K=1024, MNN, PoseLib homography, AUC of mean corner error @1/3/5 px;
  split by i_ (illumination) / v_ (viewpoint); match precision @3 px.
- MegaDepth-1500: long side 1200, K=2048, MNN, PoseLib relative pose, AUC of max(R,t) error @5/10/20°;
  match precision = GT symmetric epipolar < 2 px.
- RANSAC threshold sweep (HP: 1,2,3,4,6 px; MD: 0.5,1,1.5,2,3 px). Reported two ways: **th_fixed** (best for
  vanilla, applied to every variant — the fair one) and th_best per variant (glue-factory `ransac_th=-1`).
- Variants per method: `vanilla`, `pool_resp`, `pool_grid`, `pool_rand`, `pool_resp2x` (2K budget — headroom
  diagnostic only, NOT a fair comparison).
- `--diag`: ROC-AUC of detector response (and size) for predicting "keypoint has a GT-correct MNN match";
  correct-match rate per response quartile; fraction of pool's correct matches surviving top-K cut.
- Outputs: `results/<dataset>_<method>_k<K>.json` (summary + diag) and `_pairs.csv`.

Smoke-tested on real data (few pairs) — runs fine. Reference numbers from RIPE paper (PoseLib, MNN):
SIFT MegaDepth-1500 AUC 37.29 / 50.61 / 61.89; SIFT HPatches 29.98 / 53.93 / 65.01; RIPE 55.11/68.34/78.03 and 37.93/58.93/69.20.

---

## 4. Environment notes
- **SURF**: pip `opencv-contrib-python` does NOT include it (nonfree disabled). The chat session built
  OpenCV 4.10 with `OPENCV_ENABLE_NONFREE=ON` in its cloud workspace — not available to you. On Windows
  options: build OpenCV from source with nonfree (CMake + VS, ~1 h), or drop SURF and say so in the report.
  Pending user decision.
- `metrics.error_auc` uses `np.trapezoid` → numpy ≥ 2.0.
- poselib ≥ 2.0 (`estimate_relative_pose(p0, p1, camdict0, camdict1, {"max_epipolar_error": th, "seed": 0})`).

---

## 5. Current status (updated 2026-10-31 13:40 by Claude Code)
User decided: build it (no stop at Phase 5), include SURF, put everything under `runs/`.
**Done:** Phase 4 baselines and diagnostics; RIPE-Select (RL) and the supervised ablation trained for SIFT, ORB and SURF;
HPatches (540 pairs) and MegaDepth-1500 (500 pairs) evaluated. Results and honest conclusions: **`runs/REPORT.md`**.
Pipeline: `runs/run_all.ps1` (resumable; RL seeds 1 and 2 still running). Report: `proj/scripts/make_report.py` writes `runs/report/`.
Environment: `.venv` (py3.12, numpy 2, torch-cpu, poselib); `.venv_surf37` (py3.7 + opencv-contrib 3.4.2.17, SURF only).
Training data: `data/imc2020/{palace_of_westminster, grand_place_brussels}` (train), `reichstag` (validation).
Memory: 16 GB RAM. Run one heavy job at a time (8 workers). MNN and policy features are row-chunked.

## 5b. Original plan (superseded)
1. ⏳ User runs `run_baselines.bat` (SIFT + ORB; HPatches full + MegaDepth 500-pair subset). Consider running
   the full 1500 MegaDepth pairs (only 806 unique images) if the machine is fast — drop `--limit 500`.
2. Read `results/*.json`. Confirm/deny the hypotheses:
   - H1: on MegaDepth, response ranking is a weak predictor of matchability (ROC-AUC near 0.5–0.6), and
     `pool_resp2x` ≫ vanilla → headroom for a smarter selection at fixed K.
   - H2: HPatches gains from selection are smaller; illumination split is where SIFT/ORB are weakest.
   - H3: how much `pool_grid` (cheap heuristic) already recovers — this is the bar the RL method must beat.
3. Make plots (bar charts per method/variant; matchability-by-quartile), write Phase 4 section.
4. Write Phase 5 (below, provisional) with numbers → **STOP, wait for user choice.**

---

## 6. Phase 1 summary — RIPE mechanics (from paper + code)
- **State** image; **policy** = RIPE network logit heatmap; 8×8 cells; per cell Categorical over 64 px +
  Bernoulli accept (σ of chosen logit). log π = log Cat + log Bern (`ripe/models/ripe.py` KeypointSampler).
  One-step contextual bandit, ~C cells = many "trajectories".
- **Reward**: MNN on descriptors → F-matrix RANSAC (PoseLib, 1 px @560 px). R[p,q] = sign(λ)·1 for
  MNN∧inlier pairs, tiny penalty (−1e-7·α, sign-flipped for negatives) elsewhere (`utils.get_rewards`).
  Positive pairs (same scene) reward matchable+epipolar-consistent keypoints; negative pairs penalise
  keypoints whose matches a random F still explains (repetitive/generic structure).
- **Gradient**: REINFORCE, Σ R ⊙ (log p ⊕ log p') (`train.py` ~L258–326); regulariser kp_penalty
  −7e-7·α on log-probs (paper Tab.1 says −7e-8; supp ablation best −7e-7); α ramps over first 2/3 of training.
- **Descriptors**: hyper-columns (VGG stages 64+128+256+512=960 → 1×1 conv → 256).
  **Descriptor loss** (`losses/contrastive_loss.py`): positives — hinge(μ + d(inlier match) − d(2nd-NN hardest)),
  negatives — hinge(μ − d(match)); μ = 0.2, weight ψ = 5; if <8 inliers use all matches.
  Ablation: without desc loss or without hyper-columns, training **fails**.
- Training: 80k steps, batch 6×4 accum, 560 px, AdamW 1e-3→1e-6, ~3 days A100, DISK MegaDepth subset.
- **Transferable**: reward (black-box MNN+F-RANSAC, signed by pair label), negative pairs, grid-cell sampling,
  regulariser/curriculum, descriptor loss (needs any differentiable descriptor head), unlabeled-pair training.
  **Tied to CNN**: dense heatmap (choosing *where*), hyper-columns.

## 7. Phase 2/3 summary
See `proj/docs/analysis_draft.md` for the stage-by-stage tables with citations (Lowe 2004; Bay 2008;
Rublee 2011; Mur-Artal 2015; Hartmann 2014 "Predicting Matchability"; Papadaki 2020 "Match or No Match";
Jin et al. 2021 IMC; Arandjelović 2012 RootSIFT; Bellavia & Colombo 2020; Bian 2019; Tareen & Saleem 2018;
Mikolajczyk 2005; Morel & Yu 2009). Key point: HPatches stresses localisation/illumination/descriptor;
MegaDepth stresses *which* K keypoints are kept (foliage, people, sky, repetitive facades) and descriptor ambiguity.

Candidate ideas (A–G) scored in the draft. Cleanest transfer = **A** (RL selection) ± **B** (descriptor head).

## 8. Phase 5 — PROVISIONAL recommendation (confirm with Phase-4 numbers before presenting)
1. **SIFT + RL keypoint-selection policy ("RIPE-Select")** — top choice.
   - Candidate pool: SIFT with lowered thresholds, top-4K by response. Action: per grid cell (e.g. 32×32 px)
     Categorical over candidates + Bernoulli accept (RIPE style), or budget-normalised Bernoulli
     p_i = σ(f(x_i)+b), b set per image so Σp = K (detach b). Inference: top-K by policy score.
   - Policy f: small MLP (~150-D → 64 → 64 → 1), CPU-trainable. Per-candidate inputs: log response, response
     percentile, log size, octave/layer, #orientation duplicates, RootSIFT descriptor, intra-image
     distinctiveness (NN and NN1/NN2 distance to other descriptors in the same image → repetitive patterns),
     local density + local response rank, patch intensity/contrast/saturation (+ mean colour/HSV for
     vegetation/sky). No absolute x,y (dataset bias).
   - Reward: exactly RIPE — MNN on SIFT descriptors of kept points, F-RANSAC (PoseLib), +1 inlier match in
     positive pairs, −1 in negative pairs, tiny penalty otherwise; REINFORCE with mean-reward baseline.
   - Pre-extract pools for training images once (~10 min); ~50 ms/pair step on CPU → tens of k pairs/hour.
2. **SIFT + RIPE-Select + descriptor head (RIPE descriptor loss)** — residual MLP on RootSIFT, L2-normalised,
   trained jointly (ψ-weighted). Report separately (descriptor changes).
3. **ORB + RIPE-Select** — same policy on ORB pool (Hamming MNN); likely largest relative gain, low absolute.
Not recommended (explain in report): learned thresholds (tuned baseline erases it), orientation/upright
(confound, poor RL fit), CNN hyper-column descriptors (credit goes to CNN, needs GPU + pretrained weights).

### Evaluation plan / fairness rules
- Same pool, same K (HP 1024, MD 2048; also 512/4096 budget curve), same descriptor (for A), same MNN matcher
  (+ ratio-test variant as robustness), same PoseLib RANSAC with threshold chosen on vanilla; also report
  per-method best threshold.
- Baselines: vanilla, pool_resp, **pool_grid (ANMS-like)**, random. Ablations: no negative pairs; no
  descriptor/distinctiveness inputs; **supervised matchability classifier on the same RANSAC pseudo-labels
  (RL vs. supervised)**; policy without colour cues. 3 training seeds; report mean ± std. Runtime overhead.
- Hyper-parameters chosen on a held-out IMC scene (e.g. reichstag), never on HPatches/MegaDepth-1500.
- Metrics: AUC as above + match precision, #inliers, HPatches i/v split, plots (bar charts, budget curve,
  qualitative keypoint overlays).
