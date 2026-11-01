# RIPE ideas for classical features: research and analysis (Phases 1–5)

## Phase 1 – RIPE in one page

**Policy / state / action.** The state is the image. The policy is the RIPE network (VGG-19 encoder + DKM-style refiner decoder) producing a logit heatmap H. H is cut into 8×8 cells. In every cell the policy (i) samples one location from a softmax over the 64 logits (Categorical) and (ii) samples an accept/reject flag from Bernoulli(σ(logit of the chosen pixel)). Code: `ripe/models/ripe.py::KeypointSampler.sample`, log π = log Categorical + log Bernoulli. Every cell is an independent one-step decision, so one image gives ~C≈4900 "trajectories" at 560 px – a contextual bandit, not a sequential MDP.

**Reward.** Accepted keypoints of both images are matched by mutual nearest neighbour (Kornia, L2), and the matches are filtered by fundamental-matrix RANSAC (PoseLib, epipolar threshold 1 px at 560 px). A reward matrix R∈R^{C×C'} gets `sign(λ)·ρ` (ρ=1) at every (p,q) that is an MNN match **and** a RANSAC inlier; every other entry gets a tiny constant (`fp_penalty·α = −1e-7`, sign flipped for negative pairs). Positive pairs (same scene, λ=1) therefore reward keypoints that are matchable *and* consistent with one epipolar geometry; negative pairs (different scenes, λ=−1) punish keypoints whose matches a random F can still "explain" – repetitive/generic structures. No depth, pose or correspondences are used; only the binary pair label.

**Gradient.** REINFORCE with the outer-sum trick: ĝ = Σ_{p,q} R_{pq} ∇(log p_p + log p'_q) (Eq. 6; `train.py` lines 258–326: `dense_rewards * dense_logprobs`). A regulariser −ε·Σ log p (ε = −7e-7 in code; Tab. 1 of the paper says −7e-8 but the supp. ablation picks −7e-7) stops the network from producing low-probability keypoints, ramped in over the first 2/3 of training.

**Hyper-columns + descriptor loss.** Descriptors are bilinear samples of the four VGG stages (64+128+256+512 = 960-D) at each keypoint, projected by a 1×1 conv to 256-D. The descriptor loss (`losses/contrastive_loss.py`) uses the RANSAC inliers as *pseudo* ground truth: for positive pairs a hard-negative hinge max(0, μ + d(match) − d(2nd-nearest)), for negative pairs max(0, μ − d(match)), μ = 0.2, weight ψ = 5. The supplementary ablation is important: **without the descriptor loss, or without hyper-columns, RL training fails** – the reward is only informative once descriptors can match.

**Training cost.** 80k steps, batch 6 × 4 accumulation, 560 px, bf16, ~3 days on one A100; DISK MegaDepth subset, 50 % positive / 50 % random cross-scene negative pairs.

**Tied to RIPE's CNN vs. transferable.**

| RIPE component | Tied to the CNN? | Transfers to SIFT/SURF/ORB as… |
|---|---|---|
| Dense per-pixel heatmap, sampling *where* to put a keypoint | Yes – needs a network that outputs a full-resolution map | A policy over the *existing* candidate pool: choose *which* classical candidates to keep (smaller action space) |
| Grid-cell Categorical + Bernoulli accept | No | Cell-wise sampling over candidates → keeps spatial spread, prevents collapse |
| Reward = MNN + F-RANSAC inliers, signed by pair label | No – works with any detector, descriptor, matcher | Directly reusable, black-box |
| Negative pairs as penalty | No | Directly reusable; targets repetitive / generic structures |
| Hyper-column descriptors | Yes (needs an encoder) | Only with a CNN; otherwise as an extra "context" feature for the policy |
| Descriptor loss (Eq. 8) | No – needs any *differentiable* descriptor head | A small residual head on top of (Root)SIFT/SURF, trained with Eq. 8 |
| ε regulariser, α curriculum | No | Reusable |
| Training on arbitrary unlabeled pairs | No | Train on non-test landmark collections without 3D |

## Phase 2 – Where SIFT, SURF and ORB lose (stage by stage)

Notation for "hurts": **H** = HPatches homography (planar, 52 illumination + 56 viewpoint sequences after the standard 8 exclusions); **M** = MegaDepth-1500 relative pose (wide baseline, landmarks, seasons/day-night, transient objects, vegetation, repetitive facades, mostly upright images).

### SIFT (Lowe 2004)
| Stage | What it does | Known weaknesses | Hurts |
|---|---|---|---|
| Detection | DoG extrema in a 3-layer/octave scale space; contrast threshold 0.04 (÷ layers) and edge ratio 10 | Hand-set thresholds; fires heavily on high-frequency texture (foliage, gravel, crowds) that is not stable over time; few points in low-contrast/dark images | M strongly (trees, people, sky edges); H-illum. mildly |
| Scale / orientation | Scale from the DoG extremum; orientation = 36-bin gradient histogram, extra keypoints for peaks > 80 % (≈15 % duplicates) | Orientation is noisy under illumination change and viewpoint; rotation invariance is unnecessary for upright photo collections and costs discriminability (IMC'20 recommends upright variants, Jin et al. 2021); DoG is not affine-invariant (breaks beyond ~40–50° tilt, Mikolajczyk 2005; Morel & Yu 2009) | M (orientation, scale changes), H-view (affine) |
| Selection | Top-K by \|DoG response\| (contrast) | Contrast ≠ matchability. Most SIFT points are never matched (Hartmann et al. 2014 keep ≈30 % with little loss); points cluster on textured areas; no notion of repetitive structure or transient objects (Papadaki et al. 2020) | **M strongly** (fixed budget is wasted), H mildly |
| Description | 4×4×8 gradient histogram, L2-normalise, clip 0.2 | Local-only (no context) → ambiguous on repetitive patterns; L2/Euclidean is sub-optimal (RootSIFT, Arandjelović & Zisserman 2012); non-linear illumination (day–night) breaks gradients | H-illum., M |
| Matching | NN + ratio 0.8 or MNN | Ratio test drops correct matches on repetitive structures and keeps some wrong ones; MNN alone lets many outliers through (Bellavia & Colombo 2020; Jin et al. 2021) | M (RANSAC sees low inlier ratio) |

### SURF (Bay et al. 2006/2008)
| Stage | What it does | Known weaknesses | Hurts |
|---|---|---|---|
| Detection | Fast-Hessian (det of box-filtered Hessian), hessianThreshold = 100 | Box-filter approximation → coarser localisation and scale; blob-only; hand-set threshold | H (1 px AUC), M |
| Orientation | Haar-wavelet responses in a sliding 60° window | Less stable than SIFT under viewpoint/illumination; U-SURF (upright) variant exists exactly because it often helps | M, H-view |
| Selection | Top-K by Hessian response | Same as SIFT: response ≠ matchability | M |
| Description | 4×4×(Σdx,Σdy,Σ\|dx\|,Σ\|dy\|) = 64-D | Lower discriminability than SIFT-128 in wide-baseline evaluations (Tareen & Saleem 2018; Bian et al. 2019) | M, H |
| Matching | NN/ratio or MNN | As SIFT | M |

### ORB (Rublee et al. 2011)
| Stage | What it does | Known weaknesses | Hurts |
|---|---|---|---|
| Detection | FAST-9 (absolute intensity threshold 20) on an 8-level pyramid, scale factor 1.2 | Absolute threshold → very few points in dark/low-contrast images; only ~3.6× scale range; corners, not blobs | **H-illum., M (scale)** |
| Orientation | Intensity centroid | Noisy on low-texture patches | H-view, M |
| Selection | Per-level quota, then Harris score ranking | Strong clustering on textured areas (the reason ORB-SLAM adds grid/quadtree distribution, Mur-Artal et al. 2015); Harris score ≠ matchability | **M, H** |
| Description | rBRIEF 256-bit (pairs learned on PASCAL) | Binary + Hamming is coarse; low discriminability; not robust to large viewpoint | **M strongly**, H |
| Matching | Hamming NN / MNN | Many ties and outliers | M |

**What the benchmarks actually stress.** HPatches is planar, textured and mostly well-exposed, so it rewards precise localisation, repeatability under illumination and affine change, and descriptor robustness; keypoint *selection* matters less because most textured points are geometrically valid. MegaDepth-1500 is dominated by wide baselines, appearance change, repetitive facades and lots of unmatchable texture (trees, people, sky edges), so under a fixed budget the choice of *which* K keypoints to keep and the descriptor's ambiguity on repetitive structure dominate. RIPE's own Tab. 2 says the same: SIFT is close to learned methods on HPatches (53.9 vs 58.9 AUC@3px) but far behind on MegaDepth (37.3 vs 55.1 AUC@5°). Phase 4 below checks this directly.

## Phase 3 – Where RIPE plugs in

| # | Idea | Fixes | Why RL fits | Expected effect | Effort / training cost | Main risks |
|---|---|---|---|---|---|---|
| A | **RL keypoint-selection policy over the classical candidate pool** (RIPE reward: MNN + F-RANSAC inliers, signed by pair label; grid-cell sampling) | "response ≠ matchability", clustering, budget wasted on foliage/people/sky, repetitive structure (via negative pairs) | Selecting K of N is discrete and non-differentiable; reward needs only same/different-scene labels | M: more inliers at the same K → AUC@5/10/20 up; H: small gain (illumination-stable points) | Low. Tiny MLP on per-candidate cues; CPU-trainable in ~1 h once features are cached | A simple heuristic (grid/ANMS) might capture much of the gain; a supervised matchability classifier on the same pseudo-labels might match RL |
| B | **Descriptor refinement head** (residual MLP on RootSIFT/SURF) trained with RIPE's descriptor loss (Eq. 8) on RANSAC-inlier pseudo-matches, jointly with A | Descriptor ambiguity, illumination | The loss itself is supervised, but the pseudo-labels come from the RL loop; negatives push apart cross-scene matches | H and M: higher match precision; typically a few AUC points | Medium. Still CPU-feasible (MLP on 128-D) | Confirmation bias (only learns from already-easy matches); the descriptor changes, so the comparison is no longer "same descriptor": report A alone and A+B separately |
| C | **Policy with hyper-column context** (frozen pretrained CNN features at the keypoint as extra policy input) | Semantic unmatchability (sky, trees, people, cars) | Same as A | Larger M gain than A | Medium–high: needs pretrained weights (not downloadable from these sandboxes) and a CNN pass per image | The gain is partly credited to the CNN; needs a "CNN context only" ablation |
| D | Hyper-column descriptors at classical keypoints (RIPE descriptor + RIPE loss) | Descriptor | — | Large gain, but it is essentially RIPE's descriptor on SIFT points | High (GPU) | Not an "improved SIFT" any more – weak story for a fair comparison |
| E | Learned thresholds (contrast/edge/Hessian/FAST, ratio) as an RL bandit | Hand-set thresholds | Non-differentiable, black-box reward | Small; a grid search on the baseline gets most of it | Low | A tuned baseline erases the gain → "no improvement" |
| F | Learned orientation / upright switch | Orientation noise, unnecessary rotation invariance | Continuous action, poor RL fit | M: upright SIFT is a known +gain but is not RL; H: none | Medium | Confounds the comparison unless the baseline also gets upright |
| G | Negative-pair suppression of false-match-prone keypoints | Repetitive / generic structures | Exactly RIPE's negative-pair reward | Part of A (λ = −1 pairs) | — | Too strong a penalty kills keypoints on repetitive but valid facades |

**Bottom line of the mapping.** The cleanest transfer is A (optionally + B): it keeps the classical detector and descriptor, so "same K, same descriptor, same matcher, same RANSAC" holds exactly and any gain is attributable to the RL selection. C/D give bigger numbers but move the credit to a CNN. E/F are tuning tricks that a fair, tuned baseline would also use.
