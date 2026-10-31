# Phase 4: vanilla baselines + diagnostics (SIFT, ORB). Run from the project root.
# HPatches: all 540 pairs, full threshold sweep. MegaDepth-1500: 500 evenly spaced pairs, sweep 1.0/1.5/2.0 px
# (PoseLib RANSAC at default 100k iterations dominates runtime; see HANDOFF.md).
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$py = ".\.venv\Scripts\python.exe"
$hp = "data\hpatches-sequences-release\hpatches-sequences-release"
$md = "data\megadepth1500\megadepth1500"
$w = 12
foreach ($m in "sift", "orb") {
    & $py -u proj\scripts\eval_baselines.py --dataset hpatches --root $hp --method $m --diag --workers $w --out results
}
foreach ($m in "sift", "orb") {
    & $py -u proj\scripts\eval_baselines.py --dataset megadepth --root $md --method $m --diag --limit 500 --ths 1.0 1.5 2.0 --workers $w --out results
}
"PHASE4 DONE"
