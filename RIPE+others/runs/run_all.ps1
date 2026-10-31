# Full pipeline: train RIPE-Select (RL) + supervised ablation for SIFT / ORB / SURF, then evaluate everything.
# Resumable: every stage is skipped when its output exists. One heavy job at a time (16 GB RAM).
# Run from anywhere:  powershell -ExecutionPolicy Bypass -File runs\run_all.ps1
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"
$hp = "data\hpatches-sequences-release\hpatches-sequences-release"
$md = "data\megadepth1500\megadepth1500"
$methods = "sift", "orb", "surf", "brisk"
$seedMethods = "sift", "orb", "brisk"     # SURF seeds dropped (negative on seed 0; revisit later)
$W = 8
New-Item -ItemType Directory -Force runs\logs, runs\eval | Out-Null

function Train($m, $mode, $seed) {
    $out = "runs\train\${m}_${mode}_s$seed"
    if (Test-Path "$out\done.txt") { return }
    "[$(Get-Date -f HH:mm)] train $m $mode s$seed"
    $extra = if ($mode -eq "rlres") { @("--mode", "rl", "--residual") } else { @("--mode", $mode) }
    & $py -u proj\scripts\train_policy.py --method $m @extra --seed $seed --steps 1200 --workers $W --out $out *> "runs\logs\train_${m}_${mode}_s$seed.log"
    if (Test-Path "$out\best.npz") { "ok" | Set-Content "$out\done.txt" }
}

function Eval($m, $ds, $tag, $extra) {
    if (Test-Path "runs\eval\${ds}_${m}_*$tag.json") { return }
    "[$(Get-Date -f HH:mm)] eval $m $ds $tag"
    $root = if ($ds -eq "hpatches") { $hp } else { $md }
    $a = @("-u", "proj\scripts\eval_baselines.py", "--dataset", $ds, "--root", $root, "--method", $m,
           "--workers", $W, "--out", "runs\eval") + $extra
    if ($tag) { $a += @("--tag", $tag) }      # PS 5.1 drops empty-string args to native commands
    & $py @a *> "runs\logs\eval_${ds}_${m}$tag.log"
}

# 1) policies (seed 0): RL + supervised ablation on the same pseudo-labels
foreach ($m in $methods) { Train $m "rl" 0; Train $m "sup" 0 }
Train "surf" "rlres" 0     # SURF: plain policy never beat response ranking on val -> residual on the detector ranking

# 2) HPatches: all baselines + diagnostics + learned selection, full RANSAC sweep
foreach ($m in $methods) {
    $res = if ($m -eq "surf") { @("rl_res=runs\train\surf_rlres_s0\best.npz") } else { @() }
    Eval $m "hpatches" "" (@("--diag", "--policy", "rl=runs\train\${m}_rl_s0\best.npz", "sup=runs\train\${m}_sup_s0\best.npz") + $res)
}

# 3) MegaDepth-1500 (500 evenly spaced pairs, both scenes), RANSAC 1.0/1.5 px
foreach ($m in $methods) {
    $res = if ($m -eq "surf") { @("rl_res=runs\train\surf_rlres_s0\best.npz") } else { @() }
    $resv = if ($m -eq "surf") { @("rl_res", "rl_res_grid") } else { @() }
    Eval $m "megadepth" "" (@("--diag", "--limit", "500", "--ths", "1.0", "1.5",
        "--policy", "rl=runs\train\${m}_rl_s0\best.npz", "sup=runs\train\${m}_sup_s0\best.npz") + $res +
        @("--variants", "vanilla", "pool_resp", "pool_grid", "rl", "rl_grid", "sup", "sup_grid") + $resv)
}

# 4) seed variance for the RL policy (seeds 1, 2)
foreach ($m in $seedMethods) { Train $m "rl" 1; Train $m "rl" 2 }
foreach ($m in $seedMethods) {
    $pol = @("--policy", "rl_s1=runs\train\${m}_rl_s1\best.npz", "rl_s2=runs\train\${m}_rl_s2\best.npz")
    Eval $m "hpatches" "_seeds" ($pol + @("--variants", "vanilla", "rl_s1", "rl_s1_grid", "rl_s2", "rl_s2_grid"))
    Eval $m "megadepth" "_seeds" ($pol + @("--limit", "500", "--ths", "1.0", "1.5",
        "--variants", "vanilla", "rl_s1", "rl_s1_grid", "rl_s2", "rl_s2_grid"))
}
"[$(Get-Date -f HH:mm)] ALL DONE"
