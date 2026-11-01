# Full pipeline: train RIPE-Select (RL) + supervised ablation for SIFT / ORB / SURF, then evaluate everything.
# Resumable: every stage is skipped when its output exists. One heavy job at a time (16 GB RAM).
# Run from anywhere:  powershell -ExecutionPolicy Bypass -File runs\run_all.ps1
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\apenv\Scripts\python.exe"      # GPU env (CUDA torch 2.14.1); .venv is the original CPU-only env
$hp = "data\hpatches-sequences-release\hpatches-sequences-release"
$md = "data\megadepth1500\megadepth1500"
$methods = "sift", "orb", "surf", "brisk"
$seedMethods = "sift", "orb", "brisk"     # SURF seeds dropped (negative on seed 0; revisit later)
$W = 8                                    # 8 froze the laptop with CPU matching; with GPU matching the CPU sits at ~65% on 6
(Get-Process -Id $PID).PriorityClass = "BelowNormal"   # inherited by python children: keeps the desktop responsive
$env:OMP_NUM_THREADS = "2"; $env:MKL_NUM_THREADS = "2"; $env:OPENBLAS_NUM_THREADS = "2"
$env:PYTHONIOENCODING = "utf-8"           # make_report prints unicode; the cp1252 console encoder crashed on it
New-Item -ItemType Directory -Force runs\logs, runs\eval | Out-Null

function Train($m, $mode, $seed) {
    $out = "runs\train\${m}_${mode}_s$seed"
    if (Test-Path "$out\done.txt") { return }
    "[$(Get-Date -f HH:mm)] train $m $mode s$seed"
    $extra = if ($mode -eq "rlres") { @("--mode", "rl", "--residual") } else { @("--mode", $mode) }
    $wt = if ($m -eq "brisk") { 6 } else { $W }     # BRISK's 512-bit pool cache ran out of memory at 8 workers
    & $py -u proj\scripts\train_policy.py --method $m @extra --seed $seed --steps 1200 --workers $wt --out $out *> "runs\logs\train_${m}_${mode}_s$seed.log"
    # a crash leaves a partial best.npz behind: only a clean exit counts as done
    if ($LASTEXITCODE -eq 0 -and (Test-Path "$out\best.npz")) { "ok" | Set-Content "$out\done.txt" }
}

function Eval($m, $ds, $tag, $extra) {
    if (Test-Path "runs\eval\${ds}_${m}_*$tag.json") { return }
    "[$(Get-Date -f HH:mm)] eval $m $ds $tag"
    $root = if ($ds -eq "hpatches") { $hp } else { $md }
    $a = @("-u", "proj\scripts\eval_baselines.py", "--dataset", $ds, "--root", $root, "--method", $m,
           "--workers", $W, "--out", "runs\eval") + $extra
    if ($tag) { $a += @("--tag", $tag) }      # PS 5.1 drops empty-string args to native commands
    & $py @a *> "runs\logs\eval_${ds}_${m}$tag.log"
    & $py proj\scripts\make_report.py *> runs\logs\report.log      # keep runs/report/*.md up to date
}

# 1) policies (seed 0): RL + supervised ablation on the same pseudo-labels
foreach ($m in $methods) { Train $m "rl" 0; Train $m "sup" 0 }
Train "surf" "rlres" 0     # SURF: plain policy never beat response ranking on val -> residual on the detector ranking

# 2-4) per group of methods: evals, then extra seeds, then seed evals - so one group is fully finished
# before the next starts (SIFT + ORB first, BRISK after)
function Stage($ms) {
    # 2) HPatches: all baselines + diagnostics + learned selection, full RANSAC sweep
    foreach ($m in $ms) {
        $res = if ($m -eq "surf") { @("rl_res=runs\train\surf_rlres_s0\best.npz") } else { @() }
        Eval $m "hpatches" "" (@("--diag", "--policy", "rl=runs\train\${m}_rl_s0\best.npz", "sup=runs\train\${m}_sup_s0\best.npz") + $res)
    }
    # 3) MegaDepth-1500 (500 evenly spaced pairs, both scenes), RANSAC 1.0/1.5 px
    foreach ($m in $ms) {
        $res = if ($m -eq "surf") { @("rl_res=runs\train\surf_rlres_s0\best.npz") } else { @() }
        $resv = if ($m -eq "surf") { @("rl_res", "rl_res_grid") } else { @() }
        Eval $m "megadepth" "" (@("--diag", "--limit", "500", "--ths", "1.0", "1.5",
            "--policy", "rl=runs\train\${m}_rl_s0\best.npz", "sup=runs\train\${m}_sup_s0\best.npz") + $res +
            @("--variants", "vanilla", "pool_resp", "pool_grid", "rl", "rl_grid", "sup", "sup_grid") + $resv)
    }
    # 4) seed variance for the RL policy (seeds 1, 2)
    $sm = @($ms | Where-Object { $seedMethods -contains $_ })
    foreach ($m in $sm) { Train $m "rl" 1; Train $m "rl" 2 }
    foreach ($m in $sm) {
        # never evaluate a partial policy left behind by a crashed run
        if (-not ((Test-Path "runs\train\${m}_rl_s1\done.txt") -and (Test-Path "runs\train\${m}_rl_s2\done.txt"))) {
            "[$(Get-Date -f HH:mm)] skip $m seed evals: a seed run did not finish"; continue
        }
        $pol =@("--policy", "rl_s1=runs\train\${m}_rl_s1\best.npz", "rl_s2=runs\train\${m}_rl_s2\best.npz")
        Eval $m "hpatches" "_seeds" ($pol + @("--variants", "vanilla", "rl_s1", "rl_s1_grid", "rl_s2", "rl_s2_grid"))
        Eval $m "megadepth" "_seeds" ($pol + @("--limit", "500", "--ths", "1.0", "1.5",
            "--variants", "vanilla", "rl_s1", "rl_s1_grid", "rl_s2", "rl_s2_grid"))
    }
}
Stage @("sift", "orb", "surf")      # SURF evals already exist -> skipped
"[$(Get-Date -f HH:mm)] SIFT + ORB DONE"
Stage @("brisk")
& $py proj\scripts\make_report.py *> runs\logs\report.log
"[$(Get-Date -f HH:mm)] ALL DONE"
