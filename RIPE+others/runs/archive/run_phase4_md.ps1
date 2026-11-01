Set-Location $PSScriptRoot
$py = ".\.venv\Scripts\python.exe"; $md = "data\megadepth1500\megadepth1500"
foreach ($m in "sift", "orb") { & $py -u proj\scripts\eval_baselines.py --dataset megadepth --root $md --method $m --diag --limit 500 --ths 1.0 1.5 2.0 --workers 12 --out results }
"PHASE4 MD DONE"
