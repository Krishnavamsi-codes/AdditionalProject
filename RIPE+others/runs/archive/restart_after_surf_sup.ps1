# One-off: wait for the running SURF supervised training to finish, then restart run_all.ps1 (updated) so the
# residual SURF stage runs before the evaluations. run_all resumes; finished stages are skipped.
Set-Location (Split-Path $PSScriptRoot -Parent)
while ($true) {
    $log = Get-Content runs\logs\train_surf_sup_s0.log -Raw -ErrorAction SilentlyContinue
    if ($log -match "done; best val") { break }
    if ($log -match "MemoryError|Traceback") { break }
    Start-Sleep 30
}
if (Test-Path runs\train\surf_sup_s0\best.npz) { "ok" | Set-Content runs\train\surf_sup_s0\done.txt }
Get-CimInstance Win32_Process | Where { ($_.Name -eq 'powershell.exe' -and $_.CommandLine -match 'run_all') -or
    ($_.Name -eq 'python.exe' -and $_.CommandLine -match 'train_policy|eval_baselines|spawn_main' -and $_.CommandLine -notmatch 'uv\\cache') } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Remove-Item runs\eval\* -ErrorAction SilentlyContinue      # partial eval output from the old process, if any
Start-Sleep 5
Start-Process powershell -WindowStyle Minimized -WorkingDirectory (Get-Location) -ArgumentList "-NoProfile -ExecutionPolicy Bypass -Command `"& .\runs\run_all.ps1 *> runs\logs\run_all.log`""
"restarted $(Get-Date -f HH:mm)"
