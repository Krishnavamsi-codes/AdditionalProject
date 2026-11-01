# One-off: when SIFT RL seed 2 is finished, restart run_all.ps1 so the updated queue (BRISK first) takes effect.
Set-Location "D:\AdditionalProject\RIPE+others"
while (-not (Test-Path runs\train\sift_rl_s2\done.txt)) { Start-Sleep 30 }
Get-CimInstance Win32_Process | Where { ($_.Name -eq 'powershell.exe' -and $_.CommandLine -match 'run_all') -or
    ($_.Name -eq 'python.exe' -and $_.CommandLine -match 'train_policy|eval_baselines|spawn_main' -and $_.CommandLine -notmatch 'uv\\cache') } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Remove-Item -Recurse -Force runs\train\orb_rl_s1 -ErrorAction SilentlyContinue     # just-started run of the old queue
Start-Sleep 5
Start-Process powershell -WindowStyle Minimized -WorkingDirectory (Get-Location) -ArgumentList "-NoProfile -ExecutionPolicy Bypass -Command `"& .\runs\run_all.ps1 *> runs\logs\run_all.log`""
"restarted $(Get-Date -f HH:mm)"
