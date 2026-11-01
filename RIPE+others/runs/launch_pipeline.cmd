@echo off
rem Launched by the one-off Windows scheduled task "RIPE_pipeline" so the pipeline is not a child of the Claude app.
cd /d "D:\AdditionalProject\RIPE+others"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "& '.\runs\run_all.ps1' *> '.\runs\logs\run_all.log'"
