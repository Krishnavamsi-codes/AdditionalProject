@echo off
cd /d %~dp0
pip install opencv-contrib-python poselib pandas "numpy>=2"
start "SIFT" cmd /k python proj\scripts\eval_baselines.py --dataset hpatches --root data\hpatches-sequences-release\hpatches-sequences-release --method sift --diag ^&^& python proj\scripts\eval_baselines.py --dataset megadepth --root data\megadepth1500\megadepth1500 --method sift --limit 500
start "ORB" cmd /k python proj\scripts\eval_baselines.py --dataset hpatches --root data\hpatches-sequences-release\hpatches-sequences-release --method orb --diag ^&^& python proj\scripts\eval_baselines.py --dataset megadepth --root data\megadepth1500\megadepth1500 --method orb --limit 500
