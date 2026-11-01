@echo off
REM Runs the follow-up experiments unattended (your main 5-fold run is already done: scorer_results.csv + models\).
REM Every step writes a log into logs\ . Keep this file next to the .py files. Total: roughly 1 to 1.5 hours.
REM Keep the laptop plugged in and awake. If a step fails, the later ones still run; check logs\ for the error text.
if not exist logs mkdir logs

echo [1/4] statistics on the main experiment (seconds)
python -u analyze_results.py --csv scorer_results.csv --out logs\1_analysis_main.txt > logs\1_analysis_main_console.txt 2>&1

echo [2/4] descriptor compression with RL dimension selection (about 20 to 30 min)
python -u compress_eval.py --cache cache --models models --baseline baseline_results.csv --K 1000 > logs\2_compress.txt 2>&1

echo [3/4] hyperparameter grid with nested validation (about 40 to 60 min)
python -u train_scorer_grid.py --cache cache --baseline baseline_results.csv --K 1000 > logs\3_grid.txt 2>&1

echo [4/4] statistics on the tuned models (seconds)
python -u analyze_results.py --csv grid_results.csv --rl rl_tuned --sup supervised_tuned --out logs\4_analysis_grid.txt > logs\4_analysis_grid_console.txt 2>&1

echo finished > logs\DONE.txt
echo All done. Open the files in logs\
