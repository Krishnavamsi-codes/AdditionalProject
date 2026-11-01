@echo off
REM Second batch (run AFTER run_all.bat; it needs cache\, models\, scorer_results.csv, compress_results.csv, baseline_results.csv).
REM Everything is written to logs\ . Total: roughly 1 to 1.5 hours. Keep the laptop plugged in and awake.
if not exist logs mkdir logs

echo [1/3] extra seeds: settles whether RL really beats supervised (about 8 min per seed)
for %%s in (1 2 3) do (
  python -u train_scorer.py --cache cache --baseline baseline_results.csv --K 1000 --seed %%s --out scorer_results_s%%s.csv --models models_s%%s > logs\seed%%s.txt 2>&1
)
python -u merge_seeds.py --csvs scorer_results.csv scorer_results_s1.csv scorer_results_s2.csv scorer_results_s3.csv --out merged_seeds.csv > logs\merge.txt 2>&1
python -u analyze_results.py --csv merged_seeds.csv --out logs\7_analysis_4seeds.txt > logs\7_console.txt 2>&1

echo [2/3] significance of the compression results at 64 bytes and 32 bytes
REM in these two logs the column "vs_response_%%" means "vs rand64" / "vs rand32" (random dimension subset), and "supervised" means PCA
python -u analyze_results.py --csv compress_results.csv --rl rl64_uint8 --sup pca64_uint8 --resp rand64_uint8 --out logs\8_compress64_stats.txt > logs\8_console.txt 2>&1
python -u analyze_results.py --csv compress_results.csv --rl rl32_uint8 --sup pca32_uint8 --resp rand32_uint8 --out logs\9_compress32_stats.txt > logs\9_console.txt 2>&1

echo [3/3] feature ablation and budget curve (about 25 min)
python -u ablate_and_budget.py --cache cache --models models --baseline baseline_results.csv --K 1000 > logs\10_ablation_budget.txt 2>&1

echo finished > logs\DONE_more.txt
echo All done. Open the files in logs\
