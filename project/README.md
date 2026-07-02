# HPatches Benchmarking Framework for Classical Local Features

This repository contains a research-grade, Python-based benchmarking framework for evaluating classical local feature detectors and descriptors (SIFT, ORB, BRISK, LATCH) on the rigorous HPatches dataset. 

## Project Structure
```text
project/
├── dataset/
│   ├── __init__.py
│   └── hpatches.py            # HPatches data loader and homography parsing
├── experiments/
│   ├── __init__.py
│   └── runner.py              # Orchestrates the benchmark pipeline
├── matchers/
│   ├── __init__.py
│   └── matchers.py            # Nearest-neighbor matching and RANSAC homography estimation
├── methods/
│   ├── __init__.py
│   └── classic.py             # OOP wrappers for OpenCV local feature algorithms
├── metrics/
│   ├── __init__.py
│   └── evaluator.py           # Geometric evaluation metrics (Repeatability, MMA, AP)
├── results/                   # Generated artifacts (CSVs, metrics, plots)
│   ├── comparisons/           # Side-by-side detector visualizations
│   ├── failures/              # RANSAC inlier analysis on extreme pairs
│   ├── profiling/             # Runtime profiles
│   └── stats/                 # Histograms and boxplots of keypoint distributions
├── scripts/                   # Analytical scripts generating markdown reports and plots
├── visualization/
│   ├── __init__.py
│   └── plotter.py             # Matplotlib/Seaborn visualization utilities
└── main.py                    # Entry point for the benchmark
```

## Environment & Installation
The framework requires a standard scientific Python stack alongside `opencv-contrib-python` for access to patented/non-free algorithms. 

```bash
# Create and activate environment
conda create -n apv1 python=3.11
conda activate apv1

# Install requirements
pip install numpy scipy pandas matplotlib seaborn scikit-image scikit-learn tqdm pyyaml jupyterlab ipykernel opencv-contrib-python pillow imageio plotly rich tabulate psutil pytest
```

## Reproducibility Instructions

### 1. Running the Benchmark
To execute the full benchmark across all 580 sequence pairs in HPatches:
```bash
conda run -n apv1 python main.py
```
*Note: This process performs exhaustive geometric filtering using KD-Trees and takes approximately 45 minutes on a modern CPU. The raw data is saved to `results/all_results.csv`.*

### 2. Regenerating Figures and Tables
All figures, histograms, runtime graphs, and visual comparisons are strictly generated from the CSV data. To regenerate them, run the analytical scripts:
```bash
$env:PYTHONPATH="."
python scripts/split_analysis.py       # Illumination vs Viewpoint degradation
python scripts/failure_analysis.py     # Extreme geometric failure cases
python scripts/compute_statistics.py   # Keypoint and inlier distributions
python scripts/visual_comparison.py    # Hard-coded side-by-side evaluations
python scripts/runtime_profile.py      # Pipeline bottleneck analysis
python scripts/parameter_study.py      # OpenCV hyperparameter grid search
python scripts/cross_validation.py     # Wrapper integrity verification
```

## Known Limitations
1. **SURF Exclusion**: The Speeded-Up Robust Features (SURF) algorithm is currently mocked as "Unavailable" because the pre-built `opencv-contrib-python` binaries are compiled with `OPENCV_ENABLE_NONFREE=OFF` due to patent restrictions. 
2. **Homography Error Metric**: The matrix norm $L_2$ error ($||H_{gt} - H_{est}||_2$) is recorded but mathematically sensitive to scaling factors in extreme perspective warps. Reprojection error is recommended for geometric evaluation.
3. **Repeatability Upper Bound**: Due to KD-Tree non-mutual-exclusive matching, raw repeatability values may exhibit bounds inflation if keypoints are densely clustered. Ensure symmetric filtering is active for strict protocol compliance.
