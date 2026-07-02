import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_compute_statistics(csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    # 1. Keypoints boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Method', y='Num Keypoints 1')
    plt.title('Distribution of Detected Keypoints (Image 1)')
    plt.savefig(os.path.join(output_dir, 'dist_keypoints.png'))
    plt.close()
    
    # 2. Matches boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Method', y='Num Matches')
    plt.title('Distribution of Raw Matches')
    plt.yscale('log')
    plt.savefig(os.path.join(output_dir, 'dist_matches.png'))
    plt.close()
    
    # 3. Inlier Ratio Violin Plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x='Method', y='Inlier Ratio')
    plt.title('Distribution of Inlier Ratios')
    plt.ylim(0, 1.1)
    plt.savefig(os.path.join(output_dir, 'dist_inliers.png'))
    plt.close()
    
    # Generate Stats Table
    stats_df = df.groupby('Method')[['Num Keypoints 1', 'Num Matches', 'Inlier Ratio', 'Total Time (ms)']].agg(['mean', 'median', 'min', 'max'])
    
    with open(os.path.join(output_dir, 'detector_statistics.md'), 'w') as f:
        f.write("# Phase 5: Detector Statistics\n\n")
        f.write("This document provides statistical distributions (mean, median, bounds) across all 580 sequence pairs.\n\n")
        f.write("## Summary Statistics\n")
        f.write(stats_df.to_markdown())
        f.write("\n\n## Visual Distributions\n")
        f.write("![Keypoints](stats/dist_keypoints.png)\n\n")
        f.write("![Matches](stats/dist_matches.png)\n\n")
        f.write("![Inliers](stats/dist_inliers.png)\n\n")

if __name__ == '__main__':
    run_compute_statistics('results/all_results.csv', 'results/stats')
