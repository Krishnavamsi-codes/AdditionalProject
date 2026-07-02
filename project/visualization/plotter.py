import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from typing import List, Dict, Tuple

class Plotter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")
        
    def draw_keypoints(self, img: np.ndarray, kp: List[cv2.KeyPoint], title: str, filename: str):
        img_kp = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
        plt.figure(figsize=(10, 8))
        if len(img_kp.shape) == 3:
            plt.imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img_kp, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight')
        plt.close()
        
    def draw_matches(self, img1: np.ndarray, kp1: List[cv2.KeyPoint], img2: np.ndarray, kp2: List[cv2.KeyPoint], matches: List[cv2.DMatch], title: str, filename: str, max_matches: int = 100):
        # Draw only top N matches to avoid clutter
        if len(matches) > max_matches:
            matches = sorted(matches, key=lambda x: x.distance)[:max_matches]
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        plt.figure(figsize=(15, 6))
        if len(img_matches.shape) == 3:
            plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img_matches, cmap='gray')
        plt.title(f"{title} (Top {len(matches)})")
        plt.axis('off')
        plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight')
        plt.close()
        
    def plot_pr_curve(self, results: Dict[str, Tuple[List[float], List[float]]], title: str, filename: str):
        """
        results: dict mapping method_name to (precisions, recalls)
        """
        plt.figure(figsize=(8, 6))
        for name, (precisions, recalls) in results.items():
            if len(recalls) > 0:
                plt.plot(recalls, precisions, label=name)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc='lower left')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight')
        plt.close()
        
    def plot_comparison_bar(self, df: pd.DataFrame, metric: str, title: str, filename: str):
        """
        df: DataFrame with columns ['Method', metric]
        """
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x='Method', y=metric, hue='Method', legend=False)
        plt.title(title)
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight')
        plt.close()

    def plot_mma_curves(self, df: pd.DataFrame, filename: str):
        """
        df: DataFrame containing 'Method' and 'MMA@Xpx' columns
        """
        plt.figure(figsize=(10, 6))
        thresholds = [1, 2, 3, 5, 10]
        cols = [f'MMA@{t}px' for t in thresholds]
        
        for idx, row in df.iterrows():
            method = row['Method']
            vals = [row[c] for c in cols if c in row]
            if len(vals) == len(thresholds):
                plt.plot(thresholds, vals, marker='o', label=method)
                
        plt.xlabel('Error Threshold (pixels)')
        plt.ylabel('Mean Matching Accuracy (MMA)')
        plt.title('Mean Matching Accuracy (MMA) Curve')
        plt.legend()
        plt.grid(True)
        plt.ylim([0.0, 1.05])
        plt.xticks(thresholds)
        plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight')
        plt.close()

    def plot_runtimes(self, df: pd.DataFrame, filename: str):
        """
        df: DataFrame with columns ['Method', 'Detection Time', 'Descriptor Time', 'Matching Time']
        """
        if df.empty: return
        df_melt = pd.melt(df, id_vars=['Method'], value_vars=['Detection Time (ms)', 'Descriptor Time (ms)', 'Matching Time (ms)'], var_name='Stage', value_name='Time (ms)')
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_melt, x='Method', y='Time (ms)', hue='Stage')
        plt.title('Runtime Comparison')
        plt.yscale('log')
        plt.ylabel('Time (ms) - Log Scale')
        plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight')
        plt.close()
