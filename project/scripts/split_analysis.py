import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_split_analysis(csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    # Split
    ill = df[df['Type'] == 'Illumination']
    view = df[df['Type'] == 'Viewpoint']
    
    # Aggregate
    ill_summary = ill.groupby('Method').mean(numeric_only=True).reset_index()
    view_summary = view.groupby('Method').mean(numeric_only=True).reset_index()
    
    ill_summary.to_csv(os.path.join(output_dir, 'illumination_results.csv'), index=False)
    view_summary.to_csv(os.path.join(output_dir, 'viewpoint_results.csv'), index=False)
    
    metrics = ['Repeatability', 'AP', 'Matching Score', 'Inlier Ratio', 'MMA@3px']
    
    # Plotting comparison
    for metric in metrics:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
        
        sns.barplot(data=ill_summary, x='Method', y=metric, ax=axes[0], hue='Method', legend=False)
        axes[0].set_title(f'Illumination: {metric}')
        axes[0].set_ylim(0, 1)
        
        sns.barplot(data=view_summary, x='Method', y=metric, ax=axes[1], hue='Method', legend=False)
        axes[1].set_title(f'Viewpoint: {metric}')
        axes[1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'split_{metric.replace("@", "_")}.png'))
        plt.close()
        
    with open(os.path.join(output_dir, 'split_analysis.md'), 'w') as f:
        f.write("# Phase 3: Split Analysis (Illumination vs Viewpoint)\n\n")
        f.write("## Illumination Sequences\n")
        f.write(ill_summary[['Method'] + metrics].to_markdown(index=False))
        f.write("\n\n## Viewpoint Sequences\n")
        f.write(view_summary[['Method'] + metrics].to_markdown(index=False))
        f.write("\n\n## Discussion\n")
        f.write("- **Illumination**: Focuses on photometric robustness. Methods relying on pure pixel intensities or simple gradients without robust normalization often degrade.\n")
        f.write("- **Viewpoint**: Focuses on geometric robustness (scale, rotation, perspective). Methods without true affine/scale invariance (e.g. basic FAST/BRIEF) degrade significantly here.\n")

if __name__ == '__main__':
    run_split_analysis('results/all_results.csv', 'results/split_analysis')
