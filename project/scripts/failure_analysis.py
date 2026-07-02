import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from dataset.hpatches import HPatchesDataset
from methods import SIFTMethod, ORBMethod, BRISKMethod, LATCHMethod
from matchers import FeatureMatcher, estimate_homography
from visualization.plotter import Plotter

def plot_failure_case(img1, kp1, img2, kp2, matches, inlier_mask, title, output_path):
    # Convert mask to list of matches
    all_matches = sorted(matches, key=lambda x: x.distance)
    
    # We will draw ALL matches in red, and INLIERS in green
    # OpenCV's drawMatches can take a list of lists of matches or a matchesMask
    
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, 
                                  matchColor=(0, 255, 0), singlePointColor=None,
                                  matchesMask=inlier_mask.ravel().tolist(), flags=2)
                                  
    plt.figure(figsize=(15, 6))
    plt.imshow(img_matches, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def run_failure_analysis(csv_path, dataset_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    dataset = HPatchesDataset(dataset_path)
    
    methods_dict = {
        'SIFT': SIFTMethod(),
        'ORB': ORBMethod(),
        'BRISK': BRISKMethod(),
        'LATCH': LATCHMethod()
    }
    
    analysis_md = ["# Phase 4: Failure Analysis\n\n"]
    
    for method_name, method_obj in methods_dict.items():
        m_df = df[df['Method'] == method_name].copy()
        if m_df.empty: continue
            
        m_df.sort_values(by='Inlier Ratio', ascending=False, inplace=True)
        
        best = m_df.iloc[0]
        worst = m_df.iloc[-1]
        
        analysis_md.append(f"## {method_name}")
        
        for case_name, row in [('Best', best), ('Worst', worst)]:
            seq_name = row['Sequence']
            idx = row['Target_Idx']
            inlier_ratio = row['Inlier Ratio']
            
            # Load images
            img1, imgN, H_gt = dataset.load_pair(seq_name, idx, grayscale=True)
            
            # Compute
            kp1 = method_obj.detect(img1)
            kp2 = method_obj.detect(imgN)
            kp1, desc1 = method_obj.compute(img1, kp1)
            kp2, desc2 = method_obj.compute(imgN, kp2)
            
            matcher = FeatureMatcher(matcher_type='auto', cross_check=True, ratio_thresh=0.75 if not method_obj.is_binary else 1.0)
            matches = matcher.match(desc1, desc2)
            
            H_est, mask = estimate_homography(kp1, kp2, matches)
            
            if mask is None:
                mask = np.zeros((len(matches), 1), dtype=np.uint8)
                
            fname = f"{method_name}_{case_name}_{seq_name}_{idx}.png"
            title = f"{method_name} {case_name} Case: {seq_name} (Inlier Ratio: {inlier_ratio:.2f})"
            
            plot_failure_case(img1, kp1, imgN, kp2, matches, mask, title, os.path.join(output_dir, fname))
            
            analysis_md.append(f"### {case_name} Case: `{seq_name}` (Idx {idx})")
            analysis_md.append(f"- **Inlier Ratio**: {inlier_ratio:.2f}")
            analysis_md.append(f"- **Detected**: {len(kp1)} / {len(kp2)}")
            analysis_md.append(f"- **Matches**: {len(matches)}")
            analysis_md.append(f"- **Inliers (RANSAC)**: {np.sum(mask)}")
            analysis_md.append(f"![{case_name}](failures/{fname})")
            analysis_md.append("\n")

    with open(os.path.join(output_dir, 'failure_analysis.md'), 'w') as f:
        f.write("\n".join(analysis_md))

if __name__ == '__main__':
    run_failure_analysis('results/all_results.csv', '../hpatches-sequences-release', 'results/failures')
