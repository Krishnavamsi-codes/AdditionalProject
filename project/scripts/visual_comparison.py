import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from dataset.hpatches import HPatchesDataset
from methods import SIFTMethod, ORBMethod, BRISKMethod, LATCHMethod
from matchers import FeatureMatcher, estimate_homography

def draw_and_save(img1, kp1, imgN, kp2, matches, mask, title, out_path):
    all_matches = sorted(matches, key=lambda x: x.distance)
    img_res = cv2.drawMatches(img1, kp1, imgN, kp2, all_matches, None, 
                              matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
                              matchesMask=mask.ravel().tolist() if mask is not None else None, flags=0)
                              
    plt.figure(figsize=(15, 6))
    plt.imshow(img_res, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

def run_visual_comparison(dataset_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    dataset = HPatchesDataset(dataset_path)
    
    methods = [SIFTMethod(), ORBMethod(), BRISKMethod(), LATCHMethod()]
    
    # Select specific hard pairs
    # i_kobe idx 6 is extreme illumination
    # v_adam idx 6 is extreme viewpoint
    pairs = [('i_kobe', 6), ('v_adam', 6)]
    
    for seq_name, idx in pairs:
        try:
            img1, imgN, H = dataset.load_pair(seq_name, idx, grayscale=True)
        except Exception:
            continue # In case idx 6 doesn't exist, though HPatches has 1-6
            
        for method in methods:
            kp1 = method.detect(img1)
            kp2 = method.detect(imgN)
            kp1, desc1 = method.compute(img1, kp1)
            kp2, desc2 = method.compute(imgN, kp2)
            
            matcher = FeatureMatcher('auto', True, 0.75 if not method.is_binary else 1.0)
            matches = matcher.match(desc1, desc2)
            
            H_est, mask = estimate_homography(kp1, kp2, matches)
            if mask is None:
                mask = np.zeros((len(matches), 1), dtype=np.uint8)
                
            out_name = os.path.join(output_dir, f'compare_{seq_name}_{method.name}.png')
            draw_and_save(img1, kp1, imgN, kp2, matches, mask, f'{method.name} on {seq_name}', out_name)

if __name__ == '__main__':
    run_visual_comparison('../hpatches-sequences-release', 'results/comparisons')
