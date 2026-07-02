import time
import os
import pandas as pd
from tqdm import tqdm
from dataset.hpatches import HPatchesDataset
from methods import SIFTMethod, ORBMethod, BRISKMethod, LATCHMethod
from matchers import FeatureMatcher, estimate_homography
from metrics import Evaluator

def run_profiler(dataset_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    dataset = HPatchesDataset(dataset_path)
    evaluator = Evaluator(pixel_threshold=3.0)
    
    methods = [ORBMethod(), SIFTMethod(), BRISKMethod(), LATCHMethod()]
    
    # Take just 5 sequences for profiling
    sequences = dataset.get_all_sequences()[:5]
    
    results = []
    
    for seq in tqdm(sequences, desc="Profiling Sequences"):
        for idx in range(2, 7):
            for method in methods:
                t0 = time.time()
                # 1. Image Loading (actually we load it here to measure)
                img1, imgN, H = dataset.load_pair(seq.name, idx, grayscale=True)
                shape1, shape2 = img1.shape[:2], imgN.shape[:2]
                t1 = time.time()
                
                # 2. Detection
                kp1 = method.detect(img1)
                kp2 = method.detect(imgN)
                t2 = time.time()
                
                # 3. Description
                kp1, desc1 = method.compute(img1, kp1)
                kp2, desc2 = method.compute(imgN, kp2)
                t3 = time.time()
                
                # 4. Matching
                matcher = FeatureMatcher(matcher_type='auto', cross_check=True, ratio_thresh=0.75 if not method.is_binary else 1.0)
                matches = matcher.match(desc1, desc2)
                t4 = time.time()
                
                # 5. Homography Estimation
                H_est, mask = estimate_homography(kp1, kp2, matches)
                t5 = time.time()
                
                # 6. Metric Computation
                _ = evaluator.evaluate_detector(kp1, kp2, H, shape1, shape2)
                _ = evaluator.evaluate_descriptor(kp1, kp2, desc1, desc2, H, shape1, shape2)
                _ = evaluator.evaluate_matching(kp1, kp2, matches, H, shape1, shape2)
                _ = evaluator.evaluate_mma(kp1, kp2, matches, H, shape1, shape2)
                _ = evaluator.evaluate_geometry(H, H_est, shape1)
                t6 = time.time()
                
                results.append({
                    'Method': method.name,
                    'Image Loading (ms)': (t1 - t0) * 1000,
                    'Detection (ms)': (t2 - t1) * 1000,
                    'Description (ms)': (t3 - t2) * 1000,
                    'Matching (ms)': (t4 - t3) * 1000,
                    'Homography Est (ms)': (t5 - t4) * 1000,
                    'Metric Comp (ms)': (t6 - t5) * 1000
                })

    df = pd.DataFrame(results)
    summary = df.groupby('Method').mean().reset_index()
    
    with open(os.path.join(output_dir, 'runtime_profile.md'), 'w') as f:
        f.write("# Phase 6: Runtime Profiling\n\n")
        f.write("This document breaks down the computational cost of the benchmark pipeline on a subset of 5 sequences.\n\n")
        f.write(summary.to_markdown(index=False))

if __name__ == '__main__':
    run_profiler('../hpatches-sequences-release', 'results/profiling')
