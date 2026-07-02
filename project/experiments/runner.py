import os
import time
import pandas as pd
from tqdm import tqdm
from dataset.hpatches import HPatchesDataset
from methods import SIFTMethod, SURFMethod, ORBMethod, BRISKMethod, LATCHMethod
from matchers import FeatureMatcher, estimate_homography
from metrics import Evaluator
from visualization import Plotter

class ExperimentRunner:
    def __init__(self, dataset_path: str, output_dir: str):
        self.dataset = HPatchesDataset(dataset_path)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.methods = [
            SIFTMethod(),
            SURFMethod(),
            ORBMethod(),
            BRISKMethod(),
            LATCHMethod()
        ]
        
        self.evaluator = Evaluator(pixel_threshold=3.0)
        self.plotter = Plotter(output_dir)
        
    def run(self):
        results = []
        sample_saved = {m.name: False for m in self.methods}
        
        for seq, idx, img1, imgN, H in tqdm(self.dataset.iterate_pairs(grayscale=True), total=len(self.dataset.get_all_sequences()) * 5, desc="Evaluating Pairs"):
            shape1 = img1.shape[:2]
            shape2 = imgN.shape[:2]
            
            for method in self.methods:
                if hasattr(method, 'available') and method.available is False:
                    continue
                    
                t0 = time.time()
                kp1 = method.detect(img1)
                t1 = time.time()
                kp2 = method.detect(imgN)
                t2 = time.time()
                det_time = ((t1 - t0) + (t2 - t1)) / 2 * 1000
                
                t3 = time.time()
                kp1, desc1 = method.compute(img1, kp1)
                t4 = time.time()
                kp2, desc2 = method.compute(imgN, kp2)
                t5 = time.time()
                desc_time = ((t4 - t3) + (t5 - t4)) / 2 * 1000
                
                matcher = FeatureMatcher(matcher_type='auto', cross_check=True, ratio_thresh=0.75 if not method.is_binary else 1.0)
                t6 = time.time()
                matches = matcher.match(desc1, desc2)
                t7 = time.time()
                match_time = (t7 - t6) * 1000
                
                det_res = self.evaluator.evaluate_detector(kp1, kp2, H, shape1, shape2)
                desc_res = self.evaluator.evaluate_descriptor(kp1, kp2, desc1, desc2, H, shape1, shape2)
                match_res = self.evaluator.evaluate_matching(kp1, kp2, matches, H, shape1, shape2)
                mma_res = self.evaluator.evaluate_mma(kp1, kp2, matches, H, shape1, shape2)
                
                H_est, mask = estimate_homography(kp1, kp2, matches)
                geom_res = self.evaluator.evaluate_geometry(H, H_est, shape1)
                
                if not sample_saved[method.name] and len(matches) > 10:
                    self.plotter.draw_matches(img1, kp1, imgN, kp2, matches, f"{method.name} Matches", f"{method.name}_sample.png")
                    sample_saved[method.name] = True
                
                res = {
                    'Sequence': seq.name,
                    'Type': 'Illumination' if seq.is_illumination else 'Viewpoint',
                    'Target_Idx': idx,
                    'Method': method.name,
                    'Detection Time (ms)': det_time,
                    'Descriptor Time (ms)': desc_time,
                    'Matching Time (ms)': match_time,
                    'Total Time (ms)': det_time*2 + desc_time*2 + match_time,
                    'Num Keypoints 1': det_res['num_kp1'],
                    'Num Keypoints 2': det_res['num_kp2'],
                    'Num Matches': len(matches),
                    'Repeatability': det_res['repeatability'],
                    'Precision': desc_res['precision'],
                    'Recall': desc_res['recall'],
                    'AP': desc_res['ap'],
                    'Matching Score': match_res['matching_score'],
                    'Inlier Ratio': match_res['inlier_ratio'],
                    'MMA@1px': mma_res['mma_1px'],
                    'MMA@2px': mma_res['mma_2px'],
                    'MMA@3px': mma_res['mma_3px'],
                    'MMA@5px': mma_res['mma_5px'],
                    'MMA@10px': mma_res['mma_10px'],
                    'Reprojection Error': geom_res['reprojection_error'],
                    'Homography Error': geom_res['homography_error']
                }
                results.append(res)
                
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(self.output_dir, 'all_results.csv'), index=False)
        self.generate_summary(df)

    def generate_summary(self, df: pd.DataFrame):
        summary = df.groupby('Method').mean(numeric_only=True).reset_index()
        summary.to_csv(os.path.join(self.output_dir, 'summary_results.csv'), index=False)
        
        # Save JSON
        summary.to_json(os.path.join(self.output_dir, 'summary_results.json'), orient='records', indent=4)
        
        with open(os.path.join(self.output_dir, 'summary_results.md'), 'w') as f:
            f.write(summary.to_markdown(index=False))
            
        self.plotter.plot_comparison_bar(summary, 'Repeatability', 'Repeatability Comparison', 'repeatability.png')
        self.plotter.plot_comparison_bar(summary, 'AP', 'Average Precision Comparison', 'ap.png')
        self.plotter.plot_comparison_bar(summary, 'Matching Score', 'Matching Score Comparison', 'matching_score.png')
        self.plotter.plot_comparison_bar(summary, 'Inlier Ratio', 'Inlier Ratio Comparison', 'inlier_ratio.png')
        
        # MMA Curve
        self.plotter.plot_mma_curves(summary, 'mma_curves.png')
        
        runtime_df = summary[['Method', 'Detection Time (ms)', 'Descriptor Time (ms)', 'Matching Time (ms)']].copy()
        self.plotter.plot_runtimes(runtime_df, 'runtimes.png')
