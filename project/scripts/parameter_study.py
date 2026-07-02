import pandas as pd
import os
import cv2
from dataset.hpatches import HPatchesDataset
from methods.classic import OpenCVMethod
from matchers import FeatureMatcher
from metrics import Evaluator

def evaluate_params(dataset, sequences, method_name, constructor_fn, param_grid):
    evaluator = Evaluator()
    results = []
    
    for param_name, param_values in param_grid.items():
        for val in param_values:
            kwargs = {param_name: val}
            
            # Temporary override method
            class TempMethod(OpenCVMethod):
                def __init__(self):
                    is_bin = method_name in ['ORB', 'BRISK', 'LATCH']
                    super().__init__(f"{method_name}_{param_name}={val}", constructor_fn(**kwargs), is_bin)
                    
            method = TempMethod()
            
            mma_list = []
            for seq in sequences:
                for idx in range(2, 7):
                    img1, imgN, H = dataset.load_pair(seq.name, idx, grayscale=True)
                    kp1, desc1 = method.compute(img1, method.detect(img1))
                    kp2, desc2 = method.compute(imgN, method.detect(imgN))
                    
                    matcher = FeatureMatcher('auto', True, 0.75 if not method.is_binary else 1.0)
                    matches = matcher.match(desc1, desc2)
                    
                    mma = evaluator.evaluate_mma(kp1, kp2, matches, H, img1.shape[:2], imgN.shape[:2])['mma_3px']
                    mma_list.append(mma)
                    
            mean_mma = sum(mma_list) / max(len(mma_list), 1)
            results.append({
                'Method': method_name,
                'Parameter': param_name,
                'Value': val,
                'MMA@3px': mean_mma
            })
            
    return results

def run_parameter_study(dataset_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    dataset = HPatchesDataset(dataset_path)
    sequences = dataset.get_all_sequences()[:5]  # Subset
    
    all_res = []
    all_res += evaluate_params(dataset, sequences, 'SIFT', cv2.SIFT_create, {'contrastThreshold': [0.02, 0.04, 0.08]})
    all_res += evaluate_params(dataset, sequences, 'ORB', cv2.ORB_create, {'nfeatures': [500, 2000, 5000]})
    all_res += evaluate_params(dataset, sequences, 'BRISK', cv2.BRISK_create, {'thresh': [15, 30, 60]})
    
    df = pd.DataFrame(all_res)
    
    with open(os.path.join(output_dir, 'parameter_study.md'), 'w') as f:
        f.write("# Phase 7: Parameter Study\n\n")
        f.write("Evaluated on a 5-sequence subset to determine hyperparameter sensitivity.\n\n")
        f.write(df.to_markdown(index=False))

if __name__ == '__main__':
    run_parameter_study('../hpatches-sequences-release', 'results/parameters')
