import pandas as pd
import numpy as np
import sys
import os

def validate_results(csv_path, output_md):
    df = pd.read_csv(csv_path)
    
    issues = []
    
    # 1. Check expected row count (116 sequences * 5 target images * 4 methods = 2320 rows)
    expected_rows = 116 * 5 * 4
    if len(df) != expected_rows:
        issues.append(f"- **Row Count Error**: Expected {expected_rows} rows, found {len(df)}. Some experiments may have crashed or skipped.")
    
    # 2. Check for NaN values
    nans = df.isna().sum()
    nans = nans[nans > 0]
    if len(nans) > 0:
        issues.append("- **NaN Values Detected**:")
        for col, count in nans.items():
            issues.append(f"  - Column `{col}` has {count} NaN values.")
            
    # 3. Check for Infinity values
    # numeric_cols = df.select_dtypes(include=[np.number]).columns
    infs = df.isin([np.inf, -np.inf]).sum()
    infs = infs[infs > 0]
    if len(infs) > 0:
        issues.append("- **Infinity Values Detected**:")
        for col, count in infs.items():
            issues.append(f"  - Column `{col}` has {count} Infinity values. (Likely caused by divide-by-zero or RANSAC failure returning None).")
            
    # 4. Check impossible bounds
    # Metrics like Repeatability, AP, Precision, Recall, MMA, Inlier Ratio must be in [0, 1]
    bounds_cols = ['Repeatability', 'Precision', 'Recall', 'AP', 'Matching Score', 'Inlier Ratio', 'MMA@1px', 'MMA@2px', 'MMA@3px', 'MMA@5px', 'MMA@10px']
    for col in bounds_cols:
        if col in df.columns:
            invalid = df[(df[col] < 0.0) | (df[col] > 1.0)]
            if len(invalid) > 0:
                issues.append(f"- **Impossible Bounds**: Column `{col}` has {len(invalid)} values outside [0, 1].")
                
    # 5. Duplicated rows
    duplicates = df.duplicated(subset=['Sequence', 'Target_Idx', 'Method']).sum()
    if duplicates > 0:
        issues.append(f"- **Duplicated Runs**: Found {duplicates} duplicate experiment configurations.")

    with open(output_md, 'w') as f:
        f.write("# Phase 2: Result Validation\n\n")
        f.write("This document tracks data anomalies, NaNs, infinities, and bounds violations in `results/all_results.csv`.\n\n")
        
        if len(issues) == 0:
            f.write("## Status: PASS\n\nNo anomalies detected. Data is mathematically sound and bounded correctly.\n")
        else:
            f.write("## Status: ANOMALIES DETECTED\n\n")
            for issue in issues:
                f.write(f"{issue}\n")
            
            f.write("\n## Explanation of Anomalies\n")
            if "Homography Error" in infs:
                f.write("- **Homography Error Infinities**: These occur when OpenCV's `findHomography` fails to find a valid transformation matrix due to an insufficient number of inliers. This typically happens for SIFT on the most extreme sequence pairs where aggressive filtering yields zero geometric matches.\n")
            
    print(f"Validation complete. Wrote {output_md}")

if __name__ == '__main__':
    validate_results('results/all_results.csv', 'result_validation.md')
