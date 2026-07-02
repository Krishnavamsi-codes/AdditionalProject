import cv2
import numpy as np
import pandas as pd
import os
from visualization import Plotter

def test_plotter():
    out_dir = r"d:\AdditionalProject\project\results"
    plotter = Plotter(out_dir)
    
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    kp = [cv2.KeyPoint(10, 10, 1)]
    plotter.draw_keypoints(img, kp, "Test Keypoints", "test_kp.png")
    
    img2 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    matches = [cv2.DMatch(0, 0, 1.0)]
    plotter.draw_matches(img, kp, img2, kp, matches, "Test Matches", "test_matches.png")
    
    results = {'Method1': ([1.0, 0.8, 0.6], [0.1, 0.5, 0.9])}
    plotter.plot_pr_curve(results, "Test PR Curve", "test_pr.png")
    
    df = pd.DataFrame({'Method': ['Method1', 'Method2'], 'Metric': [0.5, 0.8]})
    plotter.plot_comparison_bar(df, 'Metric', "Test Bar", "test_bar.png")
    
    df2 = pd.DataFrame({
        'Method': ['Method1', 'Method2'],
        'Detection Time': [10, 20],
        'Descriptor Time': [15, 25],
        'Matching Time': [5, 10]
    })
    plotter.plot_runtimes(df2, "test_runtimes.png")
    
    # Check if files exist
    assert os.path.exists(os.path.join(out_dir, "test_kp.png"))
    assert os.path.exists(os.path.join(out_dir, "test_matches.png"))
    assert os.path.exists(os.path.join(out_dir, "test_pr.png"))
    assert os.path.exists(os.path.join(out_dir, "test_bar.png"))
    assert os.path.exists(os.path.join(out_dir, "test_runtimes.png"))
    
    print("Plotter tests passed!")

if __name__ == "__main__":
    test_plotter()
