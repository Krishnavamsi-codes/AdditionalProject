import cv2
import numpy as np
from metrics import Evaluator

def test_evaluator():
    evaluator = Evaluator(pixel_threshold=3.0)
    
    # Fake data
    kp1 = [cv2.KeyPoint(10, 10, 1), cv2.KeyPoint(20, 20, 1)]
    kp2 = [cv2.KeyPoint(10.5, 10.5, 1), cv2.KeyPoint(30, 30, 1)]
    H = np.eye(3)
    
    # Detector
    det_res = evaluator.evaluate_detector(kp1, kp2, H, (100, 100))
    print(det_res)
    assert det_res['repeatability'] == 0.5
    
    # Descriptor
    desc1 = np.random.rand(2, 32).astype(np.float32)
    desc2 = np.random.rand(2, 32).astype(np.float32)
    desc_res = evaluator.evaluate_descriptor(kp1, kp2, desc1, desc2, H)
    print("Descriptor Map:", desc_res['map'])
    
    # Matching
    matches = [cv2.DMatch(0, 0, 0.1), cv2.DMatch(1, 1, 0.2)]
    match_res = evaluator.evaluate_matching(kp1, kp2, matches, H)
    print(match_res)
    assert match_res['matching_score'] == 0.5
    
    # Geometry
    H_est = np.array([[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1.0]])
    geom_res = evaluator.evaluate_geometry(H, H_est, (100, 100))
    print(geom_res)
    
    print("Evaluator tests passed!")

if __name__ == "__main__":
    test_evaluator()
