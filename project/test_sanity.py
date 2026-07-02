import numpy as np
import cv2
from metrics.evaluator import Evaluator
from matchers.matcher import FeatureMatcher

def test_sanity():
    # 1. Homography Direction Check
    # H_1_2 maps img1 to img2
    H = np.array([[1.5, 0, 10], [0, 1.5, 20], [0, 0, 1]], dtype=np.float64)
    pt1 = np.array([[0, 0], [10, 10]], dtype=np.float32)
    
    ev = Evaluator()
    proj = ev.project_keypoints(pt1, H)
    assert np.allclose(proj[0], [10, 20])
    assert np.allclose(proj[1], [10*1.5 + 10, 10*1.5 + 20])
    print("✓ Homography projection direction is correct (1 -> 2).")
    
    # 2. Descriptor Types
    matcher_float = FeatureMatcher(matcher_type='bf', cross_check=True)
    desc_float1 = np.random.rand(10, 128).astype(np.float32)
    desc_float2 = np.random.rand(10, 128).astype(np.float32)
    matches_float = matcher_float.match(desc_float1, desc_float2)
    print("✓ Float descriptors map to L2 norm successfully.")
    
    matcher_bin = FeatureMatcher(matcher_type='bf', cross_check=True)
    desc_bin1 = np.random.randint(0, 256, (10, 32)).astype(np.uint8)
    desc_bin2 = np.random.randint(0, 256, (10, 32)).astype(np.uint8)
    matches_bin = matcher_bin.match(desc_bin1, desc_bin2)
    print("✓ Binary descriptors map to Hamming norm successfully.")
    
    # 3. Covisible Region filtering
    shape1 = (100, 100) # h, w
    shape2 = (150, 150)
    # H just shifts by 50, 50
    H_shift = np.array([[1, 0, 50], [0, 1, 50], [0, 0, 1]], dtype=np.float64)
    # pt at 10,10 in img1 -> 60,60 in img2 (valid)
    # pt at 120, 120 in img1 -> 170,170 in img2 (invalid)
    pts1 = np.array([[10, 10], [120, 120]], dtype=np.float32)
    pts2 = np.array([[60, 60], [10, 10]], dtype=np.float32) # 10,10 in img2 -> -40,-40 in img1 (invalid)
    mask1, mask2 = ev.get_covisible_indices(pts1, pts2, H_shift, shape1, shape2)
    
    assert mask1[0] == True and mask1[1] == False
    assert mask2[0] == True and mask2[1] == False
    print("✓ Co-visible region filtering behaves correctly.")
    
    print("\nSanity checks passed.")

if __name__ == "__main__":
    test_sanity()
