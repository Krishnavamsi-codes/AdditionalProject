import cv2
import numpy as np
from matchers import FeatureMatcher, estimate_homography

def test_matcher():
    desc1 = np.random.randint(0, 256, (100, 32), dtype=np.uint8)
    desc2 = np.random.randint(0, 256, (120, 32), dtype=np.uint8)
    
    # Test BF Matcher with Cross Check
    matcher = FeatureMatcher(matcher_type='bf', cross_check=True, ratio_thresh=1.0)
    matches = matcher.match(desc1, desc2)
    print(f"BF CrossCheck: {len(matches)} matches")
    
    # Test Ratio Test
    matcher2 = FeatureMatcher(matcher_type='bf', cross_check=False, ratio_thresh=0.75)
    matches2 = matcher2.match(desc1, desc2)
    print(f"BF Ratio 0.75: {len(matches2)} matches")
    
    # Test both
    matcher3 = FeatureMatcher(matcher_type='bf', cross_check=True, ratio_thresh=0.75)
    matches3 = matcher3.match(desc1, desc2)
    print(f"BF CrossCheck + Ratio 0.75: {len(matches3)} matches")
    
    # Test float descriptors
    desc3 = np.random.rand(100, 128).astype(np.float32)
    desc4 = np.random.rand(120, 128).astype(np.float32)
    matcher4 = FeatureMatcher(matcher_type='flann', cross_check=True, ratio_thresh=0.8)
    matches4 = matcher4.match(desc3, desc4)
    print(f"FLANN Float Ratio 0.8 + CrossCheck: {len(matches4)} matches")
    
    # Test homography
    kp1 = [cv2.KeyPoint(np.random.rand()*100, np.random.rand()*100, 1) for _ in range(100)]
    kp2 = [cv2.KeyPoint(np.random.rand()*100, np.random.rand()*100, 1) for _ in range(120)]
    
    # Fake matches
    fake_matches = [cv2.DMatch(i, i, 1.0) for i in range(10)]
    H, mask = estimate_homography(kp1, kp2, fake_matches)
    print(f"Homography shape: {H.shape if H is not None else None}")
    
    print("Matcher tests passed!")

if __name__ == "__main__":
    test_matcher()
