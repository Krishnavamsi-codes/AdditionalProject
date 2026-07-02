import cv2
import numpy as np
from typing import List, Tuple, Any

class FeatureMatcher:
    def __init__(self, matcher_type: str = 'auto', cross_check: bool = False, ratio_thresh: float = 0.75):
        """
        matcher_type: 'auto', 'bf', 'flann'
        cross_check: if True, use mutual nearest neighbor.
        ratio_thresh: Lowe's ratio test threshold. Set to 1.0 to disable.
        """
        self.matcher_type = matcher_type
        self.cross_check = cross_check
        self.ratio_thresh = ratio_thresh

    def match(self, desc1: np.ndarray, desc2: np.ndarray) -> List[cv2.DMatch]:
        """Match desc1 to desc2 and return filtered matches."""
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []
            
        norm_type = cv2.NORM_HAMMING if desc1.dtype == np.uint8 else cv2.NORM_L2
        matcher = self._create_matcher(norm_type)
        
        # Use knnMatch for ratio test
        if self.ratio_thresh < 1.0:
            # Need at least 2 descriptors in desc2 to find the 2 nearest neighbors
            k = min(2, len(desc2))
            knn_matches = matcher.knnMatch(desc1, desc2, k=k)
            good_matches = []
            for m_obj in knn_matches:
                if len(m_obj) == 2:
                    m, n = m_obj
                    if m.distance < self.ratio_thresh * n.distance:
                        good_matches.append(m)
                elif len(m_obj) == 1:
                    good_matches.append(m_obj[0])
            
            # Manual cross-check on good matches if requested
            if self.cross_check and len(good_matches) > 0:
                knn_matches_21 = matcher.knnMatch(desc2, desc1, k=1)
                valid_cross = []
                for m in good_matches:
                    m21 = knn_matches_21[m.trainIdx]
                    if len(m21) > 0 and m21[0].trainIdx == m.queryIdx:
                        valid_cross.append(m)
                return valid_cross
                
            return good_matches
        else:
            # No ratio test
            if self.matcher_type in ['auto', 'bf'] and self.cross_check:
                # Fast path using OpenCV's built-in crossCheck
                matcher = cv2.BFMatcher(norm_type, crossCheck=True)
                return matcher.match(desc1, desc2)
            else:
                matches = matcher.match(desc1, desc2)
                if self.cross_check and len(matches) > 0:
                     matches_21 = matcher.match(desc2, desc1)
                     # matches_21 is list of DMatch but sorted by desc2 index
                     # Actually match() returns a list of DMatch where queryIdx is from desc2
                     # To find it quickly we can create an array or dict
                     m21_dict = {m.queryIdx: m.trainIdx for m in matches_21}
                     valid_cross = []
                     for m in matches:
                         if m21_dict.get(m.trainIdx) == m.queryIdx:
                             valid_cross.append(m)
                     return valid_cross
                return matches

    def _create_matcher(self, norm_type):
        m_type = 'bf' if self.matcher_type == 'auto' else self.matcher_type
            
        if m_type == 'bf':
            return cv2.BFMatcher(norm_type, crossCheck=False)
        elif m_type == 'flann':
            if norm_type == cv2.NORM_HAMMING:
                index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
                search_params = dict(checks=50)
            else:
                index_params = dict(algorithm=1, trees=5) # KDTree
                search_params = dict(checks=50)
            return cv2.FlannBasedMatcher(index_params, search_params)
        else:
            raise ValueError(f"Unknown matcher type {self.matcher_type}")

def estimate_homography(kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint], matches: List[cv2.DMatch]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns estimated homography and inlier mask (nx1 array of uint8).
    """
    if len(matches) < 4:
        return None, None
        
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
    return H, mask
