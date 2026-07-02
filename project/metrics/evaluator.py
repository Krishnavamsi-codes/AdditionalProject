import numpy as np
import cv2
from typing import List, Tuple, Dict, Any

class Evaluator:
    def __init__(self, pixel_threshold: float = 3.0):
        self.pixel_threshold = pixel_threshold
        
    def project_keypoints(self, pts: np.ndarray, H: np.ndarray) -> np.ndarray:
        if len(pts) == 0:
            return np.array([])
        pts_proj = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H)
        return pts_proj.reshape(-1, 2)

    def get_covisible_indices(self, pts1: np.ndarray, pts2: np.ndarray, H: np.ndarray, shape1: Tuple[int, int], shape2: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns boolean masks for pts1 and pts2 indicating if they project within the bounds of the other image.
        shape is (height, width).
        """
        h1, w1 = shape1
        h2, w2 = shape2
        
        # Filter pts1
        if len(pts1) > 0:
            pts1_proj = self.project_keypoints(pts1, H)
            mask1 = (pts1_proj[:, 0] >= 0) & (pts1_proj[:, 0] < w2) & (pts1_proj[:, 1] >= 0) & (pts1_proj[:, 1] < h2)
        else:
            mask1 = np.array([], dtype=bool)
            
        # Filter pts2
        if len(pts2) > 0:
            H_inv = np.linalg.inv(H)
            pts2_proj = self.project_keypoints(pts2, H_inv)
            mask2 = (pts2_proj[:, 0] >= 0) & (pts2_proj[:, 0] < w1) & (pts2_proj[:, 1] >= 0) & (pts2_proj[:, 1] < h1)
        else:
            mask2 = np.array([], dtype=bool)
            
        return mask1, mask2

    def evaluate_detector(self, kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint], H: np.ndarray, shape1: Tuple[int, int], shape2: Tuple[int, int]) -> Dict[str, Any]:
        """
        Computes Number of detected keypoints and Repeatability strictly in the overlapping region.
        """
        num_kp1 = len(kp1)
        num_kp2 = len(kp2)
        
        if num_kp1 == 0 or num_kp2 == 0:
            return {'num_kp1': num_kp1, 'num_kp2': num_kp2, 'repeatability': 0.0}
            
        pts1 = np.float32([kp.pt for kp in kp1])
        pts2 = np.float32([kp.pt for kp in kp2])
        
        mask1, mask2 = self.get_covisible_indices(pts1, pts2, H, shape1, shape2)
        
        pts1_covis = pts1[mask1]
        pts2_covis = pts2[mask2]
        
        n1_covis = len(pts1_covis)
        n2_covis = len(pts2_covis)
        
        if n1_covis == 0 or n2_covis == 0:
            return {'num_kp1': num_kp1, 'num_kp2': num_kp2, 'repeatability': 0.0}
            
        pts1_proj = self.project_keypoints(pts1_covis, H)
        
        from scipy.spatial import KDTree
        
        # Find closest point in pts2_covis for each projected pts1_covis
        # using KDTree for fast O(N log N) query and O(N) memory
        tree = KDTree(pts2_covis)
        min_dists, indices = tree.query(pts1_proj, k=1)
        
        # Enforce unique matching: a point in pts2 can only be counted once
        valid_mask = min_dists <= self.pixel_threshold
        unique_matched_pts2 = np.unique(indices[valid_mask])
        
        repeatable_kp_count = len(unique_matched_pts2)
        repeatability = repeatable_kp_count / min(n1_covis, n2_covis)
        
        return {
            'num_kp1': num_kp1,
            'num_kp2': num_kp2,
            'repeatability': float(repeatability)
        }

    def evaluate_descriptor(self, kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint], desc1: np.ndarray, desc2: np.ndarray, H: np.ndarray, shape1: Tuple[int, int], shape2: Tuple[int, int]) -> Dict[str, Any]:
        """
        Computes AP, Precision, Recall strictly in the co-visible region.
        """
        if len(kp1) == 0 or len(kp2) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'ap': 0.0, 'pr_curve': ([], [])}
            
        pts1 = np.float32([kp.pt for kp in kp1])
        pts2 = np.float32([kp.pt for kp in kp2])
        
        mask1, mask2 = self.get_covisible_indices(pts1, pts2, H, shape1, shape2)
        
        if not np.any(mask1) or not np.any(mask2):
            return {'precision': 0.0, 'recall': 0.0, 'ap': 0.0, 'pr_curve': ([], [])}
            
        pts1_covis = pts1[mask1]
        desc1_covis = desc1[mask1]
        
        pts2_covis = pts2[mask2]
        desc2_covis = desc2[mask2]
        
        pts1_proj = self.project_keypoints(pts1_covis, H)
        
        norm_type = cv2.NORM_HAMMING if desc1.dtype == np.uint8 else cv2.NORM_L2
        matcher = cv2.BFMatcher(norm_type, crossCheck=False)
        matches = matcher.match(desc1_covis, desc2_covis)
        
        matches = sorted(matches, key=lambda x: x.distance)
        
        correct_matches = []
        for m in matches:
            pt1_proj = pts1_proj[m.queryIdx]
            pt2 = pts2_covis[m.trainIdx]
            dist = np.linalg.norm(pt1_proj - pt2)
            correct_matches.append(dist <= self.pixel_threshold)
            
        correct_matches = np.array(correct_matches)
        
        tp_cumsum = np.cumsum(correct_matches)
        fp_cumsum = np.cumsum(~correct_matches)
        
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        total_possible = min(len(pts1_covis), len(pts2_covis))
        recalls = tp_cumsum / total_possible if total_possible > 0 else np.zeros_like(tp_cumsum)
        
        # Average Precision (AP) replacing mAP
        ap = np.sum(precisions * correct_matches) / total_possible if total_possible > 0 else 0.0
        
        total_tp = tp_cumsum[-1] if len(tp_cumsum) > 0 else 0
        overall_precision = total_tp / len(matches) if len(matches) > 0 else 0.0
        overall_recall = total_tp / total_possible if total_possible > 0 else 0.0
        
        return {
            'precision': float(overall_precision),
            'recall': float(overall_recall),
            'ap': float(ap),
            'pr_curve': (precisions.tolist(), recalls.tolist())
        }
        
    def evaluate_mma(self, kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint], matches: List[cv2.DMatch], H: np.ndarray, shape1: Tuple[int, int], shape2: Tuple[int, int], thresholds: List[float] = [1.0, 2.0, 3.0, 5.0, 10.0]) -> Dict[str, float]:
        """
        Computes Mean Matching Accuracy (MMA) for a range of pixel thresholds.
        Only considers matches where the query point in image 1 is within the co-visible region.
        """
        if len(kp1) == 0 or len(matches) == 0:
            return {f'mma_{t}px': 0.0 for t in thresholds}
            
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        h1, w1 = shape1
        h2, w2 = shape2
        
        pts1_proj = self.project_keypoints(pts1, H)
        
        # Filter matches that lie outside the target image bounding box
        valid_mask = (pts1_proj[:, 0] >= 0) & (pts1_proj[:, 0] < w2) & (pts1_proj[:, 1] >= 0) & (pts1_proj[:, 1] < h2)
        
        pts1_proj_valid = pts1_proj[valid_mask]
        pts2_valid = pts2[valid_mask]
        
        if len(pts1_proj_valid) == 0:
             return {f'mma_{t}px': 0.0 for t in thresholds}
             
        dists = np.linalg.norm(pts1_proj_valid - pts2_valid, axis=1)
        
        mma_results = {}
        for t in thresholds:
            correct = np.sum(dists <= t)
            mma_results[f'mma_{int(t)}px'] = float(correct / len(pts1_proj_valid))
            
        return mma_results

    def evaluate_matching(self, kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint], matches: List[cv2.DMatch], H: np.ndarray, shape1: Tuple[int, int], shape2: Tuple[int, int]) -> Dict[str, Any]:
        """
        Computes Matching Score and Correct Match Ratio (Inlier Ratio) in the co-visible region.
        """
        if len(kp1) == 0 or len(matches) == 0:
            return {'matching_score': 0.0, 'inlier_ratio': 0.0}
            
        pts1 = np.float32([kp.pt for kp in kp1])
        pts2 = np.float32([kp.pt for kp in kp2])
        mask1, _ = self.get_covisible_indices(pts1, pts2, H, shape1, shape2)
        n1_covis = np.sum(mask1)
        
        if n1_covis == 0:
            return {'matching_score': 0.0, 'inlier_ratio': 0.0}
            
        m_pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        m_pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        m_pts1_proj = self.project_keypoints(m_pts1, H)
        
        h2, w2 = shape2
        valid_mask = (m_pts1_proj[:, 0] >= 0) & (m_pts1_proj[:, 0] < w2) & (m_pts1_proj[:, 1] >= 0) & (m_pts1_proj[:, 1] < h2)
        
        m_pts1_proj_valid = m_pts1_proj[valid_mask]
        m_pts2_valid = m_pts2[valid_mask]
        
        if len(m_pts1_proj_valid) == 0:
            return {'matching_score': 0.0, 'inlier_ratio': 0.0}
            
        dists = np.linalg.norm(m_pts1_proj_valid - m_pts2_valid, axis=1)
        correct = np.sum(dists <= self.pixel_threshold)
                
        matching_score = correct / n1_covis
        inlier_ratio = correct / len(m_pts1_proj_valid)
        
        return {
            'matching_score': float(matching_score),
            'inlier_ratio': float(inlier_ratio)
        }

    def evaluate_geometry(self, H_gt: np.ndarray, H_est: np.ndarray, shape1: Tuple[int, int]) -> Dict[str, Any]:
        """
        Computes Reprojection Error and Homography Estimation Error.
        """
        if H_est is None:
            return {'reprojection_error': float('inf'), 'homography_error': float('inf')}
            
        try:
            H_gt_norm = H_gt / (H_gt[2, 2] + 1e-8)
            H_est_norm = H_est / (H_est[2, 2] + 1e-8)
            h_err = np.linalg.norm(H_gt_norm - H_est_norm)
        except:
            h_err = float('inf')
        
        h, w = shape1
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        proj_gt = self.project_keypoints(corners, H_gt)
        proj_est = self.project_keypoints(corners, H_est)
        
        if len(proj_gt) == 4 and len(proj_est) == 4:
            rep_err = np.mean(np.linalg.norm(proj_gt - proj_est, axis=1))
        else:
            rep_err = float('inf')
        
        return {
            'reprojection_error': float(rep_err),
            'homography_error': float(h_err)
        }
