import numpy as np
from typing import Tuple, Any

class BaseMethod:
    def __init__(self, name: str, is_binary: bool = False):
        self.name = name
        self.is_binary = is_binary
        
    def detect(self, image: np.ndarray) -> Any:
        """Returns a list of cv2.KeyPoint"""
        raise NotImplementedError
        
    def compute(self, image: np.ndarray, keypoints: Any) -> Tuple[Any, np.ndarray]:
        """Returns (keypoints, descriptors)"""
        raise NotImplementedError
        
    def detect_and_compute(self, image: np.ndarray) -> Tuple[Any, np.ndarray]:
        """Returns (keypoints, descriptors)"""
        raise NotImplementedError
