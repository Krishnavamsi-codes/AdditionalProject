import cv2
import numpy as np
from typing import Tuple, Any
from .base import BaseMethod

class OpenCVMethod(BaseMethod):
    def __init__(self, name: str, detector, is_binary: bool):
        super().__init__(name, is_binary)
        self.detector = detector
        
    def detect(self, image: np.ndarray) -> Any:
        return self.detector.detect(image, None)
        
    def compute(self, image: np.ndarray, keypoints: Any) -> Tuple[Any, np.ndarray]:
        # Handle cases where keypoints is empty
        if len(keypoints) == 0:
            return keypoints, np.array([])
        return self.detector.compute(image, keypoints)
        
    def detect_and_compute(self, image: np.ndarray) -> Tuple[Any, np.ndarray]:
        return self.detector.detectAndCompute(image, None)


class SIFTMethod(OpenCVMethod):
    def __init__(self, nfeatures=5000):
        super().__init__("SIFT", cv2.SIFT_create(nfeatures=nfeatures), is_binary=False)


class SURFMethod(OpenCVMethod):
    def __init__(self, hessianThreshold=100):
        try:
            detector = cv2.xfeatures2d.SURF_create(hessianThreshold=hessianThreshold)
            super().__init__("SURF", detector, is_binary=False)
            self.available = True
        except cv2.error:
            # Patent error, SURF not available in this build
            super().__init__("SURF", None, is_binary=False)
            self.available = False
            
    def detect(self, image: np.ndarray) -> Any:
        if not self.available: return []
        return super().detect(image)
        
    def compute(self, image: np.ndarray, keypoints: Any) -> Tuple[Any, np.ndarray]:
        if not self.available or len(keypoints) == 0: return [], np.array([])
        return super().compute(image, keypoints)
        
    def detect_and_compute(self, image: np.ndarray) -> Tuple[Any, np.ndarray]:
        if not self.available: return [], np.array([])
        return super().detect_and_compute(image)


class ORBMethod(OpenCVMethod):
    def __init__(self, nfeatures=5000):
        super().__init__("ORB", cv2.ORB_create(nfeatures=nfeatures), is_binary=True)


class BRISKMethod(OpenCVMethod):
    def __init__(self):
        super().__init__("BRISK", cv2.BRISK_create(), is_binary=True)


class LATCHMethod(BaseMethod):
    def __init__(self, nfeatures=5000):
        super().__init__("LATCH", is_binary=True)
        # LATCH is only a descriptor, needs a detector. We use ORB.
        self.detector = cv2.ORB_create(nfeatures=nfeatures)
        self.descriptor = cv2.xfeatures2d.LATCH_create()
        
    def detect(self, image: np.ndarray) -> Any:
        return self.detector.detect(image, None)
        
    def compute(self, image: np.ndarray, keypoints: Any) -> Tuple[Any, np.ndarray]:
        if len(keypoints) == 0:
            return keypoints, np.array([])
        return self.descriptor.compute(image, keypoints)
        
    def detect_and_compute(self, image: np.ndarray) -> Tuple[Any, np.ndarray]:
        keypoints = self.detect(image)
        return self.compute(image, keypoints)
