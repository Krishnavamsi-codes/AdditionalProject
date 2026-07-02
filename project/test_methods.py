import cv2
import numpy as np
from methods import SIFTMethod, SURFMethod, ORBMethod, BRISKMethod, LATCHMethod

def test_methods():
    # Create a dummy image
    image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    
    methods = [
        SIFTMethod(),
        SURFMethod(),
        ORBMethod(),
        BRISKMethod(),
        LATCHMethod()
    ]
    
    for method in methods:
        print(f"Testing {method.name}...")
        
        # Test detect
        kps = method.detect(image)
        print(f"  Detected {len(kps)} keypoints.")
        
        # Test compute
        kps2, descs = method.compute(image, kps)
        assert len(kps2) == len(kps)
        if len(kps) > 0:
            print(f"  Descriptor shape: {descs.shape}, type: {descs.dtype}")
            if method.is_binary:
                assert descs.dtype == np.uint8
            else:
                assert descs.dtype == np.float32
        
        # Test detect_and_compute
        kps3, descs3 = method.detect_and_compute(image)
        if len(kps) > 0:
            assert len(kps3) > 0
            assert descs3 is not None
        
    print("Methods tests passed!")

if __name__ == "__main__":
    test_methods()
