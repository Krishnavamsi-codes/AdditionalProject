import sys

def verify():
    report = {}
    
    # 1. Verify Imports
    try:
        import cv2
        report['cv2_import'] = True
    except ImportError:
        report['cv2_import'] = False
        
    try:
        import numpy
        report['numpy_import'] = True
    except ImportError:
        report['numpy_import'] = False
        
    try:
        import scipy
        report['scipy_import'] = True
    except ImportError:
        report['scipy_import'] = False
        
    try:
        import pandas
        report['pandas_import'] = True
    except ImportError:
        report['pandas_import'] = False
        
    try:
        import sklearn
        report['sklearn_import'] = True
    except ImportError:
        report['sklearn_import'] = False
        
    try:
        import matplotlib
        report['matplotlib_import'] = True
    except ImportError:
        report['matplotlib_import'] = False
        
    try:
        import torch
        report['torch_import'] = True
    except ImportError:
        report['torch_import'] = False

    # 2. Verify Features
    if report.get('cv2_import'):
        import cv2
        
        try:
            cv2.SIFT_create()
            report['SIFT_create'] = True
        except AttributeError:
            report['SIFT_create'] = False
            
        try:
            cv2.ORB_create()
            report['ORB_create'] = True
        except AttributeError:
            report['ORB_create'] = False
            
        try:
            cv2.BRISK_create()
            report['BRISK_create'] = True
        except AttributeError:
            report['BRISK_create'] = False
            
        try:
            hasattr(cv2, 'xfeatures2d') and cv2.xfeatures2d is not None
            report['xfeatures2d'] = hasattr(cv2, 'xfeatures2d')
        except AttributeError:
            report['xfeatures2d'] = False
            
        try:
            cv2.xfeatures2d.SURF_create
            report['SURF_create'] = True
        except AttributeError:
            report['SURF_create'] = False
            
        try:
            cv2.xfeatures2d.LATCH_create
            report['LATCH_create'] = True
        except AttributeError:
            report['LATCH_create'] = False
    else:
        report['SIFT_create'] = False
        report['ORB_create'] = False
        report['BRISK_create'] = False
        report['xfeatures2d'] = False
        report['SURF_create'] = False
        report['LATCH_create'] = False

    print("--- Environment Verification Report ---")
    
    print("\nImports:")
    imports = ['cv2', 'numpy', 'scipy', 'pandas', 'sklearn', 'matplotlib', 'torch']
    for imp in imports:
        status = "✓ Available" if report.get(f"{imp}_import") else "✗ Missing"
        print(f"{status} - {imp}")

    print("\nFeatures:")
    features = ['SIFT_create', 'ORB_create', 'BRISK_create', 'xfeatures2d', 'SURF_create', 'LATCH_create']
    for feat in features:
        status = "✓ Available" if report.get(feat) else "✗ Missing"
        if feat == 'xfeatures2d':
            print(f"{status} - cv2.{feat}")
        elif feat in ['SURF_create', 'LATCH_create']:
            print(f"{status} - cv2.xfeatures2d.{feat}")
        else:
            print(f"{status} - cv2.{feat}()")
            
    if not report.get('SURF_create') or not report.get('LATCH_create'):
        print("\n--- Note on Missing Features ---")
        print("SURF and/or LATCH are missing.")
        print("Explanation:")
        print("Since OpenCV 3, patented algorithms like SURF and some experimental ones like LATCH")
        print("were moved to the 'xfeatures2d' module inside 'opencv_contrib'.")
        print("Furthermore, due to patent restrictions, SURF is disabled by default in standard builds (OPENCV_ENABLE_NONFREE=OFF).")
        print("Minimum fix:")
        print("To enable SURF or LATCH, you must compile OpenCV from source with the 'opencv_contrib' modules")
        print("and the CMake flag '-D OPENCV_ENABLE_NONFREE=ON'.")

if __name__ == '__main__':
    verify()
