import os
import cv2
import numpy as np
from typing import List, Tuple, Generator

class HPatchesSequence:
    def __init__(self, path: str):
        self.path = path
        self.name = os.path.basename(path)
        self.is_illumination = self.name.startswith('i_')
        self.is_viewpoint = self.name.startswith('v_')
        
        # The images are named 1.ppm to 6.ppm. 1 is reference.
        self.image_paths = [os.path.join(self.path, f"{i}.ppm") for i in range(1, 7)]
        
    def get_image(self, index: int, grayscale: bool = True) -> np.ndarray:
        """
        Read image given an index from 1 to 6.
        Returns grayscale by default as most descriptors expect it, 
        but allows color if needed.
        """
        if index < 1 or index > 6:
            raise ValueError("Index must be between 1 and 6")
        
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        img = cv2.imread(self.image_paths[index - 1], flag)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {self.image_paths[index - 1]}")
        return img
        
    def get_homography(self, index: int) -> np.ndarray:
        """
        Get homography mapping from image 1 to image `index` (2 to 6).
        """
        if index == 1:
            return np.eye(3, dtype=np.float32)
        if index < 2 or index > 6:
            raise ValueError("Index must be between 2 and 6 for homography")
            
        h_path = os.path.join(self.path, f"H_1_{index}")
        if not os.path.exists(h_path):
            raise FileNotFoundError(f"Homography file not found: {h_path}")
            
        with open(h_path, 'r') as f:
            lines = f.readlines()
            
        matrix = []
        for line in lines:
            line_str = line.strip()
            if line_str:
                matrix.append([float(x) for x in line_str.split()])
        
        return np.array(matrix, dtype=np.float32)


class HPatchesDataset:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.sequences = self._load_sequences()
        
    def _load_sequences(self) -> List[HPatchesSequence]:
        seqs = []
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")
            
        for name in os.listdir(self.root_dir):
            path = os.path.join(self.root_dir, name)
            if os.path.isdir(path) and (name.startswith('i_') or name.startswith('v_')):
                seqs.append(HPatchesSequence(path))
        return sorted(seqs, key=lambda x: x.name)
        
    def get_illumination_sequences(self) -> List[HPatchesSequence]:
        return [seq for seq in self.sequences if seq.is_illumination]
        
    def get_viewpoint_sequences(self) -> List[HPatchesSequence]:
        return [seq for seq in self.sequences if seq.is_viewpoint]
        
    def get_all_sequences(self) -> List[HPatchesSequence]:
        return self.sequences

    def load_pair(self, sequence_name: str, target_idx: int, grayscale: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        for seq in self.sequences:
            if seq.name == sequence_name:
                img1 = seq.get_image(1, grayscale=grayscale)
                imgN = seq.get_image(target_idx, grayscale=grayscale)
                H = seq.get_homography(target_idx)
                return img1, imgN, H
        raise ValueError(f"Sequence {sequence_name} not found")

    def iterate_pairs(self, grayscale: bool = True) -> Generator[Tuple[HPatchesSequence, int, np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Yields pairs of (sequence, target_index, image1, imageN, homography)
        image1 is the reference image.
        imageN is the target image (N=2..6)
        homography maps points from image1 to imageN.
        """
        for seq in self.sequences:
            img1 = seq.get_image(1, grayscale=grayscale)
            for idx in range(2, 7):
                imgN = seq.get_image(idx, grayscale=grayscale)
                H = seq.get_homography(idx)
                yield seq, idx, img1, imgN, H
