import os
from dataset.hpatches import HPatchesDataset

def test_dataset():
    dataset_path = r"d:\AdditionalProject\hpatches-sequences-release"
    dataset = HPatchesDataset(dataset_path)
    
    print(f"Total sequences: {len(dataset.get_all_sequences())}")
    print(f"Illumination sequences: {len(dataset.get_illumination_sequences())}")
    print(f"Viewpoint sequences: {len(dataset.get_viewpoint_sequences())}")
    
    assert len(dataset.get_all_sequences()) == 116
    assert len(dataset.get_illumination_sequences()) == 57
    assert len(dataset.get_viewpoint_sequences()) == 59
    
    # Test iterator
    count = 0
    for seq, idx, img1, imgN, H in dataset.iterate_pairs():
        assert img1 is not None
        assert imgN is not None
        assert H.shape == (3, 3)
        count += 1
        if count == 5:
            break
            
    print("Dataset loader tests passed!")

if __name__ == "__main__":
    test_dataset()
