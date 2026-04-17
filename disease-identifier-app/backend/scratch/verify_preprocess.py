import numpy as np
from PIL import Image
import sys
import os

# Add the backend directory to sys.path to import model_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.model_utils import preprocess

def test_preprocess():
    # Create a dummy image (e.g., 200x200 RGB)
    dummy_img = Image.new('RGB', (200, 200), color=(255, 0, 0))
    
    # Preprocess
    img_array = preprocess(dummy_img, target_size=(160, 160))
    
    # Assertions
    print(f"Shape: {img_array.shape}")
    print(f"Dtype: {img_array.dtype}")
    print(f"Max value: {img_array.max()}")
    print(f"Min value: {img_array.min()}")
    
    assert img_array.shape == (160, 160, 3)
    assert img_array.dtype == np.float32
    assert img_array.max() <= 1.0
    assert img_array.min() >= 0.0
    # For a full red image, max should be 1.0 (255/255)
    assert np.allclose(img_array[0, 0], [1.0, 0.0, 0.0])
    
    print("Preprocessing test passed!")

if __name__ == "__main__":
    test_preprocess()
