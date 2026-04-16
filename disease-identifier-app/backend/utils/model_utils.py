import numpy as np
from PIL import Image

def preprocess(image: Image.Image, target_size=(160, 160)):
    """
    Preprocess image for model prediction.
    - Ensures RGB mode
    - Resizes to target_size
    - Converts to float32
    - Normalizes by 1/255.0 (Min-Max Scaling)
    """
    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize image
    image = image.resize(target_size)
    
    # Convert to numpy array and normalize
    img_array = np.array(image)
    img_array = img_array.astype('float32') / 255.0
    
    return img_array
