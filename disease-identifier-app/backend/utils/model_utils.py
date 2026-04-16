import base64
import io

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
    img_array = img_array.astype("float32") / 255.0

    return img_array


def apply_mask(img_array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a 2-D binary/soft mask to a normalised RGB image.

    Parameters
    ----------
    img_array : np.ndarray
        Normalised float32 array of shape (H, W, 3) with values in [0, 1].
    mask : np.ndarray
        2-D float32 array of shape (H, W) with values in [0, 1].

    Returns
    -------
    np.ndarray
        Masked image of shape (H, W, 3), same dtype as img_array.
    """
    mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    return img_array * mask_3ch


def encode_image_base64(img_array: np.ndarray, fmt: str = "PNG") -> str:
    """
    Encode a uint8 or float32 RGB image array to a base64 string.

    Parameters
    ----------
    img_array : np.ndarray
        RGB image array. If float in [0, 1], it is scaled to uint8 first.
    fmt : str
        PIL image format to use for encoding (default ``"PNG"``).

    Returns
    -------
    str
        Base64-encoded image string suitable for embedding in JSON responses.
    """
    if img_array.dtype != np.uint8:
        img_array = (np.clip(img_array, 0.0, 1.0) * 255).astype(np.uint8)

    pil_image = Image.fromarray(img_array)
    buf = io.BytesIO()
    pil_image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
