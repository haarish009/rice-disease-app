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


def apply_mask(img_array: np.ndarray, mask: np.ndarray, bg_alpha: float = 0.3) -> np.ndarray:
    """
    Apply a 2-D binary/soft mask to a normalised RGB image while preserving context.

    Instead of zeroing out non-masked regions, the background is dimmed to
    ``bg_alpha`` intensity so that Stage 2 receives an image that stays within
    its expected training distribution.

    Parameters
    ----------
    img_array : np.ndarray
        Normalised float32 array of shape (H, W, 3) with values in [0, 1].
    mask : np.ndarray
        2-D float32 array of shape (H, W) with values in [0, 1].
    bg_alpha : float
        Dimming factor applied to the background (non-masked) regions.
        Default is ``0.3`` (30 % of original intensity).

    Returns
    -------
    np.ndarray
        Masked image of shape (H, W, 3) with the disease region at full
        intensity and the background dimmed by ``bg_alpha``.
    """
    mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    foreground = img_array * mask_3ch
    background = img_array * (1.0 - mask_3ch) * bg_alpha
    result = foreground + background
    return np.clip(result, 0.0, 1.0).astype("float32")


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
