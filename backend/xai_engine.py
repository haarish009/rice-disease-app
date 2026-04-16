"""
xai_engine.py
-------------
Grad-CAM++ explainability engine for rice disease detection.

The class skeleton below defines the expected interface.
TODO: implement ``generate()`` with your Grad-CAM++ logic once the full
      Keras / TensorFlow model is available alongside the TFLite file.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class GradCAMPlusPlus:
    """Generates Grad-CAM++ saliency heatmaps for a TFLite model.

    Because TFLite does not expose gradients natively, the recommended
    approach is to keep the original Keras model for gradient computation
    (Grad-CAM++) while using the TFLite file for fast inference.

    Parameters
    ----------
    interpreter:
        A ``tf.lite.Interpreter`` instance (already allocated).
    input_details:
        Output of ``interpreter.get_input_details()``.
    output_details:
        Output of ``interpreter.get_output_details()``.

    Usage
    -----
    ::

        engine = GradCAMPlusPlus(model.interpreter,
                                 model.input_details,
                                 model.output_details)
        heatmap = engine.generate(preprocessed_image, class_idx=2)
        # heatmap: float32 array of shape (H, W), values in [0, 1]
    """

    def __init__(self, interpreter, input_details, output_details) -> None:
        self.interpreter = interpreter
        self.input_details = input_details
        self.output_details = output_details

        # Optional: store a reference to the full Keras model for gradient
        # computation.  Set via ``engine.keras_model = model`` after init.
        self.keras_model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, preprocessed_image: np.ndarray, class_idx: int) -> np.ndarray:
        """Return a normalised heatmap for ``class_idx``.

        Parameters
        ----------
        preprocessed_image:
            Batch of one image, shape ``(1, H, W, 3)``, float32, values
            in ``[0, 1]``.
        class_idx:
            Target class index for which to compute the saliency map.

        Returns
        -------
        np.ndarray
            Float32 heatmap of shape ``(H, W)`` with values in ``[0, 1]``.
            Higher values indicate regions most influential for the
            predicted class.

        TODO: implement one of the strategies below and remove the stub
              return.

        Implementation strategies
        -------------------------
        **Option A – Full Keras model (recommended)**
            Use ``tf.GradientTape`` to compute the gradient of the class
            score with respect to the last convolutional layer's output.
            Apply the Grad-CAM++ weighting formula::

                alpha_k = gradient² / (2·gradient² + sum * gradient³)
                cam = ReLU(sum_k(alpha_k · A_k))

        **Option B – Score-CAM (gradient-free, TFLite-compatible)**
            For each channel of the last conv feature map:
            1. Upsample the activation to input size.
            2. Mask the input with the normalised activation.
            3. Run masked input through the TFLite model.
            4. Use the change in class score as the channel weight.
            Final CAM = ReLU(weighted sum of activation channels).

        **Option C – Finite-difference saliency (fallback)**
            Perturb small input patches, measure class-score change,
            and use the delta as a proxy saliency value.
        """
        # TODO: replace this stub with real Grad-CAM++ logic
        h = preprocessed_image.shape[1]
        w = preprocessed_image.shape[2]
        logger.warning(
            "GradCAMPlusPlus.generate() is a stub – returning blank heatmap"
        )
        return np.zeros((h, w), dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal helpers (add your own below)
    # ------------------------------------------------------------------

    def _get_conv_activations(self, preprocessed_image: np.ndarray) -> np.ndarray | None:
        """Extract the last convolutional feature map from the TFLite model.

        Returns ``None`` if no suitable tensor is found.

        TODO: implement by inspecting ``interpreter.get_tensor_details()``
              and calling ``interpreter.get_tensor(idx)`` after invoking.
        """
        # TODO: implement
        return None

    def _normalize(self, cam: np.ndarray) -> np.ndarray:
        """Min-max normalise ``cam`` to ``[0, 1]``."""
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min < 1e-8:
            return np.zeros_like(cam)
        return (cam - cam_min) / (cam_max - cam_min)
