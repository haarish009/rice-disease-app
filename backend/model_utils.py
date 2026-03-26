"""
model_utils.py
--------------
TFLite model loader and inference wrapper for rice disease detection.

TODO: Implement `preprocess` and `predict` with your real model logic.
      The stubs below define the expected interface and data shapes.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class RiceDiseaseModel:
    """Loads a TFLite model and runs inference on rice-leaf images.

    Parameters
    ----------
    model_path:
        Path to the ``model_s1.tflite`` file.
    metadata_path:
        Path to ``metadata.json`` containing class definitions.
    """

    def __init__(self, model_path: str, metadata_path: str) -> None:
        self.model_path = model_path
        self.metadata_path = metadata_path

        # --- load metadata ---
        with open(metadata_path, "r") as fh:
            self.metadata = json.load(fh)

        self.classes: list[dict] = self.metadata["classes"]
        self.input_height: int = self.metadata["input_size"][0]
        self.input_width: int = self.metadata["input_size"][1]
        self.num_classes: int = self.metadata["num_classes"]

        # --- load TFLite interpreter ---
        self.interpreter = None   # TODO: replace stub with real interpreter
        self.input_details = None
        self.output_details = None
        self._load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize and normalise a raw RGB image for model inference.

        Parameters
        ----------
        image:
            RGB uint8 array of shape ``(H, W, 3)``.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``(1, input_height, input_width, 3)``
            with values in ``[0, 1]``.

        TODO: add any additional normalisation (mean/std, channel order,
              quantisation scaling) required by your specific model.
        """
        # TODO: implement preprocessing
        raise NotImplementedError("preprocess() is not yet implemented")

    def predict(self, image: np.ndarray) -> dict[str, Any]:
        """Run inference on an RGB image and return structured results.

        Parameters
        ----------
        image:
            RGB uint8 array of shape ``(H, W, 3)``.

        Returns
        -------
        dict with keys:
            - ``class_id``     (int)   – predicted class index
            - ``class_name``   (str)   – human-readable label
            - ``confidence``   (float) – probability of the top class (0–1)
            - ``probabilities``(list)  – per-class softmax probabilities
            - ``description``  (str)   – short disease description
            - ``severity``     (str)   – one of none/low/medium/high/critical
            - ``treatment``    (list)  – recommended treatment steps

        TODO: call self.preprocess(), run the interpreter, and return the
              dict above with real values.
        """
        # TODO: implement inference
        raise NotImplementedError("predict() is not yet implemented")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Initialise the TFLite interpreter.

        TODO: uncomment the block below once ``tensorflow`` (or
              ``tflite-runtime``) is available and the model file exists.
        """
        try:
            import tensorflow as tf  # noqa: PLC0415  (local import intentional)

            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            logger.info("TFLite model loaded from %s", self.model_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not load TFLite model (%s). Running in stub mode.", exc
            )
