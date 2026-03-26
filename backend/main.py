"""
main.py
-------
FastAPI application for rice disease detection.

Endpoints
---------
GET  /health   – liveness probe
POST /predict  – accept an image, return prediction + Grad-CAM++ heatmap
"""

import base64
import io
import logging
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from model_utils import RiceDiseaseModel
from xai_engine import GradCAMPlusPlus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Rice Disease Detection API",
    description=(
        "AI-powered rice disease detection with Grad-CAM++ explainability. "
        "Upload a rice-leaf image to receive a disease diagnosis, confidence "
        "score, treatment recommendations, and a saliency heatmap."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model_s1.tflite"
METADATA_PATH = BASE_DIR / "metadata.json"

# ---------------------------------------------------------------------------
# Global model instances (populated on startup)
# ---------------------------------------------------------------------------

model: RiceDiseaseModel | None = None
xai_engine: GradCAMPlusPlus | None = None


@app.on_event("startup")
async def startup_event() -> None:
    global model, xai_engine

    if not MODEL_PATH.exists():
        logger.warning(
            "Model file not found at %s – running without inference capability.",
            MODEL_PATH,
        )
        return

    model = RiceDiseaseModel(str(MODEL_PATH), str(METADATA_PATH))

    if model.interpreter is not None:
        xai_engine = GradCAMPlusPlus(
            model.interpreter, model.input_details, model.output_details
        )

    logger.info("Startup complete – model and XAI engine ready.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", summary="Health check")
async def health_check() -> dict:
    """Return service status and whether the model is loaded."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "version": "1.0.0",
    }


@app.post("/predict", summary="Predict rice disease from an image")
async def predict(file: UploadFile = File(...)) -> dict:
    """Accept a rice-leaf image and return:

    - ``class_id``      – predicted class index
    - ``class_name``    – human-readable disease label
    - ``confidence``    – probability of the top class (0–1)
    - ``probabilities`` – per-class softmax probabilities
    - ``description``   – short disease description
    - ``treatment``     – list of recommended treatment steps
    - ``heatmap``       – base64-encoded JPEG of the Grad-CAM++ overlay
    """
    # --- validate upload ---
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. "
                "Ensure model_s1.tflite exists in the backend directory."
            ),
        )

    # --- decode image ---
    contents = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Could not decode image: {exc}"
        ) from exc

    image_rgb = np.array(pil_image, dtype=np.uint8)

    # --- inference ---
    result = model.predict(image_rgb)

    # --- XAI heatmap ---
    heatmap_b64: str | None = None
    if xai_engine is not None:
        preprocessed = model.preprocess(image_rgb)
        heatmap = xai_engine.generate(preprocessed, result["class_id"])
        overlay = _overlay_heatmap(image_rgb, heatmap)
        heatmap_b64 = _image_to_base64(overlay)

    return {
        "class_id": result["class_id"],
        "class_name": result["class_name"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "description": result["description"],
        "severity": result["severity"],
        "treatment": result["treatment"],
        "heatmap": heatmap_b64,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _overlay_heatmap(
    image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4
) -> np.ndarray:
    """Blend a [0, 1] heatmap onto an RGB image using a JET colour map.

    Parameters
    ----------
    image:
        RGB uint8 array of shape ``(H, W, 3)``.
    heatmap:
        Float32 array of shape ``(fH, fW)``, values in ``[0, 1]``.
    alpha:
        Heatmap opacity (0 = invisible, 1 = fully opaque).

    Returns
    -------
    np.ndarray
        Blended RGB uint8 array of shape ``(H, W, 3)``.
    """
    import cv2  # noqa: PLC0415  (optional dep – only needed at request time)

    h, w = image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Apply JET colour map (cv2 works in BGR)
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(image, 1 - alpha, heatmap_rgb, alpha, 0)
    return blended


def _image_to_base64(image: np.ndarray) -> str:
    """Encode an RGB numpy array as a base64 JPEG string."""
    pil_img = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
