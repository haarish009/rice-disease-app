"""
main.py
-------
FastAPI application for two-stage rice disease detection.

Endpoints
---------
GET  /health   – liveness probe
POST /predict  – accept an image, run Stage 1 + Grad-CAM++ + Stage 2, return results
"""

import io
import json
import logging
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from utils.grad_cam_plus_plus import GradCAMPlusPlus, overlay_heatmap
from utils.model_loader import load_models
from utils.model_utils import apply_mask, encode_image_base64, preprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state (populated on startup)
# ---------------------------------------------------------------------------

stage1_model = None
stage2_model = None
grad_cam_engine = None
disease_metadata: list = []

BASE_DIR = Path(__file__).parent
METADATA_PATH = BASE_DIR / "disease_metadata.json"

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Rice Disease Detection API",
    description=(
        "Two-stage AI system for rice leaf disease detection. "
        "Stage 1 performs global analysis using MobileNetV2; "
        "Stage 2 refines the diagnosis on the Grad-CAM++ masked region."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_disease_metadata() -> None:
    global disease_metadata
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r") as f:
            disease_metadata = json.load(f)
        logger.info("Loaded disease metadata with %d entries", len(disease_metadata))
    else:
        logger.warning("disease_metadata.json not found at %s", METADATA_PATH)


def _get_disease_info(class_idx: int) -> dict:
    """Return metadata entry for a given class index, or a fallback dict."""
    for entry in disease_metadata:
        if entry.get("id") == class_idx:
            return entry
    return {"id": class_idx, "name": f"Unknown (class {class_idx})", "cause": "", "remedy": ""}


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event() -> None:
    global stage1_model, stage2_model, grad_cam_engine

    _load_disease_metadata()

    try:
        stage1_model, stage2_model, grad_model = load_models()
        grad_cam_engine = GradCAMPlusPlus(grad_model)
        logger.info("Startup complete – Stage 1 & Stage 2 models ready.")
    except FileNotFoundError as exc:
        logger.error("Model files not found: %s", exc)
    except Exception as exc:
        logger.error("Failed to load models: %s", exc)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Health check")
async def health_check() -> dict:
    """Return service status and whether both models are loaded."""
    return {
        "status": "ok",
        "models_loaded": stage1_model is not None and stage2_model is not None,
        "version": "2.0.0",
    }


@app.post("/predict", summary="Predict rice disease from an image")
async def predict(file: UploadFile = File(...)) -> dict:
    """Accept a rice-leaf image and return:

    - ``stage1``          – Stage 1 label + confidence
    - ``stage2``          – Stage 2 label + confidence (refined via Grad-CAM++ mask)
    - ``heatmap``         – base64-encoded PNG of the Grad-CAM++ attention overlay
    - ``final_diagnosis`` – human-readable disease name (Stage 2 result)
    - ``metadata``        – cause and remedy for the final diagnosis
    """
    # --- validate upload ---
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    if stage1_model is None or stage2_model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Models not loaded. "
                "Ensure model files exist in the models directory and restart the server."
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

    # Preprocess: resize to 160×160, normalise to [0, 1], float32
    img_array = preprocess(pil_image, target_size=(160, 160))

    # ── Stage 1: Global Analysis ─────────────────────────────────────────────
    try:
        stage1_preds = stage1_model.predict(img_array[np.newaxis, ...], verbose=0)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Stage 1 inference failed: {exc}"
        ) from exc

    stage1_class_idx = int(np.argmax(stage1_preds[0]))
    stage1_confidence = float(stage1_preds[0][stage1_class_idx])
    stage1_info = _get_disease_info(stage1_class_idx)

    # ── Grad-CAM++ Mask Generation ────────────────────────────────────────────
    try:
        mask = grad_cam_engine.generate_mask(img_array, stage1_class_idx, threshold=0.5)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Grad-CAM++ mask generation failed: {exc}"
        ) from exc

    # ── Stage 2: Refined Diagnosis on Masked Region ───────────────────────────
    segmented_img = apply_mask(img_array, mask)

    try:
        stage2_preds = stage2_model.predict(segmented_img[np.newaxis, ...], verbose=0)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Stage 2 inference failed: {exc}"
        ) from exc

    stage2_class_idx = int(np.argmax(stage2_preds[0]))
    stage2_confidence = float(stage2_preds[0][stage2_class_idx])
    stage2_info = _get_disease_info(stage2_class_idx)

    # ── Heatmap Overlay ───────────────────────────────────────────────────────
    try:
        overlay = overlay_heatmap(mask, img_array)  # returns RGB uint8
        heatmap_b64 = encode_image_base64(overlay)
    except Exception as exc:
        logger.warning("Heatmap generation failed: %s – returning None", exc)
        heatmap_b64 = None

    # ── Response ──────────────────────────────────────────────────────────────
    final_info = stage2_info
    return {
        "stage1": {
            "label": stage1_info["name"],
            "confidence": stage1_confidence,
        },
        "stage2": {
            "label": stage2_info["name"],
            "confidence": stage2_confidence,
        },
        "heatmap": heatmap_b64,
        "final_diagnosis": final_info["name"],
        "metadata": {
            "cause": final_info.get("cause", ""),
            "remedy": final_info.get("remedy", ""),
        },
    }