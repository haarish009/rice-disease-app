from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from utils.model_loader import load_models
from utils.grad_cam_plus_plus import GradCAMPlusPlus, overlay_heatmap
import cv2
import base64

app = FastAPI(title="Dual-Stage Rice Disease Detection API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
stage1_model = None
stage2_model = None
grad_model = None
grad_cam_engine = None

# Class names (Based on standard 10-class rice disease datasets)
CLASSES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy",
    "Leaf Blast",
    "Leaf Scald",
    "Narrow Brown Leaf Spot",
    "Rice Hispa",
    "Sheath Blight",
    "Sheath Rot",
    "Tungro"
]

@app.get("/")
async def root():
    return {"status": "online", "message": "Dual-Stage Rice Disease Detection API is running"}

@app.on_event("startup")
async def startup_event():
    global stage1_model, stage2_model, grad_model, grad_cam_engine
    try:
        stage1_model, stage2_model, grad_model = load_models()
        grad_cam_engine = GradCAMPlusPlus(grad_model)
        print("Success: All models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {e}")

def preprocess_image(image: Image.Image, target_size=(160, 160)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image)
    img_array = img_array.astype('float32') / 255.0
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if stage1_model is None or stage2_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    # Read and preprocess image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    img_array = preprocess_image(image)
    
    # --- STAGE 1: Global Classification ---
    stage1_preds = stage1_model.predict(img_array[np.newaxis, ...], verbose=0)
    stage1_class_idx = np.argmax(stage1_preds[0])
    stage1_confidence = float(stage1_preds[0][stage1_class_idx])
    stage1_label = CLASSES[stage1_class_idx]
    
    result = {
        "stage1": {
            "label": stage1_label,
            "confidence": round(stage1_confidence, 4),
            "class_idx": int(stage1_class_idx)
        }
    }
    
    # --- XAI: Grad-CAM++ Attention Mask ---
    mask = grad_cam_engine.generate_mask(img_array, stage1_class_idx, threshold=0.5)
    overlay = overlay_heatmap(mask, img_array)
    
    _, buffer = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
    result["heatmap"] = heatmap_base64

    # --- STAGE 2: Refined Attention ---
    if stage1_label != "Healthy":
        mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        segmented_img = img_array * mask_3ch
        
        stage2_preds = stage2_model.predict(segmented_img[np.newaxis, ...], verbose=0)
        stage2_class_idx = np.argmax(stage2_preds[0])
        stage2_confidence = float(stage2_preds[0][stage2_class_idx])
        stage2_label = CLASSES[stage2_class_idx]
        
        result["stage2"] = {
            "label": stage2_label,
            "confidence": round(stage2_confidence, 4),
            "class_idx": int(stage2_class_idx)
        }
        result["final_diagnosis"] = stage2_label
    else:
        result["final_diagnosis"] = "Healthy"

    
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
