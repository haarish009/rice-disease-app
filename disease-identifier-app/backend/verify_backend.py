import numpy as np
import tensorflow as tf
from utils.model_loader import load_models
from utils.grad_cam_plus_plus import GradCAMPlusPlus, overlay_heatmap
import cv2

def verify_pipeline():
    print("Starting verification...")
    
    try:
        # 1. Load models
        print("Loading models...")
        stage1_model, stage2_model, grad_model = load_models()
        grad_cam_engine = GradCAMPlusPlus(grad_model)
        print("Models loaded successfully")

        # 2. Create dummy image (160x160x3)
        print("Creating dummy image...")
        dummy_img = np.random.random((160, 160, 3)).astype('float32')
        
        # 3. Stage 1 Inference
        print("Running Stage 1 inference...")
        stage1_preds = stage1_model.predict(dummy_img[np.newaxis, ...], verbose=0)
        stage1_class_idx = np.argmax(stage1_preds[0])
        print(f"Stage 1 predicted class index: {stage1_class_idx}")

        # 4. Grad-CAM++ Mask Generation
        print("Generating Grad-CAM++ mask...")
        mask = grad_cam_engine.generate_mask(dummy_img, stage1_class_idx, threshold=0.5)
        print(f"Mask generated. Shape: {mask.shape}, Sum: {mask.sum():.2f}")
        
        # 5. Segmented Image
        print("Applying mask...")
        mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        segmented_img = dummy_img * mask_3ch
        
        # 6. Stage 2 Inference
        print("Running Stage 2 inference...")
        stage2_preds = stage2_model.predict(segmented_img[np.newaxis, ...], verbose=0)
        stage2_class_idx = np.argmax(stage2_preds[0])
        print(f"Stage 2 predicted class index: {stage2_class_idx}")

        # 7. Heatmap Overlay
        print("Generating heatmap overlay...")
        overlay = overlay_heatmap(mask, dummy_img)
        print(f"Overlay generated. Shape: {overlay.shape}")

        print("\nVERIFICATION SUCCESSFUL! The backend pipeline is fully functional.")

    except Exception as e:
        print(f"\nVERIFICATION FAILED: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_pipeline()
