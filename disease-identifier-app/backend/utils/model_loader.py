import tensorflow as tf
from tensorflow.keras import layers, models, applications
import os

IMG_SIZE = (160, 160)
NUM_CLASSES_STAGE1 = 10
NUM_CLASSES_STAGE2 = 10


def build_grad_model(stage1_model):
    """
    Rebuilds the Stage 1 architecture to provide access to the last conv layer for Grad-CAM++.
    Uses weight copying to ensure stability with nested Functional models.
    """
    print("Building Grad-CAM version of Stage 1 model via weight copying...")
    
    mobilenet_base = applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(*IMG_SIZE, 3),
        alpha=0.75
    )

    # Build new model with explicit layer access
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = mobilenet_base(inputs, training=False)

    # Store the conv output for Grad-CAM
    conv_output = x 

    # Reconstruct the classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES_STAGE1, activation='softmax')(x)

    # Create the full temporary model to receive weights
    grad_model_full = models.Model(inputs, outputs)

    # Copy weights from the loaded stage1_model
    print("Copying weights to Grad-CAM model...")
    grad_model_full.set_weights(stage1_model.get_weights())

    # Create the final multi-output model
    grad_model = models.Model(
        inputs=grad_model_full.input,
        outputs=[conv_output, grad_model_full.output]
    )
    
    return grad_model

def load_models():
    """
    Loads Stage 1 and Stage 2 models.
    Returns: (stage1_model, stage2_model, grad_model)
    """
    model_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models")
    
    stage1_path = os.path.join(model_dir, "rice_stage1_best (1).h5")
    stage2_path = os.path.join(model_dir, "rice_stage2_finetuned_final.h5")
    
    if not os.path.exists(stage1_path):
        raise FileNotFoundError(f"Stage 1 model not found at {stage1_path}")
    if not os.path.exists(stage2_path):
        raise FileNotFoundError(f"Stage 2 model not found at {stage2_path}")
        
    print(f"Loading Stage 1 model from {stage1_path}...")
    # Load with compile=False to avoid issues with custom layers/optimizers if any
    stage1_model = tf.keras.models.load_model(stage1_path, compile=False)
    
    # Build the grad model based on stage 1
    grad_model = build_grad_model(stage1_model)
    
    print(f"Loading Stage 2 model from {stage2_path}...")
    stage2_model = tf.keras.models.load_model(stage2_path, compile=False)
    
    return stage1_model, stage2_model, grad_model
