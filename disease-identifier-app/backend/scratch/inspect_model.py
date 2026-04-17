import tensorflow as tf
import os

model_path = r'c:\Projects\plant-disease-fastapi\disease-identifier-app\models\rice_stage2_finetuned_final.h5'

if os.path.exists(model_path):
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"Model Summary:")
        model.summary()
        
        print("\nTop-level Layers:")
        for i, layer in enumerate(model.layers):
            print(f"Layer {i}: {layer.__class__.__name__} - {layer.name}")
            if isinstance(layer, tf.keras.Model):
                print(f"  --- Inner layers of {layer.name} ---")
                for j, inner_layer in enumerate(layer.layers[-5:]): # Last 5 inner layers
                    print(f"    Inner Layer {j}: {inner_layer.__class__.__name__} - {inner_layer.name}")
    except Exception as e:
        print(f"Error: {e}")
else:
    print(f"File not found: {model_path}")
