import numpy as np
import tensorflow as tf
import cv2

class GradCAMPlusPlus:
    def __init__(self, model):
        """
        Initialize Grad-CAM++ with a model.
        The model should be a multi-output model returning [conv_output, predictions].
        """
        self.model = model

    def generate_heatmap(self, image, class_idx):
        """Returns a soft float [0,1] heatmap for Stage 2 attention."""
        # Ensure image is float32 and normalized
        if image.max() > 1.0:
            image = image.astype('float32') / 255.0

        # Convert to tensor
        img_array = np.expand_dims(image, axis=0)
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

        # Use GradientTape to compute gradients
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            # Forward pass
            conv_outputs, predictions = self.model(img_tensor)
            # Get the score for the predicted class
            class_channel = predictions[:, class_idx]

        # Compute gradients of the class output with respect to the feature maps
        grads = tape.gradient(class_channel, conv_outputs)

        if grads is None:
            print("⚠️ Warning: No gradients computed, returning uniform heatmap")
            return np.ones((image.shape[0], image.shape[1]))

        # Grad-CAM++ computation
        conv_outputs_np = conv_outputs[0].numpy()
        grads_np = grads[0].numpy()

        # Calculate alpha weights
        numerator = grads_np
        denominator = 2.0 * grads_np + np.sum(conv_outputs_np * grads_np, axis=(0, 1), keepdims=True)
        alpha = numerator / (denominator + 1e-7)

        # Calculate channel-wise weights
        weights = np.sum(alpha * np.maximum(grads_np, 0), axis=(0, 1))

        # Generate heatmap
        heatmap = np.sum(weights * conv_outputs_np, axis=-1)
        heatmap = np.maximum(heatmap, 0)  # ReLU

        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        else:
            print("⚠️ Warning: Zero heatmap, returning uniform heatmap")
            return np.ones((image.shape[0], image.shape[1]))

        # Return soft continuous heatmap resized to image
        return cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    def generate_mask(self, image, class_idx, threshold=0.5):
        """Generate Grad-CAM++ binary mask based on heatmap threshold"""
        heatmap = self.generate_heatmap(image, class_idx)
        
        # Apply threshold
        mask = (heatmap > threshold).astype('float32')

        # Ensure mask has some content
        if mask.sum() < 10:
            threshold_val = np.percentile(heatmap, 70)
            mask = (heatmap > threshold_val).astype('float32')

        return mask

def overlay_heatmap(heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlays the heatmap on the original image.
    Heatmap should be a 2D array [0, 1].
    Image should be an RGB array [0, 255] or [0, 1].
    """
    # If image is [0, 1], convert to [0, 255]
    if image.max() <= 1.0:
        image = (image * 255).astype('uint8')
    
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert heatmap to color
    heatmap_color = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_color, colormap)
    
    # Convert image to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Superimpose
    output_image = cv2.addWeighted(image_bgr, 1 - alpha, heatmap_color, alpha, 0)
    
    # Convert back to RGB
    return cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
