import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from pathlib import Path
from ml.preprocessing import load_image_as_grayscale
from ml.quality_model import quality_classifier, QUALITY_CLASSES

def generate_gradcam_heatmap(image_path, target_class=None, output_dir=None):
    """
    Generates Grad-CAM activation heatmap overlay on MRI slice.
    Highlights regions that led the model to assign the quality classification score
    (e.g., motion distortion bands, ringing, blur, or slice truncations).
    """
    img_float, img_uint8 = load_image_as_grayscale(image_path, target_size=(224, 224))
    
    # Forward pass with gradients enabled
    model = quality_classifier
    model.eval()
    
    input_tensor = torch.from_numpy(img_float).unsqueeze(0).unsqueeze(0).float()
    input_tensor.requires_grad = True
    
    # Forward
    logits = model(input_tensor)
    
    if target_class is None:
        target_class = int(torch.argmax(logits, dim=1).item())
        
    score = logits[0, target_class]
    
    # Backward to get gradients
    model.zero_grad()
    score.backward(retain_graph=True)
    
    gradients = model.gradients
    activations = model.activations
    
    if gradients is not None and activations is not None:
        # Global average pooling of gradients
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
        # Weight activations by gradients
        for i in range(activations.size(1)):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=1).squeeze().detach().numpy()
        heatmap = np.maximum(heatmap, 0) # ReLU
        max_h = np.max(heatmap)
        if max_h > 0:
            heatmap = heatmap / max_h
    else:
        # Fallback to high-gradient edge saliency if gradients are zero
        sobelx = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=3)
        heatmap = np.sqrt(sobelx**2 + sobely**2)
        heatmap = heatmap / (np.max(heatmap) + 1e-6)

    # Resize heatmap to original slice size
    heatmap_resized = cv2.resize(heatmap, (img_uint8.shape[1], img_uint8.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    # Color map
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Convert original grayscale to BGR for overlay
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
    
    # Blend: 60% original image + 40% heatmap
    overlay = cv2.addWeighted(img_bgr, 0.65, colored_heatmap, 0.35, 0)
    
    # Save output files if output_dir provided
    preview_filename = None
    gradcam_filename = None
    
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        stem = Path(image_path).stem
        preview_filename = f"{stem}_preview.png"
        gradcam_filename = f"{stem}_gradcam.png"
        
        cv2.imwrite(str(out_path / preview_filename), img_bgr)
        cv2.imwrite(str(out_path / gradcam_filename), overlay)
        
    return {
        'preview_filename': preview_filename,
        'gradcam_filename': gradcam_filename,
        'target_class_name': QUALITY_CLASSES[target_class] if target_class < len(QUALITY_CLASSES) else "Unknown",
        'max_activation': float(np.max(heatmap_resized))
    }
