import numpy as np
import cv2
from PIL import Image
import os
from pathlib import Path

def load_image_as_grayscale(image_path, target_size=(224, 224)):
    """
    Loads an image file (PNG, JPG, TIFF, etc.) and converts it to standardized grayscale
    and resized numpy float32 array in range [0, 1].
    """
    path_str = str(image_path)
    img = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        # Fallback to PIL
        try:
            pil_img = Image.open(path_str).convert('L')
            img = np.array(pil_img)
        except Exception as e:
            raise ValueError(f"Could not load image at {image_path}: {e}")
            
    # Resize to standard dimensions
    if target_size is not None:
        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    else:
        img_resized = img

    # Normalize to [0, 1] float
    img_float = img_resized.astype(np.float32) / 255.0
    return img_float, img_resized

def segment_brain_mask(img_float, threshold_ratio=0.12):
    """
    Segments the brain tissue parenchyma from background noise using Otsu or adaptive thresholding.
    Returns binary mask (1 for brain tissue, 0 for background).
    """
    img_uint8 = (img_float * 255).astype(np.uint8)
    
    # Gaussian blur to reduce high frequency noise before thresholding
    blurred = cv2.GaussianBlur(img_uint8, (5, 5), 0)
    
    # Otsu thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological closing to fill small internal holes (ventricles/sulci)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find largest connected component (the brain)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels > 1:
        # Background is 0, largest non-zero is brain
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        brain_mask = (labels == largest_label).astype(np.uint8)
    else:
        brain_mask = (mask > 0).astype(np.uint8)
        
    return brain_mask

def standardize_intensity_zscore(img_float, brain_mask=None):
    """
    Applies standard neuroimaging Z-score intensity standardization
    computed inside the brain tissue mask.
    """
    if brain_mask is None:
        brain_mask = segment_brain_mask(img_float)
        
    foreground_pixels = img_float[brain_mask > 0]
    if len(foreground_pixels) == 0:
        return img_float
        
    mean_val = np.mean(foreground_pixels)
    std_val = np.std(foreground_pixels) + 1e-7
    
    zscored = (img_float - mean_val) / std_val
    # Clip extreme outliers (-3 to 3 sigma) and scale back to [0, 1]
    clipped = np.clip(zscored, -3.0, 3.0)
    normalized = (clipped + 3.0) / 6.0
    return normalized
