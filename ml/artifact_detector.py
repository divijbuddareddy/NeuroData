import numpy as np
import cv2
from ml.preprocessing import load_image_as_grayscale, segment_brain_mask

def calculate_blur_score(img_float):
    """
    Computes blur score using variance of Laplacian and high-frequency FFT power.
    Higher value indicates sharper image; lower indicates severe blur.
    Typical sharp brain MRI: blur_score > 150. Blurry / smoothed: < 60.
    """
    img_uint8 = (img_float * 255).astype(np.uint8)
    laplacian_var = float(cv2.Laplacian(img_uint8, cv2.CV_64F).var())
    
    # FFT spectrum high-frequency ratio
    rows, cols = img_float.shape
    crow, ccol = rows // 2, cols // 2
    f = np.fft.fft2(img_float)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-7)
    
    # High frequency annular mask
    r_outer = min(crow, ccol)
    r_inner = r_outer // 3
    y, x = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((x - ccol)**2 + (y - crow)**2)
    high_freq_mask = (dist_from_center >= r_inner) & (dist_from_center <= r_outer)
    
    high_freq_energy = float(np.mean(magnitude_spectrum[high_freq_mask])) if np.any(high_freq_mask) else 0.0
    
    return {
        'laplacian_variance': round(laplacian_var, 2),
        'high_freq_energy': round(high_freq_energy, 2),
        'blur_metric': round(laplacian_var, 2),
        'is_blurry': laplacian_var < 75.0
    }

def calculate_snr_cnr(img_float, brain_mask=None):
    """
    Computes Signal-to-Noise Ratio (SNR) and Contrast-to-Noise Ratio (CNR).
    SNR = Mean(Foreground) / Std(Background Noise)
    CNR = |Mean(White Matter) - Mean(Gray Matter)| / Std(Background Noise)
    """
    if brain_mask is None:
        brain_mask = segment_brain_mask(img_float)
        
    bg_mask = (brain_mask == 0)
    # Exclude boundary air artifacts, take corners of background
    h, w = img_float.shape
    corner_size = 20
    bg_corners = np.zeros_like(bg_mask, dtype=bool)
    bg_corners[:corner_size, :corner_size] = True
    bg_corners[:corner_size, -corner_size:] = True
    bg_corners[-corner_size:, :corner_size] = True
    bg_corners[-corner_size:, -corner_size:] = True
    bg_noise_pixels = img_float[bg_corners & bg_mask]
    
    if len(bg_noise_pixels) < 10:
        bg_noise_pixels = img_float[bg_mask]
        
    bg_std = float(np.std(bg_noise_pixels)) if len(bg_noise_pixels) > 0 else 0.05
    bg_std = max(bg_std, 1e-4)
    
    fg_pixels = img_float[brain_mask > 0]
    if len(fg_pixels) > 0:
        fg_mean = float(np.mean(fg_pixels))
        snr = fg_mean / bg_std
        
        # Approximate GM and WM using Otsu threshold on brain parenchyma
        fg_uint8 = (fg_pixels * 255).astype(np.uint8)
        ret, _ = cv2.threshold(fg_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold_val = ret / 255.0
        
        gm_pixels = fg_pixels[fg_pixels < threshold_val]
        wm_pixels = fg_pixels[fg_pixels >= threshold_val]
        
        gm_mean = float(np.mean(gm_pixels)) if len(gm_pixels) > 0 else fg_mean * 0.7
        wm_mean = float(np.mean(wm_pixels)) if len(wm_pixels) > 0 else fg_mean * 1.3
        
        cnr = abs(wm_mean - gm_mean) / bg_std
    else:
        snr = 0.0
        cnr = 0.0
        
    return {
        'snr': round(snr, 2),
        'cnr': round(cnr, 2),
        'bg_noise_std': round(bg_std, 4),
        'is_low_snr': snr < 12.0
    }

def detect_motion_ghosting(img_float, brain_mask=None):
    """
    Detects motion-related phase-encoding ghosting and ringing in background air.
    Ghosting manifests as periodic replicas of brain tissue shifted into background columns/rows.
    """
    if brain_mask is None:
        brain_mask = segment_brain_mask(img_float)
        
    bg_mask = (brain_mask == 0)
    h, w = img_float.shape
    
    # Calculate background column-wise and row-wise profile variance
    # Sample background air strictly from the outer image margins (top/bottom 16 pixels)
    top_air = img_float[:16, :]
    bottom_air = img_float[-16:, :]
    
    air_profiles = np.vstack([top_air, bottom_air])
    col_variance = float(np.var(np.mean(air_profiles, axis=0)))
    
    # Energy in background relative to foreground
    fg_pixels = img_float[brain_mask > 0]
    bg_pixels = img_float[bg_mask]
    
    fg_energy = float(np.mean(fg_pixels**2)) if len(fg_pixels) > 0 else 1.0
    # Use background corner regions to avoid skull boundary bleed
    bg_energy = float(np.mean(air_profiles**2)) if len(air_profiles) > 0 else 0.0
    ghosting_ratio = bg_energy / (fg_energy + 1e-6)
    
    # Motion ghosting causes periodic striping across columns in background air
    motion_score = min(1.0, (col_variance * 120.0 + ghosting_ratio * 1.5))
    
    return {
        'motion_score': round(float(motion_score), 3),
        'ghosting_ratio': round(float(ghosting_ratio), 4),
        'has_motion_artifact': motion_score > 0.35
    }

def detect_intensity_inhomogeneity(img_float, brain_mask=None):
    """
    Detects B1 field inhomogeneity / RF coil bias field across brain parenchyma.
    Measures spatial gradient of low-pass filtered foreground.
    """
    if brain_mask is None:
        brain_mask = segment_brain_mask(img_float)
        
    fg_pixels = img_float[brain_mask > 0]
    if len(fg_pixels) == 0:
        return {'bias_score': 0.0, 'has_bias_field': False}
        
    # Large kernel Gaussian blur to isolate low-frequency intensity bias
    low_freq = cv2.GaussianBlur(img_float, (45, 45), 0)
    low_freq_fg = low_freq[brain_mask > 0]
    
    cv_bias = float(np.std(low_freq_fg) / (np.mean(low_freq_fg) + 1e-6))
    
    return {
        'bias_score': round(cv_bias, 3),
        'has_bias_field': cv_bias > 0.40
    }

def detect_cropping_and_missing_slices(img_float, brain_mask=None):
    """
    Detects if brain anatomy is truncated/cropped by the Field of View (FOV) edges,
    or if the slice is empty / missing / corrupted.
    """
    if brain_mask is None:
        brain_mask = segment_brain_mask(img_float)
        
    h, w = img_float.shape
    fg_count = np.sum(brain_mask > 0)
    
    # Missing slice / blank image check
    if fg_count < (h * w * 0.02) or np.var(img_float) < 1e-5:
        return {
            'is_cropped': False,
            'is_missing_slice': True,
            'reason': 'Slice appears blank, zero-variance or near-empty'
        }
        
    # Check if foreground mask touches border pixels with significant area (> 8 px)
    top_touch = np.sum(brain_mask[0, :] > 0) > 8
    bottom_touch = np.sum(brain_mask[h-1, :] > 0) > 8
    left_touch = np.sum(brain_mask[:, 0] > 0) > 8
    right_touch = np.sum(brain_mask[:, w-1] > 0) > 8
    
    is_cropped = bool(top_touch or bottom_touch or left_touch or right_touch)
    
    return {
        'is_cropped': is_cropped,
        'is_missing_slice': False,
        'touch_edges': {
            'top': bool(top_touch),
            'bottom': bool(bottom_touch),
            'left': bool(left_touch),
            'right': bool(right_touch)
        }
    }

def analyze_scan_artifacts(image_path):
    """
    Master analyzer running all Module 2 artifact routines.
    """
    img_float, img_uint8 = load_image_as_grayscale(image_path)
    brain_mask = segment_brain_mask(img_float)
    
    blur_res = calculate_blur_score(img_float)
    snr_res = calculate_snr_cnr(img_float, brain_mask)
    motion_res = detect_motion_ghosting(img_float, brain_mask)
    bias_res = detect_intensity_inhomogeneity(img_float, brain_mask)
    crop_res = detect_cropping_and_missing_slices(img_float, brain_mask)
    
    return {
        'blur': blur_res,
        'snr_cnr': snr_res,
        'motion': motion_res,
        'bias': bias_res,
        'geometry': crop_res,
        'composite_artifact_penalty': round(
            (1.0 if blur_res['is_blurry'] else 0.0) * 0.25 +
            (1.0 if snr_res['is_low_snr'] else 0.0) * 0.25 +
            (motion_res['motion_score']) * 0.30 +
            (1.0 if crop_res['is_cropped'] else 0.0) * 0.20 +
            (1.0 if crop_res['is_missing_slice'] else 0.0) * 1.0, 3
        )
    }
