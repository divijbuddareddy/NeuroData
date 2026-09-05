import numpy as np
import cv2
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ml.preprocessing import load_image_as_grayscale, segment_brain_mask

def extract_scan_feature_vector(image_path):
    """
    Extracts numerical feature vector capturing spatial, texture, frequency,
    and intensity characteristics of an MRI slice.
    """
    img_float, img_uint8 = load_image_as_grayscale(image_path)
    brain_mask = segment_brain_mask(img_float)
    
    fg_pixels = img_float[brain_mask > 0]
    if len(fg_pixels) == 0:
        fg_pixels = img_float.flatten()
        
    # 1. Intensity distribution statistics
    mean_val = float(np.mean(fg_pixels))
    std_val = float(np.std(fg_pixels))
    p10 = float(np.percentile(fg_pixels, 10))
    p50 = float(np.percentile(fg_pixels, 50))
    p90 = float(np.percentile(fg_pixels, 90))
    skewness = float(np.mean(((fg_pixels - mean_val) / (std_val + 1e-7))**3))
    kurtosis = float(np.mean(((fg_pixels - mean_val) / (std_val + 1e-7))**4))
    
    # 2. Geometric & Spatial moments
    area_ratio = float(np.sum(brain_mask > 0) / (brain_mask.shape[0] * brain_mask.shape[1]))
    moments = cv2.moments(brain_mask)
    hu_moments = cv2.HuMoments(moments).flatten()
    log_hu = [-np.sign(h) * np.log10(abs(h) + 1e-12) for h in hu_moments[:3]]
    
    # 3. Texture / Gradient energy
    sobelx = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    mean_grad = float(np.mean(grad_mag[brain_mask > 0])) if np.any(brain_mask > 0) else float(np.mean(grad_mag))
    
    # 4. Frequency power
    f = np.fft.fft2(img_float)
    mag_spec = np.abs(np.fft.fftshift(f))
    freq_power = float(np.mean(mag_spec))
    
    feature_vector = np.array([
        mean_val, std_val, p10, p50, p90, skewness, kurtosis,
        area_ratio, log_hu[0], log_hu[1], log_hu[2],
        mean_grad, freq_power
    ], dtype=np.float32)
    
    # Replace potential NaNs or Infs
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1.0, neginf=-1.0)
    return feature_vector

def detect_cohort_anomalies(image_paths, contamination=0.15):
    """
    Fits Isolation Forest and PCA over a list of MRI image paths.
    Returns:
      - is_outlier list (bool)
      - anomaly_scores list (float 0 to 1, higher = more anomalous)
      - pca_coords list of tuples (x, y)
    """
    if not image_paths:
        return []
        
    features = [extract_scan_feature_vector(p) for p in image_paths]
    X = np.array(features)
    
    n_samples = len(X)
    scaler = StandardScaler()
    
    if n_samples < 3:
        # Not enough samples for statistical isolation, return neutral
        return [{
            'path': str(p),
            'is_outlier': False,
            'outlier_score': 0.1,
            'pca_x': 0.0,
            'pca_y': 0.0
        } for p in image_paths]
        
    X_scaled = scaler.fit_transform(X)
    
    # Isolation Forest
    actual_contamination = min(contamination, max(0.05, 1.0 / n_samples))
    iso = IsolationForest(
        n_estimators=100,
        contamination=actual_contamination,
        random_state=42
    )
    preds = iso.fit_predict(X_scaled)  # 1 for inlier, -1 for outlier
    raw_scores = iso.score_samples(X_scaled)  # Lower is more abnormal
    
    # Normalize anomaly score to [0, 1] range (1.0 being extreme outlier)
    min_s, max_s = np.min(raw_scores), np.max(raw_scores)
    if max_s > min_s:
        normalized_anomaly_scores = 1.0 - ((raw_scores - min_s) / (max_s - min_s))
    else:
        normalized_anomaly_scores = np.zeros(n_samples)
        
    # PCA 2D embedding for cluster visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    results = []
    for i, p in enumerate(image_paths):
        is_out = (preds[i] == -1)
        results.append({
            'path': str(p),
            'is_outlier': bool(is_out),
            'outlier_score': round(float(normalized_anomaly_scores[i]), 3),
            'pca_x': round(float(X_pca[i, 0]), 3),
            'pca_y': round(float(X_pca[i, 1]), 3)
        })
        
    return results
