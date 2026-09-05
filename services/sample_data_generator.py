import numpy as np
import cv2
import pandas as pd
from pathlib import Path

def draw_synthetic_brain_slice(size=256, slice_type="axial"):
    """
    Generates an anatomically structured synthetic 2D brain MRI slice (T1-weighted).
    Features: skull/scalp, CSF dark rim, cortex gray matter, deep white matter, ventricles.
    """
    img = np.zeros((size, size), dtype=np.float32)
    center = (size // 2, size // 2)
    
    # Outer Skull / Scalp (elliptical)
    skull_axes = (int(size * 0.40), int(size * 0.45))
    cv2.ellipse(img, center, skull_axes, 0, 0, 360, 0.35, -1)
    
    # Bone marrow / diploe (thin bright layer)
    bone_axes = (int(size * 0.38), int(size * 0.43))
    cv2.ellipse(img, center, bone_axes, 0, 0, 360, 0.75, 2)
    
    # Subdural CSF / Dura (dark band)
    csf_outer_axes = (int(size * 0.36), int(size * 0.41))
    cv2.ellipse(img, center, csf_outer_axes, 0, 0, 360, 0.10, 2)
    
    # Brain Parenchyma - Gray Matter (Cortex)
    cortex_axes = (int(size * 0.34), int(size * 0.39))
    cv2.ellipse(img, center, cortex_axes, 0, 0, 360, 0.55, -1)
    
    # White Matter Core
    wm_axes = (int(size * 0.28), int(size * 0.33))
    cv2.ellipse(img, center, wm_axes, 0, 0, 360, 0.85, -1)
    
    # Add gyri/sulci convolutions using Perlin-like spatial sine/cosine modulations
    y, x = np.ogrid[:size, :size]
    gyri = np.sin(x / 4.0) * np.cos(y / 4.0) * 0.12
    # Apply convolutions inside cortex region
    mask_cortex = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(mask_cortex, center, cortex_axes, 0, 0, 360, 255, -1)
    img[mask_cortex > 0] += gyri[mask_cortex > 0]
    
    # Lateral Ventricles (CSF - dark central butterfly shape)
    left_ventricle = (int(center[0] - size * 0.06), int(center[1] - size * 0.04))
    right_ventricle = (int(center[0] + size * 0.06), int(center[1] - size * 0.04))
    cv2.ellipse(img, left_ventricle, (int(size * 0.04), int(size * 0.12)), 15, 0, 360, 0.12, -1)
    cv2.ellipse(img, right_ventricle, (int(size * 0.04), int(size * 0.12)), -15, 0, 360, 0.12, -1)
    
    # Third Ventricle (narrow midline slit)
    cv2.ellipse(img, (center[0], center[1]), (int(size * 0.015), int(size * 0.08)), 0, 0, 360, 0.08, -1)
    
    # Thalamus / Basal Ganglia intermediate intensity
    cv2.ellipse(img, (int(center[0] - size * 0.08), center[1]), (int(size * 0.06), int(size * 0.06)), 0, 0, 360, 0.62, -1)
    cv2.ellipse(img, (int(center[0] + size * 0.08), center[1]), (int(size * 0.06), int(size * 0.06)), 0, 0, 360, 0.62, -1)
    
    # Smooth anatomical transitions
    img = cv2.GaussianBlur(img, (3, 3), 0.5)
    img = np.clip(img, 0.0, 1.0)
    return img

def apply_artifact(img_clean, artifact_type="none", severity=0.5):
    """
    Applies controlled, mathematically grounded neuroimaging degradations:
    - none (clean baseline)
    - motion (k-space phase modulation ghosting)
    - noise (Rician/Gaussian high-frequency background noise)
    - blur (Laplacian Gaussian smoothing)
    - bias_field (B1 RF field inhomogeneity low-frequency shading)
    - crop (FOV cutoff of cortex)
    - missing_slice (dropped acquisition)
    """
    size = img_clean.shape[0]
    img = img_clean.copy()
    
    if artifact_type == "none":
        # Realistic slight baseline scanner noise
        noise = np.random.normal(0, 0.015, img.shape)
        return np.clip(img + noise, 0.0, 1.0)
        
    elif artifact_type == "motion":
        # Motion ghosting: periodic phase error in k-space (Fourier domain)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        # Add phase shifts every N lines along phase encoding direction
        rows, cols = img.shape
        phase_mod = np.ones((rows, cols), dtype=np.complex64)
        for r in range(rows):
            if r % 16 == 0:
                phase_mod[r, :] = np.exp(1j * np.pi * severity * 1.5)
        fshift_ghost = fshift * phase_mod
        f_ishift = np.fft.ifftshift(fshift_ghost)
        img_back = np.abs(np.fft.ifft2(f_ishift))
        return np.clip(img_back.astype(np.float32), 0.0, 1.0)
        
    elif artifact_type == "noise":
        # Low SNR Rician/Gaussian noise
        sigma = 0.08 * severity + 0.04
        noise_real = np.random.normal(0, sigma, img.shape)
        noise_imag = np.random.normal(0, sigma, img.shape)
        # Rician distribution: sqrt((I + N_r)^2 + N_i^2)
        noisy = np.sqrt((img + noise_real)**2 + noise_imag**2)
        return np.clip(noisy.astype(np.float32), 0.0, 1.0)
        
    elif artifact_type == "blur":
        # Heavy subject micro-motion or low reconstruction filter
        ksize = int(9 * severity) * 2 + 1
        blurred = cv2.GaussianBlur(img, (ksize, ksize), 3.0 * severity)
        return np.clip(blurred, 0.0, 1.0)
        
    elif artifact_type == "bias_field":
        # Low-frequency polynomial gradient across coil array
        y, x = np.ogrid[:size, :size]
        gradient = (x / float(size)) * 0.8 + (y / float(size)) * 0.4
        bias_pattern = 1.0 + (gradient - 0.6) * severity
        biased = img * bias_pattern
        return np.clip(biased.astype(np.float32), 0.0, 1.0)
        
    elif artifact_type == "crop":
        # Truncation artifact (Field of view cutoff)
        cropped = img.copy()
        crop_px = int(size * 0.22 * severity)
        cropped[:, :crop_px] = 0.0 # Crop left temporal lobe
        return cropped
        
    elif artifact_type == "missing_slice":
        # Dropped / blank / corrupted slice
        return np.zeros((size, size), dtype=np.float32)
        
    return img

def generate_sample_dataset(target_dir):
    """
    Creates a full realistic neuroimaging test cohort with diverse quality classes:
    - 6 Good Scans
    - 4 Acceptable Scans
    - 3 Poor Scans (Motion, Blur, Noise)
    - 2 Unusable Scans (Severe Truncation, Missing Slice)
    Generates PNG slice files and clinical metadata CSV.
    """
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    specs = [
        # Good (Clean 3T acquisitions)
        {"id": "SUBJ_01_clean_T1w", "artifact": "none", "severity": 0.0, "age": 28, "sex": "F", "vendor": "Siemens Prisma 3T", "site": "Site_A (MGH)", "diag": "Control"},
        {"id": "SUBJ_02_clean_T1w", "artifact": "none", "severity": 0.0, "age": 34, "sex": "M", "vendor": "Siemens Prisma 3T", "site": "Site_A (MGH)", "diag": "Control"},
        {"id": "SUBJ_03_clean_T1w", "artifact": "none", "severity": 0.0, "age": 42, "sex": "F", "vendor": "GE Discovery 3T", "site": "Site_B (UCLA)", "diag": "Patient_A"},
        {"id": "SUBJ_04_clean_T1w", "artifact": "none", "severity": 0.0, "age": 51, "sex": "M", "vendor": "Siemens Prisma 3T", "site": "Site_A (MGH)", "diag": "Control"},
        {"id": "SUBJ_05_clean_T1w", "artifact": "none", "severity": 0.0, "age": 63, "sex": "F", "vendor": "Philips Ingenia 3T", "site": "Site_C (Oxford)", "diag": "Patient_B"},
        {"id": "SUBJ_06_clean_T1w", "artifact": "none", "severity": 0.0, "age": 22, "sex": "M", "vendor": "Siemens Prisma 3T", "site": "Site_A (MGH)", "diag": "Control"},
        
        # Acceptable (Minor noise, slight field bias)
        {"id": "SUBJ_07_mild_noise_T1w", "artifact": "noise", "severity": 0.35, "age": 59, "sex": "F", "vendor": "Siemens Skyra 1.5T", "site": "Site_C (Oxford)", "diag": "Control"},
        {"id": "SUBJ_08_mild_bias_T1w", "artifact": "bias_field", "severity": 0.40, "age": 45, "sex": "M", "vendor": "GE Discovery 3T", "site": "Site_B (UCLA)", "diag": "Patient_A"},
        {"id": "SUBJ_09_mild_blur_T1w", "artifact": "blur", "severity": 0.30, "age": 70, "sex": "F", "vendor": "Philips Ingenia 3T", "site": "Site_C (Oxford)", "diag": "Patient_B"},
        {"id": "SUBJ_10_acceptable_T1w", "artifact": "none", "severity": 0.0, "age": 39, "sex": "M", "vendor": "Siemens Prisma 3T", "site": "Site_A (MGH)", "diag": "Control"},
        
        # Poor (Substantial motion ghosting, heavy noise, strong blur)
        {"id": "SUBJ_11_motion_ghost_T1w", "artifact": "motion", "severity": 0.75, "age": 67, "sex": "M", "vendor": "Siemens Skyra 1.5T", "site": "Site_C (Oxford)", "diag": "Patient_B"},
        {"id": "SUBJ_12_heavy_noise_T1w", "artifact": "noise", "severity": 0.85, "age": 72, "sex": "F", "vendor": "GE Discovery 3T", "site": "Site_B (UCLA)", "diag": "Patient_A"},
        {"id": "SUBJ_13_heavy_blur_T1w", "artifact": "blur", "severity": 0.80, "age": 55, "sex": "M", "vendor": "Philips Ingenia 3T", "site": "Site_C (Oxford)", "diag": "Control"},
        
        # Unusable (Severe FOV truncation, missing slice)
        {"id": "SUBJ_14_truncated_fov_T1w", "artifact": "crop", "severity": 0.90, "age": 48, "sex": "F", "vendor": "Siemens Prisma 3T", "site": "Site_A (MGH)", "diag": "Patient_A"},
        {"id": "SUBJ_15_corrupted_empty_T1w", "artifact": "missing_slice", "severity": 1.0, "age": 61, "sex": "M", "vendor": "GE Discovery 3T", "site": "Site_B (UCLA)", "diag": "Patient_B"}
    ]
    
    metadata_records = []
    
    for s in specs:
        filename = f"{s['id']}.png"
        file_path = target_path / filename
        
        # Generate slice
        clean_slice = draw_synthetic_brain_slice(size=256)
        degraded = apply_artifact(clean_slice, artifact_type=s['artifact'], severity=s['severity'])
        
        # Save as 8-bit PNG
        img_uint8 = (degraded * 255).astype(np.uint8)
        cv2.imwrite(str(file_path), img_uint8)
        
        # Metadata entry
        field = 1.5 if "1.5T" in s['vendor'] else 3.0
        metadata_records.append({
            'filename': filename,
            'subject_id': s['id'].split('_')[0] + "_" + s['id'].split('_')[1],
            'age': s['age'],
            'sex': s['sex'],
            'diagnosis': s['diag'],
            'scanner_vendor': s['vendor'],
            'magnetic_field_t': field,
            'voxel_spacing': '1.0 x 1.0 x 1.0 mm',
            'slice_thickness_mm': 1.0,
            'tr_ms': 2300,
            'te_ms': 2.98,
            'site_id': s['site']
        })
        
    csv_path = target_path / "dataset_metadata.csv"
    pd.DataFrame(metadata_records).to_csv(str(csv_path), index=False)
    
    return {
        'total_scans': len(specs),
        'directory': str(target_path),
        'csv_path': str(csv_path)
    }
