import pandas as pd
import numpy as np

REQUIRED_METADATA_FIELDS = [
    'subject_id', 'age', 'sex', 'scanner_vendor', 'magnetic_field_t',
    'voxel_spacing', 'slice_thickness_mm', 'tr_ms', 'te_ms', 'site_id'
]

def analyze_metadata_dataframe(df):
    """
    Analyzes metadata DataFrame for completeness, bias, scanner heterogeneity,
    and potential research confounds.
    """
    if df is None or len(df) == 0:
        return {
            'completeness_pct': 0.0,
            'total_records': 0,
            'missing_fields': REQUIRED_METADATA_FIELDS,
            'warnings': ['No metadata CSV provided. Default neuroimaging tags assumed.'],
            'demographics': {'sex_distribution': {}, 'age_summary': {}},
            'scanner_distribution': {},
            'site_distribution': {}
        }
        
    total_records = len(df)
    warnings = []
    
    # 1. Missingness & Completeness
    present_fields = [f for f in REQUIRED_METADATA_FIELDS if f in df.columns]
    missing_critical_fields = [f for f in REQUIRED_METADATA_FIELDS if f not in df.columns]
    
    if missing_critical_fields:
        warnings.append(f"Missing critical metadata columns: {', '.join(missing_critical_fields)}")
        
    total_cells = total_records * len(REQUIRED_METADATA_FIELDS)
    filled_cells = 0
    for f in REQUIRED_METADATA_FIELDS:
        if f in df.columns:
            filled_cells += int(df[f].notna().sum())
            
    completeness_pct = round((filled_cells / total_cells) * 100.0, 1) if total_cells > 0 else 0.0
    
    # 2. Duplicate Check
    if 'subject_id' in df.columns:
        duplicate_count = int(df['subject_id'].duplicated().sum())
        if duplicate_count > 0:
            warnings.append(f"Detected {duplicate_count} duplicate Subject IDs (risk of train/test data leakage).")
            
    # 3. Sex Distribution & Parity
    sex_dist = {}
    if 'sex' in df.columns:
        sex_counts = df['sex'].astype(str).str.upper().value_counts().to_dict()
        sex_dist = {str(k): int(v) for k, v in sex_counts.items()}
        # Check imbalance
        if len(sex_dist) > 1:
            total_sex = sum(sex_dist.values())
            ratios = [v / total_sex for v in sex_dist.values()]
            if min(ratios) < 0.25:
                warnings.append("Severe demographic sex imbalance detected (< 25% minority representation).")
                
    # 4. Age Distribution
    age_summary = {}
    if 'age' in df.columns:
        age_clean = pd.to_numeric(df['age'], errors='coerce').dropna()
        if len(age_clean) > 0:
            age_summary = {
                'mean': round(float(age_clean.mean()), 1),
                'std': round(float(age_clean.std()), 1) if len(age_clean) > 1 else 0.0,
                'min': int(age_clean.min()),
                'max': int(age_clean.max()),
                'median': round(float(age_clean.median()), 1)
            }
            if age_clean.min() < 0 or age_clean.max() > 115:
                warnings.append("Anomalous age values detected (<0 or >115).")
                
    # 5. Scanner & Site Heterogeneity
    scanner_dist = {}
    if 'scanner_vendor' in df.columns:
        scanner_counts = df['scanner_vendor'].astype(str).value_counts().to_dict()
        scanner_dist = {str(k): int(v) for k, v in scanner_counts.items()}
        if len(scanner_dist) > 1:
            warnings.append(f"Multi-vendor scanner distribution detected ({len(scanner_dist)} vendors). Scanner harmonization may be required.")
            
    site_dist = {}
    if 'site_id' in df.columns:
        site_counts = df['site_id'].astype(str).value_counts().to_dict()
        site_dist = {str(k): int(v) for k, v in site_counts.items()}
        
    # 6. Field Strength Consistency
    if 'magnetic_field_t' in df.columns:
        fields = df['magnetic_field_t'].dropna().unique()
        if len(fields) > 1:
            warnings.append(f"Mixed magnetic field strengths detected ({list(fields)} T). Ensure intensity calibration across 1.5T and 3.0T scans.")
            
    return {
        'completeness_pct': completeness_pct,
        'total_records': total_records,
        'missing_fields': missing_critical_fields,
        'warnings': warnings,
        'demographics': {
            'sex_distribution': sex_dist,
            'age_summary': age_summary
        },
        'scanner_distribution': scanner_dist,
        'site_distribution': site_dist
    }

def synthesize_metadata_for_scans(filenames):
    """
    Generates realistic, standardized metadata when raw CSV is absent.
    """
    rows = []
    vendors = ["Siemens Prisma 3T", "GE Discovery 3T", "Philips Ingenia 3T", "Siemens Skyra 1.5T"]
    sites = ["Site_A (MGH)", "Site_B (UCLA)", "Site_C (Oxford)"]
    
    np.random.seed(42)
    for idx, fname in enumerate(filenames):
        sub_id = f"SUBJ_{1001 + idx:04d}"
        age = int(np.random.normal(48, 14))
        age = max(18, min(85, age))
        sex = "Female" if (idx % 2 == 0) else "Male"
        vendor = vendors[idx % len(vendors)]
        site = sites[idx % len(sites)]
        field = 1.5 if "1.5T" in vendor else 3.0
        
        rows.append({
            'filename': fname,
            'subject_id': sub_id,
            'age': age,
            'sex': sex,
            'scanner_vendor': vendor,
            'magnetic_field_t': field,
            'voxel_spacing': '1.0 x 1.0 x 1.0 mm',
            'slice_thickness_mm': 1.0,
            'tr_ms': 2300,
            'te_ms': 2.98,
            'site_id': site
        })
        
    return pd.DataFrame(rows)
