import numpy as np

def compute_dataset_feature_importance(readiness_breakdown):
    """
    Computes percentage weights and impact directions (positive vs negative impact)
    for each key readiness dimension.
    """
    if not readiness_breakdown:
        return []
        
    factors = [
        {
            'name': 'Image Quality & SNR',
            'score': readiness_breakdown.get('image_quality_score', 0),
            'max': readiness_breakdown.get('image_quality_max', 35.0),
            'description': 'Proportion of scans rated Good or Acceptable without severe noise or blur.'
        },
        {
            'name': 'Motion & Distortion Control',
            'score': readiness_breakdown.get('snr_motion_score', 0),
            'max': readiness_breakdown.get('snr_motion_max', 20.0),
            'description': 'Absence of k-space motion ghosting, phase-ringing, and intensity bias.'
        },
        {
            'name': 'Clinical Metadata Completeness',
            'score': readiness_breakdown.get('metadata_score', 0),
            'max': readiness_breakdown.get('metadata_max', 20.0),
            'description': 'Completeness of essential acquisition tags (TR/TE, voxel size, age, sex).'
        },
        {
            'name': 'Cohort Anomaly & Outliers',
            'score': readiness_breakdown.get('outlier_score', 0),
            'max': readiness_breakdown.get('outlier_max', 15.0),
            'description': 'Absence of extreme statistical or anatomical outliers in feature space.'
        },
        {
            'name': 'Multi-Site & Scanner Parity',
            'score': readiness_breakdown.get('consistency_score', 0),
            'max': readiness_breakdown.get('consistency_max', 10.0),
            'description': 'Harmonization and balance across MRI hardware vendors and field strengths.'
        }
    ]
    
    for f in factors:
        f['pct_achieved'] = round((f['score'] / f['max']) * 100.0, 1) if f['max'] > 0 else 0
        f['gap'] = round(f['max'] - f['score'], 1)
        f['status'] = 'Optimal' if f['pct_achieved'] >= 85 else ('Warning' if f['pct_achieved'] >= 60 else 'Critical')
        
    return factors
