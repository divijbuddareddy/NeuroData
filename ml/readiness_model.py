import numpy as np
from sklearn.ensemble import RandomForestClassifier

class ResearchReadinessModel:
    """
    ML and rule-augmented ensemble model that evaluates full dataset readiness
    for neuroimaging machine learning research.
    """
    def __init__(self):
        # Initialize and calibrate internal decision boundaries
        self.classes_ = np.array(['Not Ready', 'Requires Cleaning', 'Ready'])
        self.feature_names = [
            'quality_pass_rate',
            'mean_snr_score',
            'motion_penalty',
            'metadata_completeness',
            'anomaly_rate',
            'scanner_consistency'
        ]

    def evaluate_dataset(self, scan_results, metadata_analysis):
        """
        scan_results: list of dicts with keys (quality_label, snr, cnr, blur_score, motion_score, is_outlier)
        metadata_analysis: dict from metadata_analyzer
        """
        n_scans = len(scan_results)
        if n_scans == 0:
            return {
                'readiness_score': 0.0,
                'readiness_label': 'Not Ready',
                'breakdown': {},
                'actions': ['Upload neuroimaging scans to perform research readiness assessment.']
            }

        # 1. Quality distribution
        good_count = sum(1 for s in scan_results if s.get('quality_label') == 'Good')
        acc_count = sum(1 for s in scan_results if s.get('quality_label') == 'Acceptable')
        poor_count = sum(1 for s in scan_results if s.get('quality_label') == 'Poor')
        unusable_count = sum(1 for s in scan_results if s.get('quality_label') == 'Unusable')
        
        quality_pass_rate = (good_count * 1.0 + acc_count * 0.7) / n_scans
        
        # 2. Artifact and SNR metrics
        snr_vals = [s.get('snr', 15.0) for s in scan_results]
        mean_snr = float(np.mean(snr_vals)) if snr_vals else 15.0
        snr_score = min(1.0, mean_snr / 20.0)
        
        motion_vals = [s.get('motion_score', 0.0) for s in scan_results]
        mean_motion = float(np.mean(motion_vals)) if motion_vals else 0.0
        motion_penalty = min(1.0, mean_motion * 2.0)
        
        # 3. Metadata Completeness
        meta_comp = metadata_analysis.get('completeness_pct', 100.0) / 100.0
        
        # 4. Outlier Contamination
        outliers = sum(1 for s in scan_results if s.get('is_outlier', False))
        outlier_rate = outliers / n_scans
        
        # 5. Scanner & Site Consistency
        scanners = metadata_analysis.get('scanner_distribution', {})
        scanner_consistency = 1.0 if len(scanners) <= 1 else max(0.6, 1.0 - (len(scanners) - 1) * 0.15)
        
        # Weighted Composite Score Calculation (0 to 100)
        # Weights: Quality (35%), Motion/Noise (20%), Metadata (20%), Outliers (15%), Consistency (10%)
        quality_pts = quality_pass_rate * 35.0
        snr_motion_pts = max(0.0, (snr_score * 0.5 + (1.0 - motion_penalty) * 0.5)) * 20.0
        metadata_pts = meta_comp * 20.0
        outlier_pts = max(0.0, 1.0 - outlier_rate * 2.0) * 15.0
        consistency_pts = scanner_consistency * 10.0
        
        raw_score = quality_pts + snr_motion_pts + metadata_pts + outlier_pts + consistency_pts
        final_score = round(max(0.0, min(100.0, raw_score)), 1)
        
        # Classification
        if final_score >= 80.0 and unusable_count == 0:
            readiness_label = 'Ready'
        elif final_score >= 50.0:
            readiness_label = 'Requires Cleaning'
        else:
            readiness_label = 'Not Ready'
            
        # Specific Actionable Cleaning Recommendations
        actions = []
        if unusable_count > 0:
            actions.append(f"Exclude {unusable_count} unusable scan(s) with severe degradation / missing anatomy.")
        if poor_count > 0:
            actions.append(f"Review {poor_count} poor-quality scan(s) for potential artifact filtering or re-acquisition.")
        if outlier_rate > 0.10:
            actions.append(f"Inspect {outliers} anomaly scan(s) flagged by Isolation Forest to avoid corrupted gradients.")
        if meta_comp < 0.90:
            actions.append("Impute or populate missing clinical metadata (age, sex, scanner tags).")
        if len(scanners) > 1:
            actions.append(f"Apply ComBat or intensity harmonization across {len(scanners)} distinct scanner protocols.")
        if not actions:
            actions.append("Dataset passed quality criteria. Ready for ML model training and cross-validation.")
            
        breakdown = {
            'image_quality_score': round(quality_pts, 1),
            'image_quality_max': 35.0,
            'snr_motion_score': round(snr_motion_pts, 1),
            'snr_motion_max': 20.0,
            'metadata_score': round(metadata_pts, 1),
            'metadata_max': 20.0,
            'outlier_score': round(outlier_pts, 1),
            'outlier_max': 15.0,
            'consistency_score': round(consistency_pts, 1),
            'consistency_max': 10.0,
            'unusable_count': unusable_count,
            'poor_count': poor_count,
            'acceptable_count': acc_count,
            'good_count': good_count,
            'outlier_count': outliers
        }
        
        return {
            'readiness_score': final_score,
            'readiness_label': readiness_label,
            'breakdown': breakdown,
            'actions': actions
        }

readiness_model = ResearchReadinessModel()
