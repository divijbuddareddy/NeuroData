import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import create_app
from database.models import db, DatasetRun, ScanResult
from services.sample_data_generator import generate_sample_dataset
from ml.artifact_detector import analyze_scan_artifacts
from ml.quality_model import predict_scan_quality
from ml.anomaly_detector import detect_cohort_anomalies
from ml.metadata_analyzer import analyze_metadata_dataframe, synthesize_metadata_for_scans
from ml.readiness_model import readiness_model
from explainability.gradcam import generate_gradcam_heatmap
from services.gemini_service import generate_dataset_explanation

def run_all_tests():
    print("[*] Running End-to-End System Verification for NeuroData Quality AI...")
    app = create_app()
    
    with app.app_context():
        # 1. Test Sample Data Generator
        print("\n[1/7] Testing Synthetic Neuroimaging Dataset Generator...")
        sample_dir = Path(app.config['SAMPLE_DATA_FOLDER'])
        gen_res = generate_sample_dataset(sample_dir)
        pngs = list(sample_dir.glob("*.png"))
        assert len(pngs) >= 15, f"Expected 15 synthetic scans, got {len(pngs)}"
        print(f"  [OK] Successfully generated {len(pngs)} realistic MRI slices with controlled artifacts.")
        
        # 2. Test Artifact & Quality Models
        print("\n[2/7] Testing OpenCV Artifact & PyTorch Quality Classification...")
        test_img = str(pngs[0])
        artifacts = analyze_scan_artifacts(test_img)
        assert 'blur' in artifacts and 'snr_cnr' in artifacts and 'motion' in artifacts
        print(f"  [OK] Artifact Metrics computed: SNR={artifacts['snr_cnr']['snr']} dB, Blur={artifacts['blur']['blur_metric']}")
        
        quality = predict_scan_quality(test_img, artifacts)
        assert quality['quality_label'] in ['Good', 'Acceptable', 'Poor', 'Unusable']
        print(f"  [OK] PyTorch Model predicted: Label={quality['quality_label']}, Confidence={quality['confidence']}")
        
        # 3. Test Grad-CAM Generation
        print("\n[3/7] Testing PyTorch Grad-CAM Explainability Engine...")
        cam_res = generate_gradcam_heatmap(test_img, output_dir=app.config['PROCESSED_FOLDER'])
        assert cam_res['preview_filename'] is not None
        print(f"  [OK] Grad-CAM heatmap generated: {cam_res['gradcam_filename']}")
        
        # 4. Test Anomaly Detection
        print("\n[4/7] Testing Isolation Forest & PCA Anomaly Detector...")
        anom_res = detect_cohort_anomalies([str(p) for p in pngs])
        assert len(anom_res) == len(pngs)
        outlier_count = sum(1 for r in anom_res if r['is_outlier'])
        print(f"  [OK] Cohort Anomaly Detector isolated {outlier_count} outliers across {len(pngs)} scans.")
        
        # 5. Test Metadata Analyzer & Readiness Model
        print("\n[5/7] Testing Metadata Analysis & Readiness Composite Model...")
        df_meta = synthesize_metadata_for_scans([p.name for p in pngs])
        meta_eval = analyze_metadata_dataframe(df_meta)
        assert meta_eval['completeness_pct'] >= 80.0
        
        dummy_scan_results = [
            {'quality_label': 'Good', 'snr': 18.0, 'motion_score': 0.05, 'is_outlier': False},
            {'quality_label': 'Poor', 'snr': 8.0, 'motion_score': 0.70, 'is_outlier': True}
        ]
        readiness_res = readiness_model.evaluate_dataset(dummy_scan_results, meta_eval)
        assert 'readiness_score' in readiness_res
        print(f"  [OK] Readiness Model computed Score: {readiness_res['readiness_score']}/100 ({readiness_res['readiness_label']})")
        
        # 6. Test Gemini Service Synthesis
        print("\n[6/7] Testing Gemini API Structured Explanation Layer...")
        gemini_explanation = generate_dataset_explanation({
            'name': 'Test Cohort',
            'readiness_score': readiness_res['readiness_score'],
            'readiness_label': readiness_res['readiness_label'],
            'total_scans': 15,
            'good_count': 6,
            'poor_count': 3,
            'unusable_count': 2,
            'outlier_count': 2
        })
        assert 'executive_summary' in gemini_explanation
        print(f"  [OK] Gemini Synthesis generated: {gemini_explanation['executive_summary'][:80]}...")

        # 7. Test API Full Pipeline
        print("\n[7/7] Testing Full Flask API Pipeline Integration...")
        client = app.test_client()
        res = client.post('/api/analyze', json={'dataset_dir': str(sample_dir), 'dataset_name': 'Automated_CI_Test_Cohort'})
        assert res.status_code == 200
        data = res.get_json()
        assert data['dataset_id'] is not None
        print(f"  [OK] Full API /api/analyze passed! Dataset ID: #{data['dataset_id']}")
        
        print("\n[SUCCESS] ALL PIPELINE TESTS COMPLETED WITH 100% SUCCESS!")

if __name__ == '__main__':
    run_all_tests()
