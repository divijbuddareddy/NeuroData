import os
import shutil
import json
import pandas as pd
import numpy as np
from pathlib import Path
from flask import Blueprint, request, jsonify, send_file, current_app
from werkzeug.utils import secure_filename

from database.models import db, DatasetRun, ScanResult, ChatMessage
from ml.preprocessing import load_image_as_grayscale
from ml.artifact_detector import analyze_scan_artifacts
from ml.anomaly_detector import detect_cohort_anomalies
from ml.metadata_analyzer import analyze_metadata_dataframe, synthesize_metadata_for_scans
from ml.quality_model import predict_scan_quality
from ml.readiness_model import readiness_model
from explainability.gradcam import generate_gradcam_heatmap
from services.gemini_service import generate_dataset_explanation, answer_researcher_question
from services.report_service import generate_pdf_report
from services.sample_data_generator import generate_sample_dataset

api_bp = Blueprint('api', __name__, url_prefix='/api')

ALLOWED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_IMAGE_EXTENSIONS or Path(filename).suffix.lower() in {'.csv', '.zip'}

@api_bp.route('/upload', methods=['POST'])
def upload_files():
    """
    Handles upload of MRI scan slices and optional metadata CSV.
    """
    if 'files[]' not in request.files and 'file' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
        
    uploaded_files = request.files.getlist('files[]') or [request.files['file']]
    dataset_name = request.form.get('dataset_name', f"Cohort_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
    
    upload_base = Path(current_app.config['UPLOAD_FOLDER'])
    dataset_dir = upload_base / secure_filename(dataset_name)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    saved_images = []
    metadata_csv_path = None
    
    for file in uploaded_files:
        if file and file.filename:
            fname = secure_filename(file.filename)
            ext = Path(fname).suffix.lower()
            save_dest = dataset_dir / fname
            file.save(str(save_dest))
            
            if ext == '.csv':
                metadata_csv_path = str(save_dest)
            elif ext in ALLOWED_IMAGE_EXTENSIONS:
                saved_images.append(str(save_dest))
                
    if not saved_images:
        return jsonify({'error': 'No valid MRI scan image files uploaded (.png, .jpg, .tif)'}), 400
        
    return jsonify({
        'message': f'Successfully uploaded {len(saved_images)} scans',
        'dataset_name': dataset_name,
        'scan_count': len(saved_images),
        'dataset_dir': str(dataset_dir),
        'has_metadata': metadata_csv_path is not None
    })

@api_bp.route('/analyze', methods=['POST'])
def run_analysis():
    """
    Executes full pipeline:
    1. Scan Quality Classifier (PyTorch)
    2. Artifact & Motion Detection (OpenCV & Stats)
    3. Anomaly & Outlier Detection (Scikit-Learn Isolation Forest & PCA)
    4. Metadata Completeness & Bias Analysis
    5. Research-Readiness Model Ensemble
    6. Grad-CAM Explainability Generation
    7. Gemini API Synthesis
    """
    data = request.get_json() or {}
    dataset_dir = data.get('dataset_dir')
    dataset_name = data.get('dataset_name', f"Dataset_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
    
    # If no custom directory provided, check sample directory
    if not dataset_dir or not Path(dataset_dir).exists():
        sample_dir = Path(current_app.config['SAMPLE_DATA_FOLDER'])
        if not any(sample_dir.glob("*.png")):
            generate_sample_dataset(sample_dir)
        dataset_dir = str(sample_dir)
        dataset_name = "Sample NeuroImaging Cohort (T1w 3T/1.5T)"

    dataset_path = Path(dataset_dir)
    image_files = sorted([p for p in dataset_path.iterdir() if p.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS])
    
    if not image_files:
        return jsonify({'error': 'No MRI scan images found in the specified path'}), 400
        
    # Check for metadata CSV
    csv_candidates = list(dataset_path.glob("*.csv"))
    if csv_candidates:
        df_meta = pd.read_csv(str(csv_candidates[0]))
    else:
        df_meta = synthesize_metadata_for_scans([p.name for p in image_files])
        
    # Metadata Analysis
    meta_analysis = analyze_metadata_dataframe(df_meta)
    
    # Anomaly Detection (Cohort level)
    anomaly_results = detect_cohort_anomalies([str(p) for p in image_files])
    anomaly_map = {Path(r['path']).name: r for r in anomaly_results}
    
    # Create DB Record for DatasetRun
    dataset_run = DatasetRun(
        name=dataset_name,
        total_scans=len(image_files),
        metadata_completeness_pct=meta_analysis['completeness_pct'],
        demographics_json=json.dumps(meta_analysis['demographics']),
        scanner_stats_json=json.dumps(meta_analysis['scanner_distribution'])
    )
    db.session.add(dataset_run)
    db.session.flush() # get dataset_run.id
    
    processed_dir = Path(current_app.config['PROCESSED_FOLDER']) / f"dataset_{dataset_run.id}"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    scan_objects = []
    good_cnt = 0
    acc_cnt = 0
    poor_cnt = 0
    unus_cnt = 0
    
    snr_list = []
    cnr_list = []
    blur_list = []
    motion_list = []
    outlier_cnt = 0
    
    # Process each scan
    for idx, img_path in enumerate(image_files):
        fname = img_path.name
        
        # 1. Artifact detection
        artifacts = analyze_scan_artifacts(str(img_path))
        
        # 2. PyTorch Quality prediction
        quality_res = predict_scan_quality(str(img_path), artifact_metrics=artifacts)
        q_label = quality_res['quality_label']
        
        if q_label == 'Good': good_cnt += 1
        elif q_label == 'Acceptable': acc_cnt += 1
        elif q_label == 'Poor': poor_cnt += 1
        elif q_label == 'Unusable': unus_cnt += 1
        
        # 3. Grad-CAM generation
        cam_info = generate_gradcam_heatmap(str(img_path), output_dir=str(processed_dir))
        
        preview_url = f"/static/processed/dataset_{dataset_run.id}/{cam_info['preview_filename']}"
        gradcam_url = f"/static/processed/dataset_{dataset_run.id}/{cam_info['gradcam_filename']}"
        
        # 4. Outlier data
        anom = anomaly_map.get(fname, {'is_outlier': False, 'outlier_score': 0.0, 'pca_x': 0.0, 'pca_y': 0.0})
        if anom['is_outlier']:
            outlier_cnt += 1
            
        # 5. Metadata matching
        meta_row = df_meta[df_meta['filename'] == fname].to_dict('records') if 'filename' in df_meta.columns else []
        meta_dict = meta_row[0] if meta_row else {}
        
        scan_res = ScanResult(
            dataset_id=dataset_run.id,
            filename=fname,
            original_path=str(img_path),
            preview_url=preview_url,
            gradcam_url=gradcam_url,
            quality_label=q_label,
            confidence=quality_res['confidence'],
            quality_probs_json=json.dumps(quality_res['probabilities']),
            snr=artifacts['snr_cnr']['snr'],
            cnr=artifacts['snr_cnr']['cnr'],
            blur_score=artifacts['blur']['blur_metric'],
            motion_score=artifacts['motion']['motion_score'],
            ghosting_score=artifacts['motion']['ghosting_ratio'],
            intensity_bias_score=artifacts['bias']['bias_score'],
            cropping_flag=artifacts['geometry']['is_cropped'],
            missing_slice_flag=artifacts['geometry']['is_missing_slice'],
            is_outlier=anom['is_outlier'],
            outlier_score=anom['outlier_score'],
            embedding_pca_x=anom['pca_x'],
            embedding_pca_y=anom['pca_y'],
            scanner_vendor=str(meta_dict.get('scanner_vendor', 'Siemens Prisma 3T')),
            magnetic_field_t=float(meta_dict.get('magnetic_field_t', 3.0)),
            subject_id=str(meta_dict.get('subject_id', f'SUBJ_{idx+1:03d}')),
            subject_age=int(meta_dict.get('age', 45)) if pd.notna(meta_dict.get('age')) else None,
            subject_sex=str(meta_dict.get('sex', 'Unknown')),
            site_id=str(meta_dict.get('site_id', 'Site_A'))
        )
        db.session.add(scan_res)
        scan_objects.append(scan_res)
        
        snr_list.append(scan_res.snr)
        cnr_list.append(scan_res.cnr)
        blur_list.append(scan_res.blur_score)
        motion_list.append(scan_res.motion_score)
        
    # Aggregate stats on DatasetRun
    dataset_run.good_count = good_cnt
    dataset_run.acceptable_count = acc_cnt
    dataset_run.poor_count = poor_cnt
    dataset_run.unusable_count = unus_cnt
    dataset_run.outlier_count = outlier_cnt
    dataset_run.mean_snr = float(np.mean(snr_list)) if snr_list else 0.0
    dataset_run.mean_cnr = float(np.mean(cnr_list)) if cnr_list else 0.0
    dataset_run.mean_blur_score = float(np.mean(blur_list)) if blur_list else 0.0
    dataset_run.mean_motion_score = float(np.mean(motion_list)) if motion_list else 0.0
    
    # Research Readiness Evaluation
    readiness_eval = readiness_model.evaluate_dataset(
        [s.to_dict() for s in scan_objects],
        meta_analysis
    )
    
    dataset_run.readiness_score = readiness_eval['readiness_score']
    dataset_run.readiness_label = readiness_eval['readiness_label']
    dataset_run.readiness_breakdown_json = json.dumps(readiness_eval['breakdown'])
    
    # Gemini Explanation Generation
    gemini_insights = generate_dataset_explanation(dataset_run.to_dict())
    dataset_run.gemini_summary = gemini_insights.get('executive_summary')
    dataset_run.gemini_risks = gemini_insights.get('risk_analysis')
    dataset_run.gemini_recommendations = "\n".join(gemini_insights.get('recommended_actions', [])) if isinstance(gemini_insights.get('recommended_actions'), list) else str(gemini_insights.get('recommended_actions'))
    
    db.session.commit()
    
    return jsonify({
        'message': 'Analysis completed successfully',
        'dataset_id': dataset_run.id,
        'dataset': dataset_run.to_dict()
    })

@api_bp.route('/dataset/<int:dataset_id>', methods=['GET'])
def get_dataset(dataset_id):
    dataset_run = DatasetRun.query.get_or_404(dataset_id)
    scans = ScanResult.query.filter_by(dataset_id=dataset_id).all()
    
    data = dataset_run.to_dict()
    data['scans'] = [s.to_dict() for s in scans]
    return jsonify(data)

@api_bp.route('/scan/<int:scan_id>', methods=['GET'])
def get_scan(scan_id):
    scan = ScanResult.query.get_or_404(scan_id)
    return jsonify(scan.to_dict())

@api_bp.route('/explain', methods=['POST'])
def generate_explanation():
    data = request.get_json() or {}
    dataset_id = data.get('dataset_id')
    
    if dataset_id:
        dataset_run = DatasetRun.query.get_or_404(dataset_id)
        structured_data = dataset_run.to_dict()
    else:
        structured_data = data.get('structured_data', {})
        
    explanation = generate_dataset_explanation(structured_data)
    return jsonify(explanation)

from services.gemini_service import (
    generate_dataset_explanation, answer_researcher_question,
    validate_api_key, set_runtime_api_key, get_active_api_key
)

@api_bp.route('/settings/api-key', methods=['GET', 'POST'])
def handle_api_key():
    if request.method == 'GET':
        active_key = get_active_api_key()
        has_key = bool(active_key and "your_gemini" not in active_key)
        masked = f"{active_key[:6]}...{active_key[-4:]}" if (has_key and len(active_key) > 10) else ("Configured" if has_key else "Not Set")
        return jsonify({
            'configured': has_key,
            'masked_key': masked,
            'model': 'gemini-2.5-flash'
        })
        
    data = request.get_json() or {}
    api_key = data.get('api_key', '').strip()
    
    if not api_key:
        return jsonify({'error': 'API key cannot be empty'}), 400
        
    # Test key with Google AI Studio
    val_res = validate_api_key(api_key)
    if not val_res['valid']:
        return jsonify({'error': f"Invalid Google Studio API Key: {val_res.get('error', 'Authentication failed')}"}), 400
        
    # Save key
    set_runtime_api_key(api_key, persist=True)
    return jsonify({
        'message': 'Google AI Studio API Key successfully connected & saved!',
        'model': val_res.get('model', 'gemini-2.5-flash'),
        'configured': True
    })

@api_bp.route('/chat', methods=['POST'])
def chat_assistant():
    data = request.get_json() or {}
    dataset_id = data.get('dataset_id')
    message = data.get('message', '').strip()
    user_api_key = data.get('api_key') or request.headers.get('X-Gemini-Api-Key')
    
    if not message:
        return jsonify({'error': 'Message content cannot be empty'}), 400
        
    context = {}
    if dataset_id:
        dataset_run = DatasetRun.query.get(dataset_id)
        if dataset_run:
            context = dataset_run.to_dict()
            user_msg = ChatMessage(dataset_id=dataset_id, role='user', content=message)
            db.session.add(user_msg)
            
    reply = answer_researcher_question(message, context, api_key=user_api_key)
    
    if dataset_id:
        assistant_msg = ChatMessage(dataset_id=dataset_id, role='assistant', content=reply)
        db.session.add(assistant_msg)
        db.session.commit()
        
    return jsonify({'reply': reply})

@api_bp.route('/history', methods=['GET'])
def get_history():
    runs = DatasetRun.query.order_by(DatasetRun.created_at.desc()).all()
    return jsonify([r.to_dict() for r in runs])

@api_bp.route('/report/<int:dataset_id>/pdf', methods=['GET'])
def download_pdf_report(dataset_id):
    dataset_run = DatasetRun.query.get_or_404(dataset_id)
    scans = ScanResult.query.filter_by(dataset_id=dataset_id).all()
    
    data = dataset_run.to_dict()
    data['scans'] = [s.to_dict() for s in scans]
    
    pdf_filename = f"NeuroData_Audit_Report_Dataset_{dataset_id}.pdf"
    reports_dir = Path(current_app.config['REPORTS_FOLDER'])
    pdf_path = reports_dir / pdf_filename
    
    generate_pdf_report(data, pdf_path)
    return send_file(str(pdf_path), as_attachment=True, download_name=pdf_filename)

@api_bp.route('/demo/preload', methods=['POST'])
def preload_demo():
    """
    Instantly creates demo synthetic cohort if not present and executes analysis.
    """
    sample_dir = Path(current_app.config['SAMPLE_DATA_FOLDER'])
    generate_sample_dataset(sample_dir)
    
    # Trigger analysis on this folder
    with current_app.test_client() as client:
        res = client.post('/api/analyze', json={
            'dataset_dir': str(sample_dir),
            'dataset_name': 'Brain T1w Multicenter Baseline (15 Scans)'
        })
        return jsonify(res.get_json())
