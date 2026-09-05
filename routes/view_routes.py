from flask import Blueprint, render_template, request
from database.models import DatasetRun, ScanResult

view_bp = Blueprint('views', __name__)

@view_bp.route('/')
def home():
    latest_run = DatasetRun.query.order_by(DatasetRun.created_at.desc()).first()
    total_datasets = DatasetRun.query.count()
    total_scans = ScanResult.query.count()
    return render_template('home.html', latest_run=latest_run, total_datasets=total_datasets, total_scans=total_scans)

@view_bp.route('/upload')
def upload_page():
    return render_template('upload.html')

@view_bp.route('/scan-quality')
def scan_quality():
    dataset_id = request.args.get('dataset_id', type=int)
    if dataset_id:
        dataset = DatasetRun.query.get(dataset_id)
    else:
        dataset = DatasetRun.query.order_by(DatasetRun.created_at.desc()).first()
        
    scans = ScanResult.query.filter_by(dataset_id=dataset.id).all() if dataset else []
    scans_dict = [s.to_dict() for s in scans]
    all_datasets = DatasetRun.query.order_by(DatasetRun.created_at.desc()).all()
    return render_template('scan_quality.html', dataset=dataset, scans=scans, scans_dict=scans_dict, all_datasets=all_datasets)

@view_bp.route('/dataset-analysis')
def dataset_analysis():
    dataset_id = request.args.get('dataset_id', type=int)
    if dataset_id:
        dataset = DatasetRun.query.get(dataset_id)
    else:
        dataset = DatasetRun.query.order_by(DatasetRun.created_at.desc()).first()
        
    scans = ScanResult.query.filter_by(dataset_id=dataset.id).all() if dataset else []
    scans_dict = [s.to_dict() for s in scans]
    all_datasets = DatasetRun.query.order_by(DatasetRun.created_at.desc()).all()
    return render_template('dataset_analysis.html', dataset=dataset, scans=scans, scans_dict=scans_dict, all_datasets=all_datasets)

@view_bp.route('/research-readiness')
def research_readiness():
    dataset_id = request.args.get('dataset_id', type=int)
    if dataset_id:
        dataset = DatasetRun.query.get(dataset_id)
    else:
        dataset = DatasetRun.query.order_by(DatasetRun.created_at.desc()).first()
        
    all_datasets = DatasetRun.query.order_by(DatasetRun.created_at.desc()).all()
    return render_template('research_readiness.html', dataset=dataset, all_datasets=all_datasets)

@view_bp.route('/assistant')
def assistant():
    dataset_id = request.args.get('dataset_id', type=int)
    if dataset_id:
        dataset = DatasetRun.query.get(dataset_id)
    else:
        dataset = DatasetRun.query.order_by(DatasetRun.created_at.desc()).first()
        
    all_datasets = DatasetRun.query.order_by(DatasetRun.created_at.desc()).all()
    return render_template('assistant.html', dataset=dataset, all_datasets=all_datasets)

@view_bp.route('/model-evaluation')
def model_evaluation():
    return render_template('evaluation.html')

@view_bp.route('/history')
def history():
    runs = DatasetRun.query.order_by(DatasetRun.created_at.desc()).all()
    return render_template('history.html', runs=runs)

@view_bp.route('/report/<int:dataset_id>')
def view_report(dataset_id):
    dataset = DatasetRun.query.get_or_404(dataset_id)
    scans = ScanResult.query.filter_by(dataset_id=dataset_id).all()
    return render_template('report_template.html', dataset=dataset, scans=scans)
