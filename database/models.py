from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
import json

db = SQLAlchemy()

class DatasetRun(db.Model):
    __tablename__ = 'dataset_runs'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    total_scans = db.Column(db.Integer, default=0)
    
    # Quality breakdown
    good_count = db.Column(db.Integer, default=0)
    acceptable_count = db.Column(db.Integer, default=0)
    poor_count = db.Column(db.Integer, default=0)
    unusable_count = db.Column(db.Integer, default=0)
    
    # Statistical aggregates
    mean_snr = db.Column(db.Float, default=0.0)
    mean_cnr = db.Column(db.Float, default=0.0)
    mean_blur_score = db.Column(db.Float, default=0.0)
    mean_motion_score = db.Column(db.Float, default=0.0)
    outlier_count = db.Column(db.Integer, default=0)
    metadata_completeness_pct = db.Column(db.Float, default=100.0)
    
    # Research readiness
    readiness_score = db.Column(db.Float, default=0.0)  # 0 to 100
    readiness_label = db.Column(db.String(50), default='Not Ready') # Ready, Requires Cleaning, Not Ready
    readiness_breakdown_json = db.Column(db.Text, default='{}')
    
    # Demographic and Site stats (JSON strings)
    demographics_json = db.Column(db.Text, default='{}')
    scanner_stats_json = db.Column(db.Text, default='{}')
    
    # Gemini AI insights
    gemini_summary = db.Column(db.Text, nullable=True)
    gemini_risks = db.Column(db.Text, nullable=True)
    gemini_recommendations = db.Column(db.Text, nullable=True)
    
    # Scans relationship
    scans = db.relationship('ScanResult', backref='dataset', cascade="all, delete-orphan", lazy=True)
    chat_messages = db.relationship('ChatMessage', backref='dataset', cascade="all, delete-orphan", lazy=True)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'created_at': self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else None,
            'total_scans': self.total_scans,
            'good_count': self.good_count,
            'acceptable_count': self.acceptable_count,
            'poor_count': self.poor_count,
            'unusable_count': self.unusable_count,
            'mean_snr': round(self.mean_snr or 0.0, 2),
            'mean_cnr': round(self.mean_cnr or 0.0, 2),
            'mean_blur_score': round(self.mean_blur_score or 0.0, 2),
            'mean_motion_score': round(self.mean_motion_score or 0.0, 3),
            'outlier_count': self.outlier_count,
            'metadata_completeness_pct': round(self.metadata_completeness_pct or 0.0, 1),
            'readiness_score': round(self.readiness_score or 0.0, 1),
            'readiness_label': self.readiness_label,
            'readiness_breakdown': json.loads(self.readiness_breakdown_json or '{}'),
            'demographics': json.loads(self.demographics_json or '{}'),
            'scanner_stats': json.loads(self.scanner_stats_json or '{}'),
            'gemini_summary': self.gemini_summary,
            'gemini_risks': self.gemini_risks,
            'gemini_recommendations': self.gemini_recommendations,
        }

class ScanResult(db.Model):
    __tablename__ = 'scan_results'

    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset_runs.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    original_path = db.Column(db.String(512), nullable=False)
    preview_url = db.Column(db.String(512), nullable=True)
    gradcam_url = db.Column(db.String(512), nullable=True)
    
    # Model 1 Quality Predictions
    quality_label = db.Column(db.String(50), nullable=False) # Good, Acceptable, Poor, Unusable
    confidence = db.Column(db.Float, default=0.0)
    quality_probs_json = db.Column(db.Text, default='{}')
    
    # Model 2 Artifact Metrics
    snr = db.Column(db.Float, default=0.0)
    cnr = db.Column(db.Float, default=0.0)
    blur_score = db.Column(db.Float, default=0.0)
    motion_score = db.Column(db.Float, default=0.0)
    ghosting_score = db.Column(db.Float, default=0.0)
    intensity_bias_score = db.Column(db.Float, default=0.0)
    cropping_flag = db.Column(db.Boolean, default=False)
    missing_slice_flag = db.Column(db.Boolean, default=False)
    
    # Model 3 Anomaly
    is_outlier = db.Column(db.Boolean, default=False)
    outlier_score = db.Column(db.Float, default=0.0)
    embedding_pca_x = db.Column(db.Float, default=0.0)
    embedding_pca_y = db.Column(db.Float, default=0.0)
    
    # Model 4 Metadata
    dimensions = db.Column(db.String(50), default="256x256")
    voxel_spacing = db.Column(db.String(50), default="1.0 x 1.0 x 1.0 mm")
    scanner_vendor = db.Column(db.String(100), default="Siemens Prisma 3T")
    magnetic_field_t = db.Column(db.Float, default=3.0)
    subject_id = db.Column(db.String(100), default="SUBJ_001")
    subject_age = db.Column(db.Integer, nullable=True)
    subject_sex = db.Column(db.String(20), nullable=True)
    site_id = db.Column(db.String(100), default="Site_A")
    acquisition_notes = db.Column(db.String(255), nullable=True)

    def to_dict(self):
        return {
            'id': self.id,
            'dataset_id': self.dataset_id,
            'filename': self.filename,
            'preview_url': self.preview_url,
            'gradcam_url': self.gradcam_url,
            'quality_label': self.quality_label,
            'confidence': round(self.confidence or 0.0, 3),
            'quality_probs': json.loads(self.quality_probs_json or '{}'),
            'snr': round(self.snr or 0.0, 2),
            'cnr': round(self.cnr or 0.0, 2),
            'blur_score': round(self.blur_score or 0.0, 2),
            'motion_score': round(self.motion_score or 0.0, 3),
            'ghosting_score': round(self.ghosting_score or 0.0, 3),
            'intensity_bias_score': round(self.intensity_bias_score or 0.0, 3),
            'cropping_flag': self.cropping_flag,
            'missing_slice_flag': self.missing_slice_flag,
            'is_outlier': self.is_outlier,
            'outlier_score': round(self.outlier_score or 0.0, 3),
            'embedding_pca_x': round(self.embedding_pca_x or 0.0, 3),
            'embedding_pca_y': round(self.embedding_pca_y or 0.0, 3),
            'dimensions': self.dimensions,
            'voxel_spacing': self.voxel_spacing,
            'scanner_vendor': self.scanner_vendor,
            'magnetic_field_t': self.magnetic_field_t,
            'subject_id': self.subject_id,
            'subject_age': self.subject_age,
            'subject_sex': self.subject_sex,
            'site_id': self.site_id,
            'acquisition_notes': self.acquisition_notes
        }

class ChatMessage(db.Model):
    __tablename__ = 'chat_messages'

    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset_runs.id'), nullable=False)
    role = db.Column(db.String(20), nullable=False) # 'user' or 'assistant'
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'dataset_id': self.dataset_id,
            'role': self.role,
            'content': self.content,
            'created_at': self.created_at.strftime("%H:%M:%S")
        }
