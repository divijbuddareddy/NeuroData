import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "neurodata.db"
UPLOAD_FOLDER = BASE_DIR / "uploads"
PROCESSED_FOLDER = BASE_DIR / "static" / "processed"
REPORTS_FOLDER = BASE_DIR / "static" / "reports"
SAMPLE_DATA_FOLDER = BASE_DIR / "datasets" / "sample_neuro_dataset"

# Ensure runtime directories exist
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)
REPORTS_FOLDER.mkdir(parents=True, exist_ok=True)
SAMPLE_DATA_FOLDER.mkdir(parents=True, exist_ok=True)

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "neurodata-quality-ai-secret-key-2026")
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{DATABASE_PATH}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = str(UPLOAD_FOLDER)
    PROCESSED_FOLDER = str(PROCESSED_FOLDER)
    REPORTS_FOLDER = str(REPORTS_FOLDER)
    SAMPLE_DATA_FOLDER = str(SAMPLE_DATA_FOLDER)
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100 MB max upload
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
