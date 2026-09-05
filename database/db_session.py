from database.models import db, DatasetRun, ScanResult, ChatMessage

def init_db(app):
    db.init_app(app)
    with app.app_context():
        db.create_all()
