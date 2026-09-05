import os
from flask import Flask, redirect, url_for
from config import Config
from database.db_session import init_db
from routes.api_routes import api_bp
from routes.view_routes import view_bp

def create_app():
    app = Flask(__name__, static_folder='static', template_folder='templates')
    app.config.from_object(Config)
    
    # Initialize SQLite Database
    init_db(app)
    
    # Register blueprints
    app.register_blueprint(api_bp)
    app.register_blueprint(view_bp)
    
    @app.context_processor
    def inject_globals():
        return {
            'app_title': 'NeuroData Quality AI',
            'app_version': '1.0.0',
            'gemini_configured': bool(Config.GEMINI_API_KEY and "your_gemini" not in Config.GEMINI_API_KEY)
        }
        
    return app

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] NeuroData Quality AI starting on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
