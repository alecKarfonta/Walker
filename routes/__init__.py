"""
Routes package for the Walker training system web interface.
"""

from .main import main_bp
from .metrics import metrics_bp
from .interaction import interaction_bp
from .settings import settings_bp
from .elite import elite_bp

def register_routes(app, env):
    """Register all route blueprints with the Flask app."""
    
    # Register blueprints and pass the environment instance
    app.register_blueprint(main_bp)
    app.register_blueprint(metrics_bp)
    app.register_blueprint(interaction_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(elite_bp)
    
    # Store environment instance for access in routes
    app.env = env 