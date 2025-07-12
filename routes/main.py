"""
Main routes for the Walker training system web interface.
Handles core interface and status endpoints.
"""

from flask import Blueprint, render_template_string, jsonify, request, current_app
import sys
import os

# Add the src directory to the path to import WebGL renderer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.rendering.webgl_renderer import get_webgl_template

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """Serve the main web interface with WebGL rendering."""
    return render_template_string(get_webgl_template())

@main_bp.route('/status')
def status():
    """Get current training status for the web interface."""
    env = current_app.env
    
    # Get canvas dimensions and culling preference if provided via query parameters
    canvas_width = request.args.get('canvas_width', type=int, default=1200)
    canvas_height = request.args.get('canvas_height', type=int, default=800)
    viewport_culling = request.args.get('viewport_culling', default='true').lower() == 'true'
    # CRITICAL FIX: Get current frontend camera position for accurate viewport culling
    camera_x = request.args.get('camera_x', type=float, default=0.0)
    camera_y = request.args.get('camera_y', type=float, default=0.0)
    return jsonify(env.get_status(canvas_width, canvas_height, viewport_culling, camera_x, camera_y)) 