"""
Settings routes for the Walker training system web interface.
Handles configuration and settings endpoints like simulation speed, optimization settings, etc.
"""

from flask import Blueprint, jsonify, request, current_app

settings_bp = Blueprint('settings', __name__)

@settings_bp.route('/set_simulation_speed', methods=['POST'])
def set_simulation_speed():
    """Set the simulation speed multiplier."""
    env = current_app.env
    try:
        data = request.get_json()
        speed = data.get('speed', 1.0)
        
        # Validate speed range
        if not isinstance(speed, (int, float)) or speed <= 0:
            return jsonify({'status': 'error', 'message': 'Speed must be a positive number'}), 400
        
        if speed > env.max_speed_multiplier:
            return jsonify({'status': 'error', 'message': f'Speed cannot exceed {env.max_speed_multiplier}x'}), 400
        
        # Set the speed
        env.simulation_speed_multiplier = float(speed)
        
        return jsonify({
            'status': 'success', 
            'message': f'Simulation speed set to {speed}x',
            'speed': speed
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@settings_bp.route('/ai_optimization_settings', methods=['GET', 'POST'])
def ai_optimization_settings():
    """Get or set AI optimization settings."""
    env = current_app.env
    try:
        if request.method == 'GET':
            return jsonify({
                'status': 'success',
                'settings': {
                    'ai_optimization_enabled': env.ai_optimization_enabled,
                    'ai_batch_percentage': env.ai_batch_percentage,
                    'ai_spatial_culling_enabled': env.ai_spatial_culling_enabled,
                    'ai_spatial_culling_distance': env.ai_spatial_culling_distance
                }
            })
        
        elif request.method == 'POST':
            data = request.get_json()
            
            # Update settings if provided
            if 'ai_optimization_enabled' in data:
                env.ai_optimization_enabled = bool(data['ai_optimization_enabled'])
            
            if 'ai_batch_percentage' in data:
                percentage = float(data['ai_batch_percentage'])
                if 0.1 <= percentage <= 1.0:
                    env.ai_batch_percentage = percentage
                else:
                    return jsonify({'status': 'error', 'message': 'ai_batch_percentage must be between 0.1 and 1.0'}), 400
            
            if 'ai_spatial_culling_enabled' in data:
                env.ai_spatial_culling_enabled = bool(data['ai_spatial_culling_enabled'])
            
            if 'ai_spatial_culling_distance' in data:
                distance = float(data['ai_spatial_culling_distance'])
                if distance > 0:
                    env.ai_spatial_culling_distance = distance
                else:
                    return jsonify({'status': 'error', 'message': 'ai_spatial_culling_distance must be positive'}), 400
            
            return jsonify({
                'status': 'success',
                'message': 'AI optimization settings updated',
                'settings': {
                    'ai_optimization_enabled': env.ai_optimization_enabled,
                    'ai_batch_percentage': env.ai_batch_percentage,
                    'ai_spatial_culling_enabled': env.ai_spatial_culling_enabled,
                    'ai_spatial_culling_distance': env.ai_spatial_culling_distance
                }
            })
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@settings_bp.route('/toggle_health_bars', methods=['POST'])
def toggle_health_bars():
    """Toggle health bar rendering on/off for performance optimization."""
    env = current_app.env
    try:
        data = request.get_json() or {}
        enable = data.get('enable', not env.show_health_bars)  # Toggle if not specified
        
        env.show_health_bars = bool(enable)
        
        message = f"Health bars {'enabled' if env.show_health_bars else 'disabled'}"
        return jsonify({
            'status': 'success',
            'message': message,
            'show_health_bars': env.show_health_bars
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@settings_bp.route('/toggle_visualization', methods=['POST'])
def toggle_visualization():
    """Toggle robot visualization on/off for maximum performance optimization."""
    env = current_app.env
    try:
        data = request.get_json() or {}
        enable = data.get('enable', not env.enable_visualization)  # Toggle if not specified
        
        env.enable_visualization = bool(enable)
        
        message = f"Robot visualization {'enabled' if env.enable_visualization else 'disabled'}"
        status_message = message + (" (maximum speed mode)" if not env.enable_visualization else " (normal mode)")
        
        return jsonify({
            'status': 'success',
            'message': status_message,
            'enable_visualization': env.enable_visualization
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500 