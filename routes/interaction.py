"""
Interaction routes for the Walker training system web interface.
Handles user interaction endpoints like click, move, zoom, and camera controls.
"""

from flask import Blueprint, jsonify, request, current_app

interaction_bp = Blueprint('interaction', __name__)

@interaction_bp.route('/click', methods=['POST'])
def click():
    """Handle click events from the frontend for agent focusing."""
    env = current_app.env
    try:
        data = request.get_json()
        agent_id = data.get('agent_id')
        
        if agent_id:
            # Find the agent by ID - convert both to strings for comparison
            agent_to_focus = next((agent for agent in env.agents if str(agent.id) == str(agent_id)), None)
            if agent_to_focus:
                env.focus_on_agent(agent_to_focus)
                return jsonify({'status': 'success', 'message': f'Focused on agent {agent_id}', 'agent_id': agent_id})
            else:
                env.focus_on_agent(None)  # Clear focus if agent not found
                return jsonify({'status': 'error', 'message': f'Agent {agent_id} not found', 'agent_id': None})
        else:
            # Clear focus if no agent_id provided
            env.focus_on_agent(None)
            return jsonify({'status': 'success', 'message': 'Focus cleared', 'agent_id': None})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@interaction_bp.route('/get_agent_at_position', methods=['POST'])
def get_agent_at_position():
    """Get agent information at a specific world position."""
    env = current_app.env
    try:
        data = request.get_json()
        world_x = data.get('x', 0)  # Frontend sends 'x', not 'world_x'
        world_y = data.get('y', 0)  # Frontend sends 'y', not 'world_y'
        
        agent = env.get_agent_at_position(world_x, world_y)
        if agent:
            return jsonify({
                'status': 'success',
                'agent_id': agent.id,  # Frontend expects 'agent_id' directly, not nested
                'position': [agent.body.position.x, agent.body.position.y] if agent.body else [0, 0]
            })
        else:
            return jsonify({'status': 'error', 'message': 'No agent found at position', 'agent_id': None})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e), 'agent_id': None}), 500

@interaction_bp.route('/move_agent', methods=['POST'])
def move_agent():
    """Move an agent to a specific world position."""
    env = current_app.env
    try:
        data = request.get_json()
        agent_id = data.get('agent_id')
        x = data.get('x', 0)
        y = data.get('y', 0)
        
        success = env.move_agent(agent_id, x, y)
        if success:
            return jsonify({'status': 'success', 'message': f'Agent {agent_id} moved to ({x}, {y})'})
        else:
            return jsonify({'status': 'error', 'message': f'Failed to move agent {agent_id}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@interaction_bp.route('/update_zoom', methods=['POST'])
def update_zoom():
    """Update the user's zoom level preference."""
    env = current_app.env
    try:
        data = request.get_json()
        zoom_level = data.get('zoom', 1.0)
        
        env.update_user_zoom(zoom_level)
        return jsonify({'status': 'success', 'zoom': zoom_level})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@interaction_bp.route('/reset_view', methods=['POST'])
def reset_view():
    """Reset camera view to default position and zoom."""
    env = current_app.env
    try:
        env.reset_camera_position()
        env.reset_user_zoom()
        return jsonify({'status': 'success', 'message': 'View reset to default'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@interaction_bp.route('/clear_zoom_override', methods=['POST'])
def clear_zoom_override():
    """Clear zoom override flag after frontend receives it."""
    env = current_app.env
    try:
        env.clear_zoom_override()
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@interaction_bp.route('/update_agent_params', methods=['POST'])
def update_agent_params():
    """Update agent parameters."""
    env = current_app.env
    try:
        data = request.get_json()
        params = data.get('params', {})
        target_agent_id = data.get('target_agent_id')
        
        result = env.update_agent_params(params, target_agent_id)
        return jsonify({'status': 'success', 'updated_params': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500 