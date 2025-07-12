"""
Elite routes for the Walker training system web interface.
Handles elite robot management endpoints.
"""

from flask import Blueprint, jsonify, request, current_app
import random

elite_bp = Blueprint('elite', __name__)

@elite_bp.route('/elite_robots', methods=['GET', 'POST'])
def elite_robots():
    """Manage elite robots - get stats or load elites into population."""
    env = current_app.env
    try:
        if request.method == 'GET':
            # Get elite robot statistics
            stats = env.elite_manager.get_elite_statistics()
            top_elites = env.elite_manager.get_top_elites(10)
            
            return jsonify({
                'elite_statistics': stats,
                'top_elites': top_elites,
                'auto_save_enabled': env.auto_save_elites,
                'current_generation': env.evolution_engine.generation
            })
        
        elif request.method == 'POST':
            # Load elite robots into current population
            data = request.get_json() or {}
            count = data.get('count', 5)
            min_generation = data.get('min_generation', max(0, env.evolution_engine.generation - 5))
            
            # Load elite robots
            elite_robots = env.elite_manager.restore_elite_robots(
                world=env.world,
                count=count,
                min_generation=min_generation
            )
            
            if elite_robots:
                # Replace random agents with elite robots
                agents_replaced = min(len(elite_robots), len(env.agents) // 4)  # Replace up to 25%
                
                # Remove random agents
                for _ in range(agents_replaced):
                    if env.agents:
                        removed_agent = env.agents.pop(random.randint(0, len(env.agents) - 1))
                        env._safe_destroy_agent(removed_agent)
                
                # Add elite robots
                for elite_robot in elite_robots[:agents_replaced]:
                    env.agents.append(elite_robot)
                    env._initialize_single_agent_ecosystem(elite_robot)
                
                return jsonify({
                    'success': True,
                    'message': f'Loaded {agents_replaced} elite robots into population',
                    'agents_replaced': agents_replaced,
                    'elite_count': len(elite_robots),
                    'population_size': len(env.agents)
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'No elite robots found to load',
                    'agents_replaced': 0
                })
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500 