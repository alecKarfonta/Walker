"""
Metrics routes for the Walker training system web interface.
Handles monitoring, metrics, and reward signal endpoints.
"""

from flask import Blueprint, jsonify, current_app

metrics_bp = Blueprint('metrics', __name__)

@metrics_bp.route('/reward_signal_status')
def reward_signal_status():
    """Get reward signal status from training environment's adapter instance."""
    env = current_app.env
    try:
        if hasattr(env, 'reward_signal_adapter') and env.reward_signal_adapter:
            status = env.reward_signal_adapter.get_system_status()
            return jsonify(status)
        else:
            return jsonify({'error': 'Reward signal adapter not available'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@metrics_bp.route('/reward_signal_metrics')
def reward_signal_metrics():
    """Get reward signal metrics from training environment's adapter instance."""
    env = current_app.env
    try:
        if hasattr(env, 'reward_signal_adapter') and env.reward_signal_adapter:
            metrics = env.reward_signal_adapter.get_all_reward_metrics()
            return jsonify(metrics)
        else:
            return jsonify({'error': 'Reward signal adapter not available'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@metrics_bp.route('/metrics')
def metrics():
    """Prometheus metrics endpoint for training system data."""
    env = current_app.env
    try:
        if not hasattr(env, 'reward_signal_adapter') or not env.reward_signal_adapter:
            return "# No reward signal adapter available\n", 200, {'Content-Type': 'text/plain'}
        
        # Get system status and metrics
        status = env.reward_signal_adapter.get_system_status()
        reward_metrics = env.reward_signal_adapter.get_all_reward_metrics()
        
        # Build Prometheus metrics format
        metrics_output = []
        
        # System-level metrics
        metrics_output.append(f"# HELP walker_total_agents Total number of agents registered")
        metrics_output.append(f"# TYPE walker_total_agents gauge")
        metrics_output.append(f"walker_total_agents {status.get('total_agents', 0)}")
        
        metrics_output.append(f"# HELP walker_total_rewards Total number of rewards recorded")
        metrics_output.append(f"# TYPE walker_total_rewards counter")
        metrics_output.append(f"walker_total_rewards {status.get('total_rewards_recorded', 0)}")
        
        metrics_output.append(f"# HELP walker_agents_with_metrics Number of agents with quality metrics")
        metrics_output.append(f"# TYPE walker_agents_with_metrics gauge")
        metrics_output.append(f"walker_agents_with_metrics {status.get('agents_with_metrics', 0)}")
        
        # Per-agent reward quality metrics (if any)
        if reward_metrics:
            metrics_output.append(f"# HELP walker_reward_quality_score Reward quality score per agent")
            metrics_output.append(f"# TYPE walker_reward_quality_score gauge")
            
            metrics_output.append(f"# HELP walker_reward_signal_to_noise Signal to noise ratio per agent") 
            metrics_output.append(f"# TYPE walker_reward_signal_to_noise gauge")
            
            for agent_id, metrics in reward_metrics.items():
                agent_id_clean = agent_id.replace('-', '_')  # Prometheus-safe agent ID
                metrics_output.append(f'walker_reward_quality_score{{agent_id="{agent_id_clean}"}} {metrics.quality_score:.4f}')
                metrics_output.append(f'walker_reward_signal_to_noise{{agent_id="{agent_id_clean}"}} {metrics.signal_to_noise_ratio:.4f}')
        
        # Training system health
        metrics_output.append(f"# HELP walker_system_active Training system active status")
        metrics_output.append(f"# TYPE walker_system_active gauge")
        metrics_output.append(f"walker_system_active {1 if status.get('active', False) else 0}")
        
        return "\n".join(metrics_output) + "\n", 200, {'Content-Type': 'text/plain'}
        
    except Exception as e:
        # Return basic error metric
        error_output = [
            "# HELP walker_metrics_error Metrics collection error",
            "# TYPE walker_metrics_error gauge", 
            f"walker_metrics_error 1",
            f"# Error: {str(e)}"
        ]
        return "\n".join(error_output) + "\n", 200, {'Content-Type': 'text/plain'} 