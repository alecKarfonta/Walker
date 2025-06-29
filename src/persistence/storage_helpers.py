"""
Storage Helper Functions

Helper functions for robot data extraction, restoration, and file operations.
"""

import json
import gzip
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from copy import deepcopy

# Use TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.crawling_agent import CrawlingAgent
    from .robot_storage import RobotState, PerformanceHistory
else:
    # Import at runtime to avoid circular dependency issues
    try:
        from src.agents.crawling_agent import CrawlingAgent
        from .robot_storage import RobotState, PerformanceHistory
    except ImportError:
        # Fallback for linting/analysis
        CrawlingAgent = None
        RobotState = None 
        PerformanceHistory = None


def extract_neural_network_data(agent: CrawlingAgent) -> Dict[str, Any]:
    """Extract neural network data from agent."""
    if not hasattr(agent, '_learning_system') or agent._learning_system is None:
        return {}
    
    learning_system = agent._learning_system
    neural_data = {}
    
    try:
        # Extract neural network weights
        if hasattr(learning_system, 'state_dict'):
            neural_data['network_weights'] = learning_system.state_dict()
        
        # Extract experience buffer (limit size)
        if hasattr(learning_system, 'memory') and hasattr(learning_system.memory, 'buffer'):
            buffer_size = min(10000, len(learning_system.memory.buffer))  # Limit to 10k experiences
            neural_data['experience_buffer'] = list(learning_system.memory.buffer)[-buffer_size:] if buffer_size > 0 else []
        
        # Extract training statistics
        neural_data['training_stats'] = {
            'epsilon': getattr(learning_system, 'epsilon', 0.3),
            'training_steps': getattr(learning_system, '_training_steps', 0),
            'loss_history': getattr(learning_system, '_loss_history', [])[-100:]  # Last 100 losses
        }
        
        # Extract attention-specific data
        if hasattr(learning_system, 'attention_history'):
            neural_data['attention_history'] = list(learning_system.attention_history)[-100:]  # Last 100 attention records
            
    except Exception as e:
        print(f"⚠️ Error extracting neural network data: {e}")
    
    return neural_data


def extract_learning_parameters(agent: CrawlingAgent) -> Dict[str, Any]:
    """Extract learning parameters from agent."""
    return {
        'learning_rate': getattr(agent, 'learning_rate', 0.005),
        'epsilon': getattr(agent, 'epsilon', 0.3),
        'min_epsilon': getattr(agent, 'min_epsilon', 0.01),
        'max_epsilon': getattr(agent, 'max_epsilon', 0.6),
        'discount_factor': getattr(agent, 'discount_factor', 0.9),
        'epsilon_decay': getattr(agent, 'epsilon_decay', 0.9999),
        'current_goal': getattr(agent, 'current_goal', 0),
        'goal_weights': getattr(agent, 'goal_weights', {}),
        'action_interval': getattr(agent, 'action_interval', 2),
        'learning_interval': getattr(agent, 'learning_interval', 30)
    }


def extract_performance_metrics(agent: CrawlingAgent) -> Dict[str, Any]:
    """Extract current performance metrics from agent."""
    return {
        'total_reward': getattr(agent, 'total_reward', 0.0),
        'max_speed': getattr(agent, 'max_speed', 0.0),
        'steps': getattr(agent, 'steps', 0),
        'best_reward_received': getattr(agent, 'best_reward_received', 0.0),
        'worst_reward_received': getattr(agent, 'worst_reward_received', 0.0),
        'time_since_good_value': getattr(agent, 'time_since_good_value', 0.0),
        'learning_progress': getattr(agent, 'total_reward', 0.0) / max(1, getattr(agent, 'steps', 1)),  # Reward per step as progress metric
        'current_position': (agent.body.position.x, agent.body.position.y) if agent.body else (0.0, 0.0),
        'distance_traveled': agent.body.position.x - agent.initial_position[0] if agent.body else 0.0,
        'stability_score': max(0, 1.0 - abs(agent.body.angle)) if agent.body else 0.0
    }


def create_performance_history(agent: CrawlingAgent, existing_history: Optional[PerformanceHistory] = None) -> PerformanceHistory:
    """Create or update performance history for agent."""
    robot_id = str(agent.id)
    
    if existing_history:
        # Update existing history
        existing_history.update_from_agent(agent)
        return existing_history
    else:
        # Create new history
        history = PerformanceHistory(
            robot_id=robot_id,
            creation_timestamp=time.time(),
            last_updated=time.time()
        )
        history.update_from_agent(agent)
        return history


def restore_neural_network_data(agent: CrawlingAgent, neural_data: Dict[str, Any]):
    """Restore neural network data to agent."""
    if not neural_data or not hasattr(agent, '_learning_system') or agent._learning_system is None:
        return
    
    learning_system = agent._learning_system
    
    try:
        # Restore neural network weights
        if 'network_weights' in neural_data and hasattr(learning_system, 'load_state_dict'):
            learning_system.load_state_dict(neural_data['network_weights'])
        
        # Restore experience buffer
        if 'experience_buffer' in neural_data and hasattr(learning_system, 'memory'):
            for experience in neural_data['experience_buffer']:
                try:
                    learning_system.memory.push(*experience)
                except Exception as e:
                    continue  # Skip invalid experiences
        
        # Restore training statistics
        if 'training_stats' in neural_data:
            stats = neural_data['training_stats']
            if 'epsilon' in stats and hasattr(learning_system, 'epsilon'):
                learning_system.epsilon = stats['epsilon']
            if 'training_steps' in stats and hasattr(learning_system, '_training_steps'):
                learning_system._training_steps = stats['training_steps']
            if 'loss_history' in stats and hasattr(learning_system, '_loss_history'):
                learning_system._loss_history = stats['loss_history']
        
        # Restore attention history
        if 'attention_history' in neural_data and hasattr(learning_system, 'attention_history'):
            from collections import deque
            learning_system.attention_history = deque(neural_data['attention_history'], maxlen=learning_system.attention_history.maxlen)
            
    except Exception as e:
        print(f"⚠️ Error restoring neural network data: {e}")


def restore_learning_parameters(agent: CrawlingAgent, learning_params: Dict[str, Any]):
    """Restore learning parameters to agent."""
    for param, value in learning_params.items():
        if hasattr(agent, param):
            setattr(agent, param, value)


def restore_performance_metrics(agent: CrawlingAgent, perf_metrics: Dict[str, Any]):
    """Restore performance metrics to agent."""
    for metric, value in perf_metrics.items():
        if hasattr(agent, metric):
            setattr(agent, metric, value)


def save_state_to_file(state: RobotState, filepath: Path, use_compression: bool = True):
    """Save robot state to file."""
    if use_compression and filepath.suffix != '.gz':
        filepath = filepath.parent / f"{filepath.name}.gz"
    
    if use_compression:
        with gzip.open(filepath, 'wt') as f:
            json.dump(state.to_dict(), f, indent=2, default=str)
    else:
        with open(filepath, 'w') as f:
            json.dump(state.to_dict(), f, indent=2, default=str)


def load_state_from_file(filepath: Path) -> RobotState:
    """Load robot state from file."""
    if filepath.suffix == '.gz':
        with gzip.open(filepath, 'rt') as f:
            data = json.load(f)
    else:
        with open(filepath, 'r') as f:
            data = json.load(f)
    
    return RobotState.from_dict(data)


def find_robot_file(storage_dir: Path, robot_id_or_filename: str) -> Optional[Path]:
    """Find robot file by ID or filename."""
    robots_dir = storage_dir / "robots"
    
    # Try exact filename first
    possible_files = [
        robots_dir / f"{robot_id_or_filename}.json",
        robots_dir / f"{robot_id_or_filename}.json.gz"
    ]
    
    # Try pattern matching for robot ID
    pattern_files = list(robots_dir.glob(f"{robot_id_or_filename}_*.json*"))
    possible_files.extend(pattern_files)
    
    # Find the first existing file
    for file_path in possible_files:
        if file_path.exists():
            return file_path
    
    return None 