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
    from src.agents.evolutionary_crawling_agent import EvolutionaryCrawlingAgent
    from .robot_storage import RobotState, PerformanceHistory
else:
    # Import at runtime to avoid circular dependency issues
    try:
        from src.agents.evolutionary_crawling_agent import EvolutionaryCrawlingAgent
        from .robot_storage import RobotState, PerformanceHistory
    except ImportError:
        # Fallback for linting/analysis
        EvolutionaryCrawlingAgent = None
        RobotState = None 
        PerformanceHistory = None


def extract_q_table_data(agent: EvolutionaryCrawlingAgent) -> Dict[str, Any]:
    """Extract Q-table data from agent."""
    if not hasattr(agent, 'q_table'):
        return {}
    
    q_table = agent.q_table
    return {
        'q_values': dict(q_table.q_values) if hasattr(q_table, 'q_values') else {},
        'state_coverage': list(q_table.state_coverage) if hasattr(q_table, 'state_coverage') else [],
        'confidence_threshold': getattr(q_table, 'confidence_threshold', 15),
        'exploration_bonus': getattr(q_table, 'exploration_bonus', 0.15),
        'action_count': getattr(q_table, 'action_count', 6),
        'default_value': getattr(q_table, 'default_value', 0.0)
    }


def extract_learning_parameters(agent: EvolutionaryCrawlingAgent) -> Dict[str, Any]:
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


def extract_performance_metrics(agent: EvolutionaryCrawlingAgent) -> Dict[str, Any]:
    """Extract current performance metrics from agent."""
    return {
        'total_reward': getattr(agent, 'total_reward', 0.0),
        'max_speed': getattr(agent, 'max_speed', 0.0),
        'steps': getattr(agent, 'steps', 0),
        'best_reward_received': getattr(agent, 'best_reward_received', 0.0),
        'worst_reward_received': getattr(agent, 'worst_reward_received', 0.0),
        'time_since_good_value': getattr(agent, 'time_since_good_value', 0.0),
        'convergence_estimate': agent.q_table.get_convergence_estimate() if hasattr(agent, 'q_table') else 0.0,
        'current_position': (agent.body.position.x, agent.body.position.y) if agent.body else (0.0, 0.0),
        'distance_traveled': agent.body.position.x - agent.initial_position[0] if agent.body else 0.0,
        'stability_score': max(0, 1.0 - abs(agent.body.angle)) if agent.body else 0.0
    }


def create_performance_history(agent: EvolutionaryCrawlingAgent, existing_history: Optional[PerformanceHistory] = None) -> PerformanceHistory:
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


def restore_q_table_data(agent: EvolutionaryCrawlingAgent, q_data: Dict[str, Any]):
    """Restore Q-table data to agent."""
    if not q_data or not hasattr(agent, 'q_table'):
        return
    
    q_table = agent.q_table
    
    # Restore Q-values
    if 'q_values' in q_data:
        q_table.q_values = dict(q_data['q_values'])
    
    # Restore state coverage
    if 'state_coverage' in q_data:
        q_table.state_coverage = set(q_data['state_coverage'])
    
    # Restore parameters
    if 'confidence_threshold' in q_data:
        q_table.confidence_threshold = q_data['confidence_threshold']
    if 'exploration_bonus' in q_data:
        q_table.exploration_bonus = q_data['exploration_bonus']


def restore_learning_parameters(agent: EvolutionaryCrawlingAgent, learning_params: Dict[str, Any]):
    """Restore learning parameters to agent."""
    for param, value in learning_params.items():
        if hasattr(agent, param):
            setattr(agent, param, value)


def restore_performance_metrics(agent: EvolutionaryCrawlingAgent, perf_metrics: Dict[str, Any]):
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