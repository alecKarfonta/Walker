"""
Q-Learning Configuration
Centralized configuration for all Q-learning approaches to ensure consistency.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class QLearningConfig:
    """Configuration for Q-learning algorithms."""
    
    # Learning parameters
    learning_rate: float = 0.01
    min_learning_rate: float = 0.001
    max_learning_rate: float = 0.3
    learning_rate_decay: float = 0.9999
    discount_factor: float = 0.9
    
    # Exploration parameters
    epsilon: float = 0.3
    min_epsilon: float = 0.01
    max_epsilon: float = 0.6
    epsilon_decay: float = 0.9999
    
    # Q-value bounds (critical for stability)
    min_q_value: float = -5.0
    max_q_value: float = 5.0
    
    # Reward scaling (ensures consistency across approaches)
    reward_clip_min: float = -0.5
    reward_clip_max: float = 0.5
    
    # Experience replay
    replay_buffer_size: int = 3000
    batch_size: int = 32
    replay_frequency: int = 12
    
    # Q-table management
    max_q_table_states: int = 1500
    q_table_pruning_threshold: int = 1800
    confidence_threshold: int = 15
    exploration_bonus: float = 0.15


@dataclass
class SurvivalQLearningConfig(QLearningConfig):
    """Extended configuration for survival Q-learning."""
    
    # Survival-specific parameters
    energy_gain_reward_scale: float = 5.0  # Was 20.0, now scaled down
    food_approach_reward_scale: float = 1.0  # Was 3.0, now scaled down
    movement_reward_scale: float = 0.5
    survival_penalty_scale: float = 0.5
    
    # State space dimensions
    shoulder_bins: int = 8
    elbow_bins: int = 8
    energy_bins: int = 5
    food_direction_bins: int = 8
    food_distance_bins: int = 5
    velocity_bins: int = 4
    contact_bins: int = 2
    social_bins: int = 4
    
    # Curriculum learning thresholds
    basic_movement_threshold: int = 500
    food_seeking_threshold: int = 1500
    
    # Enhanced Q-table settings for survival
    confidence_threshold: int = 10  # Faster confidence for complex state space
    exploration_bonus: float = 0.2  # Higher bonus for exploration


@dataclass
class DeepQLearningConfig:
    """Configuration for deep Q-learning approach."""
    
    # Network architecture
    state_dim: int = 15
    hidden_dims: Optional[List[int]] = None  # Will default to [256, 256, 128]
    
    # Training parameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 150000
    
    # Experience replay
    buffer_size: int = 25000
    batch_size: int = 32
    target_update_freq: int = 2000
    
    # Training frequency (to balance performance) - Increased by 25% for longer intervals
    experience_collection_freq: int = 10
    training_freq: int = 18750  # Increased from 15000 (25% longer interval)
    min_buffer_size: int = 5000
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256, 128]


# Global configurations
BASIC_Q_CONFIG = QLearningConfig()
SURVIVAL_Q_CONFIG = SurvivalQLearningConfig()
DEEP_Q_CONFIG = DeepQLearningConfig()


def get_config(approach: str) -> Dict[str, Any]:
    """
    Get configuration for specified Q-learning approach.
    
    Args:
        approach: 'basic', 'survival', or 'deep'
        
    Returns:
        Configuration dictionary
    """
    configs = {
        'basic': BASIC_Q_CONFIG,
        'survival': SURVIVAL_Q_CONFIG,
        'deep': DEEP_Q_CONFIG
    }
    
    config = configs.get(approach, BASIC_Q_CONFIG)
    
    # Convert dataclass to dict for compatibility
    if hasattr(config, '__dict__'):
        return config.__dict__
    return config


def validate_reward_scaling():
    """Validate that reward scales are consistent across approaches."""
    basic_max = BASIC_Q_CONFIG.reward_clip_max
    survival_max = SURVIVAL_Q_CONFIG.reward_clip_max
    
    if abs(basic_max - survival_max) > 1.0:
        print(f"⚠️ WARNING: Reward scaling mismatch! Basic: {basic_max}, Survival: {survival_max}")
        return False
    
    print(f"✅ Reward scaling consistent: {basic_max}")
    return True


def print_config_summary():
    """Print summary of Q-learning configurations."""
    print("🧠 Q-Learning Configuration Summary:")
    print(f"   Basic Q-learning: LR={BASIC_Q_CONFIG.learning_rate}, ε={BASIC_Q_CONFIG.epsilon}")
    print(f"   Survival Q-learning: Energy scale={SURVIVAL_Q_CONFIG.energy_gain_reward_scale}")
    print(f"   Deep Q-learning: State dim={DEEP_Q_CONFIG.state_dim}, Hidden={DEEP_Q_CONFIG.hidden_dims}")
    print(f"   Reward bounds: [{BASIC_Q_CONFIG.reward_clip_min}, {BASIC_Q_CONFIG.reward_clip_max}]")
    
    validate_reward_scaling()


if __name__ == "__main__":
    print_config_summary() 