"""
Basic Q-learning agent implementation.
"""

from .base_agent import BaseAgent
from .q_table import QTable, SparseQTable
from typing import List, Tuple
import numpy as np
import random

class BasicAgent(BaseAgent):
    """Basic Q-learning agent with discrete state and action spaces."""
    def __init__(self, state_dimensions: List[int], action_count: int):
        super().__init__()
        self.state_dimensions = state_dimensions
        self.action_count = action_count
        self.q_table = QTable(state_dimensions, action_count)
        self.current_state = tuple([0] * len(state_dimensions))
        self.last_state = self.current_state
        self.last_action = 0
        self.last_reward = 0.0

    def get_state(self) -> List[float]:
        """Return the current state as a list of floats (override in subclasses)."""
        return list(self.current_state)

    def take_action(self, action: int):
        """Take the specified action (override in subclasses for environment interaction)."""
        self.last_action = action
        # In a real environment, this would affect the agent/robot
        pass

    def get_reward(self) -> float:
        """Return the reward for the current state (override in subclasses)."""
        return self.last_reward

    def select_action(self, epsilon: float = None) -> int:
        """Select an action using epsilon-greedy policy."""
        if epsilon is None:
            epsilon = self.randomness
        return self.q_table.epsilon_greedy_action(self.current_state, epsilon)

    def update(self, delta_time: float):
        """Update the agent's Q-table based on the last transition."""
        # Get new state and reward (should be set by environment)
        state = self.current_state
        action = self.last_action
        reward = self.get_reward()
        next_state = state  # In a real environment, update this after action
        
        # Q-learning update
        self.q_table.update_q_value(
            state, action, reward, next_state,
            self.learning_rate, self.future_discount
        )
        
        # Optionally decay learning rate and randomness
        self.learning_rate *= self.learning_rate_decay
        self.randomness = max(self.min_randomness, self.randomness * 0.999)

    def set_state(self, state: Tuple[int, ...]):
        self.last_state = self.current_state
        self.current_state = state

    def set_reward(self, reward: float):
        self.last_reward = reward

    def reset(self):
        self.q_table.reset()
        self.current_state = tuple([0] * len(self.state_dimensions))
        self.last_state = self.current_state
        self.last_action = 0
        self.last_reward = 0.0

    def get_stats(self):
        stats = super().get_stats()
        q_stats = self.q_table.get_stats()
        stats.append(f"Q-table mean: {q_stats['mean_value']:.4f}")
        stats.append(f"Q-table max: {q_stats['max_value']:.4f}")
        stats.append(f"Q-table min: {q_stats['min_value']:.4f}")
        return stats 