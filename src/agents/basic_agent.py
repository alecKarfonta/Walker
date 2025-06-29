"""
Basic Q-learning agent implementation.
"""

from .base_agent import BaseAgent
from typing import List, Tuple
import numpy as np
import random

class BasicAgent(BaseAgent):
    """Basic Q-learning agent with discrete state and action spaces."""
    def __init__(self, state_dimensions: List[int], action_count: int):
        super().__init__()
        self.state_dimensions = state_dimensions
        self.action_count = action_count
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
        pass

    def update(self, delta_time: float):
        """Update the agent's Q-table based on the last transition."""
        # Get new state and reward (should be set by environment)
        pass

    def set_state(self, state: Tuple[int, ...]):
        self.last_state = self.current_state
        self.current_state = state

    def set_reward(self, reward: float):
        self.last_reward = reward

    def reset(self):
        self.current_state = tuple([0] * len(self.state_dimensions))
        self.last_state = self.current_state
        self.last_action = 0
        self.last_reward = 0.0

    def get_stats(self):
        stats = super().get_stats()
        return stats 