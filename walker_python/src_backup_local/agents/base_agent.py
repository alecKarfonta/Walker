"""
Base agent class for reinforcement learning agents.
"""

import random
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Abstract base class for all reinforcement learning agents."""
    
    def __init__(self):
        # Unique identifier for this agent
        self.id = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
        
        # Learning parameters
        self.learning_rate = 0.01
        self.min_learning_rate = 0.001
        self.max_learning_rate = 1.0
        self.learning_rate_decay = 0.9999
        
        # Exploration parameters
        self.randomness = 0.2
        self.min_randomness = 0.001
        self.max_randomness = 0.2
        
        # Q-learning parameters
        self.future_discount = 0.5
        self.exploration_bonus = 100.0
        self.impatience = 0.0001
        self.mutation_rate = 0.01
        
        # Agent state
        self.is_manual_control = False
        self.is_debug = False
        self.action_count = 6
        self.goal = 0
        self.goals = ["Move Right", "Move Left", "Go Home"]
        
        # Performance tracking
        self.best_value = 0.0
        self.worst_value = 0.0
        self.value_delta = 0.0
        self.value_error = 0.0
        self.value_velocity = 0.0
        self.time_since_good_value = 0.0
        
        # Memory
        self.memory_count = 7
        self.previous_actions = deque(maxlen=self.memory_count)
        self.previous_values = deque(maxlen=self.memory_count)
        self.previous_states = deque(maxlen=self.memory_count)
        
        # Update timing
        self.update_time = 0.0
        self.update_timer = 0.1
        
        # Previous state tracking
        self.previous_action = 0
        self.previous_velocity = 0.0
        self.previous_value = 0.0
        self.previous_max_action = 0
        self.previous_max_action_value = 0.0
        self.previous_q_value = 0.0
    
    def init(self, with_randomness: bool = False):
        """Initialize the agent with learning parameters."""
        self.is_manual_control = False
        
        if with_randomness:
            self.learning_rate = min(0.8, random.random())
            self.learning_rate_decay += 0.00001 * random.random()
            self.future_discount = 0.5 + (0.5 * random.random())
            self.update_timer *= random.random()
            self.impatience *= random.random()
    
    def fast_sigmoid(self, x: float) -> float:
        """Fast sigmoid function approximation."""
        if x < -10:
            return 0
        elif x > 10:
            return 1
        else:
            return 1 / (1 + (2.718281828459045 ** (-x)))
    
    def cycle_goal(self):
        """Cycle to the next goal."""
        self.goal = (self.goal + 1) % len(self.goals)
    
    def update_timing(self, delta_time: float):
        """Update the agent's timing."""
        self.update_time += delta_time
        
        if self.update_time > self.update_timer:
            self.update_time = 0.0
            return True
        return False
    
    @abstractmethod
    def get_state(self) -> List[float]:
        """Get the current state representation."""
        pass
    
    @abstractmethod
    def take_action(self, action: int):
        """Take the specified action."""
        pass
    
    @abstractmethod
    def get_reward(self) -> float:
        """Calculate the reward for the current state."""
        pass
    
    @abstractmethod
    def update(self, delta_time: float):
        """Update the agent's learning and behavior."""
        pass
    
    # Getters and setters
    def get_learning_rate(self) -> float:
        return self.learning_rate
    
    def set_learning_rate(self, learning_rate: float):
        self.learning_rate = max(self.min_learning_rate, min(self.max_learning_rate, learning_rate))
    
    def get_randomness(self) -> float:
        return self.randomness
    
    def set_randomness(self, randomness: float):
        self.randomness = max(self.min_randomness, min(self.max_randomness, randomness))
    
    def get_future_discount(self) -> float:
        return self.future_discount
    
    def set_future_discount(self, future_discount: float):
        self.future_discount = max(0.0, min(1.0, future_discount))
    
    def get_exploration_bonus(self) -> float:
        return self.exploration_bonus
    
    def set_exploration_bonus(self, exploration_bonus: float):
        self.exploration_bonus = exploration_bonus
    
    def get_manual_control(self) -> bool:
        return self.is_manual_control
    
    def set_manual_control(self, manual_control: bool):
        self.is_manual_control = manual_control
    
    def get_update_timer(self) -> float:
        return self.update_time
    
    def set_update_timer(self, update_timer: float):
        self.update_timer = update_timer
    
    def get_goals(self) -> List[str]:
        return self.goals
    
    def set_goals(self, goals: List[str]):
        self.goals = goals
    
    def get_goal(self) -> int:
        return self.goal
    
    def set_goal(self, goal: int):
        self.goal = goal
    
    def get_mutation_rate(self) -> float:
        return self.mutation_rate
    
    def set_mutation_rate(self, mutation_rate: float):
        self.mutation_rate = mutation_rate
    
    def get_impatience(self) -> float:
        return self.impatience
    
    def set_impatience(self, impatience: float):
        self.impatience = impatience
    
    def get_stats(self) -> List[str]:
        """Get agent statistics for display."""
        stats = [
            f"Learning Rate: {self.learning_rate:.4f}",
            f"Randomness: {self.randomness:.4f}",
            f"Future Discount: {self.future_discount:.2f}",
            f"Best Value: {self.best_value:.2f}",
            f"Worst Value: {self.worst_value:.2f}",
            f"Time Since Good: {self.time_since_good_value:.1f}",
            f"Goal: {self.goal}",
            f"Manual Control: {self.is_manual_control}",
        ]
        return stats 