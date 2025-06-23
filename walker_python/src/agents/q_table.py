"""
Q-learning table implementation for reinforcement learning agents.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import random
import pickle


class QTable:
    """Q-learning table for storing state-action values."""
    
    def __init__(self, state_dimensions: List[int], action_count: int, default_value: float = 0.0):
        """
        Initialize Q-table.
        
        Args:
            state_dimensions: List of dimensions for each state component
            action_count: Number of possible actions
            default_value: Default Q-value for new state-action pairs
        """
        self.state_dimensions = state_dimensions
        self.action_count = action_count
        self.default_value = default_value
        
        # Create multi-dimensional Q-table
        self.q_values = np.full(state_dimensions + [action_count], default_value, dtype=np.float32)
        
        # Visit counts for exploration bonus
        self.visit_counts = np.zeros(state_dimensions + [action_count], dtype=np.int32)
        
        # State-action mapping for sparse representation
        self.state_action_map = {}
    
    def get_q_value(self, state: Tuple[int, ...], action: int) -> float:
        """Get Q-value for state-action pair."""
        if len(state) != len(self.state_dimensions):
            raise ValueError(f"State dimensions {len(state)} don't match table dimensions {len(self.state_dimensions)}")
        
        if action >= self.action_count:
            raise ValueError(f"Action {action} exceeds action count {self.action_count}")
        
        return self.q_values[state + (action,)]
    
    def set_q_value(self, state: Tuple[int, ...], action: int, value: float):
        """Set Q-value for state-action pair."""
        if len(state) != len(self.state_dimensions):
            raise ValueError(f"State dimensions {len(state)} don't match table dimensions {len(self.state_dimensions)}")
        
        if action >= self.action_count:
            raise ValueError(f"Action {action} exceeds action count {self.action_count}")
        
        self.q_values[state + (action,)] = value
    
    def get_best_action(self, state: Tuple[int, ...]) -> Tuple[int, float]:
        """Get the best action and its Q-value for a given state."""
        if len(state) != len(self.state_dimensions):
            raise ValueError(f"State dimensions {len(state)} don't match table dimensions {len(self.state_dimensions)}")
        
        state_slice = self.q_values[state]
        best_action = int(np.argmax(state_slice))
        best_value = float(state_slice[best_action])
        
        return best_action, best_value
    
    def get_action_values(self, state: Tuple[int, ...]) -> np.ndarray:
        """Get all action values for a given state."""
        if len(state) != len(self.state_dimensions):
            raise ValueError(f"State dimensions {len(state)} don't match table dimensions {len(self.state_dimensions)}")
        
        return self.q_values[state].copy()
    
    def update_q_value(self, state: Tuple[int, ...], action: int, reward: float, 
                      next_state: Tuple[int, ...], learning_rate: float, discount_factor: float):
        """
        Update Q-value using Q-learning update rule.
        
        Q(s,a) = Q(s,a) + α[r + γ * max Q(s',a') - Q(s,a)]
        """
        current_q = self.get_q_value(state, action)
        next_best_q = self.get_best_action(next_state)[1]
        
        new_q = current_q + learning_rate * (reward + discount_factor * next_best_q - current_q)
        self.set_q_value(state, action, new_q)
        
        # Update visit count
        self.visit_counts[state + (action,)] += 1
    
    def get_visit_count(self, state: Tuple[int, ...], action: int) -> int:
        """Get visit count for state-action pair."""
        if len(state) != len(self.state_dimensions):
            raise ValueError(f"State dimensions {len(state)} don't match table dimensions {len(self.state_dimensions)}")
        
        if action >= self.action_count:
            raise ValueError(f"Action {action} exceeds action count {self.action_count}")
        
        return self.visit_counts[state + (action,)]
    
    def epsilon_greedy_action(self, state: Tuple[int, ...], epsilon: float) -> int:
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state
            epsilon: Probability of choosing random action
            
        Returns:
            Chosen action index
        """
        if random.random() < epsilon:
            return random.randint(0, self.action_count - 1)
        else:
            return self.get_best_action(state)[0]
    
    def ucb_action(self, state: Tuple[int, ...], exploration_constant: float = 1.0) -> int:
        """
        Choose action using Upper Confidence Bound (UCB) exploration.
        
        Args:
            state: Current state
            exploration_constant: UCB exploration constant
            
        Returns:
            Chosen action index
        """
        state_slice = self.q_values[state]
        visit_counts = self.visit_counts[state]
        
        # Calculate UCB values
        total_visits = np.sum(visit_counts)
        if total_visits == 0:
            return random.randint(0, self.action_count - 1)
        
        ucb_values = state_slice + exploration_constant * np.sqrt(np.log(total_visits + 1) / (visit_counts + 1))
        return int(np.argmax(ucb_values))
    
    def reset(self):
        """Reset all Q-values to default."""
        self.q_values.fill(self.default_value)
        self.visit_counts.fill(0)
        self.state_action_map.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the Q-table."""
        return {
            'total_entries': self.q_values.size,
            'non_zero_entries': np.count_nonzero(self.q_values),
            'min_value': float(np.min(self.q_values)),
            'max_value': float(np.max(self.q_values)),
            'mean_value': float(np.mean(self.q_values)),
            'std_value': float(np.std(self.q_values)),
            'total_visits': int(np.sum(self.visit_counts)),
            'max_visits': int(np.max(self.visit_counts)),
            'mean_visits': float(np.mean(self.visit_counts)),
        }
    
    def save(self, filename: str):
        """Save Q-table to file."""
        np.savez(filename, 
                 q_values=self.q_values, 
                 visit_counts=self.visit_counts,
                 state_dimensions=np.array(self.state_dimensions),
                 action_count=self.action_count,
                 default_value=self.default_value)
    
    def load(self, filename: str):
        """Load Q-table from file."""
        data = np.load(filename)
        self.q_values = data['q_values']
        self.visit_counts = data['visit_counts']
        self.state_dimensions = data['state_dimensions'].tolist()
        self.action_count = int(data['action_count'])
        self.default_value = float(data['default_value'])


class SparseQTable:
    """Sparse Q-table implementation using dictionary for memory efficiency."""
    
    def __init__(self, action_count: int, default_value: float = 0.0):
        """
        Initialize sparse Q-table.
        
        Args:
            action_count: Number of possible actions
            default_value: Default Q-value for new state-action pairs
        """
        self.action_count = action_count
        self.default_value = default_value
        self.q_values = {}  # state -> action_values
        self.visit_counts = {}  # state -> action_counts
        self.update_count = 0
    
    def _get_state_key(self, state: Tuple[int, ...]) -> str:
        """Convert state tuple to string key."""
        return ','.join(map(str, state))
    
    def _ensure_state_exists(self, state: Tuple[int, ...]):
        """Ensure state exists in the table."""
        state_key = self._get_state_key(state)
        if state_key not in self.q_values:
            self.q_values[state_key] = [self.default_value] * self.action_count
            self.visit_counts[state_key] = [0] * self.action_count
    
    def get_q_value(self, state: Tuple[int, ...], action: int) -> float:
        """Get Q-value for state-action pair."""
        if action >= self.action_count:
            raise ValueError(f"Action {action} exceeds action count {self.action_count}")
        
        state_key = self._get_state_key(state)
        if state_key not in self.q_values:
            return self.default_value
        
        return self.q_values[state_key][action]
    
    def set_q_value(self, state: Tuple[int, ...], action: int, value: float):
        """Set Q-value for state-action pair with bounds checking."""
        if action >= self.action_count:
            raise ValueError(f"Action {action} exceeds action count {self.action_count}")
        
        # BOUND the Q-value to prevent explosion
        value = np.clip(value, -10.0, 10.0)  # Conservative bounds for Q-values
        
        self._ensure_state_exists(state)
        state_key = self._get_state_key(state)
        self.q_values[state_key][action] = value
        self.update_count += 1
    
    def get_best_action(self, state: Tuple[int, ...]) -> Tuple[int, float]:
        """Get the best action and its Q-value for a given state."""
        state_key = self._get_state_key(state)
        if state_key not in self.q_values:
            return 0, self.default_value
        
        action_values = self.q_values[state_key]
        best_action = int(max(range(self.action_count), key=lambda a: action_values[a]))
        best_value = float(action_values[best_action])
        
        return best_action, best_value
    
    def get_action_values(self, state: Tuple[int, ...]) -> List[float]:
        """Get all action values for a given state."""
        state_key = self._get_state_key(state)
        if state_key not in self.q_values:
            return [self.default_value] * self.action_count
        
        return self.q_values[state_key].copy()
    
    def update_q_value(self, state: Tuple[int, ...], action: int, reward: float, 
                      next_state: Tuple[int, ...], learning_rate: float, discount_factor: float):
        """Update Q-value using Q-learning update rule."""
        current_q = self.get_q_value(state, action)
        next_best_q = self.get_best_action(next_state)[1]
        
        new_q = current_q + learning_rate * (reward + discount_factor * next_best_q - current_q)
        self.set_q_value(state, action, new_q)
        
        # Update visit count
        state_key = self._get_state_key(state)
        self.visit_counts[state_key][action] += 1
    
    def get_visit_count(self, state: Tuple[int, ...], action: int) -> int:
        """Get visit count for state-action pair."""
        state_key = self._get_state_key(state)
        if state_key not in self.visit_counts:
            return 0
        
        return self.visit_counts[state_key][action]
    
    def epsilon_greedy_action(self, state: Tuple[int, ...], epsilon: float) -> int:
        """Choose action using epsilon-greedy policy."""
        if random.random() < epsilon:
            return random.randint(0, self.action_count - 1)
        else:
            return self.get_best_action(state)[0]
    
    def reset(self):
        """Reset all Q-values to default."""
        self.q_values.clear()
        self.visit_counts.clear()
        self.update_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the Q-table."""
        if not self.q_values:
            return {
                'total_states': 0,
                'min_value': 0.0,
                'max_value': 0.0,
                'mean_value': 0.0,
                'std_value': 0.0,
                'total_visits': 0,
                'max_visits': 0,
                'mean_visits': 0.0,
                'update_count': self.update_count,
            }

        all_values = [v for action_values in self.q_values.values() for v in action_values]
        all_visits = [v for visit_counts in self.visit_counts.values() for v in visit_counts]
        
        return {
            'total_states': len(self.q_values),
            'min_value': float(min(all_values)),
            'max_value': float(max(all_values)),
            'mean_value': float(sum(all_values) / len(all_values)),
            'std_value': float(np.std(all_values)),
            'total_visits': int(sum(all_visits)),
            'max_visits': int(max(all_visits)),
            'mean_visits': float(sum(all_visits) / len(all_visits)),
            'update_count': self.update_count,
        }
    
    def get_best_q_values_for_all_states(self) -> Dict[str, float]:
        """Get the best Q-value for each state in the table."""
        best_values = {}
        for state_key, action_values in self.q_values.items():
            best_values[state_key] = float(max(action_values))
        return best_values
    
    def save(self, filename: str):
        """Save Q-table to file."""
        with open(filename, 'wb') as f:
            pickle.dump({
                'q_values': self.q_values,
                'visit_counts': self.visit_counts,
                'action_count': self.action_count,
                'default_value': self.default_value,
                'update_count': self.update_count,
            }, f)
    
    def load(self, filename: str):
        """Load Q-table from file."""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.q_values = data['q_values']
        self.visit_counts = data['visit_counts']
        self.action_count = data['action_count']
        self.default_value = data['default_value']
        self.update_count = data.get('update_count', 0) 