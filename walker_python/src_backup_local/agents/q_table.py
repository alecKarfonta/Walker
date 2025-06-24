"""
Q-learning table implementation for reinforcement learning agents.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import random
import pickle
import math


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


class EnhancedQTable(SparseQTable):
    """
    Enhanced Q-table with adaptive learning rates, exploration bonuses, 
    and confidence-based methods inspired by the Java implementation.
    """
    
    def __init__(self, action_count: int, default_value: float = 0.0, 
                 confidence_threshold: int = 10, exploration_bonus: float = 0.1):
        """
        Initialize enhanced Q-table.
        
        Args:
            action_count: Number of possible actions
            default_value: Default Q-value for new state-action pairs
            confidence_threshold: Visits needed for confidence-based decisions
            exploration_bonus: Bonus for under-explored state-actions
        """
        super().__init__(action_count, default_value)
        self.confidence_threshold = confidence_threshold
        self.exploration_bonus = exploration_bonus
        
        # Track performance metrics
        self.q_value_history = []  # Track Q-value changes for convergence analysis
        self.state_coverage = set()  # Track unique states visited
        self.action_preferences = {}  # Track action selection patterns
        
        # Adaptive parameters
        self.min_learning_rate = 0.01
        self.max_learning_rate = 0.3
        self.base_learning_rate = 0.1
        
    def get_adaptive_learning_rate(self, state: Tuple[int, ...], action: int, 
                                 base_lr: Optional[float] = None) -> float:
        """
        Get adaptive learning rate based on visit count (Java-inspired).
        
        Args:
            state: Current state
            action: Action taken
            base_lr: Base learning rate (uses instance default if None)
            
        Returns:
            Adaptive learning rate
        """
        if base_lr is None:
            base_lr = self.base_learning_rate
            
        visit_count = self.get_visit_count(state, action)
        
        # Decrease learning rate as visits increase (like Java implementation)
        adaptive_lr = base_lr / (1 + visit_count * 0.05)
        
        # Bound the learning rate
        return np.clip(adaptive_lr, self.min_learning_rate, self.max_learning_rate)
    
    def get_exploration_bonus(self, state: Tuple[int, ...], action: int) -> float:
        """
        Get exploration bonus for under-explored state-actions.
        
        Args:
            state: Current state
            action: Action to evaluate
            
        Returns:
            Exploration bonus value
        """
        visit_count = self.get_visit_count(state, action)
        
        if visit_count < self.confidence_threshold:
            # Provide bonus inversely proportional to visit count
            bonus = self.exploration_bonus * (self.confidence_threshold - visit_count) / self.confidence_threshold
            return bonus
        
        return 0.0
    
    def confidence_based_action(self, state: Tuple[int, ...], 
                              min_confidence: Optional[int] = None) -> Tuple[int, float, bool]:
        """
        Choose action based on confidence (visit count).
        
        Args:
            state: Current state
            min_confidence: Minimum visits required for confidence
            
        Returns:
            Tuple of (action, q_value, is_confident)
        """
        if min_confidence is None:
            min_confidence = self.confidence_threshold
            
        state_key = self._get_state_key(state)
        
        if state_key not in self.q_values:
            # New state - not confident, return random action
            return random.randint(0, self.action_count - 1), self.default_value, False
        
        action_values = self.q_values[state_key]
        visit_counts = self.visit_counts[state_key]
        
        # Find actions with sufficient confidence
        confident_actions = [i for i, count in enumerate(visit_counts) if count >= min_confidence]
        
        if confident_actions:
            # Choose best action among confident ones
            best_confident_action = max(confident_actions, key=lambda a: action_values[a])
            return best_confident_action, action_values[best_confident_action], True
        else:
            # No confident actions - return best action but mark as not confident
            best_action = max(range(self.action_count), key=lambda a: action_values[a])
            return best_action, action_values[best_action], False
    
    def enhanced_epsilon_greedy(self, state: Tuple[int, ...], epsilon: float,
                              use_exploration_bonus: bool = True) -> int:
        """
        Enhanced epsilon-greedy with exploration bonuses.
        
        Args:
            state: Current state
            epsilon: Probability of random exploration
            use_exploration_bonus: Whether to use exploration bonuses
            
        Returns:
            Chosen action index
        """
        if random.random() < epsilon:
            # Exploration: but bias towards less-visited actions
            if use_exploration_bonus:
                state_key = self._get_state_key(state)
                if state_key in self.visit_counts:
                    visit_counts = self.visit_counts[state_key]
                    # Inverse probability based on visit counts
                    total_inverse_visits = sum(1.0 / (count + 1) for count in visit_counts)
                    probs = [(1.0 / (count + 1)) / total_inverse_visits for count in visit_counts]
                    
                    # Sample based on inverse visit probabilities
                    return np.random.choice(self.action_count, p=probs)
            
            # Standard random exploration
            return random.randint(0, self.action_count - 1)
        else:
            # Exploitation: choose best action with exploration bonus
            if use_exploration_bonus:
                return self._get_best_action_with_bonus(state)
            else:
                return self.get_best_action(state)[0]
    
    def _get_best_action_with_bonus(self, state: Tuple[int, ...]) -> int:
        """Get best action considering exploration bonuses."""
        state_key = self._get_state_key(state)
        
        if state_key not in self.q_values:
            return random.randint(0, self.action_count - 1)
        
        action_values = self.q_values[state_key]
        
        # Add exploration bonuses to Q-values
        enhanced_values = []
        for action in range(self.action_count):
            q_value = action_values[action]
            bonus = self.get_exploration_bonus(state, action)
            enhanced_values.append(q_value + bonus)
        
        return int(np.argmax(enhanced_values))
    
    def update_q_value_enhanced(self, state: Tuple[int, ...], action: int, reward: float, 
                              next_state: Tuple[int, ...], base_learning_rate: float, 
                              discount_factor: float, use_adaptive_lr: bool = True):
        """
        Enhanced Q-value update with adaptive learning rate and tracking.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            base_learning_rate: Base learning rate
            discount_factor: Discount factor
            use_adaptive_lr: Whether to use adaptive learning rate
        """
        # Get current Q-value
        current_q = self.get_q_value(state, action)
        
        # Get best next Q-value
        next_best_q = self.get_best_action(next_state)[1]
        
        # Use adaptive learning rate if enabled
        if use_adaptive_lr:
            learning_rate = self.get_adaptive_learning_rate(state, action, base_learning_rate)
        else:
            learning_rate = base_learning_rate
        
        # Calculate new Q-value
        td_error = reward + discount_factor * next_best_q - current_q
        new_q = current_q + learning_rate * td_error
        
        # Bound the Q-value
        new_q = np.clip(new_q, -10.0, 10.0)
        
        # Update Q-value
        self.set_q_value(state, action, new_q)
        
        # Update visit count
        state_key = self._get_state_key(state)
        self.visit_counts[state_key][action] += 1
        
        # Track performance metrics
        self.q_value_history.append({
            'state': state,
            'action': action,
            'old_q': current_q,
            'new_q': new_q,
            'td_error': td_error,
            'learning_rate': learning_rate,
            'reward': reward
        })
        
        # Keep only recent history
        if len(self.q_value_history) > 1000:
            self.q_value_history = self.q_value_history[-1000:]
        
        # Track state coverage
        self.state_coverage.add(self._get_state_key(state))
        
        # Track action preferences
        if state_key not in self.action_preferences:
            self.action_preferences[state_key] = [0] * self.action_count
        self.action_preferences[state_key][action] += 1
        
        self.update_count += 1
    
    def get_convergence_estimate(self) -> float:
        """
        Estimate Q-learning convergence based on recent Q-value changes.
        
        Returns:
            Convergence score (0-1, higher means more converged)
        """
        if len(self.q_value_history) < 100:
            return 0.0
        
        # Look at recent Q-value changes
        recent_changes = [abs(entry['td_error']) for entry in self.q_value_history[-100:]]
        
        # Calculate average absolute change
        avg_change = sum(recent_changes) / len(recent_changes)
        
        # Convert to convergence score (smaller changes = higher convergence)
        convergence = 1.0 / (1.0 + avg_change)
        
        return convergence
    
    def get_state_coverage_ratio(self, estimated_total_states: int = 1000) -> float:
        """
        Get the ratio of states covered vs estimated total states.
        
        Args:
            estimated_total_states: Estimated total number of possible states
            
        Returns:
            Coverage ratio (0-1)
        """
        return min(len(self.state_coverage) / estimated_total_states, 1.0)
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get enhanced statistics including convergence and coverage metrics."""
        base_stats = self.get_stats()
        
        base_stats.update({
            'convergence_estimate': self.get_convergence_estimate(),
            'state_coverage': len(self.state_coverage),
            'confidence_threshold': self.confidence_threshold,
            'exploration_bonus': self.exploration_bonus,
            'avg_learning_rate': np.mean([entry['learning_rate'] for entry in self.q_value_history[-100:]]) if self.q_value_history else 0.0,
            'avg_td_error': np.mean([abs(entry['td_error']) for entry in self.q_value_history[-100:]]) if self.q_value_history else 0.0,
            'history_length': len(self.q_value_history),
        })
        
        return base_stats
    
    def learn_from_other_table(self, other_table: 'EnhancedQTable', learning_rate: float = 0.1):
        """
        Learn from another Q-table (evolutionary knowledge sharing).
        
        Args:
            other_table: Another EnhancedQTable to learn from
            learning_rate: Rate of learning from other table
        """
        # Transfer Q-values from other table
        for state_key, other_action_values in other_table.q_values.items():
            # Ensure this state exists in our table
            if state_key not in self.q_values:
                self.q_values[state_key] = [self.default_value] * self.action_count
                self.visit_counts[state_key] = [0] * self.action_count
            
            # Update Q-values using weighted average
            for action in range(self.action_count):
                if action < len(other_action_values):
                    current_q = self.q_values[state_key][action]
                    other_q = other_action_values[action]
                    
                    # Weight by visit count - trust more visited states more
                    other_visits = other_table.visit_counts.get(state_key, [0] * self.action_count)[action]
                    weight = min(learning_rate, learning_rate * (other_visits / 10.0))  # Scale by visits
                    
                    # Update Q-value
                    new_q = (1 - weight) * current_q + weight * other_q
                    self.q_values[state_key][action] = np.clip(new_q, -10.0, 10.0)
    
    def copy(self) -> 'EnhancedQTable':
        """Create a copy of this Q-table."""
        new_table = EnhancedQTable(
            action_count=self.action_count,
            default_value=self.default_value,
            confidence_threshold=self.confidence_threshold,
            exploration_bonus=self.exploration_bonus
        )
        
        # Copy all data
        new_table.q_values = {k: v.copy() for k, v in self.q_values.items()}
        new_table.visit_counts = {k: v.copy() for k, v in self.visit_counts.items()}
        new_table.update_count = self.update_count
        new_table.state_coverage = self.state_coverage.copy()
        new_table.action_preferences = {k: v.copy() for k, v in self.action_preferences.items()}
        
        return new_table 