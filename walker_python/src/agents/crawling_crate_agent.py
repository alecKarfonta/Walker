"""
Q-learning agent for CrawlingCrate that learns crawling strategies.
"""

import numpy as np
from typing import Tuple, Dict, Any
from .crawling_crate import CrawlingCrate
from .q_table import QTable, SparseQTable


class CrawlingCrateAgent(CrawlingCrate):
    """
    CrawlingCrate with Q-learning capabilities for learning crawling strategies.
    """
    
    def __init__(self, world, agent_id: int, position: Tuple[float, float] = (10, 20), category_bits=0x0001, mask_bits=0xFFFF):
        super().__init__(world, agent_id, position, category_bits=category_bits, mask_bits=mask_bits)
        
        # Q-learning parameters
        self.learning_rate = 0.005
        self.discount_factor = 0.95
        self.epsilon = 0.3  # Start with higher exploration
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.9999  # Slower decay since we learn less frequently
                
        # Action space: (shoulder_motor, elbow_motor) combinations
        self.actions = [
            (0, 0),      # No movement
            (1, 0),      # Shoulder forward only
            (0, 1),      # Elbow forward only
            (1, 1),      # Both forward
            (-1, 1),     # Shoulder back, elbow forward
            (1, -1),     # Shoulder forward, elbow back
            (-1, 0),     # Shoulder back only
            (0, -1),     # Elbow back only
        ]
        
        # Initialize Q-table - use sparse version for much better performance
        self.q_table = SparseQTable(len(self.actions))
        
        # Training state
        self.current_state = None
        self.current_action = None
        self.total_reward = 0.0
        self.episode_steps = 0
        self.immediate_reward = 0.0  # Track immediate reward for Q-learning
        
        # Performance tracking (inspired by Java implementation)
        self.best_distance = 0.0
        self.consecutive_failures = 0
        
        # Q-value tracking (inspired by Java implementation)
        self.best_value = 0.0
        self.worst_value = 0.0
        self.old_value = 0.0
        self.new_value = 0.0
        self.time_since_good_value = 0.0
        
        # Adaptive parameters (inspired by Java implementation)
        self.min_learning_rate = 0.05
        self.max_learning_rate = 0.3
        self.min_epsilon = 0.01
        self.max_epsilon = 0.5
        self.impatience = 0.001  # How quickly to adapt parameters
        
        # Action interval optimization - only choose new actions every N steps
        self.action_interval = 120  # Choose new action every 2 seconds (120 steps at 60fps)
        self.learning_interval = 60  # Update Q-values every 1 second (60 steps at 60fps)
        self.steps_since_last_action = 0
        self.steps_since_last_learning = 0
        self.current_action_tuple = (0, 0)  # Store the actual action tuple
        self.prev_x = position[0]  # Track previous position for reward calculation
        
        # Action history for reporting
        self.action_history = []  # Store last 10 actions
        self.max_action_history = 10
        
        # Speed and acceleration tracking (inspired by Java implementation)
        self.speed = 0.0
        self.speed_decay = 0.8  # Moving average decay
        self.previous_speed = 0.0
        self.acceleration = 0.0
        self.max_speed = 0.0
        
        # Reward weights (inspired by Java implementation)
        self.speed_value_weight = 0.1  # Reduced from 1.0 to prevent gradient explosions
        self.acceleration_value_weight = 0.1  # Reduced from 1.0 to prevent gradient explosions
        
    def get_discretized_state(self) -> Tuple:
        """Optimized state discretization - much faster version."""
        # Cache the state to avoid multiple calls
        state = self.get_state()
        
        # Simplified state components - focus on most important features
        x_pos = state[0] - self.initial_position[0]  # Use initial_position from parent class
        vel_x = np.clip(state[2], -3, 3)   # Velocity
        angle = np.clip(state[4], -np.pi/3, np.pi/3)  # Body angle
        shoulder_angle = np.clip(state[5], -np.pi, np.pi)  # Shoulder angle
        elbow_angle = np.clip(state[6], -np.pi, np.pi)  # Elbow angle
        
        # Simplified binning with fewer calculations
        x_pos_bin = int(np.clip((x_pos + 10) / 20 * 7, 0, 7))
        vel_x_bin = int((vel_x + 3) / 6 * 4)
        angle_bin = int((angle + np.pi/3) / (2*np.pi/3) * 4)
        shoulder_angle_bin = int((shoulder_angle + np.pi) / (2*np.pi) * 6)
        elbow_angle_bin = int((elbow_angle + np.pi) / (2*np.pi) * 6)
        
        # Ground contact (simplified)
        ground_contact = 1 if state[1] < 1.5 else 0
        
        return (x_pos_bin, vel_x_bin, angle_bin, shoulder_angle_bin, elbow_angle_bin, ground_contact)
        
    def choose_action(self) -> int:
        """Choose action using decaying epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(len(self.actions))
        else:
            # Exploit: best action
            if self.current_state is not None:
                return self.q_table.get_best_action(self.current_state)[0]
            else:
                return np.random.randint(len(self.actions))
    
    def get_adaptive_learning_rate(self, state, action):
        """Get adaptive learning rate based on visit count."""
        visit_count = self.q_table.get_visit_count(state, action)
        return self.learning_rate / (1 + visit_count * 0.1)
            
    def get_reward(self, prev_x: float) -> float:
        """Improved reward function based on Java implementation - uses speed and acceleration."""
        # Get x velocity of body (like Java implementation)
        x_velocity = self.body.linearVelocity.x
        
        # Track max speed
        if x_velocity > self.max_speed:
            self.max_speed = x_velocity
        
        # Apply velocity threshold (like Java implementation)
        if abs(x_velocity) < 0.25:
            x_velocity = 0.0  # Set to zero if below threshold
        
        # Calculate speed as moving average (like Java implementation)
        self.speed = (1 - self.speed_decay) * self.speed + (self.speed_decay * x_velocity)
        
        # Calculate acceleration
        self.acceleration = self.speed - self.previous_speed
        self.previous_speed = self.speed
        
        # Calculate reward based on speed and acceleration (like Java implementation)
        # Positive reward for rightward movement, negative for leftward, zero for stationary
        if x_velocity > 0:
            # Moving right - positive reward
            reward = (self.speed_value_weight * x_velocity) + (self.acceleration * self.acceleration_value_weight)
        elif x_velocity < 0:
            # Moving left - negative reward
            reward = (self.speed_value_weight * x_velocity) + (self.acceleration * self.acceleration_value_weight)
        else:
            # Stationary - neutral reward
            reward = 0.0
        
        # DEBUG LOGGING
        print(f"Reward Debug:")
        print(f"  Raw x_velocity: {self.body.linearVelocity.x:.4f}")
        print(f"  Thresholded x_velocity: {x_velocity:.4f}")
        print(f"  Speed (moving avg): {self.speed:.4f}")
        print(f"  Acceleration: {self.acceleration:.4f}")
        print(f"  Speed weight: {self.speed_value_weight:.4f}")
        print(f"  Acceleration weight: {self.acceleration_value_weight:.4f}")
        print(f"  Final reward: {reward:.4f}")
        print("---")
        
        return reward
        
    def add_action_to_history(self, action_idx: int):
        """Add action to history for reporting."""
        self.action_history.append(action_idx)
        if len(self.action_history) > self.max_action_history:
            self.action_history.pop(0)  # Remove oldest action
    
    def step(self, dt: float):
        """Step the agent with Q-learning - optimized to only choose actions and learn at intervals."""
        # Always apply the current action (cheap operation)
        self.apply_action(self.current_action_tuple)
        
        # Track reward over time
        current_x = self.body.position.x
        reward = self.get_reward(self.prev_x)
        self.immediate_reward = reward
        self.total_reward += reward
        self.prev_x = current_x
        
        # Learning interval - update Q-values periodically
        if self.steps_since_last_learning >= self.learning_interval:
            if self.current_state is not None and self.current_action is not None:
                # Get next state for Q-learning update
                next_state = self.get_discretized_state()
                print(f"Q-Learning Update Triggered:")
                print(f"  Steps since last learning: {self.steps_since_last_learning}")
                print(f"  Learning interval: {self.learning_interval}")
                print(f"  Current state: {self.current_state}")
                print(f"  Next state: {next_state}")
                print(f"  Current action: {self.current_action}")
                print(f"  Immediate reward: {self.immediate_reward:.4f}")
                print("---")
                self.update_q_value(next_state, self.immediate_reward)
            self.steps_since_last_learning = 0
        else:
            self.steps_since_last_learning += 1
            # Track time since good value (like Java implementation)
            if self.best_value > 0 and self.new_value <= self.best_value * 0.5:
                self.time_since_good_value += 1.0
        
        # Action interval - choose new actions less frequently
        if self.steps_since_last_action >= self.action_interval:
            # Get current state (expensive)
            self.current_state = self.get_discretized_state()
            
            # Choose new action (expensive)
            action_idx = self.choose_action()
            self.current_action = action_idx
            self.current_action_tuple = self.actions[action_idx]
            
            # Add to action history
            self.add_action_to_history(action_idx)
            
            # Reset interval counter
            self.steps_since_last_action = 0
        else:
            # Just increment the counter
            self.steps_since_last_action += 1
        
        # Update episode step count
        self.episode_steps += 1
        
    def update_q_value(self, next_state: Tuple, reward: float):
        """Update Q-value for the current state-action pair using Java-inspired approach."""
        if self.current_state is not None and self.current_action is not None:
            # Get the old Q-value (like Java implementation)
            self.old_value = self.q_table.get_q_value(self.current_state, self.current_action)
            
            # Get the maximum Q-value for the next state (like Java implementation)
            max_next_q_value = self.q_table.get_best_action(next_state)[1]
            
            # Calculate the new Q-value using standard Q-learning formula (like Java)
            self.new_value = (
                (1 - self.learning_rate) * self.old_value +
                self.learning_rate * (reward + self.discount_factor * max_next_q_value)
            )
            
            # DEBUG LOGGING
            print(f"Q-Update Debug:")
            print(f"  State: {self.current_state}")
            print(f"  Action: {self.current_action}")
            print(f"  Old Q-value: {self.old_value:.4f}")
            print(f"  Reward: {reward:.4f}")
            print(f"  Max next Q-value: {max_next_q_value:.4f}")
            print(f"  Learning rate: {self.learning_rate:.4f}")
            print(f"  Discount factor: {self.discount_factor:.4f}")
            print(f"  New Q-value: {self.new_value:.4f}")
            print(f"  Q-value change: {self.new_value - self.old_value:.4f}")
            print(f"  Best value: {self.best_value:.4f}")
            print(f"  Worst value: {self.worst_value:.4f}")
            print(f"  Time since good: {self.time_since_good_value:.1f}")
            print("---")
            
            # Update the Q-table
            self.q_table.set_q_value(self.current_state, self.current_action, self.new_value)
            
            # Track best and worst values (like Java implementation)
            if self.new_value > self.best_value:
                self.best_value = self.new_value
            elif self.new_value < self.worst_value:
                self.worst_value = self.new_value
            
            # Adaptive learning based on performance (like Java implementation)
            if self.best_value > 0 and self.new_value > self.best_value * 0.5:
                # Doing well - reduce exploration, increase exploitation
                self.time_since_good_value = 0.0
                self.learning_rate = max(self.min_learning_rate, 
                                       self.learning_rate - self.impatience)
                self.epsilon = max(self.min_epsilon, 
                                 self.epsilon - self.impatience)
            else:
                # Not doing well - increase exploration
                self.time_since_good_value += 1.0
                if self.time_since_good_value > 100:  # After 100 steps of poor performance
                    self.learning_rate = min(self.max_learning_rate, 
                                           self.learning_rate + self.impatience)
                    self.epsilon = min(self.max_epsilon, 
                                     self.epsilon + self.impatience)
            
            # Decay epsilon (original approach, but now adaptive)
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
    def reset(self):
        """Reset the agent for a new episode."""
        super().reset()
        self.current_state = None
        self.current_action = None
        self.total_reward = 0.0
        self.episode_steps = 0
        self.immediate_reward = 0.0  # Track immediate reward for Q-learning
        self.best_distance = 0.0
        self.consecutive_failures = 0
        
        # Reset action interval variables
        self.steps_since_last_action = 0
        self.steps_since_last_learning = 0
        self.current_action_tuple = (0, 0)
        self.prev_x = self.initial_position[0]  # Use initial_position from parent class
        
        # Reset action history
        self.action_history = []
        self.max_action_history = 10
        
        # Reset speed and acceleration tracking
        self.speed = 0.0
        self.speed_decay = 0.8
        self.previous_speed = 0.0
        self.acceleration = 0.0
        self.max_speed = 0.0
        
        # Reset Q-value tracking
        self.best_value = 0.0
        self.worst_value = 0.0
        self.old_value = 0.0
        self.new_value = 0.0
        self.time_since_good_value = 0.0
        
    def get_fitness(self) -> float:
        """Get fitness score for evolution."""
        return self.total_reward
        
    def mutate(self, mutation_rate: float = 0.1):
        """Mutate the agent's Q-table and parameters."""
        # Mutate Q-table values (sparse version)
        for state_key, action_values in self.q_table.q_values.items():
            for action in range(len(action_values)):
                if np.random.random() < mutation_rate:
                    action_values[action] += np.random.normal(0, 0.1)
        
        # Mutate learning parameters
        if np.random.random() < mutation_rate:
            self.learning_rate = np.clip(
                self.learning_rate + np.random.normal(0, 0.02),
                0.05, 0.3
            )
        if np.random.random() < mutation_rate:
            self.epsilon = np.clip(
                self.epsilon + np.random.normal(0, 0.05),
                0.01, 0.5
            )
            
    def crossover(self, other: 'CrawlingCrateAgent') -> 'CrawlingCrateAgent':
        """Create a new agent by crossing over with another agent."""
        # Create new agent
        new_agent = CrawlingCrateAgent(
            self.world, 
            self.id, 
            self.initial_position,
            category_bits=self.filter.categoryBits,
            mask_bits=self.filter.maskBits
        )
        
        # Crossover Q-table values (sparse version)
        all_states = set(self.q_table.q_values.keys()) | set(other.q_table.q_values.keys())
        for state_key in all_states:
            if state_key in self.q_table.q_values and state_key in other.q_table.q_values:
                # Both parents have this state, crossover
                if np.random.random() < 0.5:
                    new_agent.q_table.q_values[state_key] = self.q_table.q_values[state_key].copy()
                else:
                    new_agent.q_table.q_values[state_key] = other.q_table.q_values[state_key].copy()
            elif state_key in self.q_table.q_values:
                # Only self has this state
                new_agent.q_table.q_values[state_key] = self.q_table.q_values[state_key].copy()
            else:
                # Only other has this state
                new_agent.q_table.q_values[state_key] = other.q_table.q_values[state_key].copy()
        
        # Crossover parameters
        new_agent.learning_rate = (self.learning_rate + other.learning_rate) / 2
        new_agent.epsilon = (self.epsilon + other.epsilon) / 2
        
        return new_agent
        
    def set_action_interval(self, interval: int):
        """Set the interval between action choices (in physics steps)."""
        self.action_interval = max(1, interval)  # Minimum 1 step
        
    def set_learning_interval(self, interval: int):
        """Set the interval between Q-learning updates (in physics steps)."""
        self.learning_interval = max(1, interval)  # Minimum 1 step
        
    def get_action_history_string(self) -> str:
        """Get a formatted string of recent actions."""
        if not self.action_history:
            return "No actions yet"
        
        # Map action indices to readable names
        action_names = {
            0: "None", 1: "S-Fwd", 2: "E-Fwd", 3: "Both-Fwd", 
            4: "S-Back", 5: "E-Back", 6: "S-Back", 7: "E-Back"
        }
        
        # Get the last 5 actions for display
        recent_actions = self.action_history[-5:] if len(self.action_history) > 5 else self.action_history
        action_strings = [action_names.get(idx, f"A{idx}") for idx in recent_actions]
        
        return " → ".join(action_strings)
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get enhanced debug information."""
        debug = super().get_debug_info()
        
        # Get current reward for coloring
        current_reward = self.immediate_reward
        
        # Determine reward color based on value
        if current_reward > 0:
            reward_color = "green"
        elif current_reward < 0:
            reward_color = "red"
        else:
            reward_color = "yellow"
        
        debug.update({
            'total_reward': self.total_reward,
            'episode_steps': self.episode_steps,
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'current_action': self.current_action,
            'best_distance': self.best_distance,
            'consecutive_failures': self.consecutive_failures,
            'action_interval': self.action_interval,
            'learning_interval': self.learning_interval,
            'steps_since_last_action': self.steps_since_last_action,
            'steps_since_last_learning': self.steps_since_last_learning,
            'current_action_tuple': self.current_action_tuple,
            'action_history': self.action_history.copy(),
            'action_history_string': self.get_action_history_string(),
            'speed': self.speed,
            'acceleration': self.acceleration,
            'max_speed': self.max_speed,
            'current_reward': current_reward,
            'immediate_reward': self.immediate_reward,
            'reward_color': reward_color,
            'best_value': self.best_value,
            'worst_value': self.worst_value,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'time_since_good_value': self.time_since_good_value,
        })
        return debug 