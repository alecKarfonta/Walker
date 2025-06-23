"""
Q-learning agent for CrawlingCrate that learns crawling strategies.
"""

import numpy as np
from typing import Tuple, Dict, Any, List, NamedTuple
from collections import deque
import random
from .crawling_crate import CrawlingCrate
from .q_table import QTable, SparseQTable
import Box2D as b2

from .base_agent import BaseAgent


class Experience(NamedTuple):
    """Experience tuple for replay buffer."""
    state: Tuple
    action: int
    reward: float
    next_state: Tuple
    done: bool


class ReplayBuffer:
    """Experience replay buffer for Q-learning."""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        
    def add(self, experience: Experience):
        """Add experience to buffer."""
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a batch of experiences."""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(self.buffer, batch_size)
        
    def __len__(self):
        return len(self.buffer)
        
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()


class CrawlingCrateAgent(CrawlingCrate, BaseAgent):
    """
    CrawlingCrate with Q-learning capabilities for learning crawling strategies.
    """
    
    def __init__(self, world, agent_id: int, position: Tuple[float, float] = (10, 20), category_bits=0x0001, mask_bits=0xFFFF):
        # Call parent's __init__ but don't let it create body parts yet
        BaseAgent.__init__(self)
        self.world = world
        self.initial_position = position
        self.id = agent_id
        
        # Physics properties for collision filtering
        self.filter = b2.b2Filter(
            categoryBits=category_bits,
            maskBits=mask_bits
        )
        
        # Physical properties
        self.motor_torque = 150.0
        self.motor_speed = 5.0
        self.category_bits = category_bits
        self.mask_bits = mask_bits

        # Create body parts using our own methods
        self._create_body()
        self._create_arms()
        self._create_wheels()
        self._create_joints()

        # Q-learning parameters
        self.learning_rate = 0.005
        self.discount_factor = 0.9
        self.epsilon = 0.3
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.9999
        
        self.actions = [
            (1, 0), (0, 1), (1, 1),
            (-1, 0), (0, -1), (-1, -1)
        ]  # Removed (0, 0) "none" action
        
        self.state_size = 6
        self.action_size = len(self.actions)
        self.q_table = SparseQTable(self.state_size, self.action_size)
        
        self.total_reward = 0.0
        self.steps = 0
        self.action_history = []
        
        # Adaptive learning rate and epsilon
        self.min_learning_rate = 0.05
        self.max_learning_rate = 0.3
        self.max_epsilon = 0.5
        self.impatience = 0.001

        # Reward clipping to stabilize learning
        self.reward_clip_min = -1.0
        self.reward_clip_max = 1.0

        # For state calculation
        self.last_x_position = self.body.position.x
        self.last_update_step = 0
        self.reward_count = 0
        
        # Q-value bounds to prevent explosion
        self.min_q_value = -2.0  # Minimum Q-value
        self.max_q_value = 2.0   # Maximum Q-value
        
        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(capacity=5000)
        self.batch_size = 32
        self.replay_frequency = 10  # Learn from replay every N steps
        
        # Training state
        self.current_state = None
        self.current_action = None
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
        
        # Action interval optimization - only choose new actions every N steps
        self.action_interval = 3  # Choose new action every 0.2 seconds (3 steps at 60fps) - more responsive
        self.learning_interval = 60  # Update Q-values every 1 second (60 steps at 60fps)
        self.steps_since_last_action = 0
        self.steps_since_last_learning = 0
        self.current_action_tuple = (0, 0)  # Store the actual action tuple
        self.prev_x = position[0]  # Track previous position for reward calculation
        
        # Speed and acceleration tracking (inspired by Java implementation)
        self.speed = 0.0
        self.speed_decay = 0.8  # Moving average decay
        self.previous_speed = 0.0
        self.acceleration = 0.0
        self.max_speed = 0.0
        
        # Reward weights (inspired by Java implementation)
        self.speed_value_weight = 0.05
        self.acceleration_value_weight = 0.05  # Reduced from 0.1 to prevent gradient explosions

        # Reset action history
        self.action_history = []
        self.max_action_history = 10

    def get_discretized_state(self) -> Tuple:
        """State discretization using 10-degree bucket increments."""
        state = self.get_state()
        
        # Extract shoulder and elbow angles
        shoulder_angle = state[5]
        elbow_angle = state[6]
        
        # Convert to degrees
        shoulder_deg = np.degrees(shoulder_angle)
        elbow_deg = np.degrees(elbow_angle)
        
        # Define realistic angle ranges
        shoulder_range = (-180, 180)
        elbow_range = (-180, 180)
        
        # Clamp to the new, larger ranges
        shoulder_deg = np.clip(shoulder_deg, shoulder_range[0], shoulder_range[1])
        elbow_deg = np.clip(elbow_deg, elbow_range[0], elbow_range[1])
        
        # Normalize angles to a 0-based index for binning
        # This shifts the range from [-180, 180] to [0, 360]
        normalized_shoulder = shoulder_deg + 180
        normalized_elbow = elbow_deg + 180
        
        # Use 10-degree buckets
        shoulder_bin = int(normalized_shoulder // 10)
        elbow_bin = int(normalized_elbow // 10)
        
        # Calculate the number of bins
        num_shoulder_bins = (shoulder_range[1] - shoulder_range[0]) // 10
        num_elbow_bins = (elbow_range[1] - elbow_range[0]) // 10
        
        # Ensure bins are within the calculated range
        shoulder_bin = np.clip(shoulder_bin, 0, num_shoulder_bins - 1)
        elbow_bin = np.clip(elbow_bin, 0, num_elbow_bins - 1)
        
        return (shoulder_bin, elbow_bin)
        
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
        """Improved reward function with bounded penalties to prevent negative reward explosion."""
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
        
        # Base reward based on speed and acceleration - REDUCED WEIGHTS
        if x_velocity > 0:
            # Moving right - positive reward
            base_reward = (self.speed_value_weight * x_velocity) + (self.acceleration * self.acceleration_value_weight)
        elif x_velocity < 0:
            # Moving left - negative reward (REDUCED MAGNITUDE)
            base_reward = (self.speed_value_weight * x_velocity * 0.5) + (self.acceleration * self.acceleration_value_weight * 0.5)
        else:
            # Stationary - neutral reward
            base_reward = 0.0
        
        # NO SPEED PENALTIES - robots should not be punished for existing
        speed_penalty = 0.0
        # Removed all speed penalties to prevent any negative accumulation
        # The robot should only be rewarded for positive progress, not penalized for inactivity
        
        # Add small positive reward for any forward progress to encourage exploration
        progress_reward = 0.0
        if x_velocity > 0.1:  # If making meaningful forward progress
            progress_reward = 0.01  # Small constant reward
        
        final_reward = base_reward + speed_penalty + progress_reward
        
        # BOUND THE FINAL REWARD to prevent extreme values - allow small negatives for backwards movement
        final_reward = np.clip(final_reward, -0.1, 0.5)  # Small negative for backwards, moderate positive
        
        # DEBUG LOGGING - enabled temporarily to monitor rewards
        if self.id == 0 and self.steps % 50 == 0:  # Only for first agent, every 50 steps
            print(f"💰 Reward Debug Agent {self.id}:")
            print(f"  Raw x_velocity: {self.body.linearVelocity.x:.4f}")
            print(f"  Speed (moving avg): {self.speed:.4f}")
            print(f"  Acceleration: {self.acceleration:.4f}")
            print(f"  Base reward: {base_reward:.4f}")
            print(f"  Speed penalty: {speed_penalty:.4f}")
            print(f"  Progress reward: {progress_reward:.4f}")
            print(f"  Final reward: {final_reward:.4f}")
            print(f"  Total reward: {getattr(self, 'total_reward', 'N/A'):.2f}")
            print("---")
        
        return final_reward
        
    def add_action_to_history(self, action_idx: int):
        """Add action to history for reporting."""
        self.action_history.append(action_idx)
        if len(self.action_history) > self.max_action_history:
            self.action_history.pop(0)  # Remove oldest action
    
    def add_experience(self, state, action, reward, next_state, done=False):
        """Add experience to replay buffer."""
        experience = Experience(state, action, reward, next_state, done)
        self.replay_buffer.add(experience)
        
    def learn_from_replay(self):
        """Learn from a batch of experiences in the replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return
            
        batch = self.replay_buffer.sample(self.batch_size)
        
        for experience in batch:
            # Get current Q-value
            current_q = self.q_table.get_q_value(experience.state, experience.action)
            
            # Get max Q-value for next state
            if not experience.done:
                max_next_q = self.q_table.get_best_action(experience.next_state)[1]
                target_q = experience.reward + self.discount_factor * max_next_q
            else:
                target_q = experience.reward
            
            # Update Q-value
            new_q = (1 - self.learning_rate) * current_q + self.learning_rate * target_q
            self.q_table.set_q_value(experience.state, experience.action, new_q)
            
    def step(self, dt: float):
        """Step the agent with Q-learning and experience replay."""
        # Initialize action if not set
        if self.current_action_tuple == (0, 0) and self.current_action is None:
            self.current_state = self.get_discretized_state()
            action_idx = self.choose_action()
            self.current_action = action_idx
            self.current_action_tuple = self.actions[action_idx]
            self.add_action_to_history(action_idx)
            if self.id == 0:  # Debug for first agent only
                print(f"🤖 Agent {self.id}: Initialized with action {action_idx} = {self.current_action_tuple}")
        
        # Always apply the current action (cheap operation)
        self.apply_action(self.current_action_tuple)
        
        # Track reward over time
        current_x = self.body.position.x
        reward = self.get_reward(self.prev_x)
        self.total_reward += reward
        
        # CRITICAL: Update prev_x for next step - this was missing!
        self.prev_x = current_x
        
        # Prevent total reward from exploding negatively
        if self.total_reward < -10.0:  # If accumulated reward is too negative
            self.total_reward = max(self.total_reward, -10.0)  # Cap at -10
            if self.id == 0:  # Debug for first agent
                print(f"⚠️  Agent {self.id}: Total reward capped at -10.0 to prevent explosion")
        
        # Debug for first agent every 100 steps
        if self.id == 0 and self.steps % 100 == 0:
            print(f"🤖 Agent {self.id}: Step {self.steps}, pos=({self.body.position.x:.2f}, {self.body.position.y:.2f}), "
                  f"vel=({self.body.linearVelocity.x:.2f}, {self.body.linearVelocity.y:.2f}), "
                  f"action={self.current_action_tuple}, reward={reward:.3f}, total_reward={self.total_reward:.2f}")
        
        # Action interval optimization - only choose new actions every N steps
        self.action_interval = 3  # Choose new action every 0.2 seconds (3 steps at 60fps) - more responsive
        self.learning_interval = 60  # Update Q-values every 1 second (60 steps at 60fps)
        
        if self.steps % self.action_interval == 0:
            # Store previous action for comparison
            previous_action_tuple = self.current_action_tuple
            
            # Choose new action
            self.current_state = self.get_discretized_state()
            action_idx = self.choose_action()
            self.current_action = action_idx
            self.current_action_tuple = self.actions[action_idx]
            self.add_action_to_history(action_idx)
            
            # Debug for first agent - only log if action changed
            if self.id == 0 and self.current_action_tuple != previous_action_tuple:
                print(f"🤖 Agent {self.id}: New action {action_idx} = {self.current_action_tuple}")
        
        # Q-learning update every N steps
        if self.steps % self.learning_interval == 0 and len(self.replay_buffer) > 0:
            self.learn_from_replay()
        
        self.steps += 1
        
    def update_q_value(self, next_state: Tuple, reward: float):
        """Update Q-value for the current state-action pair using Java-inspired approach with bounds."""
        if self.current_state is not None and self.current_action is not None:
            # Get the old Q-value (like Java implementation)
            self.old_value = self.q_table.get_q_value(self.current_state, self.current_action)
            
            # Get the maximum Q-value for the next state (like Java implementation)
            max_next_q_value = self.q_table.get_best_action(next_state)[1]
            
            # BOUND the next Q-value to prevent explosion
            max_next_q_value = np.clip(max_next_q_value, self.min_q_value, self.max_q_value)
            
            # Calculate the new Q-value using standard Q-learning formula (like Java)
            self.new_value = (
                (1 - self.learning_rate) * self.old_value +
                self.learning_rate * (reward + self.discount_factor * max_next_q_value)
            )
            
            # CRITICAL: BOUND the new Q-value to prevent explosion
            self.new_value = np.clip(self.new_value, self.min_q_value, self.max_q_value)
            
            # DEBUG LOGGING
            #print(f"Q-Update Debug:")
            #print(f"  State: {self.current_state}")
            #print(f"  Action: {self.current_action}")
            #print(f"  Old Q-value: {self.old_value:.4f}")
            #print(f"  Reward: {reward:.4f}")
            #print(f"  Max next Q-value (clipped): {max_next_q_value:.4f}")
            #print(f"  Learning rate: {self.learning_rate:.4f}")
            #print(f"  Discount factor: {self.discount_factor:.4f}")
            #print(f"  New Q-value (clipped): {self.new_value:.4f}")
            #print(f"  Q-value change: {self.new_value - self.old_value:.4f}")
            #print(f"  Best value: {self.best_value:.4f}")
            #print(f"  Worst value: {self.worst_value:.4f}")
            #print(f"  Time since good: {self.time_since_good_value:.1f}")
            #print("---")
            
            # Update the Q-table with bounded value
            self.q_table.set_q_value(self.current_state, self.current_action, self.new_value)
            
            # Track best and worst values (like Java implementation) - but bounded
            if self.new_value > self.best_value:
                self.best_value = min(self.new_value, self.max_q_value)  # Bound best value
            elif self.new_value < self.worst_value:
                self.worst_value = max(self.new_value, self.min_q_value)  # Bound worst value
            
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
        
        # Reset reward normalization statistics
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0
        
        # Reset velocity to prevent physics issues
        for part in [self.body, self.upper_arm, self.lower_arm] + self.wheels:
            part.linearVelocity = (0, 0)
            part.angularVelocity = 0

    def reset_position(self):
        """
        Resets the agent's position and physics state to its starting point,
        but preserves the learned Q-table and other learning parameters.
        """
        self.body.position = self.initial_position
        self.body.angle = 0
        self.body.linearVelocity = (0, 0)
        self.body.angularVelocity = 0

        # Also reset arms and wheels relative to the body
        # Get the anchor points from the joints
        upper_arm_anchor = self.upper_arm_joint.localAnchorA
        lower_arm_anchor = self.lower_arm_joint.localAnchorA

        self.upper_arm.position = self.body.GetWorldPoint(upper_arm_anchor)
        self.upper_arm.angle = 0
        self.upper_arm.linearVelocity = (0, 0)
        self.upper_arm.angularVelocity = 0
        
        self.lower_arm.position = self.upper_arm.GetWorldPoint(lower_arm_anchor)
        self.lower_arm.angle = 0
        self.lower_arm.linearVelocity = (0, 0)
        self.lower_arm.angularVelocity = 0

        for i, wheel in enumerate(self.wheels):
            wheel_anchor = self.wheel_joints[i].localAnchorA
            wheel.position = self.body.GetWorldPoint(wheel_anchor)
            wheel.linearVelocity = (0, 0)
            wheel.angularVelocity = 0
        
        # Reset reward and internal state, but not the Q-table itself
        self.total_reward = 0
        self.steps = 0
        print(f"Agent {self.id} was reset due to falling off the world.")

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
        
        # Map action indices to readable names - match Java's 6 actions
        action_names = {
            0: "None", 1: "S-Fwd", 2: "E-Fwd", 3: "Both-Fwd", 
            4: "S-Back", 5: "E-Back"
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
            # Replay buffer info
            'replay_buffer_size': len(self.replay_buffer),
            'replay_buffer_capacity': self.replay_buffer.capacity,
            'batch_size': self.batch_size,
            'replay_frequency': self.replay_frequency,
        })
        return debug 

    def _create_body(self):
        body_def = b2.b2BodyDef(
            type=b2.b2_dynamicBody,
            position=self.initial_position,
            fixtures=[
                b2.b2FixtureDef(
                    shape=b2.b2PolygonShape(box=(1.5, 0.75)),
                    density=4.0,
                    friction=0.9,
                    filter=b2.b2Filter(categoryBits=self.category_bits, maskBits=self.mask_bits)
                )
            ]
        )
        self.body = self.world.CreateBody(body_def)

    def _create_arms(self):
        # Upper Arm
        upper_arm = self.world.CreateDynamicBody(
            position=self.body.position + (-1.0, 1.0),
            fixtures=[
                b2.b2FixtureDef(
                    shape=b2.b2PolygonShape(box=(1.0, 0.2)),
                    density=0.1,
                    filter=b2.b2Filter(categoryBits=self.category_bits, maskBits=self.mask_bits)
                )
            ]
        )
        # Lower Arm
        lower_arm = self.world.CreateDynamicBody(
            position=upper_arm.position + (1.0, 0),
            fixtures=[
                b2.b2FixtureDef(
                    shape=b2.b2PolygonShape(box=(1.0, 0.2)),
                    density=0.1,
                    filter=b2.b2Filter(categoryBits=self.category_bits, maskBits=self.mask_bits)
                )
            ]
        )
        self.upper_arm, self.lower_arm = upper_arm, lower_arm

    def _create_wheels(self):
        self.wheels = []
        wheel_anchor_positions = [(-1.0, -0.75), (1.0, -0.75)]
        for anchor_pos in wheel_anchor_positions:
            wheel = self.world.CreateDynamicBody(
                position=self.body.GetWorldPoint(anchor_pos),
                fixtures=[
                    b2.b2FixtureDef(
                        shape=b2.b2CircleShape(radius=0.5),
                        density=8.0,
                        friction=0.9,
                        filter=b2.b2Filter(categoryBits=self.category_bits, maskBits=self.mask_bits)
                    )
                ]
            )
            self.wheels.append(wheel)

    def _create_joints(self):
        self.upper_arm_joint = self.world.CreateRevoluteJoint(
            bodyA=self.body,
            bodyB=self.upper_arm,
            localAnchorA=(-1.0, 1.0),
            localAnchorB=(-1.0, 0),  # Connect to LEFT end of upper arm
            enableMotor=True,
            maxMotorTorque=self.motor_torque,
            motorSpeed=0,
            enableLimit=True,
            lowerAngle=-np.pi/2,  # -90 degrees
            upperAngle=np.pi/2,   # +90 degrees
        )
        self.lower_arm_joint = self.world.CreateRevoluteJoint(
            bodyA=self.upper_arm,
            bodyB=self.lower_arm,
            localAnchorA=(1.0, 0),    # RIGHT end of upper arm
            localAnchorB=(-1.0, 0),   # LEFT end of lower arm
            enableMotor=True,
            maxMotorTorque=self.motor_torque,
            motorSpeed=0,
            enableLimit=True,
            lowerAngle=0,           # 0 degrees (fully extended)
            upperAngle=3*np.pi/4,   # +135 degrees
        )
        self.wheel_joints = []
        wheel_anchor_positions = [(-1.0, -0.75), (1.0, -0.75)]
        for i, anchor_pos in enumerate(wheel_anchor_positions):
            joint = self.world.CreateRevoluteJoint(
                bodyA=self.body,
                bodyB=self.wheels[i],
                localAnchorA=anchor_pos,
                localAnchorB=(0,0),
                enableMotor=False,
            )
            self.wheel_joints.append(joint) 

    def apply_action(self, action: Tuple[float, float]):
        """Apply action to the agent's arms using joint motors to respect limits."""
        # Convert action to target motor speeds (respecting joint limits)
        shoulder_speed = float(np.clip(action[0], -1.0, 1.0)) * self.motor_speed
        elbow_speed = float(np.clip(action[1], -1.0, 1.0)) * self.motor_speed

        # Set motor speeds on the joints (this respects the joint limits!)
        self.upper_arm_joint.motorSpeed = shoulder_speed
        self.lower_arm_joint.motorSpeed = elbow_speed
        
        # Ensure joints are enabled and awake
        self.upper_arm_joint.enableMotor = True
        self.lower_arm_joint.enableMotor = True
        
        # Wake up the bodies to ensure they respond
        self.upper_arm.awake = True
        self.lower_arm.awake = True
        
        # Debug: Print motor speeds occasionally for first agent
        if self.id == 0 and self.steps % 200 == 0:  # Every 200 steps
            print(f"🔧 Agent {self.id}: Motor speeds - shoulder: {shoulder_speed:.1f}, elbow: {elbow_speed:.1f}")
            print(f"    Joint angles: shoulder: {np.degrees(self.upper_arm.angle):.1f}°, elbow: {np.degrees(self.lower_arm.angle):.1f}°")
        
        # Removed debug print to eliminate overhead - was running 1% of the time 