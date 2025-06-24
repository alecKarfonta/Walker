"""
Q-learning agent for CrawlingCrate that learns crawling strategies.
"""

import numpy as np
from typing import Tuple, Dict, Any, List, NamedTuple, Optional
from collections import deque
import random
from .crawling_crate import CrawlingCrate
from .q_table import  EnhancedQTable
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
    Enhanced with adaptive learning, multi-goal rewards, and evolutionary knowledge sharing.
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
        self.motor_speed = 3.0  # INCREASED: Much faster motor speeds for visible movement
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
        
        self.state_size = 3  # shoulder_bin, elbow_bin, vel_x_bin
        self.action_size = len(self.actions)
        
        # ENHANCED: Use EnhancedQTable instead of SparseQTable
        self.q_table = EnhancedQTable(
            action_count=self.action_size, 
            default_value=0.0,
            confidence_threshold=15,  # Visits needed for confidence
            exploration_bonus=0.15    # Higher bonus for under-explored actions
        )
        
        # Q-table size management to prevent performance degradation
        self.max_q_table_states = 1500  # Limit Q-table size
        self.q_table_pruning_threshold = 1800  # Start pruning at this size
        
        self.total_reward = 0.0
        self.steps = 0
        self.action_history = []
        
        # Adaptive learning rate and epsilon
        self.min_learning_rate = 0.01
        self.max_learning_rate = 0.3
        self.max_epsilon = 0.6
        self.impatience = 0.002  # Increased for faster adaptation

        # Reward clipping to stabilize learning (REDUCED: to match new reward scale)
        self.reward_clip_min = -0.1
        self.reward_clip_max = 0.1

        # For state calculation
        self.last_x_position = self.body.position.x
        self.last_update_step = 0
        self.reward_count = 0
        
        # Q-value bounds to prevent explosion
        self.min_q_value = -5.0  # Increased range for better learning
        self.max_q_value = 5.0   # Increased range for better learning
        
        # Track extreme rewards seen in current episode
        self.best_reward_received = -np.inf  # Start at -inf so first reward sets it
        self.worst_reward_received = np.inf   # Start at +inf so first reward sets it
        
        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(capacity=8000)  # Increased capacity
        self.batch_size = 48  # Increased batch size
        self.replay_frequency = 8  # More frequent learning
        
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
        self.action_interval = 2  # More responsive for better learning
        self.learning_interval = 30  # More frequent learning updates
        self.steps_since_last_action = 0
        self.steps_since_last_learning = 0
        self.current_action_tuple = (1, 0)  # FIXED: Start with a real action from the action list
        self.prev_x = position[0]  # Track previous position for reward calculation
        
        # Speed and acceleration tracking (inspired by Java implementation)
        self.speed = 0.0
        self.speed_decay = 0.85  # Slightly more responsive
        self.previous_speed = 0.0
        self.acceleration = 0.0
        self.max_speed = 0.0
        
        # ENHANCED: Multi-goal reward system (inspired by Java implementation)
        self.current_goal = 0
        self.goals = ['speed', 'distance', 'stability', 'efficiency', 'combined']
        self.goal_weights = {
            'speed': 0.08,
            'distance': 0.02, 
            'stability': 0.05,
            'efficiency': 0.03,
            'combined': 0.04
        }
        self.goal_switch_interval = 500  # Switch goals every 500 steps
        
        # Reward weights (enhanced from Java implementation)
        self.speed_value_weight = 0.06  # Slightly increased
        self.acceleration_value_weight = 0.04
        self.position_weight = 0.01
        self.stability_weight = 0.03
        
        # ENHANCED: Performance-based adaptation parameters
        self.performance_window = 200  # Track performance over 200 steps
        self.recent_rewards = deque(maxlen=self.performance_window)
        self.performance_threshold = 0.02  # REDUCED: Threshold for "good" performance (adjusted for new reward scale)
        
        # ENHANCED: Enhanced state discretization
        self.use_enhanced_state = True
        self.enhanced_state_size = 3  # shoulder, elbow, velocity
        
        # Reset action history
        self.action_history = []
        self.max_action_history = 15  # Longer history
        
        # Initialize crawling-specific tracking variables
        self.recent_displacements = []
        self.action_sequence = []

    def get_enhanced_discretized_state(self) -> Tuple:
        """
        Enhanced state discretization with more features (velocity, position, body angle).
        Inspired by Java implementation's more complex state representation.
        """
        # Get basic joint angles directly (no recursion)
        state = self.get_state()
        shoulder_angle = state[5]
        elbow_angle = state[6]
        
        # Convert to degrees and discretize
        shoulder_deg = np.degrees(shoulder_angle)
        elbow_deg = np.degrees(elbow_angle)
        
        # Normalize angles and create bins - use 20-degree buckets for performance balance
        shoulder_bin = int(np.clip((shoulder_deg + 180) // 45, 0, 7))  # 18 bins (360/20)
        elbow_bin = int(np.clip((elbow_deg + 180) // 45, 0, 7))        # 18 bins (360/20)
        
        # Add velocity discretization (clamped and binned) - reduced bins for performance
        vel_x = np.clip(self.body.linearVelocity.x, -3, 3)
        vel_x_bin = int((vel_x + 3) // 1.5)  # 4 bins: [-3,-1.5), [-1.5,0), [0,1.5), [1.5,3]
        vel_x_bin = np.clip(vel_x_bin, 0, 3)  # Ensure it's in valid range
        
        return (shoulder_bin, elbow_bin, vel_x_bin)  # 3D state: angles + velocity
        
    def get_discretized_state(self) -> Tuple:
        """State discretization using 10-degree bucket increments."""
        if self.use_enhanced_state:
            return self.get_enhanced_discretized_state()
            
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
        
    def get_crawling_reward(self, prev_x: float) -> float:
        """
        Comprehensive reward function designed specifically for crawling behavior.
        Rewards proper crawling technique, not just forward movement.
        """
        current_x = self.body.position.x
        total_reward = 0.0
        
        # 1. FORWARD PROGRESS REWARD (Primary - 40% of total reward)
        # Reward consistent forward movement, not just acceleration spikes
        displacement = current_x - prev_x
        if displacement > 0.003:  # Slightly lower threshold so small but real moves count
            progress_reward = displacement * 4.0  # Boost base movement reward
            # Bonus for sustained movement (not just single-step acceleration)
            if hasattr(self, 'recent_displacements'):
                self.recent_displacements.append(displacement)
                if len(self.recent_displacements) > 10:
                    self.recent_displacements.pop(0)
                # Bonus for consistent forward movement
                if len(self.recent_displacements) >= 5:
                    avg_displacement = sum(self.recent_displacements) / len(self.recent_displacements)
                    if avg_displacement > 0.005:  # INCREASED: Consistent forward movement
                        progress_reward *= 1.2  # REDUCED: Smaller bonus
            else:
                self.recent_displacements = [displacement]
        elif displacement < -0.0005:
            progress_reward = displacement * 1.0  # REDUCED: Moderate penalty for backward movement
        else:
            progress_reward = 0.0  # FIXED: NO reward for tiny movements or standing still
        
        total_reward += progress_reward * 0.4
        
        # 2. ARM GROUND CONTACT REWARD (25% of total reward)
        # Reward arms being in contact with ground for pushing
        arm_contact_reward = 0.0
        
        # Check if arms are low enough to be "touching" ground (simplified)
        upper_arm_height = self.upper_arm.position.y
        lower_arm_height = self.lower_arm.position.y
        ground_level = 0.0  # World ground level is at y = 0 for our simulation
        
        # Contact reward independent of displacement; proxies how well arms are braced
        if upper_arm_height < ground_level + 0.4:
            arm_contact_reward += 0.003
        if lower_arm_height < ground_level + 0.25:
            arm_contact_reward += 0.005

        # Extra reward if both arms are working together (both near ground)
        if (upper_arm_height < ground_level + 0.2 and 
            lower_arm_height < ground_level + 0.1):
            arm_contact_reward += 0.002  # Coordination bonus

        # If we aren't really moving, damp the bonus by 50%
        if abs(displacement) < 0.0005:
            arm_contact_reward *= 0.5
        
        total_reward += arm_contact_reward * 0.25
        
        # 3. ARM COORDINATION REWARD (20% of total reward)
        # Reward alternating or coordinated arm movements
        coordination_reward = 0.0
        
        # Track arm velocities to detect coordinated movement
        upper_arm_vel = abs(self.upper_arm.angularVelocity)
        lower_arm_vel = abs(self.lower_arm.angularVelocity)
        
        # Reward active arm movement (not static)
        if upper_arm_vel > 0.1 or lower_arm_vel > 0.1:
            coordination_reward += 0.002  # REDUCED
            
        # Reward coordinated movement (both arms moving in useful directions)
        if upper_arm_vel > 0.05 and lower_arm_vel > 0.05:
            # Check if arms are moving in complementary ways
            upper_angle = self.upper_arm.angle
            lower_angle = self.lower_arm.angle
            
            # Reward if arm configuration suggests pushing motion
            if -np.pi/3 < upper_angle < np.pi/3 and 0 < lower_angle < 2*np.pi/3:
                coordination_reward += 0.003  # REDUCED
                
        #total_reward += coordination_reward * 0.20
        
        # 4. STABILITY REWARD (10% of total reward)
        # Reward maintaining body stability while crawling
        body_angle = abs(self.body.angle)
        if body_angle < np.pi/6:  # Within 30 degrees
            stability_reward = 0.003 * (1.0 - (body_angle / (np.pi/6)))  # REDUCED
        else:
            stability_reward = -0.003 * (body_angle - np.pi/6)  # REDUCED: Penalty for excessive tilt
            
        #total_reward += stability_reward * 0.10
        
        # 5. ENERGY EFFICIENCY REWARD (5% of total reward)
        # Reward achieving progress with reasonable energy expenditure
        energy_used = abs(self.current_action_tuple[0]) + abs(self.current_action_tuple[1])
        efficiency_reward = 0.0
        
        if displacement > 0.001 and energy_used > 0:  # Only if making progress
            # Efficiency = progress per unit energy
            efficiency = displacement / (energy_used + 0.1)
            efficiency_reward = min(0.002, efficiency * 0.02)  # REDUCED
        elif energy_used > 1.5:  # Penalty for high energy use without progress
            efficiency_reward = -0.001  # REDUCED
            
        #total_reward += efficiency_reward * 0.05
        
        # 6. BEHAVIORAL PATTERN REWARDS (Bonus)
        # Reward sequences that resemble crawling patterns
        pattern_bonus = 0.0
        
        # Track recent actions to detect crawling-like patterns
        if not hasattr(self, 'action_sequence'):
            self.action_sequence = []
        
        self.action_sequence.append(self.current_action)
        if len(self.action_sequence) > 8:  # Keep last 8 actions
            self.action_sequence.pop(0)
            
        # Look for alternating patterns in actions (simplified)
        if len(self.action_sequence) >= 4:
            # Check for variation in actions (not stuck in one action)
            unique_actions = len(set(self.action_sequence[-4:]))
            if unique_actions >= 2:  # At least 2 different actions
                pattern_bonus += 0.001  # REDUCED
                
        #total_reward += pattern_bonus
        
        # 7. GROUND INTERACTION BONUS
        # Bonus for maintaining contact with ground while moving
        body_height = self.body.position.y
        if (body_height < ground_level + 0.5 and
            displacement > 0.0005):
            ground_interaction_bonus = 0.001  # REDUCED
        else:
            ground_interaction_bonus = 0.0
            
        total_reward += ground_interaction_bonus
        
        # 8. PENALTIES
        # Penalty for getting stuck or falling
        if body_height < ground_level - 0.5:  # Fallen through ground
            total_reward -= 0.01  # REDUCED
        if abs(body_angle) > np.pi/2:  # Flipped over
            total_reward -= 0.02  # REDUCED
            
        # IMPATIENCE PENALTY: Apply negative reward when stuck (inspired by Java implementation)
        impatience_penalty = 0.0
        if self.time_since_good_value > 50.0:  # After ~1s without improvement
            if self.time_since_good_value < 100.0:  # Cap the time range to prevent explosion
                # Quadratic punishment like Java: timeSinceGoodValue^2 * impatience
                impatience_penalty = -(self.time_since_good_value ** 2) * self.impatience
                # Cap the penalty to prevent reward explosion
                impatience_penalty = max(impatience_penalty, -0.02)  # Max penalty of -0.02
                
                if self.id == 0 and self.steps % 100 == 0 and impatience_penalty < -0.001:  # Debug
                    print(f"⏰ Agent {self.id}: Impatience penalty {impatience_penalty:.4f} (stuck for {self.time_since_good_value:.1f} steps)")
        
        total_reward += impatience_penalty
        
        # SANITY CHECK: If no significant forward progress, cap positive rewards
        if displacement <= 0.0005 and total_reward > 0.005:
            total_reward = min(total_reward, 0.005)  # Cap positive reward when not progressing
        
        # Clip final reward to reasonable range (CRITICAL: Much smaller range)
        total_reward = np.clip(total_reward, -0.1, 0.2)
        
        # Debug logging for first agent (MORE FREQUENT TO CATCH THE ISSUE)
        if self.id == 0 and self.steps % 50 == 0:  # Every 50 steps instead of 300
            print(f"🐛 DETAILED Reward Debug Agent {self.id} Step {self.steps}:")
            print(f"  Position: prev_x={prev_x:.4f}, current_x={current_x:.4f}")
            print(f"  Displacement: {displacement:.6f}")
            print(f"  Progress reward: {progress_reward:.6f} (weighted: {progress_reward * 0.4:.6f})")
            print(f"  Upper arm height: {upper_arm_height:.4f}, Lower arm height: {lower_arm_height:.4f}")
            print(f"  Ground level: {ground_level}")
            print(f"  Arm contact raw: {arm_contact_reward:.6f} (weighted: {arm_contact_reward * 0.25:.6f})")
            print(f"  Body height: {body_height:.4f}")
            print(f"  Ground interaction: {ground_interaction_bonus:.6f}")
            print(f"  Body angle: {np.degrees(abs(self.body.angle)):.2f}°")
            print(f"  Impatience penalty: {impatience_penalty:.6f} (stuck for {self.time_since_good_value:.1f} steps)")
            print(f"  TOTAL BEFORE CLIP: {total_reward:.6f}")
            print(f"  TOTAL AFTER CLIP: {np.clip(total_reward, -0.1, 0.2):.6f}")
            print(f"  Episode total so far: {getattr(self, 'total_reward', 'N/A')}")
            print("---")
            
        return total_reward
        
    def enhanced_choose_action(self) -> int:
        """
        Enhanced action selection using confidence-based exploration and adaptive strategies.
        """
        if self.current_state is None:
            return np.random.randint(len(self.actions))
        
        # FORCED EXPLORATION: If stuck on same action for too long, force random selection
        if (hasattr(self, 'action_history') and 
            len(self.action_history) >= 10 and 
            len(set(self.action_history[-10:])) <= 2):  # Only 2 or fewer unique actions in last 10
            if self.id == 0:  # Debug for first agent
                print(f"🔄 Agent {self.id}: FORCED EXPLORATION - stuck on {self.action_history[-5:]}") 
            return np.random.randint(len(self.actions))
        
        # Check if we have enough confidence for exploitation
        action, q_value, is_confident = self.q_table.confidence_based_action(
            self.current_state, 
            min_confidence=self.q_table.confidence_threshold
        )
        
        # Adaptive exploration strategy with HIGHER exploration rate
        if is_confident and np.random.random() > max(0.2, self.epsilon):  # Minimum 20% exploration
            # Use confident exploitation with exploration bonus
            return self.q_table._get_best_action_with_bonus(self.current_state)
        else:
            # Use enhanced epsilon-greedy with bias towards under-explored actions
            return self.q_table.enhanced_epsilon_greedy(
                self.current_state, 
                max(0.2, self.epsilon),  # Minimum 20% exploration
                use_exploration_bonus=True
            )
    
    def choose_action(self) -> int:
        """Choose action using enhanced or traditional method."""
        return self.enhanced_choose_action()
    
    def update_adaptive_exploration(self):
        """
        Update exploration rate based on recent performance (Java-inspired).
        Adapts epsilon and learning rate based on Q-value improvements and time since good value.
        """
        # ENHANCED: Java-inspired adaptive exploration based on time_since_good_value
        if self.time_since_good_value > 5.0:
            # Been a while since good performance - increase randomness (like Java)
            self.epsilon = min(self.max_epsilon, self.epsilon * 1.001)  # Slight increase
            if self.time_since_good_value > 150:  # After long poor performance
                self.learning_rate = min(self.max_learning_rate, 
                                       self.learning_rate + self.impatience * 0.5)
                                       
            # Debug output for first agent when stuck
            if self.id == 0 and int(self.time_since_good_value) % 20 == 0:  # Every 20 steps when stuck
                print(f"🔄 Agent {self.id}: Stuck for {self.time_since_good_value:.0f} steps, increasing exploration")
                print(f"   Epsilon: {self.epsilon:.3f}, Learning rate: {self.learning_rate:.3f}")
        else:
            # Doing well - reduce exploration, focus on exploitation (like Java)
            self.epsilon = max(self.min_epsilon, self.epsilon * 0.999)  # Slight decrease
            self.learning_rate = max(self.min_learning_rate, 
                                   self.learning_rate * 0.9995)  # Very slight decrease
        
        # Also apply traditional epsilon decay but at much slower rate when not stuck
        if self.time_since_good_value <= 5.0:
            self.epsilon = max(self.min_epsilon, self.epsilon * 0.9998)  # Slower decay only when doing well
        
    def cycle_goal(self):
        """Cycle to the next goal (like Java implementation)."""
        if self.steps % self.goal_switch_interval == 0 and self.steps > 0:
            self.current_goal = (self.current_goal + 1) % len(self.goals)
            if self.id == 0:  # Debug for first agent
                print(f"🎯 Agent {self.id}: Switched to goal '{self.goals[self.current_goal]}'")
    
    def get_adaptive_learning_rate(self, state, action):
        """Get adaptive learning rate based on visit count and performance."""
        # Use the enhanced Q-table's adaptive learning rate
        return self.q_table.get_adaptive_learning_rate(state, action, self.learning_rate)
            
    def get_reward(self, prev_x: float) -> float:
        """Use comprehensive crawling-specific reward system."""
        reward = self.get_crawling_reward(prev_x)
        
        # Track immediate reward for debugging
        self.immediate_reward = reward
        
        # Add to recent rewards for performance tracking
        self.recent_rewards.append(reward)
        
        # Update best / worst rewards seen so far this episode
        if reward > self.best_reward_received:
            self.best_reward_received = reward
        if reward < self.worst_reward_received:
            self.worst_reward_received = reward
        
        # UPDATE TIME SINCE GOOD VALUE (Java-inspired pattern)
        # If this reward is good enough, reset the timer; otherwise increment it
        if reward > self.performance_threshold:  # Good reward threshold
            self.time_since_good_value = 0.0  # Reset timer on good performance
        else:
            self.time_since_good_value += 1.0  # Increment timer when performance is poor
        
        # DEBUG LOGGING - monitor crawling reward components (less frequent to avoid spam)
        if self.id == 0 and self.steps % 200 == 0:  # Only for first agent, every 200 steps
            current_x = self.body.position.x
            displacement = current_x - prev_x
            print(f"🐛 Crawling Reward Summary Agent {self.id}:")
            print(f"  Position: {current_x:.4f}, Displacement: {displacement:.4f}")
            print(f"  Reward: {reward:.4f}")
            print(f"  Total reward: {getattr(self, 'total_reward', 'N/A'):.2f}")
            print(f"  Recent avg: {np.mean(list(self.recent_rewards)[-50:]):.4f}")
            print("---")
        
        return reward
        
    def add_action_to_history(self, action_idx: int):
        """Add action to history for reporting."""
        self.action_history.append(action_idx)
        if len(self.action_history) > self.max_action_history:
            self.action_history.pop(0)  # Remove oldest action
    
    def add_experience(self, state, action, reward, next_state, done=False):
        """Add experience to replay buffer."""
        experience = Experience(state, action, reward, next_state, done)
        self.replay_buffer.add(experience)
        
    def learn_from_replay_enhanced(self):
        """Enhanced experience replay learning using the enhanced Q-table features."""
        if len(self.replay_buffer) < self.batch_size:
            return
            
        batch = self.replay_buffer.sample(self.batch_size)
        
        for experience in batch:
            # Use enhanced Q-table update method
            self.q_table.update_q_value_enhanced(
                state=experience.state,
                action=experience.action,
                reward=experience.reward,
                next_state=experience.next_state,
                base_learning_rate=self.learning_rate,
                discount_factor=self.discount_factor,
                use_adaptive_lr=True
            )
    
    def learn_from_replay(self):
        """Traditional experience replay learning (fallback method)."""
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
        """Enhanced step method with adaptive exploration and improved Q-learning."""
        # Using comprehensive crawling-specific reward system
        
        # Initialize action if not set (but we start with a real action now)
        if self.current_action is None:
            self.current_state = self.get_discretized_state()
            action_idx = self.choose_action()
            self.current_action = action_idx
            self.current_action_tuple = self.actions[action_idx]
            self.add_action_to_history(action_idx)
            if self.id == 0:  # Debug for first agent only
                print(f"🤖 Agent {self.id}: Initialized with action {action_idx} = {self.current_action_tuple}")
        
        # Always apply the current action (cheap operation)
        self.apply_action(self.current_action_tuple)
        
        # DEBUG: Show what action is being applied for first agent
        if self.id == 0 and self.steps % 100 == 0:
            print(f"🎮 Agent {self.id} Step {self.steps}: Applying action {self.current_action_tuple}")
            print(f"  Motor speeds: shoulder={self.current_action_tuple[0] * self.motor_speed:.2f}, elbow={self.current_action_tuple[1] * self.motor_speed:.2f}")
            print(f"  Arm angles: shoulder={np.degrees(self.upper_arm.angle):.1f}°, elbow={np.degrees(self.lower_arm.angle):.1f}°")
        
        # Track reward over time using multi-goal system
        current_x = self.body.position.x
        reward = self.get_reward(self.prev_x)
        self.total_reward += reward
        
        # CRITICAL: Update prev_x for next step
        self.prev_x = current_x
        
        # Prevent total reward from exploding negatively (REDUCED: adjusted for new reward scale)
        if self.total_reward < -5.0:  # REDUCED threshold for scaled-down reward system  
            self.total_reward = max(self.total_reward, -5.0)
        
        # Combined debug logging for agent 0, on a longer interval
        if self.id == 0 and self.steps % 600 == 0:
            convergence = self.q_table.get_convergence_estimate()
            q_table_size = len(self.q_table.q_values)
            total_state_actions = sum(len(actions) for actions in self.q_table.q_values.values())
            print(f"🤖 Agent {self.id}: Step {self.steps}")
            print(f"    Pos: ({self.body.position.x:.2f}, {self.body.position.y:.2f})")
            print(f"    Vel: ({self.body.linearVelocity.x:.2f}, {self.body.linearVelocity.y:.2f})")
            print(f"    Action: {self.current_action_tuple}, Reward: {reward:.3f}")
            print(f"    Total reward: {self.total_reward:.2f}, Epsilon: {self.epsilon:.3f}")
            print(f"    🧠 Q-table: {q_table_size} states, {total_state_actions} state-actions, Convergence: {convergence:.3f}")

            # Performance warning
            if q_table_size > 1000:
                print(f"    ⚠️  Q-table getting large! {q_table_size} states may slow performance")
        
        # Enhanced action selection with adaptive intervals
        if self.steps % self.action_interval == 0:
            # Store previous state and action for Q-learning update
            prev_state = self.current_state
            prev_action = self.current_action
            
            # Get new state
            self.current_state = self.get_discretized_state()
            
            # Update Q-value for previous state-action pair if we have one
            if prev_state is not None and prev_action is not None:
                self.q_table.update_q_value_enhanced(
                    state=prev_state,
                    action=prev_action,
                    reward=reward,
                    next_state=self.current_state,
                    base_learning_rate=self.learning_rate,
                    discount_factor=self.discount_factor,
                    use_adaptive_lr=True
                )
            
            # Choose new action using enhanced method
            action_idx = self.choose_action()
            previous_action_tuple = self.current_action_tuple
            self.current_action = action_idx
            self.current_action_tuple = self.actions[action_idx]
            self.add_action_to_history(action_idx)
            
            # ENHANCED DEBUG: Show action changes for first agent
            if self.id == 0 and (self.current_action_tuple != previous_action_tuple or self.steps % 300 == 0):
                confidence_info = self.q_table.confidence_based_action(self.current_state)
                print(f"🎯 Agent {self.id} Step {self.steps}: ACTION CHANGE")
                print(f"  {previous_action_tuple} -> {self.current_action_tuple} (action {action_idx})")
                print(f"  State: {self.current_state}, Confident: {confidence_info[2]}")
                print(f"  Reward: {reward:.4f}, Epsilon: {self.epsilon:.3f}")
                print(f"  Recent actions: {self.action_history[-5:] if len(self.action_history) >= 5 else self.action_history}")
        
        # Adaptive exploration and parameter updates
        if self.steps % 50 == 0:  # Update adaptation every 50 steps
            self.update_adaptive_exploration()
        
        # Experience replay learning (less frequent with enhanced Q-table)
        if self.steps % self.replay_frequency == 0 and len(self.replay_buffer) > self.batch_size:
            self.learn_from_replay_enhanced()
        
        # Q-table size management - prune if getting too large
        if self.steps % 200 == 0:  # Check every 200 steps
            self.prune_q_table_if_needed()
        
        self.steps += 1
        
    def prune_q_table_if_needed(self):
        """Prune Q-table by removing least-visited states to maintain performance."""
        current_size = len(self.q_table.q_values)
        
        if current_size > self.q_table_pruning_threshold:
            # Calculate visit counts for each state
            state_visit_counts = {}
            for state, action_values in self.q_table.visit_counts.items():
                state_visit_counts[state] = sum(action_values)
            
            # Sort states by visit count (least visited first)
            sorted_states = sorted(state_visit_counts.items(), key=lambda x: x[1])
            
            # Remove least visited states until we reach target size
            states_to_remove = current_size - self.max_q_table_states
            for i in range(min(states_to_remove, len(sorted_states))):
                state_to_remove = sorted_states[i][0]
                
                # Only remove states with very few visits to avoid losing important learning
                if state_visit_counts[state_to_remove] <= 2:
                    if state_to_remove in self.q_table.q_values:
                        del self.q_table.q_values[state_to_remove]
                    if state_to_remove in self.q_table.visit_counts:
                        del self.q_table.visit_counts[state_to_remove]
                    if state_to_remove in self.q_table.state_coverage:
                        self.q_table.state_coverage.remove(state_to_remove)
            
            new_size = len(self.q_table.q_values)
            if self.id == 0:  # Debug for first agent
                print(f"🧹 Agent {self.id}: Pruned Q-table from {current_size} to {new_size} states")
        
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
        self.current_action_tuple = (1, 0)  # FIXED: Start with a real action from the action list
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
        
        # Initialize velocity tracking for acceleration-based rewards
        self.prev_velocity = 0.0
        
        # Reset crawling-specific tracking variables
        self.recent_displacements = []
        self.action_sequence = []
        
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
        
        # Reset reward tracking
        self.best_reward_received = -np.inf  # Start at -inf so first reward sets it
        self.worst_reward_received = np.inf   # Start at +inf so first reward sets it
        
        # Reset velocity to prevent physics issues
        for part in [self.body, self.upper_arm, self.lower_arm] + self.wheels:
            part.linearVelocity = (0, 0)
            part.angularVelocity = 0

    def reset_position(self):
        """
        Resets the agent's position and physics state to its starting point,
        but preserves the learned Q-table and other learning parameters.
        """
        try:
            # Disable motors during reset to prevent issues
            if hasattr(self, 'upper_arm_joint') and self.upper_arm_joint:
                self.upper_arm_joint.enableMotor = False
            if hasattr(self, 'lower_arm_joint') and self.lower_arm_joint:
                self.lower_arm_joint.enableMotor = False
            
            # Reset main body safely
            if self.body and hasattr(self.body, 'position'):
                self.body.position = self.initial_position
                self.body.angle = 0
                self.body.linearVelocity = (0, 0)
                self.body.angularVelocity = 0
                self.body.awake = True  # Ensure body is active

            # Reset arms and wheels relative to the body using the original offsets
            base_x, base_y = float(self.initial_position[0]), float(self.initial_position[1])

            # Reset upper arm safely
            if hasattr(self, 'upper_arm') and self.upper_arm:
                self.upper_arm.position = (base_x - 1.0, base_y + 1.0)
                self.upper_arm.angle = 0
                self.upper_arm.linearVelocity = (0, 0)
                self.upper_arm.angularVelocity = 0
                self.upper_arm.awake = True

            # Reset lower arm safely
            if hasattr(self, 'lower_arm') and self.lower_arm:
                self.lower_arm.position = (base_x + 1.0, base_y + 1.0)  # Use base position, not upper arm
                self.lower_arm.angle = 0
                self.lower_arm.linearVelocity = (0, 0)
                self.lower_arm.angularVelocity = 0
                self.lower_arm.awake = True

            # Reset wheels safely
            if hasattr(self, 'wheels') and self.wheels:
                wheel_offsets = [(-1.0, -0.75), (1.0, -0.75)]
                for wheel, offset in zip(self.wheels, wheel_offsets):
                    if wheel:
                        wheel.position = (base_x + offset[0], base_y + offset[1])
                        wheel.linearVelocity = (0, 0)
                        wheel.angularVelocity = 0
                        wheel.awake = True
            
            # Re-enable motors after reset
            if hasattr(self, 'upper_arm_joint') and self.upper_arm_joint:
                self.upper_arm_joint.enableMotor = True
            if hasattr(self, 'lower_arm_joint') and self.lower_arm_joint:
                self.lower_arm_joint.enableMotor = True
            
            # Reset reward and internal state, but not the Q-table itself
            self.total_reward = 0
            self.steps = 0
            
        except Exception as e:
            print(f"⚠️  Error resetting position for agent {self.id}: {e}")
            # Don't print reset message on error to avoid spam

    def get_fitness(self) -> float:
        """Get fitness score for evolution."""
        return self.total_reward
        
    def enhanced_crossover(self, other: 'CrawlingCrateAgent') -> 'CrawlingCrateAgent':
        """Enhanced crossover with parameter and goal mixing."""
        # Create new agent
        new_agent = CrawlingCrateAgent(
            self.world, 
            self.id, 
            self.initial_position,
            category_bits=self.filter.categoryBits,
            mask_bits=self.filter.maskBits
        )
        
        # Enhanced Q-table crossover using the new copy and learn methods
        new_agent.q_table = self.q_table.copy()
        new_agent.q_table.learn_from_other_table(other.q_table, learning_rate=0.5)
        
        # Crossover parameters with averaging and small random variations
        new_agent.learning_rate = (self.learning_rate + other.learning_rate) / 2
        new_agent.learning_rate += np.random.normal(0, 0.01)  # Small variation
        new_agent.learning_rate = np.clip(new_agent.learning_rate, 
                                        new_agent.min_learning_rate, 
                                        new_agent.max_learning_rate)
        
        new_agent.epsilon = (self.epsilon + other.epsilon) / 2
        new_agent.epsilon += np.random.normal(0, 0.02)  # Small variation
        new_agent.epsilon = np.clip(new_agent.epsilon, 
                                  new_agent.min_epsilon, 
                                  new_agent.max_epsilon)
        
        # Crossover goal preferences
        if np.random.random() < 0.5:
            new_agent.current_goal = self.current_goal
            new_agent.goal_weights = self.goal_weights.copy()
        else:
            new_agent.current_goal = other.current_goal
            new_agent.goal_weights = other.goal_weights.copy()
        
        # Mix some goal weights
        for goal in new_agent.goal_weights:
            if np.random.random() < 0.3:  # 30% chance to mix each weight
                self_weight = self.goal_weights.get(goal, new_agent.goal_weights[goal])
                other_weight = other.goal_weights.get(goal, new_agent.goal_weights[goal])
                new_agent.goal_weights[goal] = (self_weight + other_weight) / 2
        
        return new_agent
    
    def crossover(self, other: 'CrawlingCrateAgent') -> 'CrawlingCrateAgent':
        """Create a new agent by crossing over with another agent."""
        return self.enhanced_crossover(other)
        
    def mutate(self, mutation_rate: float = 0.1):
        """Mutate the agent using enhanced mutation."""
        self.enhanced_mutate(mutation_rate)
        
    def enhanced_mutate(self, mutation_rate: float = 0.1):
        """
        Enhanced mutation that includes Q-table, parameters, and goals.
        
        Args:
            mutation_rate: Base mutation rate
        """
        # Mutate Q-table values with adaptive rate
        mutation_count = 0
        for state_key, action_values in self.q_table.q_values.items():
            for action in range(len(action_values)):
                if np.random.random() < mutation_rate:
                    # Adaptive mutation strength based on visit count
                    visit_count = self.q_table.visit_counts[state_key][action]
                    mutation_strength = 0.2 / (1 + visit_count * 0.1)  # Weaker mutation for well-explored states
                    
                    mutation = np.random.normal(0, mutation_strength)
                    action_values[action] = np.clip(action_values[action] + mutation, -10.0, 10.0)
                    mutation_count += 1
        
        # Mutate learning parameters
        if np.random.random() < mutation_rate:
            self.learning_rate = np.clip(
                self.learning_rate + np.random.normal(0, 0.02),
                self.min_learning_rate, self.max_learning_rate
            )
        
        if np.random.random() < mutation_rate:
            self.epsilon = np.clip(
                self.epsilon + np.random.normal(0, 0.05),
                self.min_epsilon, self.max_epsilon
            )
        
        # Mutate goal preferences
        if np.random.random() < mutation_rate * 0.5:  # Lower rate for goal mutation
            self.current_goal = np.random.randint(len(self.goals))
            
        # Mutate reward weights
        if np.random.random() < mutation_rate * 0.3:
            for goal in self.goal_weights:
                if np.random.random() < 0.4:
                    self.goal_weights[goal] = np.clip(
                        self.goal_weights[goal] + np.random.normal(0, 0.01),
                        0.01, 0.15
                    )
        
        if self.id == 0 and mutation_count > 0:  # Debug for first agent
            print(f"🧬 Agent {self.id}: Mutated {mutation_count} Q-values and parameters")
        
    def get_advanced_debug_info(self) -> Dict[str, Any]:
        """Enhanced debugging information with Q-learning metrics."""
        debug = self.get_debug_info()
        
        # Enhanced Q-table statistics
        q_stats = self.q_table.get_enhanced_stats()
        current_goal = self.goals[self.current_goal]
        
        debug.update({
            # Q-learning specific metrics
            'q_table_states': q_stats['total_states'],
            'q_convergence': q_stats['convergence_estimate'],
            'q_coverage': q_stats['state_coverage'],
            'avg_q_value': q_stats['mean_value'],
            'q_value_range': f"{q_stats['min_value']:.2f} to {q_stats['max_value']:.2f}",
            'avg_learning_rate': q_stats['avg_learning_rate'],
            'avg_td_error': q_stats['avg_td_error'],
            
            # Goal and performance metrics
            'current_goal': current_goal,
            'goal_weights': self.goal_weights.copy(),
            'performance_window_avg': np.mean(list(self.recent_rewards)[-50:]) if len(self.recent_rewards) > 50 else 0.0,
            'time_since_good_value': self.time_since_good_value,
            
            # Adaptive parameters
            'adaptive_epsilon': self.epsilon,
            'adaptive_learning_rate': self.learning_rate,
            'exploration_confidence': self.q_table.confidence_threshold,
            
            # Enhanced state information
            'state_type': 'enhanced' if self.use_enhanced_state else 'basic',
            'state_dimensions': self.enhanced_state_size if self.use_enhanced_state else 2,
            
            # Physical performance
            'max_speed_achieved': self.max_speed,
            'current_speed_avg': self.speed,
            'body_angle_degrees': np.degrees(self.body.angle),
            'stability_score': max(0, 1.0 - abs(self.body.angle)),
            
            # Learning efficiency
            'replay_buffer_size': len(self.replay_buffer),
            'replay_buffer_usage': len(self.replay_buffer) / self.replay_buffer.capacity,
            'action_interval': self.action_interval,
            'learning_interval': self.learning_interval,
            
            # Evolution readiness
            'mutation_readiness': self.time_since_good_value / 200.0,  # Normalized readiness for mutation
            'teaching_value': max(0, self.total_reward / 10.0),  # How valuable this agent is as a teacher
        })
        
        return debug
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get enhanced debug information."""
        return self.get_advanced_debug_info()
    
    def estimate_convergence(self) -> float:
        """Estimate how converged the Q-learning is."""
        return self.q_table.get_convergence_estimate()
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get comprehensive performance metrics for analysis."""
        return {
            'total_reward': float(self.total_reward),
            'average_recent_reward': float(np.mean(list(self.recent_rewards)[-100:])) if len(self.recent_rewards) > 100 else 0.0,
            'max_speed': float(self.max_speed),
            'distance_traveled': float(self.body.position.x - self.initial_position[0]),
            'stability_score': float(max(0, 1.0 - abs(self.body.angle))),
            'learning_convergence': float(self.estimate_convergence()),
            'exploration_rate': float(self.epsilon),
            'q_table_coverage': float(len(self.q_table.state_coverage)),
            'steps_completed': float(self.steps),
            'current_goal_index': float(self.current_goal),
            'time_since_improvement': float(self.time_since_good_value),
        }

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
        # Upper Arm (keep as rectangle)
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
        
        # Lower Arm (tapered to a point)
        # Create a tapered polygon that comes to a point at the end
        # Vertices define the shape: wide at base, narrow at tip
        tapered_vertices = [
            (-1.0, -0.2),  # Bottom left (wide end)
            (-1.0, 0.2),   # Top left (wide end)
            (0.5, 0.1),    # Top middle (narrowing)
            (1.0, 0.0),    # Point at the tip
            (0.5, -0.1),   # Bottom middle (narrowing)
        ]
        
        lower_arm = self.world.CreateDynamicBody(
            position=upper_arm.position + (1.0, 0),
            fixtures=[
                b2.b2FixtureDef(
                    shape=b2.b2PolygonShape(vertices=tapered_vertices),
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
        
        # Removed debug print to eliminate overhead - was running 1% of the time 