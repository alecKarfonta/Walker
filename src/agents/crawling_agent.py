"""
Consolidated CrawlingAgent: Box2D physics + neural network learning + evolutionary features.
Replaces the complex hierarchy: CrawlingCrate -> CrawlingCrateAgent -> EvolutionaryCrawlingAgent
"""

import Box2D as b2
import numpy as np
import time
import uuid
from typing import Tuple, List, Dict, Any, Optional, Union
from collections import deque

from .base_agent import BaseAgent


class CrawlingAgent(BaseAgent):
    """
    Consolidated crawling robot with Box2D physics, neural network learning, and evolutionary features.
    Supports variable morphology (1-6 limbs with 2-3 segments each).
    """
    
    def __init__(self, world: b2.b2World, agent_id: Optional[str] = None, 
                 position: Tuple[float, float] = (10, 20), 
                 category_bits=0x0001, mask_bits=0xFFFF,
                 physical_params: Optional[Any] = None,  # Will import PhysicalParameters later
                 parent_lineage: Optional[List[str]] = None,
                 learning_approach: str = "attention_deep_q_learning"):
        
        super().__init__()
        
        # Core identity
        self.id = agent_id or str(uuid.uuid4())[:8]
        self.world = world
        self.initial_position = position
        self.category_bits = category_bits
        self.mask_bits = mask_bits
        
        # Physical parameters (will be properly imported later)
        if physical_params is None:
            # Create default simple robot
            self.num_limbs = 1
            self.segments_per_limb = 2
        else:
            self.physical_params = physical_params
            self.num_limbs = physical_params.num_arms
            self.segments_per_limb = physical_params.segments_per_limb
        
        # Learning system (will be properly initialized after action_size is set)
        self.learning_approach = learning_approach
        self._learning_system = None
        
        # Evolution tracking
        self.generation = 0
        self.parent_lineage = parent_lineage or []
        self.mutation_count = 0
        self.crossover_count = 0
        
        # Performance tracking
        self.total_reward = 0.0
        self.immediate_reward = 0.0
        self.last_reward = 0.0
        self.steps = 0
        self.max_speed = 0.0
        self.best_reward_received = 0.0
        self.worst_reward_received = 0.0
        self.time_since_good_value = 0.0
        
        # ENHANCED: Better action system for locomotion - MUST be set before calling _calculate_state_size()
        self.action_size = 15  # Expanded action space for better locomotion control
        self.actions = self._generate_locomotion_action_combinations()
        self.state_size = self._calculate_state_size()  # For learning manager compatibility
        
        # Initialize learning system now that action_size is set
        self._initialize_learning_system()
        self.current_action = None
        self.current_action_tuple = (1, 0)  # Default: slight forward
        self.current_state = None
        
        # Timing
        self.last_action_time = time.time()
        self.action_persistence_duration = 0.1  # Hold actions for 100ms
        
        # History tracking
        self.action_history = []
        self.max_action_history = 50
        self.recent_displacements = []
        self.prev_x = position[0]
        self.prev_food_distance = float('inf')
        
        # Motor parameters - reduced for stable crawling motion
        self.motor_speed = 3.0  # Reduced from 8.0 to match evolutionary agents
        self.motor_torque = 2000.0
        
        # Physics bodies (will be created)
        self.body = None
        self.upper_arm = None  # For compatibility
        self.lower_arm = None  # For compatibility
        self.upper_arm_joint = None  # For compatibility
        self.lower_arm_joint = None  # For compatibility
        self.limbs = []  # List of limb structures
        self.joints = []  # All joints for easy access
        
        # Create physical robot
        self._create_physics_bodies()
        
        print(f"🤖 Created CrawlingAgent {self.id} with {self.num_limbs} limbs, "
              f"{self.segments_per_limb} segments each")
    
    def _initialize_learning_system(self):
        """
        Initialize the neural network learning system with consistent parameters.
        
        CRITICAL: These parameters must match the robot's state and action generation:
        - state_dim=19: Must match get_state_representation() output size
        - action_dim=15: Must match _generate_locomotion_action_combinations() size
        
        Any dimension mismatch will cause "expected sequence of length X (got Y)" errors.
        """
        try:
            # Import here to avoid circular imports
            from .attention_deep_q_learning import AttentionDeepQLearning
            
            # FIXED PARAMETERS: No dynamic calculation to avoid inconsistencies
            state_size = 19  # Fixed - matches get_state_representation()
            action_size = 15  # Fixed - matches locomotion action combinations
            
            self._learning_system = AttentionDeepQLearning(
                state_dim=state_size,
                action_dim=action_size,
                learning_rate=0.001
            )
                
        except Exception as e:
            print(f"⚠️ Error initializing learning system for agent {self.id}: {e}")
            self._learning_system = None
    
    def _calculate_state_size(self) -> int:
        """
        Calculate state size for enhanced robot representation.
        
        ENHANCED STATE SPACE (19 dimensions total):
        ===========================================
        - Joint angles and velocities: 4 values 
          * shoulder_angle, elbow_angle, shoulder_velocity, elbow_velocity
        - Body physics state: 6 values
          * position_x, position_y, velocity_x, velocity_y, body_angle, angular_velocity  
        - Food targeting information: 4 values
          * food_distance, food_direction_x, food_direction_y, approach_angle
        - Environmental feedback: 3 values
          * ground_contact, arm_contact, stability_measure
        - Temporal context: 2 values
          * recent_shoulder_action, recent_elbow_action
        
        TOTAL: 4 + 6 + 4 + 3 + 2 = 19 dimensions
        
        This matches the neural network architecture in AttentionDeepQLearning.
        All robots use this consistent state representation for proper learning.
        """
        return 19  # Fixed size for consistent neural network architecture
    
    def _generate_locomotion_action_combinations(self) -> List[Tuple[float, float]]:
        """Generate enhanced action combinations optimized for crawling locomotion."""
        combinations = []
        
        # Original 3x3 grid (9 actions) - keep for baseline
        for shoulder in [-1, 0, 1]:  # Backward, stop, forward
            for elbow in [-1, 0, 1]:  # Contract, stop, extend
                combinations.append((shoulder, elbow))
        
        # Enhanced locomotion patterns (6 additional actions)
        # These are common crawling patterns that should be easier to learn
        combinations.extend([
            # Coordinated crawling motions
            (0.5, -0.5),   # Gentle reach forward while contracting elbow
            (-0.5, 0.5),   # Pull back while extending elbow  
            (0.8, -0.2),   # Strong reach with slight elbow contract
            (-0.8, 0.2),   # Strong pull with slight elbow extend
            (0.3, 0.7),    # Gentle shoulder with strong elbow extend
            (-0.3, -0.7),  # Gentle shoulder back with strong elbow contract
        ])
        
        return combinations
    
    def _generate_action_combinations(self) -> List[Tuple[float, float]]:
        """Backward compatibility - delegate to locomotion action generator."""
        return self._generate_locomotion_action_combinations()
    
    def _create_physics_bodies(self):
        """Create Box2D physics bodies based on physical parameters."""
        # Main body (chassis)
        body_def = b2.b2BodyDef(
            type=b2.b2_dynamicBody,
            position=self.initial_position,
            linearDamping=0.1,
            angularDamping=0.1
        )
        self.body = self.world.CreateBody(body_def)
        
        # Main chassis fixture
        chassis_shape = b2.b2PolygonShape(box=(1.5, 0.75))
        chassis_fixture = self.body.CreateFixture(
            shape=chassis_shape, 
            density=4.0, 
            friction=0.8,
            filter=b2.b2Filter(categoryBits=self.category_bits, maskBits=self.mask_bits)
        )
        
        # Create wheels for compatibility with rendering system
        self._create_wheels()
        
        # Create simple 2-segment arm for now (for compatibility)
        self._create_simple_arm()
    
    def _create_simple_arm(self):
        """Create a simple 2-segment arm (for compatibility with existing code)."""
        # Upper arm
        upper_arm_def = b2.b2BodyDef(
            type=b2.b2_dynamicBody,
            position=self.body.GetWorldPoint((-1.0, 1.0)) if self.body else (0, 0),
            linearDamping=0.05,
            angularDamping=0.05
        )
        self.upper_arm = self.world.CreateBody(upper_arm_def)
        
        upper_arm_fixture = self.upper_arm.CreateFixture(
            shape=b2.b2PolygonShape(box=(1.0, 0.2)),
            density=0.1,
            friction=0.6,
            filter=b2.b2Filter(categoryBits=self.category_bits, maskBits=self.mask_bits)
        )
        
        # Lower arm
        lower_arm_def = b2.b2BodyDef(
            type=b2.b2_dynamicBody,
            position=self.upper_arm.GetWorldPoint((1.0, 0)),
            linearDamping=0.05,
            angularDamping=0.05
        )
        self.lower_arm = self.world.CreateBody(lower_arm_def)
        
        # Tapered shape for lower arm
        tapered_vertices = [
            (-1.0, -0.2), (-1.0, 0.2), (0.5, 0.1),
            (1.0, 0.0), (0.5, -0.1),
        ]
        
        lower_arm_fixture = self.lower_arm.CreateFixture(
            shape=b2.b2PolygonShape(vertices=tapered_vertices),
            density=0.1,
            friction=0.6,
            filter=b2.b2Filter(categoryBits=self.category_bits, maskBits=self.mask_bits)
        )
        
        # Create joints
        self.upper_arm_joint = self.world.CreateRevoluteJoint(
            bodyA=self.body,
            bodyB=self.upper_arm,
            localAnchorA=(-1.0, 1.0),
            localAnchorB=(-1.0, 0),
            enableMotor=True,
            maxMotorTorque=self.motor_torque,
            motorSpeed=0,
            enableLimit=True,
            lowerAngle=-np.pi/2,
            upperAngle=np.pi/2,
        )
        
        self.lower_arm_joint = self.world.CreateRevoluteJoint(
            bodyA=self.upper_arm,
            bodyB=self.lower_arm,
            localAnchorA=(1.0, 0),
            localAnchorB=(-1.0, 0),
            enableMotor=True,
            maxMotorTorque=self.motor_torque,
            motorSpeed=0,
            enableLimit=True,
            lowerAngle=0,
            upperAngle=3*np.pi/4,
        )
        
        self.joints = [self.upper_arm_joint, self.lower_arm_joint]
    
    def _create_wheels(self):
        """Create wheels for the robot (for rendering compatibility)."""
        self.wheels = []
        wheel_anchor_positions = [(-1.0, -0.75), (1.0, -0.75)]
        
        for anchor_pos in wheel_anchor_positions:
            wheel_world_pos = self.body.GetWorldPoint(anchor_pos) if self.body else (0, 0)
            wheel_def = b2.b2BodyDef(
                type=b2.b2_dynamicBody,
                position=wheel_world_pos,
                linearDamping=0.05,
                angularDamping=0.05
            )
            wheel = self.world.CreateBody(wheel_def)
            
            wheel_fixture = wheel.CreateFixture(
                shape=b2.b2CircleShape(radius=0.5),
                density=8.0,
                friction=0.9,
                filter=b2.b2Filter(categoryBits=self.category_bits, maskBits=self.mask_bits)
            )
            
            # Create wheel joint
            wheel_joint = self.world.CreateRevoluteJoint(
                bodyA=self.body,
                bodyB=wheel,
                localAnchorA=anchor_pos,
                localAnchorB=(0, 0),
                enableMotor=False
            )
            
            self.wheels.append(wheel)
        
        # Initialize wheel joints list for compatibility
        self.wheel_joints = []
    
    def get_state_representation(self) -> np.ndarray:
        """
        Get enhanced state representation for neural network training.
        
        Returns exactly 19 dimensions as documented in _calculate_state_size().
        This ensures consistent input to the neural network architecture.
        """
        try:
            state_array = np.zeros(19, dtype=np.float32)  # Fixed size for consistency
            
            # DEBUG: Minimal state generation confirmation (only once per agent)
            if not hasattr(self, '_state_gen_confirmed'):
                print(f"✅ Agent {str(self.id)[:8]}: Using 19D state representation")
                self._state_gen_confirmed = True
            
            # 1. Joint angles and velocities (4 values)
            if self.upper_arm_joint and self.lower_arm_joint:
                # Normalize angles to [-1, 1]
                shoulder_angle = np.tanh(self.upper_arm_joint.angle)
                elbow_angle = np.tanh(self.lower_arm_joint.angle)
                # Normalize angular velocities
                shoulder_vel = np.tanh(self.upper_arm_joint.motorSpeed / 10.0)
                elbow_vel = np.tanh(self.lower_arm_joint.motorSpeed / 10.0)
                
                state_array[0] = shoulder_angle
                state_array[1] = elbow_angle
                state_array[2] = shoulder_vel
                state_array[3] = elbow_vel
            
            # 2. Body state (6 values)
            if self.body:
                # Position normalized relative to starting position
                state_array[4] = np.tanh((self.body.position.x - self.initial_position[0]) / 20.0)
                state_array[5] = np.tanh((self.body.position.y - self.initial_position[1]) / 10.0)
                # Velocity
                state_array[6] = np.tanh(self.body.linearVelocity.x / 5.0)
                state_array[7] = np.tanh(self.body.linearVelocity.y / 5.0)
                # Orientation
                state_array[8] = np.tanh(self.body.angle * 2.0)
                state_array[9] = np.tanh(self.body.angularVelocity / 3.0)
            
            # 3. Food targeting information (4 values)
            # Get food info from training environment if available
            food_info = None
            if hasattr(self, '_training_env') and self._training_env:
                try:
                    food_info = self._training_env._get_closest_food_distance_for_agent(self)
                except:
                    pass
            
            if food_info and food_info.get('distance', 999999) < 100:
                # Food distance normalized
                state_array[10] = np.tanh(food_info['distance'] / 50.0)
                # Food direction as x/y components
                food_direction = food_info.get('direction', 0)
                state_array[11] = np.tanh(food_direction / 25.0)  # X direction
                # Y direction (if available)
                food_y_dir = food_info.get('direction_y', 0)
                state_array[12] = np.tanh(food_y_dir / 25.0)
                # Approach angle (angle between body direction and food direction)
                if self.body:
                    body_angle = self.body.angle
                    food_angle = np.arctan2(food_y_dir, food_direction)
                    approach_angle = food_angle - body_angle
                    state_array[13] = np.tanh(approach_angle / np.pi)
                else:
                    state_array[13] = 0.0
            else:
                # No food information available
                state_array[10:14] = 0.0
            
            # 4. Physics feedback (3 values)
            # Ground contact detection
            ground_contact = 0.0
            arm_contact = 0.0
            if self.body:
                for contact_edge in self.body.contacts:
                    if contact_edge.contact and contact_edge.contact.touching:
                        # Check for ground contact (assume ground has category bit 0x0001)
                        fixture_a = contact_edge.contact.fixtureA
                        fixture_b = contact_edge.contact.fixtureB
                        if ((fixture_a.filterData.categoryBits & 0x0001) or 
                            (fixture_b.filterData.categoryBits & 0x0001)):
                            ground_contact = 1.0
                        else:
                            arm_contact = 1.0  # Contact with other objects
            
            state_array[14] = ground_contact
            state_array[15] = arm_contact
            
            # Stability measure (combination of velocity and angle)
            if self.body:
                stability = 1.0 / (1.0 + abs(self.body.angle) + abs(self.body.angularVelocity))
                state_array[16] = stability
            else:
                state_array[16] = 0.0
            
            # 5. Action history (2 values) - recent actions for temporal context
            if hasattr(self, 'current_action_tuple') and self.current_action_tuple:
                state_array[17] = self.current_action_tuple[0]  # Recent shoulder action
                state_array[18] = self.current_action_tuple[1]  # Recent elbow action
            else:
                state_array[17:19] = 0.0
            
            # Ensure no NaN values
            state_array = np.nan_to_num(state_array, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return state_array
            
        except Exception as e:
            print(f"⚠️ Error getting expanded state for agent {self.id}: {e}")
            return np.zeros(19, dtype=np.float32)  # Return fixed size on error
    
    def _get_legacy_state_representation(self, expected_size: int) -> np.ndarray:
        """Generate legacy state representation for backward compatibility."""
        try:
            state_array = np.zeros(expected_size, dtype=np.float32)
            
            # Legacy format: joint angles + metadata
            idx = 0
            
            # Joint angles (simplified)
            if self.upper_arm_joint and self.lower_arm_joint and idx < expected_size:
                state_array[idx] = np.tanh(self.upper_arm_joint.angle)
                idx += 1
                if idx < expected_size:
                    state_array[idx] = np.tanh(self.lower_arm_joint.angle)
                    idx += 1
            
            # Fill remaining with basic physics metadata
            if self.body and idx < expected_size:
                state_array[idx] = np.tanh(self.body.linearVelocity.x / 5.0)
                idx += 1
                if idx < expected_size:
                    state_array[idx] = np.tanh(self.body.angle * 2.0)
                    idx += 1
                if idx < expected_size:
                    state_array[idx] = np.tanh((self.body.position.x - self.prev_x) * 10.0)
                    idx += 1
            
            # Ensure no NaN values
            state_array = np.nan_to_num(state_array, nan=0.0, posinf=1.0, neginf=-1.0)
            return state_array
            
        except Exception as e:
            print(f"⚠️ Error getting legacy state for agent {self.id}: {e}")
            return np.zeros(expected_size, dtype=np.float32)
    
    def choose_action(self, state: np.ndarray) -> int:
        """Choose action using neural network."""
        if self._learning_system:
            action = self._learning_system.choose_action(state)
            return max(0, min(action, self.action_size - 1))
        else:
            return np.random.randint(0, self.action_size)
    
    def learn_from_experience(self, prev_state, action, reward, new_state, done=False):
        """Learn from experience using neural network."""
        # Ensure states are numpy arrays
        if prev_state is None:
            return
        
        self._learning_system.store_experience(prev_state, action, reward, new_state, done)
        
        # Train every 30 steps (reduced frequency for better performance)
        if self.steps % 30 == 0:
            # Log first training session
            if not hasattr(self, '_training_started'):
                print(f"🚀 Agent {str(self.id)[:8]}: Neural network training STARTED at step {self.steps}")
                self._training_started = True
                # Initialize moving average tracking
                self._loss_history = []
                self._qval_history = []
                self._moving_avg_window = 10  # Track last 10 values
            
            training_stats = self._learning_system.learn()
            
            # Track moving averages
            if training_stats:
                loss = training_stats.get('loss', 0.0)
                qval = training_stats.get('mean_q_value', 0.0)
                
                # Add to history and maintain window size
                self._loss_history.append(loss)
                self._qval_history.append(qval)
                
                if len(self._loss_history) > self._moving_avg_window:
                    self._loss_history.pop(0)
                if len(self._qval_history) > self._moving_avg_window:
                    self._qval_history.pop(0)
            
            # Log training activity every 150 steps (LCM of 30 and 50) for detailed metrics
            if self.steps % 150 == 0:
                if training_stats and hasattr(self, '_loss_history'):
                    # Calculate moving averages
                    loss_avg = sum(self._loss_history) / len(self._loss_history) if self._loss_history else 0.0
                    qval_avg = sum(self._qval_history) / len(self._qval_history) if self._qval_history else 0.0
                    
                    current_loss = training_stats.get('loss', 0.0)
                    current_qval = training_stats.get('mean_q_value', 0.0)
                    
                    print(f"🧠 Agent {str(self.id)[:8]}: Training step {self.steps}, "
                            f"Loss: {current_loss:.4f} (avg: {loss_avg:.4f}), "
                            f"Q-val: {current_qval:.3f} (avg: {qval_avg:.3f})")
                else:
                    print(f"🧠 Agent {str(self.id)[:8]}: Training step {self.steps}, No stats returned")
            
            # Also log first few training sessions for each agent
            if hasattr(self, '_training_count'):
                self._training_count += 1
            else:
                self._training_count = 1
            
            if self._training_count <= 3 and training_stats:
                print(f"🔥 Agent {str(self.id)[:8]}: Training session #{self._training_count}, "
                        f"Experience buffer size: {getattr(self._learning_system, 'replay_buffer_size', 'unknown')}")
            
    
    def apply_action(self, action: Tuple[float, float]):
        """Apply action to the robot's arms."""
        try:
            if self.upper_arm_joint and self.lower_arm_joint:
                shoulder_speed = float(np.clip(action[0], -1.0, 1.0)) * self.motor_speed
                elbow_speed = float(np.clip(action[1], -1.0, 1.0)) * self.motor_speed
                
                self.upper_arm_joint.motorSpeed = shoulder_speed
                self.lower_arm_joint.motorSpeed = elbow_speed
                
                # Wake up bodies
                if self.upper_arm:
                    self.upper_arm.awake = True
                if self.lower_arm:
                    self.lower_arm.awake = True
        except Exception as e:
            print(f"⚠️ Error applying action for agent {self.id}: {e}")
    
    def get_crawling_reward(self, prev_x: float, food_info = None) -> float:
        """Calculate reward for crawling behavior with food-seeking incentive."""
        if not self.body:
            return 0.0
            
        current_x = self.body.position.x
        current_y = self.body.position.y
        total_reward = 0.0
        
        # ENHANCED: Much stronger forward progress reward
        displacement = current_x - prev_x
        if displacement > 0.001:  # Lower threshold for sensitivity
            progress_reward = displacement * 5.0  # MUCH stronger multiplier (was 0.5)
            
            # Sustained movement bonus with stronger incentive
            if hasattr(self, 'recent_displacements'):
                self.recent_displacements.append(displacement)
                if len(self.recent_displacements) > 10:
                    self.recent_displacements.pop(0)
                if len(self.recent_displacements) >= 5:
                    avg_displacement = sum(self.recent_displacements) / len(self.recent_displacements)
                    if avg_displacement > 0.002:  # Lower threshold for bonus
                        progress_reward *= 1.5  # Stronger bonus (was 1.1)
            else:
                self.recent_displacements = [displacement]
        elif displacement < -0.0005:
            progress_reward = displacement * 2.0  # Stronger penalty for going backward
        else:
            progress_reward = 0.0
        
        total_reward += progress_reward
        
        # ENHANCED: Much stronger food-seeking reward
        if food_info and food_info.get('distance', 999999) < 100:
            current_food_distance = food_info['distance']
            
            # Initialize previous food distance if not exists
            if not hasattr(self, 'prev_food_distance'):
                self.prev_food_distance = current_food_distance
            
            # Reward getting closer to food
            food_distance_change = self.prev_food_distance - current_food_distance
            if food_distance_change > 0.05:  # Lower threshold for sensitivity
                food_seeking_reward = food_distance_change * 2.0  # Much stronger reward (was 0.3)
                total_reward += food_seeking_reward
                
                # Extra bonus if very close to food
                if current_food_distance < 5.0:
                    total_reward += 0.2  # Much stronger bonus (was 0.02)
                elif current_food_distance < 10.0:
                    total_reward += 0.1  # Medium bonus for being close
            elif food_distance_change < -0.05:  # Moving away from food
                food_seeking_penalty = food_distance_change * 0.5  # Stronger penalty (was 0.1)
                total_reward += food_seeking_penalty
            
            # Update previous food distance
            self.prev_food_distance = current_food_distance
        
        # ENHANCED: Stability reward with better scaling
        body_angle = abs(self.body.angle)
        if body_angle < 0.2:
            total_reward += 0.05  # Stronger stability reward (was 0.01)
        elif body_angle > 1.5:
            total_reward -= 0.02  # Stronger penalty for instability (was 0.005)
        
        # NEW: Velocity-based reward for maintaining good speed
        if self.body.linearVelocity.x > 0.1:
            velocity_reward = min(0.1, self.body.linearVelocity.x * 0.1)
            total_reward += velocity_reward
        
        # NEW: Efficiency reward - penalize excessive joint movement
        if hasattr(self, 'upper_arm_joint') and hasattr(self, 'lower_arm_joint'):
            if self.upper_arm_joint and self.lower_arm_joint:
                joint_efficiency = abs(self.upper_arm_joint.motorSpeed) + abs(self.lower_arm_joint.motorSpeed)
                if joint_efficiency > 10.0:  # Penalize excessive movement
                    total_reward -= 0.01
        
        # EXPANDED: Much wider reward range for stronger learning signals
        total_reward = np.clip(total_reward, -1.0, 1.0)  # Was [-0.1, 0.1]
        
        return total_reward
    
    def step(self, dt: float):
        """Main step function - combines physics and learning."""
        # Initialize action if needed
        if self.current_action is None:
            self.current_state = self.get_state_representation()
            action_idx = self.choose_action(self.current_state)
            self.current_action = action_idx
            self.current_action_tuple = self.actions[action_idx]
        
        # Apply current action
        self.apply_action(self.current_action_tuple)
        
        # Calculate reward
        current_x = self.body.position.x if self.body else 0.0
        
        # Get food information from training environment if available
        food_info = None
        if hasattr(self, '_training_env') and self._training_env:
            try:
                food_info = self._training_env._get_closest_food_distance_for_agent(self)
            except Exception as e:
                # If food info fails, continue without it
                pass
        
        reward = self.get_crawling_reward(self.prev_x, food_info)
        self.total_reward += reward
        self.immediate_reward = reward
        self.last_reward = reward
        
        # Time-based action selection
        current_time = time.time()
        if current_time - self.last_action_time >= self.action_persistence_duration:
            # Store previous state and action for learning
            prev_state = self.current_state
            prev_action = self.current_action
            
            # Get new state and action
            self.current_state = self.get_state_representation()
            
            # Learn from experience
            if prev_state is not None and prev_action is not None:
                self.learn_from_experience(prev_state, prev_action, reward, self.current_state)
            
            # Choose new action
            action_idx = self.choose_action(self.current_state)
            self.current_action = action_idx
            self.current_action_tuple = self.actions[action_idx]
            
            # Add to history
            self.action_history.append(action_idx)
            if len(self.action_history) > self.max_action_history:
                self.action_history.pop(0)
            
            self.last_action_time = current_time
        
        # Update tracking
        self.prev_x = current_x
        self.steps += 1
    
    def reset(self):
        """Reset agent for new episode while preserving learning."""
        self.reset_position()
        
        # Reset state tracking
        self.current_state = None
        self.current_action = None
        self.total_reward = 0.0
        self.immediate_reward = 0.0
        self.last_reward = 0.0
        self.steps = 0
        
        # Reset timing
        self.last_action_time = time.time()
        self.current_action_tuple = (1, 0)
        self.prev_x = self.initial_position[0]
        
        # Reset tracking
        self.action_history = []
        self.recent_displacements = []
    
    def reset_position(self):
        """Reset physical position while preserving learning state."""
        try:
            if not self.body:
                return
                
            # Reset main body
            self.body.position = self.initial_position
            self.body.angle = 0
            self.body.linearVelocity = (0, 0)
            self.body.angularVelocity = 0
            self.body.awake = True
            
            # Reset arms
            if self.upper_arm:
                self.upper_arm.position = self.body.GetWorldPoint((-1.0, 1.0))
                self.upper_arm.angle = 0
                self.upper_arm.linearVelocity = (0, 0)
                self.upper_arm.angularVelocity = 0
                self.upper_arm.awake = True
            
            if self.lower_arm:
                self.lower_arm.position = self.upper_arm.GetWorldPoint((1.0, 0)) if self.upper_arm else (0, 0)
                self.lower_arm.angle = 0
                self.lower_arm.linearVelocity = (0, 0)
                self.lower_arm.angularVelocity = 0
                self.lower_arm.awake = True
            
            # Reset wheels
            if hasattr(self, 'wheels') and self.wheels:
                wheel_offsets = [(-1.0, -0.75), (1.0, -0.75)]
                for wheel, offset in zip(self.wheels, wheel_offsets):
                    if wheel:
                        wheel.position = self.body.GetWorldPoint(offset)
                        wheel.linearVelocity = (0, 0)
                        wheel.angularVelocity = 0
                        wheel.awake = True
            
            self.total_reward = 0
            self.steps = 0
            
        except Exception as e:
            print(f"⚠️ Error resetting position for agent {self.id}: {e}")
    
    def destroy(self):
        """Clean up physics bodies."""
        try:
            # Destroy wheels first
            if hasattr(self, 'wheels') and self.wheels:
                for wheel in self.wheels:
                    if wheel and wheel in self.world.bodies:
                        self.world.DestroyBody(wheel)
                self.wheels = []
            
            # Destroy bodies in correct order
            if self.lower_arm and self.lower_arm in self.world.bodies:
                self.world.DestroyBody(self.lower_arm)
            if self.upper_arm and self.upper_arm in self.world.bodies:
                self.world.DestroyBody(self.upper_arm)
            if self.body and self.body in self.world.bodies:
                self.world.DestroyBody(self.body)
                
        except Exception as e:
            print(f"⚠️ Error destroying agent {self.id}: {e}")
    
    def get_fitness(self) -> float:
        """Get fitness score for evolution."""
        return self.total_reward
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information."""
        return {
            'agent_id': self.id,
            'learning_approach': self.learning_approach,
            'total_reward': self.total_reward,
            'steps': self.steps,
            'position': (self.body.position.x, self.body.position.y) if self.body else (0, 0),
            'velocity': (self.body.linearVelocity.x, self.body.linearVelocity.y) if self.body else (0, 0),
            'body_angle': self.body.angle if self.body else 0,
            'current_action': self.current_action_tuple,
            'num_limbs': self.num_limbs,
            'segments_per_limb': self.segments_per_limb,
            'state_size': self._calculate_state_size(),
            'action_size': self.action_size,
        }
    
    # Backward compatibility methods
    def get_state(self):
        """Backward compatibility - return state as list."""
        if self.body and self.upper_arm and self.lower_arm:
            return [
                self.body.position.x, self.body.position.y,
                self.body.linearVelocity.x, self.body.linearVelocity.y,
                self.body.angle, self.upper_arm.angle, self.lower_arm.angle
            ]
        return [0.0] * 7
    
    def take_action(self, action):
        """Backward compatibility."""
        if isinstance(action, (list, tuple)) and len(action) >= 2:
            self.apply_action((float(action[0]), float(action[1])))
    
    def get_reward(self, prev_x: float) -> float:
        """Backward compatibility."""
        # Get food info for backward compatibility
        food_info = None
        if hasattr(self, '_training_env') and self._training_env:
            try:
                food_info = self._training_env._get_closest_food_distance_for_agent(self)
            except:
                pass
        return self.get_crawling_reward(prev_x, food_info)
    
    def update(self, delta_time: float):
        """Backward compatibility."""
        self.step(delta_time)
    
    # Evolution methods
    def clone_with_mutation(self, mutation_rate: float = 0.1) -> 'CrawlingAgent':
        """Create a mutated copy of this agent for evolution."""
        import copy
        from src.agents.physical_parameters import PhysicalParameters
        
        # Create new physical parameters by mutating current ones
        if hasattr(self, 'physical_params') and self.physical_params:
            mutated_params = self.physical_params.mutate(mutation_rate)
        else:
            mutated_params = PhysicalParameters.random_parameters()
        
        # Create new agent with mutated parameters
        mutated_agent = CrawlingAgent(
            world=self.world,
            agent_id=None,  # Generate new ID
            position=self.initial_position,
            category_bits=self.category_bits,
            mask_bits=self.mask_bits,
            physical_params=mutated_params,
            parent_lineage=getattr(self, 'parent_lineage', []) + [self.id],
            learning_approach=self.learning_approach
        )
        
        # Copy neural network weights if available
        if hasattr(self, '_attention_dqn') and self._attention_dqn and hasattr(self._attention_dqn, 'get_network_state'):
            try:
                network_state = self._attention_dqn.get_network_state()
                if hasattr(mutated_agent._attention_dqn, 'set_network_state'):
                    mutated_agent._attention_dqn.set_network_state(network_state)
            except Exception as e:
                print(f"Warning: Could not copy network state: {e}")
        
        return mutated_agent
    
    def crossover(self, other: 'CrawlingAgent') -> 'CrawlingAgent':
        """Create a child agent by crossing over with another agent."""
        from src.agents.physical_parameters import PhysicalParameters
        
        # Crossover physical parameters
        if hasattr(self, 'physical_params') and hasattr(other, 'physical_params'):
            child_params = self.physical_params.crossover(other.physical_params)
        else:
            child_params = PhysicalParameters.random_parameters()
        
        # Create child agent
        child_agent = CrawlingAgent(
            world=self.world,
            agent_id=None,  # Generate new ID
            position=self.initial_position,
            category_bits=self.category_bits,
            mask_bits=self.mask_bits,
            physical_params=child_params,
            parent_lineage=[self.id, other.id],
            learning_approach=self.learning_approach
        )
        
        # Try to blend neural network weights (simplified approach)
        if (hasattr(self, '_attention_dqn') and self._attention_dqn and 
            hasattr(other, '_attention_dqn') and other._attention_dqn):
            try:
                # For now, just use one parent's network (could be improved with actual weight blending)
                import random
                parent_to_use = self if random.random() < 0.5 else other
                network_state = parent_to_use._attention_dqn.get_network_state()
                if hasattr(child_agent._attention_dqn, 'set_network_state'):
                    child_agent._attention_dqn.set_network_state(network_state)
            except Exception as e:
                print(f"Warning: Could not blend network states: {e}")
        
        return child_agent
    
    def get_evolutionary_fitness(self) -> float:
        """Get fitness value for evolutionary selection."""
        # Use distance traveled as primary fitness
        if hasattr(self, 'body') and self.body:
            current_x = self.body.position.x
            distance = current_x - self.initial_position[0]
            return max(0.0, distance)  # Ensure non-negative fitness
        return 0.0 