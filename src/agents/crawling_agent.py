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
        
        # 🌊 INFINITE EXPLORATION REWARD SYSTEM: Replace episodic rewards with rolling windows
        self.total_reward = 0.0  # Keep for backward compatibility, but deprecate usage
        self.immediate_reward = 0.0
        self.last_reward = 0.0
        
        # 🎯 NEW: Rolling reward system for infinite exploration
        self.reward_window_size = 1000  # Track last 1000 steps
        self.recent_rewards = deque(maxlen=self.reward_window_size)  # Sliding window of rewards
        self.recent_steps_window = deque(maxlen=self.reward_window_size)  # Corresponding step numbers
        
        # Performance metrics over different time scales
        self.short_term_window = deque(maxlen=100)   # Last 100 steps (short-term performance)
        self.medium_term_window = deque(maxlen=500)  # Last 500 steps (medium-term performance)
        self.long_term_window = deque(maxlen=2000)   # Last 2000 steps (long-term performance)
        
        # 📊 Performance tracking for infinite exploration
        self.reward_rate = 0.0          # Reward per step (current performance)
        self.short_term_avg = 0.0       # Average reward over last 100 steps
        self.medium_term_avg = 0.0      # Average reward over last 500 steps
        self.long_term_avg = 0.0        # Average reward over last 2000 steps
        self.performance_trend = 0.0    # Improvement rate (positive = getting better)
        
        # 🎖️ Achievement tracking for motivation
        self.best_short_term_avg = -float('inf')
        self.best_medium_term_avg = -float('inf')
        self.best_long_term_avg = -float('inf')
        self.achievement_count = 0      # Number of personal records broken
        
        # 📈 Progressive benchmarks instead of episodes
        self.benchmark_intervals = [100, 500, 1000, 2000, 5000, 10000]  # Steps at which to evaluate
        self.benchmark_scores = {}      # Store scores at each benchmark
        self.next_benchmark_step = 100  # Next milestone to evaluate
        
        # 🕒 Time-based performance windows
        self.performance_history = []   # (timestamp, reward_rate, avg_reward) tuples
        self.last_performance_update = time.time()
        self.performance_update_interval = 30.0  # Update every 30 seconds
        
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
        self.prev_y = position[1]
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
        
        # Restore missing basic attributes
        self.steps = 0
        self.max_speed = 0.0
        self.best_reward_received = 0.0
        self.worst_reward_received = 0.0
        self.time_since_good_value = 0.0
        
        # Optional training environment reference (for food info)
        self._training_env = None
        
        # Ray casting system for forward perception
        self.ray_sensor_range = 8.0  # 8 meter max range
        self.num_rays = 5  # 5 forward-facing rays
        self.ray_angles = self._calculate_ray_angles()  # Pre-calculate ray directions
        self.last_ray_results = []  # Cache last ray cast results
    
    def _initialize_learning_system(self):
        """
        Initialize the neural network learning system with consistent parameters.
        
        CRITICAL: These parameters must match the robot's state and action generation:
        - state_dim=19: Must match get_state_representation() output size
        - action_dim=15: Must match _generate_locomotion_action_combinations() size
        
        Any dimension mismatch will cause "expected sequence of length X (got Y)" errors.
        """
        try:
            # CRITICAL: Check if network is already assigned by Robot Memory Pool
            if hasattr(self, '_learning_system') and self._learning_system is not None:
                print(f"🧠 Agent {self.id}: Using pre-assigned network from Robot Memory Pool")
                return
            
            # Import here to avoid circular imports
            from .attention_deep_q_learning import AttentionDeepQLearning
            
            # FIXED PARAMETERS: No dynamic calculation to avoid inconsistencies
            state_size = 29  # Fixed - matches get_state_representation() with ray sensing
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
        Calculate state size for enhanced robot representation with ray sensing.
        
        ENHANCED STATE SPACE (29 dimensions total):
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
        - Ray sensing data: 10 values (5 rays × 2 values each)
          * ray1_distance, ray1_object_type, ray2_distance, ray2_object_type, ...
        
        TOTAL: 4 + 6 + 4 + 3 + 2 + 10 = 29 dimensions
        
        Ray sensing provides forward-facing environmental awareness for navigation.
        """
        return 29  # Updated size including ray sensing data
    
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
    
    def _calculate_ray_angles(self) -> List[float]:
        """Calculate 5 ray angles for forward-right sensing."""
        # 5 rays spread in a 90-degree cone facing forward-right
        # Center ray points forward-right (45 degrees), others spread ±45 degrees
        center_angle = np.pi / 4  # 45 degrees to the right
        spread = np.pi / 4  # ±45 degrees spread
        
        angles = []
        for i in range(self.num_rays):
            # Spread rays evenly across the cone
            angle_offset = (i - 2) * (spread / 2)  # -2, -1, 0, 1, 2 gives even spread
            angles.append(center_angle + angle_offset)
        
        return angles
    
    def _cast_ray(self, start_pos: Tuple[float, float], angle: float) -> Tuple[float, int]:
        """
        Cast a single ray and return distance and object type.
        Only considers ground/terrain and obstacle bodies, ignoring other robots.
        
        Returns:
            Tuple of (distance, object_type_code)
            object_type_code: 0=clear, 1=obstacle, 2=terrain
        """
        # Calculate ray endpoint
        end_x = start_pos[0] + np.cos(angle) * self.ray_sensor_range
        end_y = start_pos[1] + np.sin(angle) * self.ray_sensor_range
        
        class RayCallback(b2.b2RayCastCallback):
            def __init__(self, robot_instance, ray_range):
                super().__init__()
                self.hit_distance = ray_range  # Start with max range
                self.object_type = 0  # Default: clear
                self.robot_instance = robot_instance
                self.ray_range = ray_range
                
            def ReportFixture(self, fixture, point, normal, fraction):
                # Skip the robot's own body parts
                robot_bodies = []
                if self.robot_instance.body:
                    robot_bodies.append(self.robot_instance.body)
                if self.robot_instance.upper_arm:
                    robot_bodies.append(self.robot_instance.upper_arm)
                if self.robot_instance.lower_arm:
                    robot_bodies.append(self.robot_instance.lower_arm)
                if hasattr(self.robot_instance, 'wheels') and self.robot_instance.wheels:
                    robot_bodies.extend(self.robot_instance.wheels)
                
                if fixture.body in robot_bodies:
                    return -1  # Ignore robot's own body parts
                
                # Check if this is something we should ignore
                if fixture.body.userData:
                    body_data = fixture.body.userData
                    if isinstance(body_data, dict):
                        obj_type = body_data.get('type')
                        # Skip other robots and food
                        if obj_type == 'robot' or obj_type == 'food':
                            return -1  # Ignore
                
                # Calculate distance to hit point
                distance = fraction * self.ray_range
                
                # Only update if this is closer than previous hits
                if distance < self.hit_distance:
                    self.hit_distance = distance
                    
                    # Determine object type - only obstacles and terrain
                    if fixture.body.userData:
                        body_data = fixture.body.userData
                        if isinstance(body_data, dict):
                            if body_data.get('type') == 'obstacle':
                                self.object_type = 1  # Obstacle
                            elif body_data.get('type') == 'terrain' or body_data.get('type') == 'ground':
                                self.object_type = 2  # Terrain/Ground
                            else:
                                self.object_type = 1  # Unknown static object treated as obstacle
                        else:
                            self.object_type = 1  # Generic obstacle
                    else:
                        # No userData - assume it's ground/terrain if it's static
                        if fixture.body.type == b2.b2_staticBody:
                            self.object_type = 2  # Static body = terrain
                        else:
                            self.object_type = 1  # Dynamic body = obstacle
                
                # Return fraction to clip ray to closest hit so far
                return fraction
        
        # Cast the ray
        callback = RayCallback(self, self.ray_sensor_range)
        try:
            self.world.RayCast(callback, start_pos, (end_x, end_y))
        except:
            # If ray casting fails, return max range and clear
            return self.ray_sensor_range, 0
        
        # Return results
        if callback.hit_distance < self.ray_sensor_range:
            return callback.hit_distance, callback.object_type
        else:
            return self.ray_sensor_range, 0  # No hit - clear path
    
    def _perform_ray_scan(self) -> List[Tuple[float, int]]:
        """Perform a complete ray scan and return results for all 5 rays."""
        if not self.body:
            # Return default values if no body
            return [(self.ray_sensor_range, 0)] * self.num_rays
        
        # Get robot position and orientation
        robot_pos = (self.body.position.x, self.body.position.y)
        robot_angle = self.body.angle
        
        results = []
        for ray_angle in self.ray_angles:
            # Convert relative ray angle to world angle
            world_angle = robot_angle + ray_angle
            distance, obj_type = self._cast_ray(robot_pos, world_angle)
            results.append((distance, obj_type))
            
            # Ray visualization removed for performance
        
        self.last_ray_results = results
        return results
    

    
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
        Get enhanced state representation for neural network training with ray sensing.
        
        Returns exactly 29 dimensions as documented in _calculate_state_size().
        This ensures consistent input to the neural network architecture.
        """
        try:
            state_array = np.zeros(29, dtype=np.float32)  # Fixed size for consistency
            
            # DEBUG: Minimal state generation confirmation (only once per agent)
            if not hasattr(self, '_state_gen_confirmed'):
                print(f"✅ Agent {str(self.id)[:8]}: Using 29D state representation with ray sensing")
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
            
            # 2. Body state (6 values) - NO ABSOLUTE COORDINATES
            if self.body:
                # Recent movement direction instead of absolute position
                displacement_x = self.body.position.x - self.prev_x if hasattr(self, 'prev_x') else 0.0
                displacement_y = self.body.position.y - getattr(self, 'prev_y', self.initial_position[1])
                state_array[4] = np.tanh(displacement_x * 100.0)  # Recent x movement
                state_array[5] = np.tanh(displacement_y * 100.0)  # Recent y movement
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
            
            # 6. Ray sensing data (10 values) - 5 rays × 2 values each
            ray_results = self._perform_ray_scan()
            for i, (distance, obj_type) in enumerate(ray_results):
                base_idx = 19 + (i * 2)  # Start at index 19, 2 values per ray
                if base_idx < 29:  # Safety check
                    # Normalize distance to [0, 1] range
                    state_array[base_idx] = min(distance / self.ray_sensor_range, 1.0)
                    # Object type as categorical value [0-2] normalized to [0, 1] 
                    # 0=clear, 1=obstacle, 2=terrain
                    state_array[base_idx + 1] = obj_type / 2.0
            
            # Ensure no NaN values
            state_array = np.nan_to_num(state_array, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return state_array
            
        except Exception as e:
            print(f"⚠️ Error getting expanded state for agent {self.id}: {e}")
            return np.zeros(29, dtype=np.float32)  # Return fixed size on error
    
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
        
        # Train every 38 steps (increased by 25% from 30 steps for better performance)
        if self.steps % 38 == 0:
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
        
        # ENHANCED: Much stronger forward progress reward + rightward movement incentive
        displacement = current_x - prev_x
        if displacement > 0.001:  # Lower threshold for sensitivity
            progress_reward = displacement * 5.0  # MUCH stronger multiplier (was 0.5)
            
            # 🎯 RIGHTWARD MOVEMENT BONUS: Small consistent reward for moving right
            rightward_bonus = displacement * 2.0  # Additional bonus for rightward movement
            progress_reward += rightward_bonus
            
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
            # 🚫 STRONG PENALTY FOR STANDING STILL: Much stronger to prevent exploitation
            progress_reward = -0.01 if abs(self.body.linearVelocity.x) < 0.01 else -0.005
        
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
        elif abs(self.body.linearVelocity.x) < 0.01:
            # 🚫 MUCH STRONGER PENALTY FOR STANDING STILL: Prevent exploitation
            total_reward -= 0.05  # INCREASED: from -0.02 to -0.05 (stronger penalty)
        
        # NEW: Additional inactivity penalty based on recent movement history
        if hasattr(self, 'recent_displacements') and len(self.recent_displacements) >= 5:
            recent_avg_movement = sum(abs(d) for d in self.recent_displacements[-5:]) / 5
            if recent_avg_movement < 0.001:  # Very little movement in last 5 steps
                total_reward -= 0.03  # INCREASED: from -0.015 to -0.03 (stronger sustained inactivity penalty)
        
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
        
        # 🌊 NEW: Update rolling reward system instead of just accumulating total_reward
        self.immediate_reward = reward
        self.last_reward = reward
        
        # Add to rolling windows
        self.recent_rewards.append(reward)
        self.recent_steps_window.append(self.steps)
        self.short_term_window.append(reward)
        self.medium_term_window.append(reward)
        self.long_term_window.append(reward)
        
        # 📊 Calculate rolling performance metrics
        self._update_performance_metrics()
        
        # 🎖️ Check for achievements (personal records)
        self._check_achievements()
        
        # 📈 Check for benchmark milestones
        self._check_benchmarks()
        
        # 🕒 Update time-based performance history
        self._update_performance_history()
        
        # Keep total_reward for backward compatibility, but it's deprecated
        self.total_reward += reward
        
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
        self.prev_y = self.body.position.y if self.body else 0.0
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
        self.prev_y = self.initial_position[1]
        
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
    
    # 🌊 INFINITE EXPLORATION: Rolling reward system helper methods
    def _update_performance_metrics(self):
        """Update rolling performance metrics from windowed rewards."""
        import numpy as np
        
        # IMPROVED: Apply decay factor for inactivity to make metrics more responsive
        current_velocity = abs(self.body.linearVelocity.x) if self.body else 0.0
        is_inactive = current_velocity < 0.01  # Robot is barely moving
        
        # Calculate rolling averages for different time scales
        if len(self.short_term_window) > 0:
            if is_inactive and len(self.short_term_window) >= 10:
                # When inactive, give more weight to recent rewards (decay older ones faster)
                weights = np.exp(-np.arange(len(self.short_term_window)) * 0.1)  # Exponential decay
                weights = weights[::-1]  # Reverse so recent rewards get higher weight
                weighted_rewards = np.array(self.short_term_window) * weights
                self.short_term_avg = float(np.sum(weighted_rewards) / np.sum(weights))
            else:
                # Normal calculation when active
                self.short_term_avg = float(np.mean(self.short_term_window))
        
        if len(self.medium_term_window) > 0:
            if is_inactive and len(self.medium_term_window) >= 50:
                # Apply stronger decay for medium-term when inactive
                weights = np.exp(-np.arange(len(self.medium_term_window)) * 0.05)  
                weights = weights[::-1]
                weighted_rewards = np.array(self.medium_term_window) * weights
                self.medium_term_avg = float(np.sum(weighted_rewards) / np.sum(weights))
            else:
                self.medium_term_avg = float(np.mean(self.medium_term_window))
            
        if len(self.long_term_window) > 0:
            self.long_term_avg = float(np.mean(self.long_term_window))
        
        # Calculate current reward rate (recent performance) - more sensitive to inactivity
        if len(self.recent_rewards) >= 10:  # Need at least 10 samples
            recent_10 = list(self.recent_rewards)[-10:]
            if is_inactive:
                # When inactive, weight the most recent rewards much more heavily
                weights = np.exp(np.arange(10) * 0.2)  # Strong exponential weighting toward recent
                weighted_recent = np.array(recent_10) * weights
                self.reward_rate = float(np.sum(weighted_recent) / np.sum(weights))
            else:
                self.reward_rate = float(np.mean(recent_10))
        
        # Calculate performance trend (improvement over time)
        if len(self.medium_term_window) >= 100:
            first_half = list(self.medium_term_window)[:len(self.medium_term_window)//2]
            second_half = list(self.medium_term_window)[len(self.medium_term_window)//2:]
            
            first_half_avg = np.mean(first_half)
            second_half_avg = np.mean(second_half)
            self.performance_trend = float(second_half_avg - first_half_avg)
    
    def _check_achievements(self):
        """Check for personal record achievements to maintain motivation."""
        # Check short-term achievement
        if self.short_term_avg > self.best_short_term_avg:
            self.best_short_term_avg = self.short_term_avg
            self.achievement_count += 1
            if self.steps % 1000 == 0:  # Log occasionally to avoid spam
                print(f"🎖️ Agent {str(self.id)[:8]}: New short-term record! Avg reward = {self.short_term_avg:.4f}")
        
        # Check medium-term achievement  
        if self.medium_term_avg > self.best_medium_term_avg:
            self.best_medium_term_avg = self.medium_term_avg
            self.achievement_count += 1
            if self.steps % 2000 == 0:  # Log occasionally
                print(f"🏆 Agent {str(self.id)[:8]}: New medium-term record! Avg reward = {self.medium_term_avg:.4f}")
        
        # Check long-term achievement
        if self.long_term_avg > self.best_long_term_avg:
            self.best_long_term_avg = self.long_term_avg
            self.achievement_count += 1
            if self.steps % 5000 == 0:  # Log occasionally
                print(f"🌟 Agent {str(self.id)[:8]}: New long-term record! Avg reward = {self.long_term_avg:.4f}")
    
    def _check_benchmarks(self):
        """Check if we've reached a benchmark milestone for evaluation."""
        if self.steps >= self.next_benchmark_step:
            # Record performance at this benchmark
            self.benchmark_scores[self.next_benchmark_step] = {
                'short_term_avg': self.short_term_avg,
                'medium_term_avg': self.medium_term_avg,
                'long_term_avg': self.long_term_avg,
                'reward_rate': self.reward_rate,
                'performance_trend': self.performance_trend,
                'achievement_count': self.achievement_count,
                'timestamp': time.time()
            }
            
            # Find next benchmark
            for milestone in self.benchmark_intervals:
                if milestone > self.steps:
                    self.next_benchmark_step = milestone
                    break
            else:
                # If we've passed all predefined benchmarks, add more
                self.next_benchmark_step = self.steps + 5000
    
    def _update_performance_history(self):
        """Update time-based performance history."""
        current_time = time.time()
        if current_time - self.last_performance_update >= self.performance_update_interval:
            # Record current performance
            self.performance_history.append((
                current_time,
                self.reward_rate,
                self.short_term_avg,
                self.medium_term_avg,
                self.performance_trend
            ))
            
            # Keep only last 100 entries (about 50 minutes of history)
            if len(self.performance_history) > 100:
                self.performance_history.pop(0)
            
            self.last_performance_update = current_time
    
    # 🎯 NEW: Methods to get meaningful performance metrics for infinite exploration
    def get_current_performance(self):
        """Get current performance metrics instead of total_reward."""
        return {
            'reward_rate': self.reward_rate,
            'short_term_avg': self.short_term_avg,
            'medium_term_avg': self.medium_term_avg, 
            'long_term_avg': self.long_term_avg,
            'performance_trend': self.performance_trend,
            'achievement_count': self.achievement_count,
            'steps': self.steps,
            'benchmarks_reached': len(self.benchmark_scores)
        }
    
    def get_fitness_score(self):
        """Get a comparable fitness score for infinite exploration."""
        # Combine multiple metrics for a comprehensive fitness score
        base_score = self.medium_term_avg * 100  # Base performance
        trend_bonus = max(0, self.performance_trend * 50)  # Improvement bonus
        achievement_bonus = self.achievement_count * 5  # Achievement bonus
        consistency_bonus = max(0, 10 - abs(self.short_term_avg - self.medium_term_avg) * 100)  # Consistency
        
        return base_score + trend_bonus + achievement_bonus + consistency_bonus 