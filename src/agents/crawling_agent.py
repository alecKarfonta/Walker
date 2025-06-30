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
        
        # Action system - MUST be set before calling _calculate_state_size()
        self.action_size = 9  # 3x3 grid for shoulder/elbow control
        self.actions = self._generate_action_combinations()
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
        
        # Motor parameters
        self.motor_speed = 8.0
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
        """Initialize the neural network learning system."""
        try:
            # Import here to avoid circular imports
            from .attention_deep_q_learning import AttentionDeepQLearning
            
            # Calculate state size
            state_size = self._calculate_state_size()
            
            self._learning_system = AttentionDeepQLearning(
                state_dim=state_size,
                action_dim=self.action_size,
                learning_rate=0.001
            )
                
        except Exception as e:
            print(f"⚠️ Error initializing learning system for agent {self.id}: {e}")
            self._learning_system = None
    
    def _calculate_state_size(self) -> int:
        """Calculate the state size based on robot morphology."""
        # Dynamic size: 2 values per joint + 3 metadata values
        total_joints = self.num_limbs * self.segments_per_limb
        return total_joints + 3
    
    def _generate_action_combinations(self) -> List[Tuple[float, float]]:
        """Generate all possible action combinations for shoulder/elbow control."""
        combinations = []
        for shoulder in [-1, 0, 1]:  # Backward, stop, forward
            for elbow in [-1, 0, 1]:  # Contract, stop, extend
                combinations.append((shoulder, elbow))
        return combinations
    
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
        """Get current state as numpy array for neural network."""
        try:
            state_size = self._calculate_state_size()
            state_array = np.zeros(state_size, dtype=np.float32)
            
            # For simple robot: shoulder and elbow angles
            if self.upper_arm_joint and self.lower_arm_joint:
                shoulder_angle = np.tanh(self.upper_arm_joint.angle)
                elbow_angle = np.tanh(self.lower_arm_joint.angle)
                
                state_array[0] = shoulder_angle
                state_array[1] = elbow_angle
            
            # Physical state metadata (last 3 elements)
            velocity = np.tanh(self.body.linearVelocity.x / 5.0) if self.body else 0.0
            stability = np.tanh(self.body.angle * 2.0) if self.body else 0.0
            progress = np.tanh((self.body.position.x - self.prev_x) * 10.0) if self.body else 0.0
            
            state_array[-3] = velocity
            state_array[-2] = stability
            state_array[-1] = progress
            
            return state_array
            
        except Exception as e:
            print(f"⚠️ Error getting state for agent {self.id}: {e}")
            return np.zeros(self._calculate_state_size(), dtype=np.float32)
    
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
            
            training_stats = self._learning_system.learn()
            
            # Log training activity more frequently (every 50 steps instead of 100)
            if self.steps % 50 == 0 and training_stats:
                print(f"🧠 Agent {str(self.id)[:8]}: Training step {self.steps}, "
                        f"Loss: {training_stats.get('loss', 0.0):.4f}, "
                        f"Q-val: {training_stats.get('mean_q_value', 0.0):.3f}")
            
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
        
        # Forward progress reward (FIXED: Much lower scaling)
        displacement = current_x - prev_x
        if displacement > 0.001:  # Lower threshold
            progress_reward = displacement * 0.5  # Much lower multiplier (was 8.0)
            
            # Reduced sustained movement bonus (FIXED: Lower bonus)
            if hasattr(self, 'recent_displacements'):
                self.recent_displacements.append(displacement)
                if len(self.recent_displacements) > 10:
                    self.recent_displacements.pop(0)
                if len(self.recent_displacements) >= 5:
                    avg_displacement = sum(self.recent_displacements) / len(self.recent_displacements)
                    if avg_displacement > 0.003:  # Higher threshold
                        progress_reward *= 1.1  # Much smaller bonus (was 1.5)
            else:
                self.recent_displacements = [displacement]
        elif displacement < -0.0005:
            progress_reward = displacement * 0.5  # Lower penalty multiplier
        else:
            progress_reward = 0.0
        
        total_reward += progress_reward
        
        # Food-seeking reward (NEW: Reward moving toward closest edible food)
        if food_info and food_info.get('distance', 999999) < 100:  # Only if food is reasonably close
            current_food_distance = food_info['distance']
            
            # Initialize previous food distance if not exists
            if not hasattr(self, 'prev_food_distance'):
                self.prev_food_distance = current_food_distance
            
            # Reward getting closer to food
            food_distance_change = self.prev_food_distance - current_food_distance
            if food_distance_change > 0.1:  # Getting closer to food
                food_seeking_reward = food_distance_change * 0.3  # Moderate reward for approaching food
                total_reward += food_seeking_reward
                
                # Extra bonus if very close to food
                if current_food_distance < 5.0:
                    total_reward += 0.02  # Bonus for being near food
            elif food_distance_change < -0.1:  # Moving away from food
                food_seeking_penalty = food_distance_change * 0.1  # Small penalty for moving away
                total_reward += food_seeking_penalty
            
            # Update previous food distance
            self.prev_food_distance = current_food_distance
        
        # Stability reward (FIXED: Much smaller values)
        body_angle = abs(self.body.angle)
        if body_angle < 0.2:
            total_reward += 0.01  # Was 0.08
        elif body_angle > 1.5:
            total_reward -= 0.005  # Was 0.04
        
        # Much tighter reward clipping (FIXED: Smaller range)
        total_reward = np.clip(total_reward, -0.1, 0.1)
        
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