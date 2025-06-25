"""
Evolutionary Crawling Agent with Physical Parameter Evolution.

This agent combines Q-learning with evolutionary physical parameters,
creating robots with diverse body characteristics that can evolve over time.
"""

import numpy as np
import Box2D as b2
from typing import Tuple, Dict, Any, List, Optional, Union
from copy import deepcopy
import random
import uuid
import time

from .base_agent import BaseAgent
from .crawling_crate_agent import CrawlingCrateAgent
from .physical_parameters import PhysicalParameters


class EvolutionaryCrawlingAgent(CrawlingCrateAgent):
    """
    Enhanced CrawlingCrate agent with evolvable physical parameters.
    Each agent has unique body characteristics that can be evolved.
    """
    
    def __init__(self, 
                 world: b2.b2World, 
                 agent_id: Optional[int] = None, 
                 position: Tuple[float, float] = (10, 20),
                 category_bits: int = 0x0002,  # AGENT_CATEGORY 
                 mask_bits: int = 0x0005,      # GROUND_CATEGORY | OBSTACLE_CATEGORY
                 physical_params: Optional[PhysicalParameters] = None,
                 parent_lineage: Optional[List[str]] = None):
        """
        Initialize evolutionary crawling agent.
        
        Args:
            world: Box2D world
            agent_id: Unique agent identifier
            position: Starting position
            category_bits: Collision category bits
            mask_bits: Collision mask bits
            physical_params: Physical parameters (random if None)
            parent_lineage: List of parent agent IDs for tracking evolution
        """
        # Initialize physical parameters first
        if physical_params is None:
            self.physical_params = PhysicalParameters.random_parameters()
        else:
            self.physical_params = physical_params.validate_and_repair()
        
        # Generate UUID if agent_id is None
        if agent_id is None:
            self.id = str(uuid.uuid4())[:8]  # Use short UUID
        else:
            self.id = str(agent_id)  # Ensure ID is always string
        
        # Store lineage information (convert any integer IDs to strings)
        if parent_lineage is None:
            self.parent_lineage = []
        else:
            self.parent_lineage = [str(pid) for pid in parent_lineage]
        self.generation = len(self.parent_lineage)
        
        # Store parameters for body creation
        self.world = world
        self.initial_position = position
        self.category_bits = category_bits
        self.mask_bits = mask_bits
        
        # Initialize base agent
        BaseAgent.__init__(self)
        
        # Physics properties for collision filtering
        self.filter = b2.b2Filter(
            categoryBits=category_bits,
            maskBits=mask_bits
        )
        
        # Apply physical parameters to motor and learning settings
        self.motor_torque = self.physical_params.motor_torque
        self.motor_speed = self.physical_params.motor_speed
        
        # Create body parts using physical parameters
        self._create_evolutionary_body()
        self._create_evolutionary_arms()
        self._create_evolutionary_wheels()
        self._create_evolutionary_joints()
        
        # Initialize Q-learning with physical parameters
        self._initialize_learning_from_params()
        
        # Evolution tracking
        self.mutation_count = 0
        self.crossover_count = 0
        self.fitness_history = []
        self.diversity_score = 0.0
        
        # Destruction tracking to prevent core dumps
        self._destroyed = False
        
        # Initialize crawling-specific tracking
        self.recent_displacements = []
        self.action_sequence = []
        
        # ACTION PERSISTENCE: Time-based action selection (0.5 seconds) - INHERITED
        self.action_persistence_duration = 0.25  # 0.25 seconds
        self.last_action_time = time.time()  # Track when last action was selected
        self.action_persisted = False  # Track if we're in persistence mode
        
        # Learning approach inheritance for evolution (can be None or LearningApproach)
        self._inherited_learning_approach = None  # Type: Optional[LearningApproach]
    
    def _create_evolutionary_body(self):
        """Create body using evolved physical parameters."""
        body_def = b2.b2BodyDef(
            type=b2.b2_dynamicBody,
            position=(float(self.initial_position[0]), float(self.initial_position[1])),
            linearDamping=self.physical_params.body_linear_damping,
            angularDamping=self.physical_params.body_angular_damping
        )
        self.body = self.world.CreateBody(body_def)
        
        # Create chassis with evolved dimensions
        chassis_fixture = self.body.CreateFixture(
            shape=b2.b2PolygonShape(box=(
                self.physical_params.body_width,
                self.physical_params.body_height
            )),
            density=self.physical_params.body_density,
            friction=self.physical_params.body_friction,
            restitution=self.physical_params.body_restitution,
            filter=b2.b2Filter(categoryBits=self.category_bits, maskBits=self.mask_bits)
        )
    
    def _create_evolutionary_arms(self):
        """Create arms using evolved physical parameters."""
        # Upper Arm with evolved dimensions
        upper_arm_pos = (
            self.body.position[0] - 1.0,
            self.body.position[1] + 1.0
        )
        
        upper_arm = self.world.CreateDynamicBody(
            position=upper_arm_pos,
            fixtures=[
                b2.b2FixtureDef(
                    shape=b2.b2PolygonShape(box=(
                        self.physical_params.arm_length,
                        self.physical_params.arm_width
                    )),
                    density=self.physical_params.arm_density,
                    friction=self.physical_params.arm_friction,
                    restitution=self.physical_params.arm_restitution,
                    filter=b2.b2Filter(categoryBits=self.category_bits, maskBits=self.mask_bits)
                )
            ]
        )
        
        # Lower Arm (wrist) with evolved dimensions
        # Create a tapered polygon for the lower arm
        wrist_vertices = self._create_tapered_arm_shape(
            self.physical_params.wrist_length,
            self.physical_params.wrist_width
        )
        
        lower_arm = self.world.CreateDynamicBody(
            position=(upper_arm.position[0] + self.physical_params.arm_length, 
                     upper_arm.position[1]),
            fixtures=[
                b2.b2FixtureDef(
                    shape=b2.b2PolygonShape(vertices=wrist_vertices),
                    density=self.physical_params.arm_density,
                    friction=self.physical_params.arm_friction,
                    restitution=self.physical_params.arm_restitution,
                    filter=b2.b2Filter(categoryBits=self.category_bits, maskBits=self.mask_bits)
                )
            ]
        )
        
        self.upper_arm, self.lower_arm = upper_arm, lower_arm
    
    def _create_tapered_arm_shape(self, length: float, width: float) -> List[Tuple[float, float]]:
        """Create a tapered arm shape that comes to a point."""
        half_width = width / 2
        return [
            (-length, -half_width),  # Bottom left (wide end)
            (-length, half_width),   # Top left (wide end)
            (length * 0.5, half_width * 0.5),  # Top middle (narrowing)
            (length, 0.0),           # Point at the tip
            (length * 0.5, -half_width * 0.5), # Bottom middle (narrowing)
        ]
    
    def _create_evolutionary_wheels(self):
        """Create wheels using evolved physical parameters."""
        self.wheels = []
        
        # Calculate wheel positions based on evolved leg spread and suspension
        leg_spread = self.physical_params.leg_spread
        suspension = self.physical_params.suspension
        
        wheel_positions = [(-leg_spread/2, -suspension), (leg_spread/2, -suspension)]
        
        for wheel_pos in wheel_positions:
            wheel = self.world.CreateDynamicBody(
                position=self.body.GetWorldPoint(wheel_pos),
                fixtures=[
                    b2.b2FixtureDef(
                        shape=b2.b2CircleShape(radius=self.physical_params.wheel_radius),
                        density=self.physical_params.wheel_density,
                        friction=self.physical_params.wheel_friction,
                        restitution=self.physical_params.wheel_restitution,
                        filter=b2.b2Filter(categoryBits=self.category_bits, maskBits=self.mask_bits)
                    )
                ]
            )
            self.wheels.append(wheel)
    
    def _create_evolutionary_joints(self):
        """Create joints using evolved physical parameters."""
        # Shoulder joint with evolved parameters
        self.upper_arm_joint = self.world.CreateRevoluteJoint(
            bodyA=self.body,
            bodyB=self.upper_arm,
            localAnchorA=(-1.0, 1.0),
            localAnchorB=(-self.physical_params.arm_length, 0),
            enableMotor=True,
            maxMotorTorque=self.physical_params.arm_torque,
            motorSpeed=0,
            enableLimit=True,
            lowerAngle=self.physical_params.shoulder_lower_limit,
            upperAngle=self.physical_params.shoulder_upper_limit,
        )
        
        # Elbow joint with evolved parameters
        self.lower_arm_joint = self.world.CreateRevoluteJoint(
            bodyA=self.upper_arm,
            bodyB=self.lower_arm,
            localAnchorA=(self.physical_params.arm_length, 0),
            localAnchorB=(-self.physical_params.wrist_length, 0),
            enableMotor=True,
            maxMotorTorque=self.physical_params.wrist_torque,
            motorSpeed=0,
            enableLimit=True,
            lowerAngle=self.physical_params.elbow_lower_limit,
            upperAngle=self.physical_params.elbow_upper_limit,
        )
        
        # Wheel joints using evolved leg spread and suspension
        self.wheel_joints = []
        leg_spread = self.physical_params.leg_spread
        suspension = self.physical_params.suspension
        wheel_positions = [(-leg_spread/2, -suspension), (leg_spread/2, -suspension)]
        
        for i, wheel_pos in enumerate(wheel_positions):
            joint = self.world.CreateRevoluteJoint(
                bodyA=self.body,
                bodyB=self.wheels[i],
                localAnchorA=wheel_pos,
                localAnchorB=(0, 0),
                enableMotor=False,
            )
            self.wheel_joints.append(joint)
    
    def _initialize_learning_from_params(self):
        """Initialize Q-learning parameters from physical parameters."""
        # Copy learning parameters from physical params
        self.learning_rate = self.physical_params.learning_rate
        self.min_learning_rate = self.physical_params.min_learning_rate
        self.max_learning_rate = self.physical_params.max_learning_rate
        self.epsilon = self.physical_params.epsilon
        self.min_epsilon = self.physical_params.min_epsilon
        self.max_epsilon = self.physical_params.max_epsilon
        self.discount_factor = self.physical_params.discount_factor
        self.impatience = self.physical_params.impatience
        self.epsilon_decay = 0.9999
        
        # Copy reward weights
        self.speed_value_weight = self.physical_params.speed_value_weight
        self.acceleration_value_weight = self.physical_params.acceleration_value_weight
        self.position_weight = self.physical_params.position_weight
        self.stability_weight = self.physical_params.stability_weight
        
        # Copy motor parameters
        self.motor_torque = self.physical_params.motor_torque
        self.motor_speed = self.physical_params.motor_speed
        self.action_interval = self.physical_params.action_interval
        self.learning_interval = self.physical_params.learning_interval
        
        # Initialize actions
        self.actions = [
            (1, 0), (0, 1), (1, 1),
            (-1, 0), (0, -1), (-1, -1)
        ]
        
        self.state_size = 3  # shoulder_bin, elbow_bin, vel_x_bin
        self.action_size = len(self.actions)
        
        # Initialize Q-learning components
        from .q_table import EnhancedQTable
        self.q_table = EnhancedQTable(
            action_count=len(self.actions),
            default_value=0.0,
            confidence_threshold=15,
            exploration_bonus=self.physical_params.exploration_bonus
        )
        
        # Q-table size management
        self.max_q_table_states = 1500
        self.q_table_pruning_threshold = 1800
        
        # Initialize tracking variables
        self.total_reward = 0.0
        self.steps = 0
        self.action_history = []
        self.max_action_history = 15
        self.current_state = None
        self.current_action = None
        self.current_action_tuple = (1, 0)
        
        # Reward clipping
        self.reward_clip_min = -0.1
        self.reward_clip_max = 0.1
        
        # Position tracking
        self.last_x_position = self.body.position.x
        self.last_update_step = 0
        self.reward_count = 0
        self.prev_x = self.initial_position[0]
        
        # Q-value bounds
        self.min_q_value = -5.0
        self.max_q_value = 5.0
        
        # Performance tracking
        self.best_reward_received = -np.inf
        self.worst_reward_received = np.inf
        
        # Experience replay (optimized for performance)
        from .crawling_crate_agent import Experience, ReplayBuffer
        self.replay_buffer = ReplayBuffer(capacity=2500)  # Reduced for performance
        self.batch_size = 28  # Smaller batch size
        self.replay_frequency = 15  # Less frequent for performance
        
        # Java-inspired tracking
        self.best_value = 0.0
        self.worst_value = 0.0
        self.old_value = 0.0
        self.new_value = 0.0
        self.time_since_good_value = 0.0
        
        # Speed tracking
        self.speed = 0.0
        self.speed_decay = 0.85
        self.previous_speed = 0.0
        self.acceleration = 0.0
        self.max_speed = 0.0
        
        # Multi-goal system (from physical params or defaults)
        self.current_goal = 0
        self.goals = ['speed', 'distance', 'stability', 'efficiency', 'combined']
        self.goal_weights = {
            'speed': self.physical_params.speed_value_weight,
            'distance': 0.02,
            'stability': self.physical_params.stability_weight,
            'efficiency': self.physical_params.acceleration_value_weight,
            'combined': 0.04
        }
        self.goal_switch_interval = 500
        
        # Performance adaptation
        from collections import deque
        self.performance_window = 200
        self.recent_rewards = deque(maxlen=self.performance_window)
        self.performance_threshold = 0.02
        
        # Enhanced state discretization (MISSING ATTRIBUTES!)
        self.use_enhanced_state = True
        self.enhanced_state_size = 3  # shoulder, elbow, velocity
        
    def get_evolutionary_reward(self, prev_x: float) -> float:
        """
        Enhanced reward function that considers evolved physical parameters.
        """
        base_reward = self.get_crawling_reward(prev_x)
        
        # Apply evolved reward weights
        weighted_reward = (
            base_reward * self.physical_params.speed_value_weight +
            self._get_stability_bonus() * self.physical_params.stability_weight +
            self._get_efficiency_bonus() * self.physical_params.acceleration_value_weight
        )
        
        return weighted_reward
    
    def _get_stability_bonus(self) -> float:
        """Calculate stability bonus based on body angle."""
        body_angle = abs(self.body.angle)
        if body_angle < np.pi/6:  # Within 30 degrees
            return 0.01 * (1.0 - (body_angle / (np.pi/6)))
        else:
            return -0.01 * (body_angle - np.pi/6)
    
    def _get_efficiency_bonus(self) -> float:
        """Calculate efficiency bonus based on movement vs energy."""
        current_x = self.body.position.x
        if hasattr(self, 'prev_x'):
            displacement = current_x - self.prev_x
            energy_used = abs(self.current_action_tuple[0]) + abs(self.current_action_tuple[1])
            
            if displacement > 0.001 and energy_used > 0:
                efficiency = displacement / (energy_used + 0.1)
                return min(0.005, efficiency * 0.02)
        
        return 0.0
    
    def evolve_with(self, other: 'EvolutionaryCrawlingAgent', 
                   mutation_rate: float = 0.1) -> 'EvolutionaryCrawlingAgent':
        """
        Create offspring through crossover and mutation with another agent.
        
        Args:
            other: Other parent agent
            mutation_rate: Mutation rate for offspring
            
        Returns:
            New evolutionary agent offspring
        """
        # Crossover physical parameters
        if random.random() < 0.7:  # 70% chance of crossover
            child_params = self.physical_params.crossover(other.physical_params)
            child_params = child_params.mutate(mutation_rate)
        else:
            # Direct mutation without crossover
            child_params = self.physical_params.mutate(mutation_rate * 1.5)
        
        # Create lineage tracking
        child_lineage = self.parent_lineage + [self.id]
        
        # Create new agent with evolved parameters
        child = EvolutionaryCrawlingAgent(
            world=self.world,
            agent_id=None,  # Generate new UUID automatically
            position=self.initial_position,
            category_bits=self.category_bits,
            mask_bits=self.mask_bits,
            physical_params=child_params,
            parent_lineage=child_lineage
        )
        
        # COMPREHENSIVE LEARNING TRANSFER: Preserve ALL learning weights for each approach
        try:
            # Determine primary parent (more learning data) for approach inheritance
            if len(self.q_table.q_values) >= len(other.q_table.q_values):
                primary_parent = self
                secondary_parent = other
            else:
                primary_parent = other
                secondary_parent = self
            
            # Inherit learning approach from primary parent
            child._inherited_learning_approach = getattr(primary_parent, '_inherited_learning_approach', None)
            
            # Copy Q-table from primary parent (preserves all learning weights)
            child.q_table = primary_parent.q_table.copy()
            
            # Cross-train with secondary parent if both have significant learning
            if (len(primary_parent.q_table.q_values) > 10 and 
                len(secondary_parent.q_table.q_values) > 10):
                
                if hasattr(child.q_table, 'learn_from_other_table'):
                    child.q_table.learn_from_other_table(secondary_parent.q_table, learning_rate=0.3)
                    print(f"🧬 Crossover learning: {primary_parent.id[:6]} + {secondary_parent.id[:6]} → {child.id[:6]}")
                
                # Store both parents' learning data for comprehensive transfer
                setattr(child, '_parent_qtables', {
                    'primary': primary_parent.q_table,
                    'secondary': secondary_parent.q_table,
                    'primary_approach': getattr(primary_parent, '_inherited_learning_approach', None),
                    'secondary_approach': getattr(secondary_parent, '_inherited_learning_approach', None)
                })
            else:
                # Store single parent data
                setattr(child, '_parent_qtables', {
                    'primary': primary_parent.q_table,
                    'primary_approach': getattr(primary_parent, '_inherited_learning_approach', None)
                })
                
        except Exception as e:
            print(f"⚠️ Error in comprehensive learning transfer during crossover: {e}")
            # Fallback to basic transfer
            child.q_table = self.q_table.copy()
            child._inherited_learning_approach = getattr(self, '_inherited_learning_approach', None)
        
        child.crossover_count = 1
        return child
    
    def clone_with_mutation(self, mutation_rate: float = 0.1) -> 'EvolutionaryCrawlingAgent':
        """
        Create a mutated clone of this agent.
        
        Args:
            mutation_rate: Mutation rate for clone
            
        Returns:
            New mutated agent
        """
        # Mutate physical parameters
        mutated_params = self.physical_params.mutate(mutation_rate)
        
        # Create lineage tracking
        child_lineage = self.parent_lineage + [self.id]
        
        # Create cloned agent
        clone = EvolutionaryCrawlingAgent(
            world=self.world,
            agent_id=None,  # Generate new UUID automatically
            position=self.initial_position,
            category_bits=self.category_bits,
            mask_bits=self.mask_bits,
            physical_params=mutated_params,
            parent_lineage=child_lineage
        )
        
        # COMPREHENSIVE LEARNING TRANSFER: Copy Q-table and learning approach inheritance
        clone.q_table = self.q_table.copy()
        clone._inherited_learning_approach = getattr(self, '_inherited_learning_approach', None)
        
        # Store parent learning data for comprehensive transfer
        setattr(clone, '_parent_qtables', {
            'primary': self.q_table,
            'primary_approach': getattr(self, '_inherited_learning_approach', None)
        })
        
        clone.mutation_count = self.mutation_count + 1
        
        return clone
    
    def get_evolutionary_fitness(self) -> float:
        """
        Calculate comprehensive fitness including physical efficiency.
        
        Returns:
            Fitness score considering multiple factors
        """
        base_fitness = self.total_reward
        
        # Efficiency bonus: reward for achieving good results with reasonable physical parameters
        efficiency_score = 0.0
        
        # Penalize extreme parameter values that might be unrealistic
        param_penalty = 0.0
        if self.physical_params.body_width > 2.5 or self.physical_params.body_height > 1.2:
            param_penalty += 0.1  # Penalty for extremely large bodies
        if self.physical_params.motor_torque > 250:
            param_penalty += 0.05  # Penalty for excessive motor power
        
        # Bonus for moderate, efficient designs
        if (0.8 < self.physical_params.body_width < 2.0 and
            100 < self.physical_params.motor_torque < 200):
            efficiency_score += 0.1
        
        return base_fitness + efficiency_score - param_penalty
    
    def get_diversity_metrics(self) -> Dict[str, float]:
        """Get diversity metrics for this agent."""
        return self.physical_params.get_diversity_metrics()
    
    def get_evolutionary_debug_info(self) -> Dict[str, Any]:
        """Get comprehensive debug information including evolution data."""
        base_info = self.get_advanced_debug_info()
        
        evolutionary_info = {
            'generation': self.generation,
            'parent_lineage': self.parent_lineage,
            'mutation_count': self.mutation_count,
            'crossover_count': self.crossover_count,
            'diversity_score': self.diversity_score,
            'evolutionary_fitness': self.get_evolutionary_fitness(),
            
            # Physical parameter summary
            'body_dimensions': f"{self.physical_params.body_width:.2f}x{self.physical_params.body_height:.2f}",
            'arm_lengths': f"{self.physical_params.arm_length:.2f}/{self.physical_params.wrist_length:.2f}",
            'wheel_config': f"r={self.physical_params.wheel_radius:.2f}, spread={self.physical_params.leg_spread:.2f}",
            'motor_config': f"torque={self.physical_params.motor_torque:.0f}, speed={self.physical_params.motor_speed:.1f}",
            
            # Learning parameter summary
            'learning_config': f"lr={self.physical_params.learning_rate:.3f}, ε={self.physical_params.epsilon:.3f}",
            'reward_weights': {
                'speed': self.physical_params.speed_value_weight,
                'stability': self.physical_params.stability_weight,
                'acceleration': self.physical_params.acceleration_value_weight,
            }
        }
        
        base_info.update(evolutionary_info)
        return base_info
    
    def reset_with_new_position(self, position: Tuple[float, float]):
        """Reset agent to a new position while preserving learned behavior."""
        self.initial_position = position
        self.reset_position()
        
        # Reset action persistence timing
        self.last_action_time = time.time()
        self.action_persisted = False
        
    def destroy(self):
        """Clean up physics bodies safely."""
        try:
            # Mark as destroyed to prevent further use
            self._destroyed = True
            
            # Disable motors first to prevent issues during destruction
            if hasattr(self, 'upper_arm_joint') and self.upper_arm_joint:
                try:
                    self.upper_arm_joint.enableMotor = False
                except:
                    pass
            if hasattr(self, 'lower_arm_joint') and self.lower_arm_joint:
                try:
                    self.lower_arm_joint.enableMotor = False
                except:
                    pass
            
            # Destroy bodies in order (Box2D will handle joints automatically)
            if hasattr(self, 'wheels'):
                for wheel in self.wheels:
                    if wheel:
                        try:
                            self.world.DestroyBody(wheel)
                        except:
                            pass
            
            if hasattr(self, 'lower_arm') and self.lower_arm:
                try:
                    self.world.DestroyBody(self.lower_arm)
                except:
                    pass
            
            if hasattr(self, 'upper_arm') and self.upper_arm:
                try:
                    self.world.DestroyBody(self.upper_arm)
                except:
                    pass
            
            if hasattr(self, 'body') and self.body:
                try:
                    self.world.DestroyBody(self.body)
                except:
                    pass
                
        except Exception as e:
            print(f"⚠️  Error in destroy() for agent {getattr(self, 'id', 'unknown')}: {e}")
            self._destroyed = True 