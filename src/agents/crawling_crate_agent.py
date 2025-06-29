"""
Physics-based CrawlingCrate agent using attention-based deep Q-learning.
Clean, modern architecture focused on neural network learning.
"""

import numpy as np
from typing import Tuple, Dict, Any, List, NamedTuple, Optional
from collections import deque
import random
import time
import Box2D as b2
from .crawling_crate import CrawlingCrate
from .base_agent import BaseAgent


class CrawlingCrateAgent(CrawlingCrate, BaseAgent):
    """
    CrawlingCrate with attention-based deep Q-learning.
    Modern neural network architecture for optimal learning performance.
    """
    
    def __init__(self, world, agent_id, position: Tuple[float, float] = (10, 20), 
                 category_bits=0x0001, mask_bits=0xFFFF, learning_approach: str = "attention_deep_q_learning"):
        
        # Initialize base classes
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
        self.motor_speed = 3.0
        self.category_bits = category_bits
        self.mask_bits = mask_bits

        # Create body parts using our own methods
        self._create_body()
        self._create_arms()
        self._create_wheels()
        self._create_joints()

        # FORCE attention-based learning - no other options
        self.learning_approach = "attention_deep_q_learning"
        # DON'T initialize learning during __init__ - will be assigned by Learning Manager later
        self._learning_system = None
        
        # Action space definition
        self.actions = [
            (1, 0), (0, 1), (1, 1),
            (-1, 0), (0, -1), (-1, -1)
        ]
        
        # Dynamic state size: 2 joints + 3 metadata for basic robots
        self.state_size = 5  # For attention networks: 2 joint angles + 3 metadata (velocity, stability, progress)
        self.action_size = len(self.actions)
        
        # Basic tracking
        self.total_reward = 0.0
        self.steps = 0
        self.immediate_reward = 0.0
        self.last_reward = 0.0
        
        # Position tracking
        self.prev_x = position[0]
        self.last_x_position = self.body.position.x
        
        # Timing and action persistence - FAST for responsive control
        self.last_action_time = time.time()
        self.action_persistence_duration = 0.1  # Reduced from 0.25s to 0.1s (6 frames at 60 FPS)
        self.current_action_tuple = (1, 0)
        self.current_action = 0
        self.current_state = None
        
        # Performance tracking
        self.recent_displacements = []
        self.action_sequence = []
        self.action_history = []
        self.max_action_history = 100
        
        # Food approach tracking
        self.prev_food_distance = float('inf')
        self.food_distance_history = []

    def _initialize_attention_learning(self):
        """Initialize attention-based deep Q-learning system via Learning Manager."""
        # CRITICAL: Never create networks directly - ONLY get from Learning Manager
        # This prevents the constant GPU network recreation that's killing performance
        from .learning_manager import LearningManager
        
        # Try multiple ways to get learning manager instance
        learning_manager = None
        if hasattr(self, 'world') and hasattr(self.world, '_training_env'):
            learning_manager = getattr(self.world._training_env, 'learning_manager', None)
        
        if learning_manager:
            # Use Learning Manager's pooling system - this is the ONLY way to get networks
            self._learning_system = learning_manager._acquire_attention_network(self.id, self.action_size, self.state_size)
            if self._learning_system:
                print(f"🧠 Agent {self.id}: Got attention network from Learning Manager pool")
                return
            else:
                print(f"❌ Agent {self.id}: Learning Manager failed to provide network")
        else:
            print(f"❌ Agent {self.id}: No Learning Manager available - agent will have no learning")
        
        # CRITICAL: NO FALLBACK NETWORK CREATION - this was the performance killer
        # If Learning Manager can't provide a network, agent just won't learn
        # This is better than constant GPU thrashing
        self._learning_system = None
        print(f"⚠️ Agent {self.id}: No learning system - will use random actions")


    @property
    def q_table(self):
        """Backward compatibility: Return learning system."""
        if self._learning_system:
            return LearningSystemWrapper(self._learning_system, self.learning_approach)
        return None
    
    @q_table.setter
    def q_table(self, value):
        """Backward compatibility: Set learning system."""
        if isinstance(value, LearningSystemWrapper):
            self._learning_system = value._learning_system
        else:
            self._learning_system = value

    def get_state_representation(self):
        """
        Get state representation for attention-based neural networks.
        Uses dynamic size based on robot morphology (2 joints + 3 metadata for basic robots).
        Always returns numpy array for neural network input.
        """
        # DYNAMIC STATE SIZE: 2 joints + 3 metadata = 5 elements for basic robots
        joint_count = 2  # Basic robots have 2 joints (shoulder, elbow)
        state_size = joint_count + 3
        
        # Initialize state array with exact size needed
        state_array = np.zeros(state_size, dtype=np.float32)
        
        # Get basic robot joint angles
        state = self.get_state()
        shoulder_angle = np.tanh(state[5])  # Normalize to [-1, 1]
        elbow_angle = np.tanh(state[6])     # Normalize to [-1, 1]
        
        # Fill joint positions
        state_array[0] = shoulder_angle  # First joint
        state_array[1] = elbow_angle     # Second joint
        
        # Physical state information (last 3 elements)
        velocity = np.tanh(self.body.linearVelocity.x / 5.0)
        stability = np.tanh(self.body.angle * 2.0)
        progress = np.tanh((self.body.position.x - self.prev_x) * 10.0)
        
        state_array[joint_count] = velocity      # velocity
        state_array[joint_count + 1] = stability  # stability  
        state_array[joint_count + 2] = progress   # progress
        
        return state_array

    def choose_action(self, state) -> int:
        """Choose action using attention-based learning system."""
        #state = self.get_state_representation()
        action = self._learning_system.choose_action(state)
        return action
        

    def learn_from_experience(self, prev_state, action, reward, new_state, done=False):
        """Learn from experience using attention-based learning system."""
        try:
            if self._learning_system:
                # Ensure states are numpy arrays
                if not isinstance(prev_state, np.ndarray):
                    prev_state = self.get_state_representation()
                if not isinstance(new_state, np.ndarray):
                    new_state = self.get_state_representation()
                
                self._learning_system.store_experience(prev_state, action, reward, new_state, done)
                
                # Train frequently for neural networks (every 2 steps for rapid learning)
                if self.steps % 10 == 0:
                    # Log first training session
                    if not hasattr(self, '_training_started'):
                        print(f"🚀 Agent {self.id}: Neural network training STARTED at step {self.steps}")
                        self._training_started = True
                    
                    training_stats = self._learning_system.learn()
                    
                    # Log training activity periodically
                    if self.steps % 100 == 0 and training_stats:
                        print(f"🧠 Agent {self.id}: Training step {self.steps}, Loss: {training_stats.get('loss', 0.0):.4f}, Q-val: {training_stats.get('mean_q_value', 0.0):.3f}")
                    
        except Exception as e:
            print(f"❌ Error in learning for agent {self.id}: {e}")

    def get_crawling_reward(self, prev_x: float) -> float:
        """Calculate enhanced reward for crawling behavior with multi-limb robot support."""
        current_x = self.body.position.x
        total_reward = 0.0
        
        # 1. FORWARD PROGRESS REWARD (Core movement reward)
        displacement = current_x - prev_x
        if displacement > 0.003:
            progress_reward = displacement * 8.0
            
            # Sustained movement bonus
            if hasattr(self, 'recent_displacements'):
                self.recent_displacements.append(displacement)
                if len(self.recent_displacements) > 10:
                    self.recent_displacements.pop(0)
                if len(self.recent_displacements) >= 5:
                    avg_displacement = sum(self.recent_displacements) / len(self.recent_displacements)
                    if avg_displacement > 0.005:
                        progress_reward *= 1.5  # Sustained movement bonus
            else:
                self.recent_displacements = [displacement]
        elif displacement < -0.0005:
            progress_reward = displacement * 1.0  # Small backward penalty
        else:
            progress_reward = 0.0
        
        total_reward += progress_reward * 0.35  # Reduced from 0.4 to make room for other rewards
        
        # 2. FOOD APPROACH REWARD (Enhanced weight)
        food_approach_reward = self._get_food_approach_reward()
        total_reward += food_approach_reward * 0.20  # Increased from 0.15
        
        # 3. STABILITY REWARD (New - especially important for multi-limb robots)
        stability_reward = self._get_stability_reward()
        total_reward += stability_reward * 0.15
        
        # 4. ENERGY EFFICIENCY REWARD (New - rewards efficient movement)
        efficiency_reward = self._get_energy_efficiency_reward(displacement)
        total_reward += efficiency_reward * 0.15
        
        # 5. COORDINATION REWARD (New - for multi-limb robots)
        coordination_reward = self._get_coordination_reward()
        total_reward += coordination_reward * 0.15
        
        # Clip final reward
        total_reward = np.clip(total_reward, -0.75, 0.75)  # Slightly expanded range
        
        return total_reward
        
    def _get_stability_reward(self) -> float:
        """Calculate reward for maintaining stable posture."""
        try:
            if not hasattr(self, 'body') or not self.body:
                return 0.0
            
            # Reward for staying upright (angle close to 0)
            body_angle = abs(self.body.angle)
            if body_angle < 0.2:  # Very stable
                return 0.08
            elif body_angle < 0.5:  # Moderately stable  
                return 0.04
            elif body_angle > 1.5:  # Very unstable
                return -0.04
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _get_energy_efficiency_reward(self, displacement: float) -> float:
        """Calculate reward for energy-efficient movement."""
        try:
            if displacement <= 0:
                return 0.0
            
            # Calculate "energy cost" based on action intensity
            if hasattr(self, 'current_action_tuple') and self.current_action_tuple:
                action_intensity = sum(abs(a) for a in self.current_action_tuple) / len(self.current_action_tuple)
                
                # Reward high movement with low action intensity (efficiency)
                if action_intensity > 0.1:
                    efficiency = displacement / action_intensity
                    return min(0.06, efficiency * 0.02)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _get_coordination_reward(self) -> float:
        """Calculate reward for good limb coordination (for multi-limb robots)."""
        try:
            # This is overridden in EvolutionaryCrawlingAgent for multi-limb robots
            # Basic robots get a small base coordination reward for smooth joint movement
            if hasattr(self, 'upper_arm_joint') and hasattr(self, 'lower_arm_joint'):
                if self.upper_arm_joint and self.lower_arm_joint:
                    # Reward for coordinated joint movement (not fighting each other)
                    upper_speed = abs(self.upper_arm_joint.motorSpeed)
                    lower_speed = abs(self.lower_arm_joint.motorSpeed)
                    
                    # Small bonus for active but coordinated movement
                    if 0.1 < upper_speed < 2.0 and 0.1 < lower_speed < 2.0:
                        return 0.02
            
            return 0.0
            
        except Exception:
            return 0.0

    def step(self, dt: float):
        """Main step function with clean learning integration."""
        # Initialize action if needed
        if self.current_action is None:
            self.current_state = self.get_state_representation()
            action_idx = self.choose_action()
            self.current_action = action_idx
            self.current_action_tuple = self.actions[action_idx]
        
        # Apply current action
        self.apply_action(self.current_action_tuple)
        
        # Calculate reward
        current_x = self.body.position.x
        reward = self.get_crawling_reward(self.prev_x)
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
            action_idx = self.choose_action()
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
        """Reset the agent for a new episode."""
        super().reset()
        self.current_state = None
        self.current_action = None
        self.total_reward = 0.0
        self.immediate_reward = 0.0
        self.last_reward = 0.0
        
        # Reset timing
        self.last_action_time = time.time()
        self.current_action_tuple = (1, 0)
        self.prev_x = self.initial_position[0]
        
        # Reset tracking
        self.action_history = []
        self.recent_displacements = []
        self.action_sequence = []
        self.prev_food_distance = float('inf')
        self.food_distance_history = []
        
        # Reset velocities
        for part in [self.body, self.upper_arm, self.lower_arm] + self.wheels:
            part.linearVelocity = (0, 0)
            part.angularVelocity = 0

    def get_fitness(self) -> float:
        """Get fitness score for evolution."""
        return self.total_reward
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information including learning system details."""
        return {
            'agent_id': self.id,
            'learning_approach': self.learning_approach,
            'total_reward': self.total_reward,
            'steps': self.steps,
            'position': (self.body.position.x, self.body.position.y),
            'velocity': (self.body.linearVelocity.x, self.body.linearVelocity.y),
            'body_angle': self.body.angle,
            'current_action': self.current_action_tuple,
            'action_history_length': len(self.action_history),
            'state_size': self.state_size,
            'action_size': self.action_size,
        }

    # Physics methods (unchanged)
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
        
        tapered_vertices = [
            (-1.0, -0.2), (-1.0, 0.2), (0.5, 0.1),
            (1.0, 0.0), (0.5, -0.1),
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
            bodyA=self.body, bodyB=self.upper_arm,
            localAnchorA=(-1.0, 1.0), localAnchorB=(-1.0, 0),
            enableMotor=True, maxMotorTorque=self.motor_torque,
            motorSpeed=0, enableLimit=True,
            lowerAngle=-np.pi/2, upperAngle=np.pi/2,
        )
        self.lower_arm_joint = self.world.CreateRevoluteJoint(
            bodyA=self.upper_arm, bodyB=self.lower_arm,
            localAnchorA=(1.0, 0), localAnchorB=(-1.0, 0),
            enableMotor=True, maxMotorTorque=self.motor_torque,
            motorSpeed=0, enableLimit=True,
            lowerAngle=0, upperAngle=3*np.pi/4,
        )
        self.wheel_joints = []
        wheel_anchor_positions = [(-1.0, -0.75), (1.0, -0.75)]
        for i, anchor_pos in enumerate(wheel_anchor_positions):
            joint = self.world.CreateRevoluteJoint(
                bodyA=self.body, bodyB=self.wheels[i],
                localAnchorA=anchor_pos, localAnchorB=(0,0),
                enableMotor=False,
            )
            self.wheel_joints.append(joint) 

    def apply_action(self, action: Tuple[float, float]):
        """Apply action to the agent's arms."""
        shoulder_speed = float(np.clip(action[0], -1.0, 1.0)) * self.motor_speed
        elbow_speed = float(np.clip(action[1], -1.0, 1.0)) * self.motor_speed

        self.upper_arm_joint.motorSpeed = shoulder_speed
        self.lower_arm_joint.motorSpeed = elbow_speed
        
        self.upper_arm.awake = True
        self.lower_arm.awake = True

    def _get_food_approach_reward(self) -> float:
        """Calculate reward for moving toward food."""
        try:
            current_food_distance = self._get_nearest_food_distance()
            
            if not hasattr(self, 'prev_food_distance'):
                self.prev_food_distance = current_food_distance
                return 0.0
            
            if current_food_distance == float('inf') or self.prev_food_distance == float('inf'):
                return 0.0
            
            distance_change = self.prev_food_distance - current_food_distance
            food_reward = 0.0
            
            if distance_change > 0.5:
                food_reward += min(0.08, distance_change * 0.3)
            elif distance_change < -0.5:
                food_reward -= min(0.008, abs(distance_change) * 0.03)
            
            if current_food_distance < 5.0:
                food_reward += 0.04 * (5.0 - current_food_distance) / 5.0
            
            self.prev_food_distance = current_food_distance
            return food_reward
            
        except Exception:
            return 0.0
    
    def _get_nearest_food_distance(self) -> float:
        """Get distance to nearest food source."""
        try:
            training_env = self.world._training_env
            agent_pos = (self.body.position.x, self.body.position.y)
            
            nearest_distance = float('inf')
            for food_source in training_env.ecosystem_dynamics.food_sources:
                if food_source.amount <= 0.1:
                    continue
                food_x, food_y = food_source.position
                distance = ((agent_pos[0] - food_x) ** 2 + (agent_pos[1] - food_y) ** 2) ** 0.5
                if distance < nearest_distance:
                    nearest_distance = distance
            
            return nearest_distance
        except Exception:
            return float('inf')

    def reset_position(self):
        """Reset agent position while preserving learning state."""
        try:
            if hasattr(self, 'upper_arm_joint') and self.upper_arm_joint:
                self.upper_arm_joint.enableMotor = False
            if hasattr(self, 'lower_arm_joint') and self.lower_arm_joint:
                self.lower_arm_joint.enableMotor = False
            
            base_x, base_y = float(self.initial_position[0]), float(self.initial_position[1])
            
            self.body.position = self.initial_position
            self.body.angle = 0
            self.body.linearVelocity = (0, 0)
            self.body.angularVelocity = 0
            self.body.awake = True

            if hasattr(self, 'upper_arm') and self.upper_arm:
                self.upper_arm.position = (base_x - 1.0, base_y + 1.0)
                self.upper_arm.angle = 0
                self.upper_arm.linearVelocity = (0, 0)
                self.upper_arm.angularVelocity = 0
                self.upper_arm.awake = True

            if hasattr(self, 'lower_arm') and self.lower_arm:
                self.lower_arm.position = (base_x + 1.0, base_y + 1.0)
                self.lower_arm.angle = 0
                self.lower_arm.linearVelocity = (0, 0)
                self.lower_arm.angularVelocity = 0
                self.lower_arm.awake = True

            if hasattr(self, 'wheels') and self.wheels:
                wheel_offsets = [(-1.0, -0.75), (1.0, -0.75)]
                for wheel, offset in zip(self.wheels, wheel_offsets):
                    if wheel:
                        wheel.position = (base_x + offset[0], base_y + offset[1])
                        wheel.linearVelocity = (0, 0)
                        wheel.angularVelocity = 0
                        wheel.awake = True
            
            if hasattr(self, 'upper_arm_joint') and self.upper_arm_joint:
                self.upper_arm_joint.enableMotor = True
            if hasattr(self, 'lower_arm_joint') and self.lower_arm_joint:
                self.lower_arm_joint.enableMotor = True
            
            self.total_reward = 0
            self.steps = 0

        except Exception as e:
            print(f"⚠️ Error resetting position for agent {self.id}: {e}") 

class LearningSystemWrapper:
    """Compatibility wrapper for learning systems to provide q_table interface."""
    
    def __init__(self, learning_system, learning_approach):
        self._learning_system = learning_system
        self._learning_approach = learning_approach
    
    def get_convergence_estimate(self):
        """Return convergence estimate."""
        if hasattr(self._learning_system, 'get_convergence_estimate'):
            return self._learning_system.get_convergence_estimate()
        return 0.5  # Default estimate for neural networks
    
    def get_stats(self):
        """Return learning statistics."""
        if hasattr(self._learning_system, 'get_stats'):
            return self._learning_system.get_stats()
        return {'approach': self._learning_approach, 'active': True}
    
    @property
    def q_values(self):
        """Return Q-values for compatibility."""
        if hasattr(self._learning_system, 'q_values'):
            return self._learning_system.q_values
        return {}  # Neural networks don't have explicit Q-values
    
    @property
    def state_coverage(self):
        """Return state coverage for compatibility."""
        if hasattr(self._learning_system, 'state_coverage'):
            return self._learning_system.state_coverage
        return set()  # Neural networks don't track explicit state coverage
    
    @property
    def q_value_history(self):
        """Return Q-value history for compatibility."""
        if hasattr(self._learning_system, 'q_value_history'):
            return self._learning_system.q_value_history
        return []  # Neural networks don't track explicit Q-value history
    
    def __getattr__(self, name):
        """Forward unknown attributes to the underlying learning system."""
        return getattr(self._learning_system, name) 