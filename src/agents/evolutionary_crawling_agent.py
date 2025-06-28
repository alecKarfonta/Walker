"""
Evolutionary Crawling Agent with Attention-based Deep Q-Learning.
Clean, modern architecture focused on attention-based neural networks.
"""

import numpy as np
import Box2D as b2
from typing import Tuple, Dict, Any, List, Optional, Union
from copy import deepcopy
import random
import uuid
import time

from .crawling_crate_agent import CrawlingCrateAgent
from .physical_parameters import PhysicalParameters


class EvolutionaryCrawlingAgent(CrawlingCrateAgent):
    """
    Enhanced CrawlingCrate agent with evolvable physical parameters.
    Uses ONLY attention-based deep Q-learning for optimal performance.
    """
    
    def __init__(self, 
                 world: b2.b2World, 
                 agent_id: Optional[int] = None, 
                 position: Tuple[float, float] = (10, 20),
                 category_bits: int = 0x0002,  # AGENT_CATEGORY 
                 mask_bits: int = 0x0005,      # GROUND_CATEGORY | OBSTACLE_CATEGORY
                 physical_params: Optional[PhysicalParameters] = None,
                 parent_lineage: Optional[List[str]] = None,
                 learning_approach: str = "attention_deep_q_learning"):
        """
        Initialize evolutionary crawling agent with attention-based learning ONLY.
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
        
        # Store lineage information
        if parent_lineage is None:
            self.parent_lineage = []
        else:
            self.parent_lineage = [str(pid) for pid in parent_lineage]
        self.generation = len(self.parent_lineage)
        
        # Pre-calculate action space for learning system initialization
        total_joints = self.physical_params.num_arms * self.physical_params.segments_per_limb
        temp_actions = self._generate_dynamic_action_space_static(total_joints)
        
        # Set required attributes before calling parent constructor
        self.temp_action_size = len(temp_actions)
        # Dynamic state size based on actual morphology: joints + 3 metadata
        actual_joints = self.physical_params.num_arms * self.physical_params.segments_per_limb
        self.state_size = actual_joints + 3  # Dynamic size for neural networks
        
        # FORCE attention deep Q-learning - no other options
        self.learning_approach = "attention_deep_q_learning"
        
        # Initialize parent class with attention learning
        super().__init__(
            world=world,
            agent_id=self.id,  # Keep as string
            position=position,
            category_bits=category_bits,
            mask_bits=mask_bits,
            learning_approach="attention_deep_q_learning"  # Force attention learning
        )
        
        # Override physical properties with evolved parameters
        self.motor_torque = self.physical_params.motor_torque
        self.motor_speed = self.physical_params.motor_speed
        
        # Recreate body parts with evolved parameters
        self._destroy_existing_body()
        self._create_evolutionary_body()
        self._create_evolutionary_arms()
        self._create_evolutionary_wheels()
        self._create_evolutionary_joints()
        
        # Evolution tracking
        self.mutation_count = 0
        self.crossover_count = 0
        self.fitness_history = []
        self.diversity_score = 0.0
        
        # Generate dynamic action space based on morphology
        self.actions = self._generate_dynamic_action_space()
        self.action_size = len(self.actions)
        
        # Remove temporary action size
        if hasattr(self, 'temp_action_size'):
            delattr(self, 'temp_action_size')
        
        # DON'T initialize learning during __init__ - will be assigned by Learning Manager later
        self._learning_system = None
        
        # MORPHOLOGY-AWARE ACTION PERSISTENCE: Complex robots need more time per action
        self._set_morphology_aware_timing()
        
        # MULTI-ACTION SYSTEM: Very complex robots can execute multiple actions per step
        self._set_multi_action_capability()
        
        print(f"🧠 Created agent {self.id} with {self.physical_params.num_arms} limbs, {self.physical_params.segments_per_limb} segments each (learning will be assigned later)")

    def _set_morphology_aware_timing(self):
        """Set action persistence duration based on robot complexity."""
        try:
            total_joints = self.physical_params.num_arms * self.physical_params.segments_per_limb
            
            # DRAMATICALLY REDUCED action persistence for better limb control
            # Complex robots need rapid action updates for fine-grained coordination
            if total_joints <= 4:
                # Simple robots: 0.1s (6 frames at 60 FPS) - much faster than before
                self.action_persistence_duration = 0.1
            elif total_joints <= 8:
                # Medium complexity: 0.067s (4 frames) - very responsive
                self.action_persistence_duration = 0.067
            elif total_joints <= 12:
                # Complex robots: 0.05s (3 frames) - near real-time control
                self.action_persistence_duration = 0.05
            else:
                # Very complex robots: 0.033s (2 frames) - maximum responsiveness
                self.action_persistence_duration = 0.033
            
            # Log timing for complex robots
            if total_joints > 4:
                frames_per_action = int(self.action_persistence_duration * 60)
                print(f"⚡ {total_joints}-joint robot {self.id[:8]}: {self.action_persistence_duration:.3f}s ({frames_per_action} frames) - HIGH FREQUENCY CONTROL")
                
        except Exception as e:
            print(f"⚠️ Error setting morphology-aware timing for agent {self.id}: {e}")
            self.action_persistence_duration = 0.1  # Fast fallback

    def _set_multi_action_capability(self):
        """Set up multi-action capability for very complex robots."""
        try:
            total_joints = self.physical_params.num_arms * self.physical_params.segments_per_limb
            
            # Very complex robots can execute multiple actions per physics step
            if total_joints >= 15:
                self.actions_per_step = 3  # Triple action rate for very complex robots
                print(f"🚀 {total_joints}-joint robot {self.id[:8]}: TRIPLE ACTION RATE (3 actions per step)")
            elif total_joints >= 10:
                self.actions_per_step = 2  # Double action rate for complex robots
                print(f"🚀 {total_joints}-joint robot {self.id[:8]}: DOUBLE ACTION RATE (2 actions per step)")
            else:
                self.actions_per_step = 1  # Standard action rate
                
            self.action_counter = 0  # Track actions within each step
                
        except Exception as e:
            print(f"⚠️ Error setting multi-action capability for agent {self.id}: {e}")
            self.actions_per_step = 1  # Fallback to single action

    def _initialize_attention_learning(self):
        """Initialize ONLY attention-based deep Q-learning via Learning Manager."""
        try:
            # CRITICAL: Never create networks directly - ONLY get from Learning Manager
            # This prevents the constant GPU network recreation that's killing performance
            from .learning_manager import LearningManager
            
            # Try to get learning manager instance from training environment
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
            
        except Exception as e:
            print(f"❌ Error getting learning system for agent {self.id}: {e}")
            self._learning_system = None

    @property 
    def q_table(self):
        """Backward compatibility: Return learning system."""
        return self._learning_system
    
    @q_table.setter
    def q_table(self, value):
        """Backward compatibility: Set learning system."""
        self._learning_system = value

    @property
    def action_size(self):
        """Return action space size, using temp_action_size during initialization if available."""
        if hasattr(self, 'temp_action_size'):
            return self.temp_action_size
        return getattr(self, '_action_size', 6)  # Default fallback
    
    @action_size.setter
    def action_size(self, value):
        """Set action space size."""
        self._action_size = value

    def _destroy_existing_body(self):
        """Safely destroy existing body parts before creating evolved ones."""
        try:
            # Disable motors first
            if hasattr(self, 'upper_arm_joint') and self.upper_arm_joint:
                self.upper_arm_joint.enableMotor = False
            if hasattr(self, 'lower_arm_joint') and self.lower_arm_joint:
                self.lower_arm_joint.enableMotor = False
            
            # Destroy in reverse order of creation
            if hasattr(self, 'wheels'):
                for wheel in self.wheels:
                    if wheel:
                        try:
                            self.world.DestroyBody(wheel)
                        except (RuntimeError, AttributeError) as e:
                            print(f"⚠️ Error destroying wheel for agent {self.id}: {e}")
            
            if hasattr(self, 'lower_arm') and self.lower_arm:
                try:
                    self.world.DestroyBody(self.lower_arm)
                except (RuntimeError, AttributeError) as e:
                    print(f"⚠️ Error destroying lower_arm for agent {self.id}: {e}")
            
            if hasattr(self, 'upper_arm') and self.upper_arm:
                try:
                    self.world.DestroyBody(self.upper_arm)
                except (RuntimeError, AttributeError) as e:
                    print(f"⚠️ Error destroying upper_arm for agent {self.id}: {e}")
            
            if hasattr(self, 'body') and self.body:
                try:
                    self.world.DestroyBody(self.body)
                except (RuntimeError, AttributeError) as e:
                    print(f"⚠️ Error destroying body for agent {self.id}: {e}")
                    
        except Exception as e:
            print(f"⚠️ Error destroying existing body for agent {self.id}: {e}")

    def get_actual_joint_count(self) -> int:
        """Get the actual number of joints for this robot's morphology."""
        try:
            if hasattr(self, 'limb_joints') and self.limb_joints:
                joint_count = 0
                for limb_joints in self.limb_joints:
                    joint_count += len(limb_joints)
                return joint_count
            else:
                # Fallback: calculate from physical parameters
                return self.physical_params.num_arms * self.physical_params.segments_per_limb
        except AttributeError as e:
            print(f"⚠️ AttributeError in get_actual_joint_count for agent {self.id}: {e}")
            # Safe fallback only for attribute errors
            return 2
        except Exception as e:
            print(f"❌ CRITICAL ERROR in get_actual_joint_count for agent {self.id}: {e}")
            # DO NOT return 2 - this could cause silent state corruption
            # Let the error propagate to reveal the real problem
            raise

    def get_state_representation(self) -> np.ndarray:
        """
        Get comprehensive state representation for attention-based neural networks.
        Includes ALL joint angles for multi-limb robots with VARIABLE state size.
        Always returns numpy array for consistent neural network input.
        """
        try:
            # DYNAMIC STATE SIZE: Actual number of joints + 3 metadata elements
            actual_joints = self.get_actual_joint_count()
            state_size = actual_joints + 3  # joint angles + velocity + stability + progress
            
            # Initialize state array with exact size needed
            state = np.zeros(state_size, dtype=np.float32)
            
            # Get ALL joint angles for multi-limb robots
            joint_idx = 0
            if hasattr(self, 'limb_joints') and self.limb_joints:
                for limb_joints in self.limb_joints:
                    for joint in limb_joints:
                        if joint and joint_idx < actual_joints:
                            # Normalize joint angles to [-1, 1] range
                            state[joint_idx] = np.tanh(joint.angle)
                            joint_idx += 1
            
            # Physical state information (last 3 elements)
            velocity = np.tanh(self.body.linearVelocity.x / 5.0) if hasattr(self, 'body') else 0.0
            stability = np.tanh(self.body.angle * 2.0) if hasattr(self, 'body') else 0.0
            progress = np.tanh((self.body.position.x - self.prev_x) * 10.0) if hasattr(self, 'body') else 0.0
            
            state[actual_joints] = velocity      # velocity
            state[actual_joints + 1] = stability  # stability  
            state[actual_joints + 2] = progress   # progress
            
            return state
                
        except Exception as e:
            print(f"⚠️ Error getting evolutionary state for agent {self.id}: {e}")
            # Safe fallback with proper error handling
            try:
                actual_joints = self.get_actual_joint_count()
                return np.zeros(actual_joints + 3, dtype=np.float32)  # Dynamic size based on morphology
            except Exception as e2:
                print(f"❌ CRITICAL: Could not determine joint count for agent {self.id}: {e2}")
                # Ultimate fallback to prevent complete failure, but log it prominently
                print(f"🚨 USING EMERGENCY FALLBACK STATE for agent {self.id} - this may indicate serious issues")
                return np.zeros(5, dtype=np.float32)  # Emergency fallback: 2 joints + 3 metadata

    def choose_action(self) -> int:
        """Choose action using attention-based learning system."""
        try:
            state = self.get_state_representation()
            if self._learning_system:
                action = self._learning_system.choose_action(state)
                return max(0, min(action, len(self.actions) - 1))
            else:
                return random.randint(0, len(self.actions) - 1)
            
        except Exception as e:
            print(f"❌ Error choosing evolutionary action for agent {self.id}: {e}")
            return random.randint(0, len(self.actions) - 1)

    def apply_action(self, action: Tuple[float, float]):
        """Apply action to evolved joint configuration."""
        try:
            if hasattr(self, 'limb_joints') and self.limb_joints:
                # CRITICAL FIX: Multi-limb robots need to use their dynamic action space
                # Convert action index to proper joint action tuple for multi-limb robots
                if isinstance(action, tuple) and len(action) == 2:
                    # This is a basic 2-joint action from parent class - need to map to multi-limb
                    if hasattr(self, 'current_action') and self.current_action is not None:
                        # Use the current action index to get the proper multi-limb action
                        if 0 <= self.current_action < len(self.actions):
                            multi_limb_action = self.actions[self.current_action]
                            self.apply_action_to_joints(multi_limb_action)
                        else:
                            # Fallback: apply action to first 2 joints only
                            self.apply_action_to_joints(action)
                    else:
                        # Fallback: apply action to first 2 joints only
                        self.apply_action_to_joints(action)
                else:
                    # Already a proper multi-joint action tuple
                    self.apply_action_to_joints(action)
            else:
                # Fallback to basic joint control
                super().apply_action(action)
                
        except Exception as e:
            print(f"⚠️ Error applying action for agent {self.id}: {e}")
            # Fallback to parent class method
            super().apply_action(action)

    def get_evolutionary_fitness(self) -> float:
        """Calculate comprehensive fitness including morphological efficiency and coordination."""
        base_fitness = self.total_reward
        
        # Enhanced fitness calculation for multi-limb robots
        total_joints = self.physical_params.num_arms * self.physical_params.segments_per_limb
        
        # 1. MORPHOLOGICAL COMPLEXITY BONUS
        # Reward for successfully controlling complex morphologies
        complexity_bonus = 0.0
        if base_fitness > 0.1:  # Only if robot is actually successful
            if total_joints > 6:  # Complex robots
                # Graduated complexity bonus
                if total_joints <= 10:
                    complexity_bonus = 0.05  # Moderate complexity
                elif total_joints <= 15:
                    complexity_bonus = 0.08  # High complexity  
                else:
                    complexity_bonus = 0.12  # Very high complexity
                
                # Scale bonus by actual performance
                complexity_bonus *= min(1.0, base_fitness / 0.5)
        
        # 2. EFFICIENCY SCORING
        # Reward for achieving good results with reasonable parameters
        efficiency_score = 0.0
        
        # Parameter efficiency (existing logic enhanced)
        if (0.8 < self.physical_params.body_width < 2.0 and
            100 < self.physical_params.motor_torque < 200):
            efficiency_score += 0.08
        
        # Joint count efficiency - reward for not over-engineering
        if 4 <= total_joints <= 12:  # Sweet spot for complexity
            efficiency_score += 0.04
        elif total_joints > 18:  # Potentially over-engineered
            efficiency_score -= 0.02
        
        # 3. PARAMETER PENALTIES (Enhanced)
        param_penalty = 0.0
        
        # Size penalties
        if self.physical_params.body_width > 2.5 or self.physical_params.body_height > 1.2:
            param_penalty += 0.08
        
        # Motor penalties  
        if self.physical_params.motor_torque > 250:
            param_penalty += 0.04
        
        # Extreme complexity penalty
        if total_joints > 20:  # Very high complexity
            param_penalty += 0.06 * ((total_joints - 20) / 10.0)
        
        # 4. SPECIALIZATION BONUS
        # Reward for specialized morphologies that perform well
        specialization_bonus = 0.0
        if hasattr(self.physical_params, 'limb_specialization') and base_fitness > 0.2:
            if self.physical_params.limb_specialization != "general":
                # Specialized robots get bonus if they're performing well
                specialization_bonus = 0.03
        
        # 5. GENERATION BONUS
        # Small bonus for evolved robots (not first generation)
        generation_bonus = 0.0
        if self.generation > 0 and base_fitness > 0.1:
            # Reward for successful evolution (up to 5 generations)
            generation_bonus = min(0.02, self.generation * 0.005)
        
        # 6. LONGEVITY BONUS  
        # Reward for robots that survive longer (if step count available)
        longevity_bonus = 0.0
        if hasattr(self, 'steps') and self.steps > 100:
            # Small bonus for long-lived successful robots
            if base_fitness > 0.05:
                longevity_bonus = min(0.03, (self.steps - 100) / 5000.0)
        
        # Calculate final fitness
        final_fitness = (base_fitness + 
                        complexity_bonus + 
                        efficiency_score + 
                        specialization_bonus + 
                        generation_bonus + 
                        longevity_bonus - 
                        param_penalty)
        
        return final_fitness

    def evolve_with(self, other: 'EvolutionaryCrawlingAgent', 
                   mutation_rate: float = 0.1) -> 'EvolutionaryCrawlingAgent':
        """Create offspring through crossover and mutation with attention learning transfer."""
        # Crossover physical parameters
        if random.random() < 0.7:
            child_params = self.physical_params.crossover(other.physical_params)
            child_params = child_params.mutate(mutation_rate)
        else:
            child_params = self.physical_params.mutate(mutation_rate * 1.5)
        
        # Create lineage tracking
        child_lineage = self.parent_lineage + [self.id]
        
        # Create new agent with evolved parameters
        child = EvolutionaryCrawlingAgent(
            world=self.world,
            agent_id=None,  # Generate new UUID
            position=self.initial_position,
            category_bits=self.category_bits,
            mask_bits=self.mask_bits,
            physical_params=child_params,
            parent_lineage=child_lineage,
            learning_approach="attention_deep_q_learning"  # Always attention learning
        )
        
        # Transfer attention neural network weights
        try:
            if hasattr(self, '_learning_system') and self._learning_system and hasattr(child, '_learning_system') and child._learning_system:
                # Transfer neural network weights from parent to child
                child._learning_system.q_network.load_state_dict(self._learning_system.q_network.state_dict())
                child._learning_system.target_network.load_state_dict(self._learning_system.target_network.state_dict())
                print(f"🧠 Transferred attention network: {self.id[:6]} → {child.id[:6]}")
        except Exception as e:
            print(f"⚠️ Error transferring attention learning in crossover: {e}")
        
        child.crossover_count = 1
        return child

    def clone_with_mutation(self, mutation_rate: float = 0.1) -> 'EvolutionaryCrawlingAgent':
        """Create a mutated clone with attention learning transfer."""
        # Mutate physical parameters
        mutated_params = self.physical_params.mutate(mutation_rate)
        
        # Create lineage tracking
        child_lineage = self.parent_lineage + [self.id]
        
        # Create cloned agent
        clone = EvolutionaryCrawlingAgent(
            world=self.world,
            agent_id=None,  # Generate new UUID
            position=self.initial_position,
            category_bits=self.category_bits,
            mask_bits=self.mask_bits,
            physical_params=mutated_params,
            parent_lineage=child_lineage,
            learning_approach="attention_deep_q_learning"  # Always attention learning
        )
        
        # Transfer attention neural network weights
        try:
            if hasattr(self, '_learning_system') and self._learning_system and hasattr(clone, '_learning_system') and clone._learning_system:
                # Transfer neural network weights from parent to clone
                clone._learning_system.q_network.load_state_dict(self._learning_system.q_network.state_dict())
                clone._learning_system.target_network.load_state_dict(self._learning_system.target_network.state_dict())
                print(f"🧠 Cloned attention network: {self.id[:6]} → {clone.id[:6]}")
        except Exception as e:
            print(f"⚠️ Error transferring attention learning in cloning: {e}")
        
        clone.mutation_count = self.mutation_count + 1
        return clone

    def get_debug_info(self) -> Dict[str, Any]:
        """Get comprehensive debug information."""
        debug = super().get_debug_info()
        
        debug.update({
            'generation': self.generation,
            'parent_lineage': self.parent_lineage,
            'mutation_count': self.mutation_count,
            'crossover_count': self.crossover_count,
            'evolutionary_fitness': self.get_evolutionary_fitness(),
            
            # Physical parameter summary
            'body_dimensions': f"{self.physical_params.body_width:.2f}x{self.physical_params.body_height:.2f}",
            'limb_config': f"{self.physical_params.num_arms} arms, {self.physical_params.segments_per_limb} segments",
            'motor_config': f"torque={self.physical_params.motor_torque:.0f}, speed={self.physical_params.motor_speed:.1f}",
            'action_space_size': len(self.actions),
            'total_joints': self.physical_params.num_arms * self.physical_params.segments_per_limb,
        })
        
        return debug

    # Keep all the existing evolutionary body creation methods
    # (These are preserved from the original implementation)
    
    def _create_evolutionary_body(self):
        """Create body using evolved physical parameters."""
        scale = self.physical_params.overall_scale
        scaled_width = self.physical_params.body_width * scale
        scaled_height = self.physical_params.body_height * scale
        
        body_def = b2.b2BodyDef(
            type=b2.b2_dynamicBody,
            position=(float(self.initial_position[0]), float(self.initial_position[1])),
            linearDamping=self.physical_params.body_linear_damping * self.physical_params.body_rigidity,
            angularDamping=self.physical_params.body_angular_damping * self.physical_params.body_rigidity
        )
        self.body = self.world.CreateBody(body_def)
        
        body_vertices = self._create_evolved_body_shape(scaled_width, scaled_height)
        
        adjusted_density = self.physical_params.body_density * self.physical_params.structural_reinforcement
        adjusted_friction = self.physical_params.body_friction * self.physical_params.body_rigidity
        
        chassis_fixture = self.body.CreateFixture(
            shape=b2.b2PolygonShape(vertices=body_vertices),
            density=adjusted_density,
            friction=adjusted_friction,
            restitution=self.physical_params.body_restitution,
            filter=b2.b2Filter(categoryBits=self.category_bits, maskBits=self.mask_bits)
        )
        
    def _create_evolved_body_shape(self, width: float, height: float) -> List[Tuple[float, float]]:
        """Create body shape based on evolved parameters."""
        half_width = width / 2
        half_height = height / 2
        aspect_ratio = self.physical_params.body_aspect_ratio
        taper = self.physical_params.body_taper
        curve = self.physical_params.body_curve
        
        length = width * aspect_ratio
        half_length = length / 2
        
        if self.physical_params.body_shape == "rectangle":
            return [
                (-half_length, -half_height),
                (half_length, -half_height),
                (half_length, half_height),
                (-half_length, half_height),
            ]
        elif self.physical_params.body_shape == "oval":
            points = []
            num_points = 8
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                x = half_length * np.cos(angle) * (1.0 + curve * 0.2)
                y = half_height * np.sin(angle)
                if x < 0:  # Back end
                    y *= taper
                points.append((x, y))
            return points
        else:
            # Fallback to rectangle
            return [
                (-half_length, -half_height),
                (half_length, -half_height),
                (half_length, half_height),
                (-half_length, half_height),
            ]

    # Keep existing methods for creating evolutionary arms, wheels, joints, etc.
    # (All the morphology creation code remains the same)
    
    def _create_evolutionary_arms(self):
        """Create arms using evolved physical parameters with variable segments."""
        scale = self.physical_params.overall_scale
        body_length = self.physical_params.body_width * self.physical_params.body_aspect_ratio * scale
        body_height = self.physical_params.body_height * scale
        
        attach_x = self.physical_params.arm_attachment_x * body_length / 2
        attach_y = (self.physical_params.arm_attachment_y - 0.5) * body_height
        
        self.limbs = []
        self.limb_joints = []
        self.upper_arms = []
        self.lower_arms = []
        
        for arm_index in range(self.physical_params.num_arms):
            limb_segments = []
            limb_joints_for_arm = []
            
            variation = 0.0 if arm_index == 0 else (2.0 - self.physical_params.arm_symmetry - 1.0) * 0.5
            
            arm_spacing = 0.3 * scale if self.physical_params.num_arms > 1 else 0.0
            arm_attach_x = attach_x + (arm_index - self.physical_params.num_arms/2) * arm_spacing
            arm_attach_y = attach_y
            
            prev_body = self.body
            prev_anchor = (arm_attach_x, arm_attach_y)
            
            for segment_index in range(self.physical_params.segments_per_limb):
                base_length = self.physical_params.arm_length if segment_index == 0 else self.physical_params.wrist_length
                segment_length_ratio = self.physical_params.segment_length_ratios[segment_index] if segment_index < len(self.physical_params.segment_length_ratios) else 1.0
                segment_width_ratio = self.physical_params.segment_width_ratios[segment_index] if segment_index < len(self.physical_params.segment_width_ratios) else 0.8
                
                segment_length = base_length * segment_length_ratio * scale * (1.0 + random.uniform(-variation, variation))
                segment_width = self.physical_params.arm_width * segment_width_ratio * scale * (1.0 + random.uniform(-variation, variation))
                
                if segment_index == 0:
                    segment_pos = (
                        self.body.position[0] + arm_attach_x - segment_length * np.cos(self.physical_params.arm_angle_offset),
                        self.body.position[1] + arm_attach_y + segment_length * np.sin(self.physical_params.arm_angle_offset)
                    )
                else:
                    segment_pos = (
                        prev_body.position[0] + segment_length,
                        prev_body.position[1]
                    )
                
                segment_density, segment_friction = self._get_specialized_limb_properties()
                segment_vertices = self._create_specialized_segment_shape(
                    segment_length, segment_width, segment_index, self.physical_params.segments_per_limb
                )
                
                segment = self.world.CreateDynamicBody(
                    position=segment_pos,
                fixtures=[
                    b2.b2FixtureDef(
                            shape=b2.b2PolygonShape(vertices=segment_vertices),
                            density=segment_density,
                            friction=segment_friction,
                        restitution=self.physical_params.arm_restitution,
                        filter=b2.b2Filter(categoryBits=self.category_bits, maskBits=self.mask_bits)
                    )
                ]
            )
            
                limb_segments.append(segment)
                
                if segment_index < len(self.physical_params.joint_torques):
                    joint_torque = self.physical_params.joint_torques[segment_index]
                    joint_speed = self.physical_params.joint_speeds[segment_index]
                    joint_lower = self.physical_params.joint_lower_limits[segment_index]
                    joint_upper = self.physical_params.joint_upper_limits[segment_index]
                else:
                    joint_torque = 100.0
                    joint_speed = 3.0
                    joint_lower = -np.pi/4
                    joint_upper = np.pi/2
                
                # STABILITY FIX: Reduce torque for complex robots to prevent physics instability
                total_joints = self.physical_params.num_arms * self.physical_params.segments_per_limb
                if total_joints > 10:  # Complex robots (more than 10 joints)
                    stability_factor = max(0.3, 10.0 / total_joints)  # Scale down torque
                    joint_torque *= stability_factor
                    joint_speed *= stability_factor
                
                joint = self.world.CreateRevoluteJoint(
                    bodyA=prev_body,
                    bodyB=segment,
                    localAnchorA=prev_anchor if prev_body == self.body else (segment_length, 0),
                    localAnchorB=(-segment_length, 0),
                    enableMotor=True,
                    maxMotorTorque=joint_torque,
                    motorSpeed=0,
                    enableLimit=True,
                    lowerAngle=joint_lower,
                    upperAngle=joint_upper,
                )
                
                limb_joints_for_arm.append(joint)
                
                prev_body = segment
                prev_anchor = (segment_length, 0)
            
            self.limbs.append(limb_segments)
            self.limb_joints.append(limb_joints_for_arm)
            
            if len(limb_segments) >= 1:
                self.upper_arms.append(limb_segments[0])
            if len(limb_segments) >= 2:
                self.lower_arms.append(limb_segments[1])
        
        # For backward compatibility
        self.upper_arm = self.upper_arms[0] if self.upper_arms else None
        self.lower_arm = self.lower_arms[0] if self.lower_arms else None
        
    def _get_specialized_limb_properties(self) -> Tuple[float, float]:
        """Get limb properties based on specialization type."""
        base_density = self.physical_params.arm_density
        base_friction = self.physical_params.arm_friction
        
        if self.physical_params.limb_specialization == "digging":
            return base_density * 1.5, base_friction * 1.8
        elif self.physical_params.limb_specialization == "climbing":
            return base_density * 0.9, base_friction * 2.2
        elif self.physical_params.limb_specialization == "swimming":
            return base_density * 0.7, base_friction * 0.4
        elif self.physical_params.limb_specialization == "grasping":
            return base_density * 1.1, base_friction * 1.5
        else:  # general
            return base_density, base_friction
        
    def _create_specialized_segment_shape(self, length: float, width: float, 
                                        segment_index: int, total_segments: int) -> List[Tuple[float, float]]:
        """Create segment shape based on specialization and position."""
        half_width = width / 2
        taper_factor = 1.0 - (segment_index / max(1, total_segments)) * 0.3
        tip_width = half_width * taper_factor
        
        # Simple tapered shape for all specializations
        return [
            (-length, -half_width),
            (-length, half_width),
            (length, tip_width),
            (length, -tip_width),
        ]
    
    def _create_evolutionary_wheels(self):
        """Create wheels using evolved physical parameters with advanced variability."""
        self.wheels = []
        
        # Check if robot has any wheels at all
        if self.physical_params.num_wheels == 0:
            return  # Wheel-less robot (pure arm-based locomotion)
        
        # Use advanced wheel configuration
        for i in range(self.physical_params.num_wheels):
            if i < len(self.physical_params.wheel_positions):
                wheel_pos = self.physical_params.wheel_positions[i]
                wheel_size = self.physical_params.wheel_sizes[i] if i < len(self.physical_params.wheel_sizes) else 0.5
                wheel_angle = self.physical_params.wheel_angles[i] if i < len(self.physical_params.wheel_angles) else 0.0
                wheel_type = self.physical_params.wheel_types[i] if i < len(self.physical_params.wheel_types) else "circle"
                wheel_stiffness = self.physical_params.wheel_stiffness[i] if i < len(self.physical_params.wheel_stiffness) else 1.0
                
                # Apply asymmetry if specified
                if "wheels" in self.physical_params.asymmetric_features:
                    asymmetry_factor = self.physical_params.left_right_asymmetry
                    if self.physical_params.dominant_side == "left" and wheel_pos[0] < 0:
                        wheel_size *= (1.0 + asymmetry_factor * 0.5)
                    elif self.physical_params.dominant_side == "right" and wheel_pos[0] > 0:
                        wheel_size *= (1.0 + asymmetry_factor * 0.5)
                    elif wheel_pos[0] < 0:  # Left side smaller if right dominant
                        wheel_size *= (1.0 - asymmetry_factor * 0.3)
                    elif wheel_pos[0] > 0:  # Right side smaller if left dominant
                        wheel_size *= (1.0 - asymmetry_factor * 0.3)
                
                # Create wheel shape based on type
                wheel_shape = self._create_wheel_shape(wheel_type, wheel_size)
                
                # Calculate world position with angle offset
                local_pos = (wheel_pos[0], wheel_pos[1] + wheel_angle * 0.2)  # Slight height adjustment for angled wheels
                world_pos = self.body.GetWorldPoint(local_pos)
                
                # Create wheel body with dynamic properties
                wheel_density = self.physical_params.wheel_density * wheel_stiffness
                wheel_friction = self.physical_params.wheel_friction
                
                # Adjust friction based on wheel type
                if wheel_type == "bumpy":
                    wheel_friction *= 1.4  # Better grip
                elif wheel_type == "star":
                    wheel_friction *= 1.2  # Slightly better grip
                elif wheel_type == "oval":
                    wheel_friction *= 0.9  # Slightly less grip
                
                wheel = self.world.CreateDynamicBody(
                    position=world_pos,
                    fixtures=[
                        b2.b2FixtureDef(
                            shape=wheel_shape,
                            density=wheel_density,
                            friction=wheel_friction,
                            restitution=self.physical_params.wheel_restitution,
                            filter=b2.b2Filter(categoryBits=self.category_bits, maskBits=self.mask_bits)
                        )
                    ]
                )
                self.wheels.append(wheel)
    
    def _create_wheel_shape(self, wheel_type: str, radius: float):
        """Create different wheel shapes based on type."""
        if wheel_type == "circle":
            return b2.b2CircleShape(radius=radius)
        elif wheel_type == "oval":
            # Approximate oval with elongated circle
            return b2.b2CircleShape(radius=radius * 0.8)  # Will be stretched during attachment
        elif wheel_type == "star":
            # Star shape using polygon (simplified to pentagon for physics stability)
            vertices = []
            num_points = 5
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                # Alternate between inner and outer radius for star effect
                r = radius if i % 2 == 0 else radius * 0.6
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                vertices.append((x, y))
            return b2.b2PolygonShape(vertices=vertices)
        elif wheel_type == "bumpy":
            # Bumpy wheel using octagon for better ground contact
            vertices = []
            num_sides = 8
            for i in range(num_sides):
                angle = 2 * np.pi * i / num_sides
                # Slight radius variation for bumpy effect
                r = radius * (0.9 + 0.2 * (i % 2))
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                vertices.append((x, y))
            return b2.b2PolygonShape(vertices=vertices)
        else:
            # Default to circle
            return b2.b2CircleShape(radius=radius)
    
    def _create_evolutionary_joints(self):
        """Create joints using evolved physical parameters with advanced wheel support."""
        # Set primary joint references for backward compatibility
        if hasattr(self, 'limb_joints') and self.limb_joints:
            first_limb_joints = self.limb_joints[0]
            if len(first_limb_joints) >= 1:
                self.upper_arm_joint = first_limb_joints[0]
            if len(first_limb_joints) >= 2:
                self.lower_arm_joint = first_limb_joints[1]
        
        # Create wheel joints for advanced wheel system
        self.wheel_joints = []
        if hasattr(self, 'wheels') and self.wheels and self.physical_params.num_wheels > 0:
            for i, wheel in enumerate(self.wheels):
                if i < len(self.physical_params.wheel_positions):
                    wheel_pos = self.physical_params.wheel_positions[i]
                    wheel_angle = self.physical_params.wheel_angles[i] if i < len(self.physical_params.wheel_angles) else 0.0
                    wheel_stiffness = self.physical_params.wheel_stiffness[i] if i < len(self.physical_params.wheel_stiffness) else 1.0
                    
                    # Create different joint types based on wheel configuration
                    if abs(wheel_angle) < 0.1:  # Standard vertical wheel
                        joint = self.world.CreateRevoluteJoint(
                            bodyA=self.body,
                            bodyB=wheel,
                            localAnchorA=wheel_pos,
                            localAnchorB=(0, 0),
                            enableMotor=False,
                        )
                    else:  # Angled wheel (like castor wheels or splayed legs)
                        # Use distance joint for more flexible attachment
                        joint = self.world.CreateDistanceJoint(
                            bodyA=self.body,
                            bodyB=wheel,
                            localAnchorA=wheel_pos,
                            localAnchorB=(0, 0),
                            length=0.0,  # Rigid connection
                            frequency=wheel_stiffness * 5.0,  # Suspension frequency
                            dampingRatio=0.7
                        )
                    
                    self.wheel_joints.append(joint)

    def apply_action_to_joints(self, action_tuple: Tuple) -> None:
        """Apply action tuple to the robot's joints."""
        if not hasattr(self, 'limb_joints') or not self.limb_joints:
            return
        
        joint_index = 0
        
        for limb_idx, limb_joints in enumerate(self.limb_joints):
            for segment_idx, joint in enumerate(limb_joints):
                if joint_index < len(action_tuple) and joint:
                    action_value = action_tuple[joint_index]
                    
                    if segment_idx < len(self.physical_params.joint_torques):
                        torque = self.physical_params.joint_torques[segment_idx]
                        speed = self.physical_params.joint_speeds[segment_idx]
                    else:
                        torque = self.physical_params.motor_torque
                        speed = self.physical_params.motor_speed
                    
                    if action_value != 0:
                        # STABILITY FIX: Reduce action intensity for complex robots
                        total_joints = self.physical_params.num_arms * self.physical_params.segments_per_limb
                        if total_joints > 10:  # Complex robots need gentler movements
                            action_intensity = max(0.3, 10.0 / total_joints)
                            action_value *= action_intensity
                        
                        joint.motorSpeed = action_value * speed
                        joint.maxMotorTorque = torque
                        joint.enableMotor = True
                    else:
                        joint.motorSpeed = 0
                        joint.enableMotor = False
                
                joint_index += 1

    def _generate_dynamic_action_space(self) -> List[Tuple]:
        """Generate action space based on robot morphology."""
        total_joints = self.physical_params.num_arms * self.physical_params.segments_per_limb
        return self._generate_dynamic_action_space_static(total_joints)
    
    @staticmethod
    def _generate_dynamic_action_space_static(total_joints: int) -> List[Tuple]:
        """Static method to generate action space based on joint count."""
        actions = []
        
        # Add "no movement" action
        actions.append((0,) * total_joints)
        
        # Single joint movements
        for joint_idx in range(total_joints):
            action = [0] * total_joints
            action[joint_idx] = 1
            actions.append(tuple(action))
            
            action = [0] * total_joints
            action[joint_idx] = -1
            actions.append(tuple(action))
        
        # Simple combinations for first two joints
        if total_joints >= 2:
            actions.append(tuple([1, 1] + [0] * (total_joints - 2)))
            actions.append(tuple([-1, -1] + [0] * (total_joints - 2)))
            actions.append(tuple([1, -1] + [0] * (total_joints - 2)))
            actions.append(tuple([-1, 1] + [0] * (total_joints - 2)))
        
        return actions
        
    def step(self, dt: float):
        """Enhanced step function with multi-action capability for complex robots."""
        # For very complex robots, execute multiple actions per step
        if hasattr(self, 'actions_per_step') and self.actions_per_step > 1:
            for _ in range(self.actions_per_step):
                super().step(dt)  # Execute parent step multiple times
        else:
            super().step(dt)  # Standard single step
        
    def destroy(self):
        """Clean up physics bodies safely."""
        try:
            self._destroyed = True
            
            # Disable motors first
            if hasattr(self, 'limb_joints') and self.limb_joints:
                for limb_joints in self.limb_joints:
                    for joint in limb_joints:
                        if joint:
                            try:
                                joint.enableMotor = False
                            except (RuntimeError, AttributeError) as e:
                                print(f"⚠️ Error disabling motor for agent {self.id}: {e}")
            
            # Destroy bodies in reverse order
            if hasattr(self, 'wheels'):
                for wheel in self.wheels:
                    if wheel:
                        try:
                            self.world.DestroyBody(wheel)
                        except (RuntimeError, AttributeError) as e:
                            print(f"⚠️ Error destroying wheel in destroy() for agent {self.id}: {e}")
            
            if hasattr(self, 'limbs'):
                for limb_segments in self.limbs:
                    for segment in limb_segments:
                        if segment:
                            try:
                                self.world.DestroyBody(segment)
                            except (RuntimeError, AttributeError) as e:
                                print(f"⚠️ Error destroying limb segment in destroy() for agent {self.id}: {e}")
            
            if hasattr(self, 'body') and self.body:
                try:
                    self.world.DestroyBody(self.body)
                except (RuntimeError, AttributeError) as e:
                    print(f"⚠️ Error destroying main body in destroy() for agent {self.id}: {e}")
                
        except Exception as e:
            print(f"⚠️ Error in destroy() for agent {getattr(self, 'id', 'unknown')}: {e}")
            self._destroyed = True 

    def _get_coordination_reward(self) -> float:
        """Calculate advanced coordination reward for multi-limb robots."""
        try:
            if not hasattr(self, 'limb_joints') or not self.limb_joints:
                return super()._get_coordination_reward()  # Fall back to basic coordination
            
            total_coordination_reward = 0.0
            total_joints = self.physical_params.num_arms * self.physical_params.segments_per_limb
            
            # 1. INTER-LIMB COORDINATION REWARD
            # Reward for symmetric limb movement patterns  
            if self.physical_params.num_arms >= 2:
                limb_speeds = []
                for limb_joints in self.limb_joints:
                    limb_speed = 0.0
                    for joint in limb_joints:
                        if joint:
                            limb_speed += abs(joint.motorSpeed)
                    limb_speeds.append(limb_speed)
                
                if len(limb_speeds) >= 2:
                    # Reward for balanced limb usage (avoid over-reliance on one limb)
                    speed_variance = np.var(limb_speeds) if len(limb_speeds) > 1 else 0
                    if speed_variance < 1.0:  # Low variance = good balance
                        total_coordination_reward += 0.04
                    
                    # Bonus for symmetric movement (front limbs similar, back limbs similar)
                    if len(limb_speeds) == 2:  # Two limbs
                        speed_diff = abs(limb_speeds[0] - limb_speeds[1])
                        if speed_diff < 0.5:  # Similar speeds
                            total_coordination_reward += 0.03
            
            # 2. INTRA-LIMB COORDINATION REWARD  
            # Reward for smooth joint chains within each limb
            for limb_joints in self.limb_joints:
                if len(limb_joints) >= 2:
                    joint_speeds = [abs(joint.motorSpeed) if joint else 0 for joint in limb_joints]
                    
                    # Reward for proximal-to-distal coordination (base joint leads, tip follows)
                    if len(joint_speeds) >= 2:
                        base_speed = joint_speeds[0]
                        tip_speed = joint_speeds[-1]
                        
                        # Natural coordination: base joint slightly more active
                        if 0.1 < base_speed < 3.0 and 0.1 < tip_speed < 2.0:
                            if base_speed >= tip_speed * 0.8:  # Base leads or matches
                                total_coordination_reward += 0.02
            
            # 3. COMPLEXITY BONUS/PENALTY
            # More complex robots get bonus for successful coordination
            # But penalty if they're just flailing around
            if total_joints > 6:  # Complex robots (more than 6 joints)
                active_joints = sum(1 for limb_joints in self.limb_joints 
                                  for joint in limb_joints 
                                  if joint and abs(joint.motorSpeed) > 0.1)
                
                joint_usage_ratio = active_joints / total_joints
                
                # Reward for using reasonable portion of joints (not all or none)
                if 0.3 <= joint_usage_ratio <= 0.8:
                    complexity_bonus = 0.03 * (total_joints / 18.0)  # Scale with complexity
                    total_coordination_reward += complexity_bonus
                elif joint_usage_ratio < 0.2:  # Under-utilizing complex morphology
                    total_coordination_reward -= 0.02
                elif joint_usage_ratio > 0.9:  # Over-activating (likely inefficient)
                    total_coordination_reward -= 0.01
            
            # 4. MORPHOLOGY-SPECIFIC REWARDS
            # Reward based on limb specialization
            if hasattr(self.physical_params, 'limb_specialization'):
                if self.physical_params.limb_specialization == "climbing":
                    # Climbing robots benefit from alternating limb patterns
                    # (Implementation would check for alternating patterns)
                    total_coordination_reward += 0.01
                elif self.physical_params.limb_specialization == "digging":
                    # Digging robots benefit from synchronized power strokes
                    # (Implementation would check for synchronized patterns)
                    total_coordination_reward += 0.01
            
            # Cap the coordination reward to prevent it from dominating
            return np.clip(total_coordination_reward, -0.05, 0.08)
            
        except Exception as e:
            print(f"⚠️ Error calculating coordination reward for agent {self.id}: {e}")
            return 0.0
    
    def _get_energy_efficiency_reward(self, displacement: float) -> float:
        """Enhanced energy efficiency reward for multi-limb robots."""
        try:
            if displacement <= 0:
                return 0.0
            
            # Multi-limb robots have more complex energy calculations
            if hasattr(self, 'limb_joints') and self.limb_joints:
                total_energy_cost = 0.0
                active_joints = 0
                
                for limb_joints in self.limb_joints:
                    for joint in limb_joints:
                        if joint and abs(joint.motorSpeed) > 0.1:
                            # Energy cost proportional to torque and speed
                            energy_cost = joint.maxMotorTorque * abs(joint.motorSpeed) / 1000.0
                            total_energy_cost += energy_cost
                            active_joints += 1
                
                if total_energy_cost > 0 and active_joints > 0:
                    # Efficiency = movement achieved per unit energy
                    efficiency = displacement / total_energy_cost
                    
                    # Scale reward based on morphology complexity
                    total_joints = self.physical_params.num_arms * self.physical_params.segments_per_limb
                    complexity_factor = min(1.0, total_joints / 10.0)  # More complex = higher standards
                    
                    efficiency_reward = min(0.08, efficiency * 0.05 * complexity_factor)
                    return efficiency_reward
            
            # Fallback to parent class method
            return super()._get_energy_efficiency_reward(displacement)
            
        except Exception as e:
            return super()._get_energy_efficiency_reward(displacement) 