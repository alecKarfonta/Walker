"""
Evolutionary Crawling Agent with Attention-based Deep Q-Learning.
Clean, modern architecture focused on attention-based neural networks.
STANDALONE - No inheritance from CrawlingAgent.
"""

import numpy as np
import Box2D as b2
from typing import Tuple, Dict, Any, List, Optional, Union
from copy import deepcopy
import random
import uuid
import time

from .physical_parameters import PhysicalParameters


class EvolutionaryCrawlingAgent:
    """
    Enhanced CrawlingCrate agent with evolvable physical parameters.
    Uses ONLY attention-based deep Q-learning for optimal performance.
    STANDALONE - No inheritance from CrawlingAgent.
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
        # Store basic attributes
        self.world = world
        self.initial_position = position
        self.category_bits = category_bits
        self.mask_bits = mask_bits
        self.learning_approach = "attention_deep_q_learning"
        
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
        
        # Set required attributes for neural network compatibility
        self.temp_action_size = len(temp_actions)
        # FIXED: Use standardized 29D state size for neural network compatibility
        # ALL agents must use 29D states to match the neural network architecture
        self.state_size = 29  # Fixed size for neural network consistency
        
        # Initialize basic robot attributes
        self.steps = 0
        self.total_reward = 0.0
        self.current_action = None
        self.current_action_tuple = None
        self.last_action_time = 0.0
        self.action_persistence_duration = 0.1  # Default, will be overridden
        
        # Initialize body parts (will be created later)
        self.body = None
        self.limb_joints = []
        self.limb_segments = []
        self.wheels = []
        
        # Initialize learning system (will be assigned by Learning Manager)
        self._learning_system = None
        
        # Initialize reward tracking
        self.previous_state = None
        self.previous_action = None
        self.last_positions = []
        
        # Motor parameters
        self.motor_torque = self.physical_params.motor_torque
        self.motor_speed = self.physical_params.motor_speed
        
        # Create body parts with evolved parameters
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
        
        # Initialize learning system (will be assigned by Learning Manager later)
        self._learning_system = None
        
        # MORPHOLOGY-AWARE ACTION PERSISTENCE: Complex robots need more time per action
        self._set_morphology_aware_timing()
        
        # UI COMPATIBILITY: Ensure limb references are set for UI rendering
        self._ensure_limb_references()
        
        # CRITICAL FIX: Initialize learning system (was missing!)
        # BUT ONLY if no network was preserved from Robot Memory Pool
        if not hasattr(self, '_learning_system') or self._learning_system is None:
            self._initialize_learning_system()
        else:
            print(f"🔄 Agent {self.id}: Preserved network from Robot Memory Pool (action_size: {self.action_size})")
    
    def step(self, dt: float):
        """Step the robot simulation with action persistence and learning."""
        self.steps += 1
        
        # Initialize action if needed
        if self.current_action is None:
            self.previous_state = self.get_state_representation()
            action_idx = self.choose_action(self.previous_state)
            self.current_action = action_idx
            self.current_action_tuple = self.actions[action_idx] if action_idx < len(self.actions) else self.actions[0]
            self.last_action_time = time.time()
        
        # Apply current action
        if hasattr(self, 'current_action_tuple') and self.current_action_tuple:
            self.apply_action_to_joints(self.current_action_tuple)
        
        # Calculate reward
        current_x = self.body.position.x if self.body else 0.0
        prev_x = self.last_positions[-1][0] if self.last_positions else current_x
        reward = self.get_reward(prev_x)
        self.total_reward += reward
        
        # Check action persistence timing for learning trigger
        current_time = time.time()
        time_since_action = current_time - self.last_action_time
        
        # Only learn when action has persisted long enough
        if time_since_action >= self.action_persistence_duration:
            # Store current state for learning
            current_state = self.get_state_representation()
            
            # Learn from the experience of the persistent action
            if self.previous_state is not None:
                # Use the existing reward calculation from earlier in the step
                self.learn_from_experience(
                    self.previous_state, 
                    self.current_action,
                    reward,
                    current_state,
                    done=False
                )
            
            # Update state for next learning cycle
            self.previous_state = current_state
            self.last_action_time = current_time
            
            # Choose new action
            action_idx = self.choose_action(current_state)
            self.current_action = action_idx
            self.current_action_tuple = self.actions[action_idx] if action_idx < len(self.actions) else self.actions[0]
        
        # Update position tracking for reward calculation
        if self.body:
            self.last_positions.append((self.body.position.x, self.body.position.y))
            if len(self.last_positions) > 10:  # Keep last 10 positions
                self.last_positions.pop(0)
    
    def apply_action(self, action: Union[int, Tuple[float, float]]):
        """Apply action to evolved joint configuration."""
        try:
            # Convert action index to action tuple if needed
            if isinstance(action, int):
                if 0 <= action < len(self.actions):
                    action_tuple = self.actions[action]
                    self.current_action = action
                    self.current_action_tuple = action_tuple
                else:
                    return  # Invalid action index
            else:
                action_tuple = action
                self.current_action_tuple = action_tuple
            
            # Apply action to joints
            self.apply_action_to_joints(action_tuple)
            # REMOVED: Don't update last_action_time here - it breaks persistence timing!
            # self.last_action_time = time.time()
            
        except Exception as e:
            print(f"⚠️ Error applying action for agent {self.id}: {e}")
    
    def learn_from_experience(self, prev_state, action, reward, new_state, done=False):
        """Learn from experience using neural network."""
        if prev_state is None:
            return
        
        # IMMOBILIZATION SAFEGUARD: Don't learn when immobilized (low energy)
        if hasattr(self, '_is_immobilized') and getattr(self, '_is_immobilized', False):
            return  # Skip learning for immobilized agents
        
        if self._learning_system is None:
            raise RuntimeError(f"Agent {self.id}: No learning system assigned!")
        
        # Store the experience
        self._learning_system.store_experience(prev_state, action, reward, new_state, done)
        
        # Train when we have enough experiences
        self._maybe_train_network()
    
    def _maybe_train_network(self):
        """Train the neural network when conditions are met."""
        # CRITICAL FIX: Recovery mechanism for agents that lose their learning systems
        if not self._learning_system:
            if not hasattr(self, '_learning_system_recovery_attempts'):
                self._learning_system_recovery_attempts = 0
            
            # Try to recover the learning system
            if self._learning_system_recovery_attempts < 3:  # Limit recovery attempts
                try:
                    print(f"🔄 RECOVERY: Agent {self.id[:8]} lost learning system - attempting recovery (attempt {self._learning_system_recovery_attempts + 1})")
                    self._initialize_attention_learning()
                    self._learning_system_recovery_attempts += 1
                    
                    if self._learning_system:
                        print(f"✅ RECOVERY: Agent {self.id[:8]} successfully recovered learning system")
                    else:
                        print(f"❌ RECOVERY: Agent {self.id[:8]} failed to recover learning system")
                        return
                except Exception as e:
                    print(f"❌ RECOVERY: Agent {self.id[:8]} learning system recovery failed: {e}")
                    self._learning_system_recovery_attempts += 1
                    return
            else:
                # Too many failed attempts - just log and return
                if not hasattr(self, '_recovery_failed_logged'):
                    print(f"❌ RECOVERY FAILED: Agent {self.id[:8]} cannot recover learning system after 3 attempts")
                    self._recovery_failed_logged = True
                return
        
        # Check if memory exists
        if not hasattr(self._learning_system, 'memory'):
            if not hasattr(self, '_no_memory_warned'):
                print(f"❌ TRAINING BLOCKED: Agent {self.id[:8]} learning system has no memory attribute")
                self._no_memory_warned = True
            return
        
        if not self._learning_system.memory:
            if not hasattr(self, '_memory_none_warned'):
                print(f"❌ TRAINING BLOCKED: Agent {self.id[:8]} memory is None")
                self._memory_none_warned = True
            return
        
        # FIXED: Use len() directly on memory object instead of accessing .buffer
        # PrioritizedReplayBuffer implements __len__ method
        try:
            buffer_size = len(self._learning_system.memory)
        except Exception as e:
            if not hasattr(self, '_buffer_access_warned'):
                print(f"❌ TRAINING BLOCKED: Agent {self.id[:8]} cannot access buffer size: {e}")
                print(f"   Memory type: {type(self._learning_system.memory)}")
                print(f"   Memory attributes: {dir(self._learning_system.memory)}")
                self._buffer_access_warned = True
            return
        
        # SIMPLIFIED TRAINING LOGIC: Train more frequently for faster learning
        min_buffer_size = 32  # Need at least 32 experiences
        
        # STEP-BASED TRAINING: Much more reliable than buffer-size tracking
        # Train every N steps instead of tracking buffer size changes
        train_frequency = 5  # Train every 5 steps when buffer is full enough
        
        if buffer_size >= min_buffer_size and self.steps % train_frequency == 0:
            # Train the network
            try:
                training_stats = self._learning_system.learn()
                
                # Increment training count
                if not hasattr(self, '_training_count'):
                    self._training_count = 0
                self._training_count += 1
                
                # LOGGING: Simple training progress
                if training_stats:
                    loss = training_stats.get('loss', 0.0)
                    q_val = training_stats.get('mean_q_value', 0.0)
                    
                    # Only log occasionally to avoid spam
                    if self._training_count % 10 == 1:  # Log every 10th training session
                        print(f"🧠 Agent {self.id[:8]}: Training #{self._training_count} (attempts: {self._learning_system_recovery_attempts if hasattr(self, '_learning_system_recovery_attempts') else 0}) - Loss: {loss:.4f}, Q-val: {q_val:.3f}, Buffer: {buffer_size}")
                
            except Exception as e:
                print(f"❌ TRAINING ERROR: Agent {self.id[:8]} training failed: {e}")
                # Don't crash - just skip this training step
                import traceback
                traceback.print_exc()
        
        # DIAGNOSTIC: Log buffer status occasionally
        if self.steps % 100 == 0 and buffer_size > 0:
            print(f"📊 Agent {self.id[:8]}: Buffer {buffer_size}/{min_buffer_size}, Training every {train_frequency} steps")
    
    def get_training_count(self) -> int:
        """Get the number of training runs completed by this agent."""
        return getattr(self, '_training_count', 0)
    
    def get_q_updates(self) -> int:
        """Get the number of Q-learning updates (training runs) for compatibility with metrics."""
        return self.get_training_count()

    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information for the agent."""
        return {
            'id': self.id,
            'generation': self.generation,
            'steps': self.steps,
            'total_reward': self.total_reward,
            'current_action': self.current_action,
            'current_action_tuple': self.current_action_tuple,
            'learning_system': str(type(self._learning_system)) if self._learning_system else None,
            'physical_params': {
                'num_arms': self.physical_params.num_arms,
                'segments_per_limb': self.physical_params.segments_per_limb,
                'motor_torque': self.physical_params.motor_torque,
                'motor_speed': self.physical_params.motor_speed
            },
            'morphology': {
                'total_joints': self.physical_params.num_arms * self.physical_params.segments_per_limb,
                'action_space_size': len(self.actions) if hasattr(self, 'actions') else 0
            }
        }

    # UI COMPATIBILITY: Add backward compatibility methods that the UI and other systems expect
    def get_state(self):
        """Backward compatibility - return state as list for UI systems."""
        if self.body and hasattr(self, 'upper_arm') and hasattr(self, 'lower_arm'):
            return [
                self.body.position.x, self.body.position.y,
                self.body.linearVelocity.x, self.body.linearVelocity.y,
                self.body.angle, 
                self.upper_arm.angle if self.upper_arm else 0, 
                self.lower_arm.angle if self.lower_arm else 0
            ]
        return [0.0] * 7

    def take_action(self, action):
        """Backward compatibility for UI systems."""
        if isinstance(action, (list, tuple)) and len(action) >= 2:
            self.apply_action((float(action[0]), float(action[1])))
        elif isinstance(action, int):
            self.apply_action(action)

    def get_reward(self, prev_x: float) -> float:
        """Backward compatibility for UI systems."""
        # Simple reward calculation for UI display
        if self.body:
            current_x = self.body.position.x
            return current_x - prev_x
        return 0.0

    def update(self, delta_time: float):
        """Backward compatibility for UI systems."""
        self.step(delta_time)

    def get_fitness(self) -> float:
        """Get fitness score for evolution and UI display."""
        return self.get_evolutionary_fitness()

    def reset(self):
        """Reset agent for new episode while preserving learning."""
        self.reset_position()
        
        # Reset state tracking
        self.current_action = None
        self.current_action_tuple = None
        self.total_reward = 0.0
        self.steps = 0
        
        # Reset timing
        self.last_action_time = 0.0
        
        # Reset tracking
        self.last_positions = []

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
            
            # Reset limb segments
            if hasattr(self, 'limbs') and self.limbs:
                for limb_segments in self.limbs:
                    for segment in limb_segments:
                        if segment:
                            segment.linearVelocity = (0, 0)
                            segment.angularVelocity = 0
                            segment.awake = True
            
            # Reset wheels
            if hasattr(self, 'wheels') and self.wheels:
                for wheel in self.wheels:
                    if wheel:
                        wheel.linearVelocity = (0, 0)
                        wheel.angularVelocity = 0
                        wheel.awake = True
            
        except Exception as e:
            print(f"⚠️ Error resetting position for agent {self.id}: {e}")

    # UI COMPATIBILITY: Ensure limb references are properly set for UI rendering
    def _ensure_limb_references(self):
        """Ensure upper_arm and lower_arm references are set for UI compatibility."""
        if not hasattr(self, 'upper_arm') or not hasattr(self, 'lower_arm'):
            # Set references for UI compatibility
            self.upper_arm = self.upper_arms[0] if hasattr(self, 'upper_arms') and self.upper_arms else None
            self.lower_arm = self.lower_arms[0] if hasattr(self, 'lower_arms') and self.lower_arms else None
            
            # Set joint references for UI compatibility
            if hasattr(self, 'limb_joints') and self.limb_joints and len(self.limb_joints) > 0:
                first_limb_joints = self.limb_joints[0]
                if not hasattr(self, 'upper_arm_joint'):
                    self.upper_arm_joint = first_limb_joints[0] if len(first_limb_joints) >= 1 else None
                if not hasattr(self, 'lower_arm_joint'):
                    self.lower_arm_joint = first_limb_joints[1] if len(first_limb_joints) >= 2 else None

    def _initialize_learning_system(self):
        """Initialize learning system by getting network from Learning Manager."""
        # Initialize as None first
        self._learning_system = None
        print(f"⏳ Agent {self.id}: Deferring network creation to Learning Manager")
        
        # MULTI-ACTION SYSTEM: Very complex robots can execute multiple actions per step
        self._set_multi_action_capability()
        
        self._initialize_attention_learning()

        print(f"🧠 Created agent {self.id} with {self.physical_params.num_arms} limbs, {self.physical_params.segments_per_limb} segments each")

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
        # CRITICAL: Check if network is already assigned by Robot Memory Pool
        if hasattr(self, '_learning_system') and self._learning_system is not None:
            print(f"🧠 Agent {self.id}: Using pre-assigned network from Robot Memory Pool")
            return
        
        # CRITICAL: Learning Manager is REQUIRED - no fallback patterns
        from .learning_manager import LearningManager
        
        # Get learning manager instance from training environment
        if not hasattr(self, 'world') or not hasattr(self.world, '_training_env'):
            raise RuntimeError(f"Agent {self.id}: No training environment available for Learning Manager access")
        
        training_env = getattr(self.world, '_training_env', None)
        learning_manager = getattr(training_env, 'learning_manager', None) if training_env else None
        if not learning_manager:
            raise RuntimeError(f"Agent {self.id}: Learning Manager is required but not available in training environment")
        
        # CRITICAL FIX: Calculate correct action space size for Learning Manager
        # The action_size property might not be set yet during initialization
        total_joints = self.physical_params.num_arms * self.physical_params.segments_per_limb
        actions = self._generate_dynamic_action_space_static(total_joints)
        correct_action_size = len(actions)
        
        # 🔧 DIAGNOSTIC LOGGING: Track action space calculations
        print(f"🔧 DIAGNOSTIC: Agent {self.id[:8]} ({self.physical_params.num_arms} limbs × {self.physical_params.segments_per_limb} segments)")
        print(f"🔧 DIAGNOSTIC: {total_joints} joints → {correct_action_size} actions")
        
        # Verify action space calculation
        if correct_action_size <= 5:
            print(f"⚠️  WARNING: Agent {self.id[:8]} has very small action space ({correct_action_size} actions) - may indicate calculation error")
        
        # Use Learning Manager's pooling system - this is the ONLY way to get networks
        self._learning_system = learning_manager._acquire_attention_network(self.id, correct_action_size, self.state_size)
        if not self._learning_system:
            raise RuntimeError(f"Agent {self.id}: Learning Manager failed to provide network - cannot proceed without learning system")
        
        # 🔧 DIAGNOSTIC LOGGING: Verify network assignment
        if hasattr(self._learning_system, 'action_dim'):
            actual_action_dim = self._learning_system.action_dim
            print(f"🔧 DIAGNOSTIC: Agent {self.id[:8]} received network with {actual_action_dim} actions (requested {correct_action_size})")
            if actual_action_dim != correct_action_size:
                print(f"❌ CRITICAL: Action space mismatch! Agent {self.id[:8]} requested {correct_action_size} but got {actual_action_dim}")
        
        print(f"🧠 Agent {self.id}: Got attention network from Learning Manager pool")

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
        FIXED: Use standardized 29-dimensional state representation for neural network compatibility.
        
        This ensures ALL agents (evolutionary and standard) use the same state format
        to prevent neural network dimension mismatch errors.
        """
        try:
            # FIXED: Always return 29 dimensions for neural network compatibility
            state_array = np.zeros(29, dtype=np.float32)
            
            # DEBUG: Minimal state generation confirmation (only once per agent)
            if not hasattr(self, '_state_gen_confirmed'):
                print(f"✅ Agent {str(self.id)[:8]}: Using 29D evolutionary state representation")
                self._state_gen_confirmed = True
            
            # 1. Joint angles and velocities (4 values) - Use primary joints
            if hasattr(self, 'limb_joints') and self.limb_joints and len(self.limb_joints) > 0:
                # Use the first limb's first two joints (primary arm)
                primary_limb = self.limb_joints[0]
                if len(primary_limb) >= 1 and primary_limb[0]:
                    state_array[0] = np.tanh(primary_limb[0].angle)  # First joint angle
                    state_array[2] = np.tanh(primary_limb[0].motorSpeed / 10.0)  # First joint velocity
                if len(primary_limb) >= 2 and primary_limb[1]:
                    state_array[1] = np.tanh(primary_limb[1].angle)  # Second joint angle  
                    state_array[3] = np.tanh(primary_limb[1].motorSpeed / 10.0)  # Second joint velocity
            
            # 2. Body physics state (6 values)
            if hasattr(self, 'body') and self.body:
                # Position normalized relative to starting position
                state_array[4] = np.tanh((self.body.position.x - self.initial_position[0]) / 20.0)
                state_array[5] = np.tanh((self.body.position.y - self.initial_position[1]) / 10.0)
                # Velocity
                state_array[6] = np.tanh(self.body.linearVelocity.x / 5.0)
                state_array[7] = np.tanh(self.body.linearVelocity.y / 5.0)
                # Orientation
                state_array[8] = np.tanh(self.body.angle * 2.0)
                state_array[9] = np.tanh(self.body.angularVelocity / 3.0)
            
            # 3. Food targeting information (4 values) - Placeholder for now
            # TODO: Connect to actual food system when available
            state_array[10:14] = 0.0
            
            # 4. Environmental feedback (3 values)
            # Ground contact detection  
            ground_contact = 0.0
            if hasattr(self, 'body') and self.body:
                for contact_edge in self.body.contacts:
                    if contact_edge.contact and contact_edge.contact.touching:
                        fixture_a = contact_edge.contact.fixtureA
                        fixture_b = contact_edge.contact.fixtureB
                        if ((fixture_a.filterData.categoryBits & 0x0001) or 
                            (fixture_b.filterData.categoryBits & 0x0001)):
                            ground_contact = 1.0
                            break
            
            state_array[14] = ground_contact
            state_array[15] = 0.0  # arm_contact placeholder
            
            # Stability measure
            if hasattr(self, 'body') and self.body:
                stability = 1.0 / (1.0 + abs(self.body.angle) + abs(self.body.angularVelocity))
                state_array[16] = stability
            
            # 5. Action history (2 values) - recent actions for temporal context
            if hasattr(self, 'current_action_tuple') and self.current_action_tuple:
                state_array[17] = self.current_action_tuple[0] if len(self.current_action_tuple) > 0 else 0.0
                state_array[18] = self.current_action_tuple[1] if len(self.current_action_tuple) > 1 else 0.0
            
            # 6. Ray sensing data (10 values: 5 rays × 2 values each)
            # Indices 19-28 for distance and object type of each ray
            for i in range(5):  # 5 rays
                distance_idx = 19 + (i * 2)      # Indices 19, 21, 23, 25, 27
                object_type_idx = 19 + (i * 2) + 1  # Indices 20, 22, 24, 26, 28
                
                # Placeholder ray data (TODO: implement actual ray casting)
                state_array[distance_idx] = 1.0      # Normalized max distance (no obstacle)
                state_array[object_type_idx] = 0.0   # Clear path
            
            # Ensure no NaN values
            state_array = np.nan_to_num(state_array, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return state_array
                
        except Exception as e:
            print(f"⚠️ Error getting evolutionary state for agent {self.id}: {e}")
            # Safe fallback: return zero-filled 29-dimensional state
            return np.zeros(29, dtype=np.float32)

    def choose_action(self, state: Optional[np.ndarray] = None) -> int:
        """Choose action using attention-based learning system."""
        try:
            # Use provided state or get current state
            if state is None:
                state = self.get_state_representation()
            
            if self._learning_system:
                action = self._learning_system.choose_action(state)
                return max(0, min(action, len(self.actions) - 1))
            else:
                return random.randint(0, len(self.actions) - 1)
            
        except Exception as e:
            print(f"❌ Error choosing evolutionary action for agent {self.id}: {e}")
            return random.randint(0, len(self.actions) - 1)



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
        """Create offspring through crossover and mutation with proper network handling."""
        # Crossover physical parameters (may create new morphology)
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
        
        # CRITICAL FIX: Only transfer network weights if action spaces match
        try:
            if (hasattr(self, '_learning_system') and self._learning_system and 
                hasattr(child, '_learning_system') and child._learning_system):
                
                # Check if action spaces are compatible  
                parent_action_size = getattr(self._learning_system, 'action_dim', 0)
                child_action_size = getattr(child._learning_system, 'action_dim', 0)
                
                if parent_action_size == child_action_size:
                    # Safe to transfer - action spaces match
                    child._learning_system.q_network.load_state_dict(self._learning_system.q_network.state_dict())
                    child._learning_system.target_network.load_state_dict(self._learning_system.target_network.state_dict())
                    
                    # Transfer learning parameters with some variation
                    child._learning_system.epsilon = min(0.9, self._learning_system.epsilon * 1.05)  # Slightly more exploration
                    child._learning_system.steps_done = max(0, self._learning_system.steps_done - 50)  # Reset some progress
                    
                    print(f"🧠 Transferred attention network: {self.id[:6]} → {child.id[:6]} (action_size: {parent_action_size})")
                else:
                    # Action spaces don't match - let child use fresh network from Learning Manager
                    print(f"🔄 Child {child.id[:6]} has different action space ({parent_action_size} → {child_action_size}) - using fresh network")
            else:
                print(f"⚠️ Cannot transfer network in crossover - one or both agents missing learning system")
                
        except Exception as e:
            print(f"⚠️ Error transferring attention learning in crossover: {e}")
            # Child will use its fresh network from Learning Manager
        
        child.crossover_count = 1
        return child

    def clone_with_mutation(self, mutation_rate: float = 0.1) -> 'EvolutionaryCrawlingAgent':
        """Create a mutated clone with proper network handling for different morphologies."""
        # Mutate physical parameters (may change morphology)
        mutated_params = self.physical_params.mutate(mutation_rate)
        
        # Create lineage tracking
        child_lineage = self.parent_lineage + [self.id]
        
        # Create cloned agent with potentially different morphology
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
        
        # CRITICAL FIX: Only transfer network weights if action spaces match
        try:
            if (hasattr(self, '_learning_system') and self._learning_system and 
                hasattr(clone, '_learning_system') and clone._learning_system):
                
                # Check if action spaces are compatible
                parent_action_size = getattr(self._learning_system, 'action_dim', 0)
                clone_action_size = getattr(clone._learning_system, 'action_dim', 0)
                
                if parent_action_size == clone_action_size:
                    # Safe to transfer - action spaces match
                    clone._learning_system.q_network.load_state_dict(self._learning_system.q_network.state_dict())
                    clone._learning_system.target_network.load_state_dict(self._learning_system.target_network.state_dict())
                    
                    # Transfer learning parameters
                    clone._learning_system.epsilon = self._learning_system.epsilon * 1.1  # Slightly more exploration
                    clone._learning_system.steps_done = max(0, self._learning_system.steps_done - 100)  # Reset some progress
                    
                    print(f"🧠 Cloned attention network: {self.id[:6]} → {clone.id[:6]} (action_size: {parent_action_size})")
                else:
                    # Action spaces don't match - let clone use fresh network from Learning Manager
                    print(f"🔄 Clone {clone.id[:6]} has different action space ({parent_action_size} → {clone_action_size}) - using fresh network")
            else:
                print(f"⚠️ Cannot transfer network in cloning - one or both agents missing learning system")
                
        except Exception as e:
            print(f"⚠️ Error in network transfer during cloning: {e}")
            # Clone will use its fresh network from Learning Manager
        
        clone.mutation_count = self.mutation_count + 1
        return clone



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
                    if self.body:
                        segment_pos = (
                            self.body.position[0] + arm_attach_x - segment_length * np.cos(self.physical_params.arm_angle_offset),
                            self.body.position[1] + arm_attach_y + segment_length * np.sin(self.physical_params.arm_angle_offset)
                        )
                    else:
                        segment_pos = (arm_attach_x, arm_attach_y)
                else:
                    if prev_body:
                        segment_pos = (
                            prev_body.position[0] + segment_length,
                            prev_body.position[1]
                        )
                    else:
                        segment_pos = (0, 0)
                
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
                world_pos = self.body.GetWorldPoint(local_pos) if self.body else local_pos
                
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
        """
        Apply continuous action tuple to the robot's joints with enhanced coordination.
        
        This improved version:
        1. Supports continuous action values (not just binary)
        2. Maintains full motor effectiveness for complex robots
        3. Adds coordination smoothing between joints
        4. Provides better diagnostic information
        """
        # DIAGNOSTIC LOGGING: Check for critical failures
        if not hasattr(self, 'limb_joints') or not self.limb_joints:
            print(f"❌ CRITICAL: Agent {self.id[:8]} has no limb_joints!")
            return
        
        if not action_tuple:
            print(f"❌ CRITICAL: Agent {self.id[:8]} received empty action_tuple!")
            return
        
        # Log first action application for debugging
        if not hasattr(self, '_action_debug_logged'):
            print(f"🔧 DEBUG: Agent {self.id[:8]} first continuous action application:")
            print(f"   💪 Limb joints: {len(self.limb_joints)} limbs")
            print(f"   🎯 Action tuple: {action_tuple}")
            print(f"   📊 Action range: {min(action_tuple):.3f} to {max(action_tuple):.3f}")
            self._action_debug_logged = True
        
        joints_activated = 0
        joint_index = 0
        total_joints = self.physical_params.num_arms * self.physical_params.segments_per_limb
        
        # Track joint coordination for smoother movement
        joint_activations = []
        
        for limb_idx, limb_joints in enumerate(self.limb_joints):
            limb_activations = []
            
            for segment_idx, joint in enumerate(limb_joints):
                if joint_index < len(action_tuple) and joint:
                    action_value = float(action_tuple[joint_index])
                    
                    # Get joint parameters
                    if segment_idx < len(self.physical_params.joint_torques):
                        base_torque = self.physical_params.joint_torques[segment_idx]
                        base_speed = self.physical_params.joint_speeds[segment_idx]
                    else:
                        base_torque = self.physical_params.motor_torque
                        base_speed = self.physical_params.motor_speed
                    
                    # ENHANCED: Apply continuous action value directly (no binary constraints)
                    # Removed previous stability fixes that reduced effectiveness
                    if abs(action_value) > 0.01:  # Small threshold to avoid micro-movements
                        # Scale motor speed by action value (continuous control)
                        motor_speed = action_value * base_speed
                        
                        # Apply torque scaling for very small actions (fine control)
                        if abs(action_value) < 0.3:
                            # Reduce torque for fine movements to prevent instability
                            torque_multiplier = 0.5 + (abs(action_value) / 0.3) * 0.5
                            torque = base_torque * torque_multiplier
                        else:
                            # Full torque for strong movements
                            torque = base_torque
                        
                        # Apply to joint
                        joint.motorSpeed = motor_speed
                        joint.maxMotorTorque = torque
                        joint.enableMotor = True
                        joints_activated += 1
                        
                        limb_activations.append(abs(action_value))
                        
                        # Log detailed activation for first few times
                        if not hasattr(self, '_joint_activation_logged'):
                            print(f"🔧 DEBUG: Agent {self.id[:8]} activated joint {joint_index}")
                            print(f"   ⚡ Continuous action: {action_value:.3f}")
                            print(f"   🚀 Motor speed: {motor_speed:.3f}")
                            print(f"   💪 Torque: {torque:.1f}")
                    else:
                        # Disable motor for very small actions
                        joint.motorSpeed = 0.0
                        joint.enableMotor = False
                        limb_activations.append(0.0)
                
                joint_index += 1
            
            joint_activations.append(limb_activations)
        
        # COORDINATION ANALYSIS: Track coordination quality
        if joints_activated > 0:
            # Calculate coordination metrics
            coordination_score = self._calculate_coordination_score(joint_activations)
            
            # Store coordination data for reward system
            if not hasattr(self, 'coordination_history'):
                self.coordination_history = []
            
            self.coordination_history.append({
                'joints_activated': joints_activated,
                'coordination_score': coordination_score,
                'action_variance': float(np.var(action_tuple)) if len(action_tuple) > 1 else 0.0,
                'total_energy': sum(abs(a) for a in action_tuple)
            })
            
            # Keep only recent coordination data
            if len(self.coordination_history) > 100:
                self.coordination_history = self.coordination_history[-50:]
            
            # Mark first successful continuous action
            if not hasattr(self, '_joint_activation_logged'):
                print(f"✅ Agent {self.id[:8]} continuous action applied:")
                print(f"   🎯 {joints_activated}/{total_joints} joints activated")
                print(f"   🤝 Coordination score: {coordination_score:.3f}")
                self._joint_activation_logged = True
        
        # MOVEMENT DETECTION: Enhanced tracking
        self._update_movement_tracking(joints_activated, action_tuple)
        
        # DIAGNOSTICS: Alert for potential issues
        if joints_activated == 0 and max(abs(a) for a in action_tuple) > 0.1:
            if not hasattr(self, '_no_joints_warned'):
                print(f"⚠️  WARNING: Agent {self.id[:8]} strong action but NO joints activated!")
                print(f"   🎯 Action range: {min(action_tuple):.3f} to {max(action_tuple):.3f}")
                self._no_joints_warned = True
    
    def _calculate_coordination_score(self, joint_activations: List[List[float]]) -> float:
        """Calculate coordination quality score for joint activations."""
        if not joint_activations:
            return 0.0
        
        total_score = 0.0
        scored_limbs = 0
        
        for limb_activations in joint_activations:
            if len(limb_activations) <= 1:
                continue
            
            # Proximal-to-distal coordination (natural movement pattern)
            proximal_distal_score = 0.0
            for i in range(len(limb_activations) - 1):
                proximal = limb_activations[i]
                distal = limb_activations[i + 1]
                
                # Reward gradual decrease in activation from proximal to distal
                if proximal > 0.1 and distal > 0.1:
                    if proximal >= distal * 0.7:  # Proximal should be stronger or similar
                        proximal_distal_score += 0.5
                    
                    # Reward smooth activation differences
                    activation_diff = abs(proximal - distal)
                    if activation_diff < 0.4:  # Smooth transition
                        proximal_distal_score += 0.3
            
            total_score += proximal_distal_score
            scored_limbs += 1
        
        return total_score / max(1, scored_limbs)
    
    def _update_movement_tracking(self, joints_activated: int, action_tuple: Tuple):
        """Update movement tracking with enhanced metrics."""
        if not hasattr(self, 'movement_activity'):
            self.movement_activity = []
        
        self.movement_activity.append({
            'timestamp': time.time(),
            'joints_activated': joints_activated,
            'action_tuple': action_tuple,
            'action_energy': sum(abs(a) for a in action_tuple),
            'max_action': max(abs(a) for a in action_tuple),
            'coordination_quality': getattr(self, '_last_coordination_score', 0.0)
        })
        
        # Keep only recent activity
        if len(self.movement_activity) > 100:
            self.movement_activity = self.movement_activity[-50:]

    def _generate_dynamic_action_space(self) -> List[Tuple]:
        """Generate action space based on robot morphology."""
        total_joints = self.physical_params.num_arms * self.physical_params.segments_per_limb
        return self._generate_dynamic_action_space_static(total_joints)
    
    @staticmethod
    def _generate_dynamic_action_space_static(total_joints: int) -> List[Tuple]:
        """
        Generate enhanced action space with continuous values and coordination patterns.
        
        This new approach provides:
        1. Continuous joint speeds (not just binary)
        2. Sophisticated coordination patterns
        3. Morphology-aware action combinations
        4. Progressive complexity scaling
        """
        actions = []
        
        # 1. BASELINE ACTIONS
        # Rest position (all joints at 0)
        actions.append((0.0,) * total_joints)
        
        # 2. SINGLE JOINT CONTINUOUS ACTIONS
        # Use multiple speed levels for each joint
        speed_levels = [0.25, 0.5, 0.75, 1.0]  # Fine-grained control
        for joint_idx in range(total_joints):
            for speed in speed_levels:
                # Positive direction
                action = [0.0] * total_joints
                action[joint_idx] = speed
                actions.append(tuple(action))
                
                # Negative direction
                action = [0.0] * total_joints
                action[joint_idx] = -speed
                actions.append(tuple(action))
        
        # 3. COORDINATED LIMB PATTERNS
        if total_joints >= 2:
            # Proximal-to-distal waves (natural movement patterns)
            for intensity in [0.5, 0.8]:
                # Forward wave
                wave_action = []
                for i in range(total_joints):
                    wave_strength = intensity * (1.0 - i * 0.1)  # Decreasing intensity
                    wave_action.append(max(0.1, wave_strength))
                actions.append(tuple(wave_action))
                
                # Backward wave
                wave_action = []
                for i in range(total_joints):
                    wave_strength = intensity * (1.0 - i * 0.1)
                    wave_action.append(-max(0.1, wave_strength))
                actions.append(tuple(wave_action))
        
        # 4. ALTERNATING PATTERNS (like walking gaits)
        if total_joints >= 4:
            # Alternating limb activation
            for base_speed in [0.4, 0.7]:
                # Even joints forward, odd joints backward
                alt_action = []
                for i in range(total_joints):
                    if i % 2 == 0:
                        alt_action.append(base_speed)
                    else:
                        alt_action.append(-base_speed * 0.8)
                actions.append(tuple(alt_action))
                
                # Reverse alternating pattern
                alt_action = []
                for i in range(total_joints):
                    if i % 2 == 0:
                        alt_action.append(-base_speed)
                    else:
                        alt_action.append(base_speed * 0.8)
                actions.append(tuple(alt_action))
        
        # 5. LIMB-SPECIFIC COORDINATION
        # For multi-limb robots, coordinate within each limb
        if total_joints >= 6:  # Multiple limbs
            segments_per_limb = 3 if total_joints >= 9 else 2
            num_limbs = total_joints // segments_per_limb
            
            for limb_idx in range(min(num_limbs, 3)):  # Max 3 limbs for action space
                # Limb-specific actions
                limb_action = [0.0] * total_joints
                start_joint = limb_idx * segments_per_limb
                end_joint = min(start_joint + segments_per_limb, total_joints)
                
                # Coordinated limb extension
                for joint_offset in range(end_joint - start_joint):
                    joint_idx = start_joint + joint_offset
                    # Proximal joints stronger than distal
                    strength = 0.8 - (joint_offset * 0.15)
                    limb_action[joint_idx] = strength
                actions.append(tuple(limb_action))
                
                # Coordinated limb contraction
                limb_action = [0.0] * total_joints
                for joint_offset in range(end_joint - start_joint):
                    joint_idx = start_joint + joint_offset
                    strength = 0.8 - (joint_offset * 0.15)
                    limb_action[joint_idx] = -strength
                actions.append(tuple(limb_action))
        
        # 6. STABILIZATION PATTERNS
        # Gentle all-joint coordination for stability
        for overall_intensity in [0.2, 0.4]:
            # All joints slight forward
            actions.append((overall_intensity,) * total_joints)
            # All joints slight backward
            actions.append((-overall_intensity,) * total_joints)
        
        print(f"📈 Generated {len(actions)} continuous actions for {total_joints} joints")
        return actions
        

        
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

    def _get_coordination_reward(self, displacement: float) -> float:
        """Calculate advanced coordination reward for multi-limb robots."""
        try:
            if not hasattr(self, 'limb_joints') or not self.limb_joints:
                return 0.0  # No coordination reward without joints
            
            total_coordination_reward = 0.0
            total_joints = self.physical_params.num_arms * self.physical_params.segments_per_limb
            
            # 1. INTER-LIMB COORDINATION REWARD
            # Reward for coordinating multiple limbs together
            if self.physical_params.num_arms > 1:
                coordination_reward = 0.0
                
                # Compare activity across limbs
                limb_activities = []
                segments_per_limb = self.physical_params.segments_per_limb
                
                for limb_idx in range(self.physical_params.num_arms):
                    limb_activity = 0.0
                    start_joint = limb_idx * segments_per_limb
                    end_joint = min(start_joint + segments_per_limb, len(self.limb_joints[0]) if self.limb_joints else 0)
                    
                    limb_joints = self.limb_joints[limb_idx] if limb_idx < len(self.limb_joints) else []
                    for joint_idx in range(len(limb_joints)):
                        joint = limb_joints[joint_idx]
                        if joint:
                            limb_activity += abs(joint.motorSpeed)
                    
                    limb_activities.append(limb_activity)
                
                # Reward for balanced limb usage
                if limb_activities and max(limb_activities) > 0.1:
                    activity_balance = 1.0 - (max(limb_activities) - min(limb_activities)) / max(limb_activities)
                    coordination_reward += activity_balance * 0.015  # Max 0.015
                
                total_coordination_reward += coordination_reward
            
            # 2. ENHANCED COORDINATION REWARD
            # Use the new sophisticated coordination reward system
            enhanced_coordination_reward = self._get_enhanced_coordination_reward(displacement)
            total_coordination_reward += enhanced_coordination_reward
            
            # 3. PROGRESSIVE MOVEMENT REWARD
            # Replace simple displacement with progressive reward
            progressive_movement_reward = self._get_progressive_movement_reward(displacement)
            total_coordination_reward += progressive_movement_reward
            
            # 4. MULTI-LIMB BONUS
            # Extra reward for multi-limb robots that achieve good coordination
            if self.physical_params.num_arms > 1 and displacement > 0.01:
                multi_limb_bonus = min(0.02, self.physical_params.num_arms * 0.005)
                total_coordination_reward += multi_limb_bonus
            
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
            
            # Simple fallback calculation
            return min(0.05, displacement * 0.01)
            
        except Exception as e:
            return 0.0 

    def _get_enhanced_coordination_reward(self, displacement: float) -> float:
        """
        Enhanced coordination reward system that promotes effective multi-limb movement.
        
        This reward system specifically addresses the joint activation problem by:
        1. Rewarding coordinated joint usage patterns
        2. Penalizing inefficient energy expenditure
        3. Promoting smooth, progressive movement
        4. Encouraging exploration of complex coordination patterns
        """
        try:
            if not hasattr(self, 'coordination_history') or not self.coordination_history:
                return 0.0
            
            total_coordination_reward = 0.0
            recent_coordination = self.coordination_history[-5:]  # Last 5 coordination events
            
            # 1. JOINT ACTIVATION EFFICIENCY REWARD
            # Reward for activating an appropriate number of joints
            if recent_coordination:
                avg_joints_activated = sum(c['joints_activated'] for c in recent_coordination) / len(recent_coordination)
                total_joints = self.physical_params.num_arms * self.physical_params.segments_per_limb
                
                # Optimal joint usage ratio depends on robot complexity
                if total_joints <= 4:  # Simple robots
                    optimal_ratio = 0.5  # Use about half the joints
                elif total_joints <= 8:  # Medium complexity
                    optimal_ratio = 0.4  # Use 40% of joints
                else:  # Complex robots
                    optimal_ratio = 0.3  # Use 30% of joints efficiently
                
                actual_ratio = avg_joints_activated / total_joints
                
                # Reward for being near optimal ratio
                ratio_diff = abs(actual_ratio - optimal_ratio)
                if ratio_diff < 0.2:  # Within 20% of optimal
                    efficiency_reward = 0.02 * (0.2 - ratio_diff) * 5  # Max 0.02 reward
                    total_coordination_reward += efficiency_reward
            
            # 2. COORDINATION QUALITY REWARD
            # Reward for smooth, coordinated movement patterns
            if recent_coordination:
                avg_coordination_score = sum(c['coordination_score'] for c in recent_coordination) / len(recent_coordination)
                
                # Scale reward by coordination quality
                coordination_quality_reward = avg_coordination_score * 0.03  # Max 0.03 for perfect coordination
                total_coordination_reward += coordination_quality_reward
            
            # 3. ENERGY EFFICIENCY REWARD
            # Reward for achieving displacement with reasonable energy expenditure
            if displacement > 0.001 and recent_coordination:
                total_energy = sum(c['total_energy'] for c in recent_coordination)
                avg_energy = total_energy / len(recent_coordination)
                
                # Calculate energy efficiency: movement per unit energy
                if avg_energy > 0.1:  # Avoid division by zero
                    energy_efficiency = displacement / avg_energy
                    
                    # Reward efficiency, but cap it to prevent exploitation
                    efficiency_reward = min(0.025, energy_efficiency * 0.01)
                    total_coordination_reward += efficiency_reward
            
            # 4. MOVEMENT PROGRESSION REWARD
            # Extra reward for consistent forward movement with good coordination
            if len(recent_coordination) >= 3:
                # Check if coordination is improving over time
                early_coord = sum(c['coordination_score'] for c in recent_coordination[:2]) / 2
                late_coord = sum(c['coordination_score'] for c in recent_coordination[-2:]) / 2
                
                if late_coord > early_coord + 0.1:  # Coordination improving
                    improvement_reward = 0.015  # Small bonus for improvement
                    total_coordination_reward += improvement_reward
            
            # 5. COMPLEXITY BONUS
            # Bonus for successfully coordinating complex morphologies
            total_joints = self.physical_params.num_arms * self.physical_params.segments_per_limb
            if total_joints > 6 and displacement > 0.01:  # Complex robots achieving movement
                complexity_bonus = min(0.02, (total_joints - 6) * 0.002)  # Scale with complexity
                total_coordination_reward += complexity_bonus
            
            # 6. ANTI-FLAILING PENALTY
            # Penalize excessive action variance without effective movement
            if recent_coordination:
                avg_variance = sum(c['action_variance'] for c in recent_coordination) / len(recent_coordination)
                
                # High variance with low displacement suggests flailing
                if avg_variance > 0.3 and displacement < 0.005:
                    flailing_penalty = -0.01 * (avg_variance - 0.3)
                    total_coordination_reward += flailing_penalty
            
            # Ensure reasonable bounds
            total_coordination_reward = max(-0.05, min(0.08, total_coordination_reward))
            
            return total_coordination_reward
            
        except Exception as e:
            # Don't let coordination reward calculation break the system
            return 0.0
    
    def _get_progressive_movement_reward(self, displacement: float) -> float:
        """
        Progressive movement reward that adapts to robot complexity and encourages learning.
        
        This reward system:
        1. Provides stronger initial rewards for any movement
        2. Gradually increases standards as robot improves
        3. Adapts to robot morphology complexity
        4. Encourages sustained forward progress
        """
        try:
            if displacement <= 0:
                return 0.0
            
            # Base movement reward - stronger for initial learning
            base_reward = displacement * 8.0  # Increased from 5.0
            
            # 1. COMPLEXITY ADAPTATION
            # Simpler robots should move more easily, complex robots get bonus for any movement
            total_joints = self.physical_params.num_arms * self.physical_params.segments_per_limb
            
            if total_joints <= 4:  # Simple robots
                complexity_multiplier = 1.0  # Standard reward
            elif total_joints <= 8:  # Medium complexity
                complexity_multiplier = 1.2  # 20% bonus for medium complexity
            else:  # Complex robots
                complexity_multiplier = 1.5  # 50% bonus for complex robots
            
            base_reward *= complexity_multiplier
            
            # 2. LEARNING STAGE ADAPTATION
            # Provide stronger rewards early in learning
            if hasattr(self, 'steps') and self.steps > 0:
                learning_stage_multiplier = max(0.5, 1.0 - (self.steps / 10000.0))  # Reduce over time
                base_reward *= (1.0 + learning_stage_multiplier)
            
            # 3. SUSTAINED MOVEMENT BONUS
            # Reward for maintaining movement over time
            if hasattr(self, 'last_positions') and len(self.last_positions) >= 5:
                recent_displacements = []
                for i in range(1, min(6, len(self.last_positions))):
                    recent_disp = self.last_positions[-i][0] - self.last_positions[-i-1][0]
                    recent_displacements.append(recent_disp)
                
                if recent_displacements:
                    avg_recent_displacement = sum(recent_displacements) / len(recent_displacements)
                    if avg_recent_displacement > 0.001:  # Sustained forward movement
                        sustained_bonus = min(0.02, avg_recent_displacement * 2.0)
                        base_reward += sustained_bonus
            
            # 4. VELOCITY CONSISTENCY REWARD
            # Reward for maintaining good velocity
            if hasattr(self, 'body') and self.body:
                current_velocity = self.body.linearVelocity.x
                if current_velocity > 0.1:
                    velocity_bonus = min(0.015, current_velocity * 0.1)
                    base_reward += velocity_bonus
            
            return base_reward
            
        except Exception as e:
            # Fallback to simple displacement reward
            return displacement * 5.0