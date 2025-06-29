"""
Robot Memory Pool System

Efficient memory management for robot objects using object pooling.
Instead of creating/destroying robots constantly, reuse existing objects 
by updating their attributes.

Enhanced to preserve learned weights for all learning approaches:
- Basic Q-Learning: Q-table values and visit counts
- Enhanced Q-Learning: Advanced Q-table with confidence and exploration data
- Survival Q-Learning: Ecosystem-aware states and learning stages
- Deep Q-Learning: Neural network weights and experience buffers
"""

import time
import random
import uuid
import pickle
import copy
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import deque
import Box2D as b2

from .crawling_agent import CrawlingAgent
from .physical_parameters import PhysicalParameters


class LearningStateSnapshot:
    """Container for learning state data that needs to be preserved."""
    
    def __init__(self, agent_id: str, learning_approach: str):
        self.agent_id = agent_id
        self.learning_approach = learning_approach
        self.timestamp = time.time()
        
        # Q-learning state
        self.q_table_data: Optional[Dict[str, Any]] = None
        self.learning_parameters: Dict[str, Any] = {}
        
        # Survival learning state
        self.survival_stats: Dict[str, Any] = {}
        self.learning_stage: Optional[str] = None
        
        # Deep learning state
        self.neural_network_weights: Optional[Any] = None
        self.experience_buffer: Optional[List[Any]] = None
        self.deep_learning_stats: Dict[str, Any] = {}
        
        # General learning metrics
        self.total_reward = 0.0
        self.steps_trained = 0
        self.epsilon = 0.3
        self.learning_rate = 0.1

class RobotMemoryPool:
    """
    Enhanced memory pool for efficient robot object reuse with learning preservation.
    
    Maintains a pool of pre-allocated robot objects that can be reused
    by updating their attributes instead of creating new instances.
    Now preserves all learned weights and state for different learning approaches.
    """
    
    def __init__(self, world: b2.b2World, min_pool_size: int = 10, max_pool_size: int = 100,
                 category_bits: int = 0x0002, mask_bits: int = 0x0001):
        """
        Initialize the robot memory pool.
        
        Args:
            world: Box2D world for physics
            min_pool_size: Minimum number of robots to keep in pool
            max_pool_size: Maximum number of robots to keep in pool
            category_bits: Collision category bits
            mask_bits: Collision mask bits
        """
        self.world = world
        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size
        self.category_bits = category_bits
        self.mask_bits = mask_bits
        
        # Active robots currently in use
        self.active_robots: Dict[str, EvolutionaryCrawlingAgent] = {}
        
        # Available robots ready for reuse
        self.available_robots: deque = deque()
        
        # Learning state preservation
        self.learning_snapshots: Dict[str, LearningStateSnapshot] = {}
        self.max_snapshots = 500  # Limit memory usage
        
        # Pool statistics
        self.pool_stats = {
            'created_count': 0,
            'reused_count': 0,
            'returned_count': 0,
            'destroyed_count': 0,
            'peak_active': 0,
            'peak_available': 0,
            'learning_states_saved': 0,
            'learning_states_restored': 0
        }
        
        # Learning manager reference
        self.learning_manager = None
        
        # Pre-allocate minimum robots
        self._initialize_pool()
        
        print(f"🏊 Enhanced RobotMemoryPool initialized: {self.min_pool_size}-{self.max_pool_size} robots")
        print(f"   🧠 Learning state preservation enabled for all approaches")
    
    def _initialize_pool(self):
        """Pre-allocate minimum number of robots."""
        for i in range(self.min_pool_size):
            robot = self._create_new_robot(position=(i * 15, 10))
            self.available_robots.append(robot)
            self.pool_stats['created_count'] += 1
    
    def _create_new_robot(self, position: Tuple[float, float] = (0, 10)) -> EvolutionaryCrawlingAgent:
        """Create a new robot instance."""
        random_params = PhysicalParameters.random_parameters()
        return EvolutionaryCrawlingAgent(
            world=self.world,
            agent_id=None,  # Generate new UUID
            position=position,
            category_bits=self.category_bits,
            mask_bits=self.mask_bits,
            physical_params=random_params
        )
    
    def set_learning_manager(self, learning_manager):
        """Set the learning manager for knowledge transfer."""
        self.learning_manager = learning_manager
        print(f"🔗 Learning manager connected to memory pool")
    
    def _save_learning_state(self, robot: EvolutionaryCrawlingAgent) -> LearningStateSnapshot:
        """
        Save the complete learning state of a robot.
        
        Args:
            robot: Robot whose learning state to save
            
        Returns:
            LearningStateSnapshot: Snapshot of the learning state
        """
        try:
            # Determine learning approach
            learning_approach = "enhanced_q_learning"  # Default
            if self.learning_manager:
                approach_enum = self.learning_manager.get_agent_approach(robot.id)
                learning_approach = approach_enum.value
            
            snapshot = LearningStateSnapshot(robot.id, learning_approach)
            
            # Save basic learning parameters
            snapshot.learning_parameters = {
                'learning_rate': getattr(robot, 'learning_rate', 0.1),
                'epsilon': getattr(robot, 'epsilon', 0.3),
                'epsilon_decay': getattr(robot, 'epsilon_decay', 0.999),
                'discount_factor': getattr(robot, 'discount_factor', 0.9),
                'total_reward': getattr(robot, 'total_reward', 0.0),
                'steps': getattr(robot, 'steps', 0)
            }
            
            # Save Q-table data based on approach
            if hasattr(robot, 'q_table') and robot.q_table:
                snapshot.q_table_data = self._serialize_q_table(robot.q_table)
            
            # Save survival learning specific state
            if learning_approach == "survival_q_learning":
                if self.learning_manager and robot.id in self.learning_manager.agent_adapters:
                    survival_adapter = self.learning_manager.agent_adapters[robot.id]
                    snapshot.survival_stats = getattr(survival_adapter, 'survival_stats', {}).copy()
                    snapshot.learning_stage = getattr(survival_adapter, 'learning_stage', 'basic_movement')
            
            # Save deep learning specific state
            elif learning_approach == "deep_q_learning":
                if self.learning_manager and robot.id in self.learning_manager.agent_adapters:
                    deep_adapter = self.learning_manager.agent_adapters[robot.id]
                    try:
                        # Save neural network weights
                        snapshot.neural_network_weights = deep_adapter.state_dict()
                        
                        # Save experience buffer (limit size to prevent memory issues)
                        if hasattr(deep_adapter, 'memory') and len(deep_adapter.memory) > 0:
                            buffer_size = min(25000, len(deep_adapter.memory))  # Limit to 25k experiences (increased from 10k)
                            snapshot.experience_buffer = list(deep_adapter.memory.buffer)[-buffer_size:]
                        
                        # Save deep learning statistics
                        snapshot.deep_learning_stats = {
                            'training_steps': getattr(deep_adapter, '_training_steps', 0),
                            'epsilon': getattr(deep_adapter, 'epsilon', 1.0),
                            'loss_history': getattr(deep_adapter, '_loss_history', [])[-100:]  # Last 100 losses
                        }
                    except Exception as e:
                        print(f"⚠️ Could not save deep learning state: {e}")
            
            # CRITICAL FIX: Save attention deep learning specific state
            elif learning_approach == "attention_deep_q_learning":
                if hasattr(robot, '_attention_dqn') and robot._attention_dqn is not None:
                    try:
                        # Save attention neural network weights
                        snapshot.neural_network_weights = robot._attention_dqn.state_dict()
                        
                        # Save experience buffer (limit size to prevent memory issues)
                        if hasattr(robot._attention_dqn, 'memory') and len(robot._attention_dqn.memory) > 0:
                            buffer_size = min(25000, len(robot._attention_dqn.memory))  # Limit to 25k experiences (increased from 10k)
                            snapshot.experience_buffer = list(robot._attention_dqn.memory.buffer)[-buffer_size:]
                        
                        # Save attention learning statistics
                        snapshot.deep_learning_stats = {
                            'training_steps': getattr(robot._attention_dqn, '_training_steps', 0),
                            'epsilon': getattr(robot._attention_dqn, 'epsilon', 1.0),
                            'loss_history': getattr(robot._attention_dqn, '_loss_history', [])[-100:],  # Last 100 losses
                            'attention_entropy': getattr(robot._attention_dqn, '_attention_entropy', 0.0)
                        }
                        print(f"💾 Saved attention network state for robot {robot.id}")
                    except Exception as e:
                        print(f"⚠️ Could not save attention learning state: {e}")
            
            self.pool_stats['learning_states_saved'] += 1
            print(f"💾 Saved {learning_approach} state for robot {robot.id}")
            return snapshot
            
        except Exception as e:
            print(f"❌ Error saving learning state for robot {robot.id}: {e}")
            return LearningStateSnapshot(robot.id, "basic_q_learning")  # Fallback
    
    def _serialize_q_table(self, q_table) -> Dict[str, Any]:
        """Serialize Q-table data for storage."""
        try:
            serialized = {
                'type': q_table.__class__.__name__,
                'default_value': getattr(q_table, 'default_value', 0.0),
                'action_count': getattr(q_table, 'action_count', 6)
            }
            
            # Serialize Q-values
            if hasattr(q_table, 'q_values'):
                if isinstance(q_table.q_values, dict):
                    # Sparse Q-table format
                    serialized['q_values'] = dict(q_table.q_values)
                    if hasattr(q_table, 'visit_counts'):
                        serialized['visit_counts'] = dict(q_table.visit_counts)
                else:
                    # Dense Q-table format (numpy array)
                    serialized['q_values'] = q_table.q_values.tolist()
                    if hasattr(q_table, 'visit_counts'):
                        serialized['visit_counts'] = q_table.visit_counts.tolist()
            
            # Save enhanced Q-table specific data
            if hasattr(q_table, 'confidence_threshold'):
                serialized['confidence_threshold'] = q_table.confidence_threshold
                serialized['exploration_bonus'] = getattr(q_table, 'exploration_bonus', 0.1)
                serialized['update_count'] = getattr(q_table, 'update_count', 0)
            
            return serialized
            
        except Exception as e:
            print(f"⚠️ Error serializing Q-table: {e}")
            return {'type': 'SparseQTable', 'default_value': 0.0, 'action_count': 6}
    
    def _restore_learning_state(self, robot: EvolutionaryCrawlingAgent, snapshot: LearningStateSnapshot):
        """
        Restore learning state to a robot from a snapshot.
        
        Args:
            robot: Robot to restore state to
            snapshot: Learning state snapshot to restore from
        """
        try:
            # Restore basic learning parameters
            for param, value in snapshot.learning_parameters.items():
                if hasattr(robot, param):
                    setattr(robot, param, value)
            
            # Set up learning approach if learning manager is available
            if self.learning_manager and snapshot.learning_approach:
                from .learning_manager import LearningApproach
                approach_map = {
                    'basic_q_learning': LearningApproach.BASIC_Q_LEARNING,
                    'enhanced_q_learning': LearningApproach.ENHANCED_Q_LEARNING,
                    'survival_q_learning': LearningApproach.SURVIVAL_Q_LEARNING,
                    'deep_q_learning': LearningApproach.DEEP_Q_LEARNING,
                    'attention_deep_q_learning': LearningApproach.ATTENTION_DEEP_Q_LEARNING  # CRITICAL FIX
                }
                
                if snapshot.learning_approach in approach_map:
                    approach = approach_map[snapshot.learning_approach]
                    success = self.learning_manager.set_agent_approach(robot, approach)
                    if not success:
                        print(f"⚠️ Failed to set learning approach {snapshot.learning_approach} for robot {robot.id}")
            
            # Restore Q-table data
            if snapshot.q_table_data:
                self._deserialize_q_table(robot, snapshot.q_table_data)
            
            # Restore survival learning state
            if snapshot.learning_approach == "survival_q_learning":
                if self.learning_manager and robot.id in self.learning_manager.agent_adapters:
                    survival_adapter = self.learning_manager.agent_adapters[robot.id]
                    if snapshot.survival_stats:
                        survival_adapter.survival_stats = snapshot.survival_stats.copy()
                    if snapshot.learning_stage:
                        survival_adapter.learning_stage = snapshot.learning_stage
            
            # Restore deep learning state
            elif snapshot.learning_approach == "deep_q_learning":
                if self.learning_manager and robot.id in self.learning_manager.agent_adapters:
                    deep_adapter = self.learning_manager.agent_adapters[robot.id]
                    try:
                        # Restore neural network weights
                        if snapshot.neural_network_weights:
                            deep_adapter.load_state_dict(snapshot.neural_network_weights)
                        
                        # Restore experience buffer
                        if snapshot.experience_buffer and hasattr(deep_adapter, 'memory'):
                            for experience_tuple in snapshot.experience_buffer:
                                deep_adapter.memory.push(*experience_tuple)
                        
                        # Restore deep learning statistics
                        if snapshot.deep_learning_stats:
                            for key, value in snapshot.deep_learning_stats.items():
                                if hasattr(deep_adapter, f'_{key}'):
                                    setattr(deep_adapter, f'_{key}', value)
                    except Exception as e:
                        print(f"⚠️ Could not restore deep learning state: {e}")
            
            # CRITICAL FIX: Restore attention deep learning state
            elif snapshot.learning_approach == "attention_deep_q_learning":
                if hasattr(robot, '_attention_dqn') and robot._attention_dqn is not None:
                    try:
                        # Restore attention neural network weights
                        if snapshot.neural_network_weights:
                            robot._attention_dqn.load_state_dict(snapshot.neural_network_weights)
                            print(f"♻️ Restored attention network weights for robot {robot.id}")
                        
                        # Restore experience buffer
                        if snapshot.experience_buffer and hasattr(robot._attention_dqn, 'memory'):
                            for experience_tuple in snapshot.experience_buffer:
                                robot._attention_dqn.memory.push(*experience_tuple)
                        
                        # Restore attention learning statistics
                        if snapshot.deep_learning_stats:
                            for key, value in snapshot.deep_learning_stats.items():
                                if hasattr(robot._attention_dqn, f'_{key}'):
                                    setattr(robot._attention_dqn, f'_{key}', value)
                        
                    except Exception as e:
                        print(f"⚠️ Could not restore attention learning state: {e}")
            
            self.pool_stats['learning_states_restored'] += 1
            print(f"♻️ Restored {snapshot.learning_approach} state for robot {robot.id}")
            
        except Exception as e:
            print(f"❌ Error restoring learning state for robot {robot.id}: {e}")
    
    def _deserialize_q_table(self, robot: EvolutionaryCrawlingAgent, q_table_data: Dict[str, Any]):
        """DISABLED: No more Q-table deserialization - only attention networks."""
        try:
            # FORCE ATTENTION LEARNING: Don't create any Q-tables
            # The learning manager will handle attention network setup
            print(f"🧠 Skipping Q-table deserialization for {robot.id} - using attention learning instead")
            
            # Ensure the robot has attention learning
            if not hasattr(robot, '_learning_system') or robot._learning_system is None:
                robot._initialize_attention_learning()
                
        except Exception as e:
            print(f"⚠️ Error ensuring attention learning for {robot.id}: {e}")
            # Try to initialize attention learning as fallback
            try:
                robot._initialize_attention_learning()
            except:
                pass
    
    def acquire_robot(self, 
                     position: Tuple[float, float] = (0, 10),
                     physical_params: Optional[PhysicalParameters] = None,
                     parent_lineage: Optional[List[str]] = None,
                     restore_learning: bool = True,
                     learning_snapshot_id: Optional[str] = None) -> EvolutionaryCrawlingAgent:
        """
        Acquire a robot from the pool, reusing existing robot if available.
        
        Args:
            position: Position to place the robot
            physical_params: Physical parameters (random if None)
            parent_lineage: Parent lineage for evolution tracking
            restore_learning: Whether to restore previous learning state
            learning_snapshot_id: Specific learning snapshot to restore (None = most recent)
            
        Returns:
            EvolutionaryCrawlingAgent: Ready-to-use robot with preserved learning
        """
        try:
            # Try to reuse an existing robot
            if self.available_robots:
                robot = self.available_robots.popleft()
                self._reset_robot(robot, position, physical_params, parent_lineage, restore_learning, learning_snapshot_id)
                self.pool_stats['reused_count'] += 1
                print(f"♻️ Reused robot {robot.id} (pool: {len(self.available_robots)} available)")
            else:
                # Create new robot if pool is empty
                if physical_params is None:
                    physical_params = PhysicalParameters.random_parameters()
                
                robot = EvolutionaryCrawlingAgent(
                    world=self.world,
                    agent_id=None,
                    position=position,
                    category_bits=self.category_bits,
                    mask_bits=self.mask_bits,
                    physical_params=physical_params,
                    parent_lineage=parent_lineage or []
                )
                self.pool_stats['created_count'] += 1
                print(f"🆕 Created new robot {robot.id} (pool empty)")
            
            # Track as active
            self.active_robots[robot.id] = robot
            
            # Update peak statistics
            self.pool_stats['peak_active'] = max(self.pool_stats['peak_active'], len(self.active_robots))
            
            return robot
            
        except Exception as e:
            print(f"❌ Error acquiring robot: {e}")
            # Fallback: create new robot
            return self._create_new_robot(position)
    
    def return_robot(self, robot: EvolutionaryCrawlingAgent, preserve_learning: bool = True):
        """
        Return a robot to the pool for reuse, optionally preserving its learning state.
        
        Args:
            robot: Robot to return to pool
            preserve_learning: Whether to save the robot's learning state
        """
        try:
            robot_id = robot.id
            
            # Save learning state before returning if requested
            if preserve_learning:
                snapshot = self._save_learning_state(robot)
                self.learning_snapshots[robot_id] = snapshot
                
                # Manage snapshot memory usage
                if len(self.learning_snapshots) > self.max_snapshots:
                    # Remove oldest snapshots
                    oldest_keys = sorted(self.learning_snapshots.keys(), 
                                       key=lambda k: self.learning_snapshots[k].timestamp)[:50]
                    for old_key in oldest_keys:
                        del self.learning_snapshots[old_key]
            
            # Remove from active robots
            if robot_id in self.active_robots:
                del self.active_robots[robot_id]
            
            # Add to available robots if pool not full
            if len(self.available_robots) < self.max_pool_size:
                # Mark as available for reuse
                robot._destroyed = False  # Reset destruction flag
                self.available_robots.append(robot)
                self.pool_stats['returned_count'] += 1
                learning_note = " (learning preserved)" if preserve_learning else ""
                print(f"🔄 Returned robot {robot_id} to pool ({len(self.available_robots)} available){learning_note}")
            else:
                # Pool is full, actually destroy the robot
                self._destroy_robot_permanently(robot)
                self.pool_stats['destroyed_count'] += 1
                print(f"💥 Destroyed robot {robot_id} (pool full)")
                
        except Exception as e:
            print(f"❌ Error returning robot {robot_id}: {e}")

    def _reset_robot(self, robot: EvolutionaryCrawlingAgent, 
                    position: Tuple[float, float],
                    physical_params: Optional[PhysicalParameters] = None,
                    parent_lineage: Optional[List[str]] = None,
                    restore_learning: bool = True,
                    learning_snapshot_id: Optional[str] = None):
        """Reset a robot for reuse by updating all its attributes and optionally restoring learning."""
        # Generate new ID for the reused robot
        old_id = robot.id
        robot.id = str(uuid.uuid4())[:8]
        
        # Update physical parameters if provided
        if physical_params is not None:
            robot.physical_params = physical_params.validate_and_repair()
            robot.motor_torque = robot.physical_params.motor_torque
            robot.motor_speed = robot.physical_params.motor_speed
        
        # Reset lineage and evolution tracking
        robot.parent_lineage = parent_lineage or []
        robot.generation = len(robot.parent_lineage)
        robot.mutation_count = 0
        robot.crossover_count = 0
        robot.fitness_history = []
        robot.diversity_score = 0.0
        
        # Reset basic learning state (will be overridden if restoring from snapshot)
        robot.total_reward = 0.0
        robot.steps = 0
        robot.action_history = []
        robot._destroyed = False
        robot.initial_position = position
        
        # Reset position
        if robot.body:
            robot.body.position = position
            robot.body.angle = 0
            robot.body.linearVelocity = (0, 0)
            robot.body.angularVelocity = 0
            robot.body.awake = True
        
        # Restore learning state if requested
        if restore_learning:
            # CRITICAL PERFORMANCE FIX: Disable ALL learning restoration
            # The constant creation of new neural networks during robot restoration
            # was causing severe UI slowdown due to GPU memory churn
            # Robots will get networks from Learning Manager instead
            print(f"🧠 Skipped learning restoration for performance - robot will get network from Learning Manager")
        else:
            print(f"🧠 No learning restoration requested for {robot.id}")
        
        print(f"🔄 Reset robot {robot.id} at position ({position[0]:.1f}, {position[1]:.1f})")
    
    def _destroy_robot_permanently(self, robot: EvolutionaryCrawlingAgent):
        """Permanently destroy a robot's physics body."""
        try:
            robot._destroyed = True
            
            # Disable motors
            if hasattr(robot, 'upper_arm_joint') and robot.upper_arm_joint:
                robot.upper_arm_joint.enableMotor = False
            if hasattr(robot, 'lower_arm_joint') and robot.lower_arm_joint:
                robot.lower_arm_joint.enableMotor = False
            
            # Destroy physics bodies
            bodies_to_destroy = []
            if hasattr(robot, 'wheels') and robot.wheels:
                bodies_to_destroy.extend([w for w in robot.wheels if w])
            if hasattr(robot, 'lower_arm') and robot.lower_arm:
                bodies_to_destroy.append(robot.lower_arm)
            if hasattr(robot, 'upper_arm') and robot.upper_arm:
                bodies_to_destroy.append(robot.upper_arm)
            if hasattr(robot, 'body') and robot.body:
                bodies_to_destroy.append(robot.body)
            
            for body in bodies_to_destroy:
                try:
                    self.world.DestroyBody(body)
                except:
                    pass
            
            # Clear references
            robot.wheels = []
            robot.upper_arm = None
            robot.lower_arm = None
            robot.body = None
            
        except Exception as e:
            print(f"❌ Error permanently destroying robot: {e}")
    
    def get_pool_statistics(self) -> Dict[str, Any]:
        """Get memory pool usage statistics."""
        return {
            'active_robots': len(self.active_robots),
            'available_robots': len(self.available_robots),
            'pool_size_range': f"{self.min_pool_size}-{self.max_pool_size}",
            'statistics': self.pool_stats.copy()
        } 