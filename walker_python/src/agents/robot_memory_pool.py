"""
Robot Memory Pool System

Efficient memory management for robot objects using object pooling.
Instead of creating/destroying robots constantly, reuse existing objects 
by updating their attributes.
"""

import time
import random
import uuid
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import deque
import Box2D as b2

from .evolutionary_crawling_agent import EvolutionaryCrawlingAgent
from .physical_parameters import PhysicalParameters


class RobotMemoryPool:
    """
    Memory pool for efficient robot object reuse.
    
    Maintains a pool of pre-allocated robot objects that can be reused
    by updating their attributes instead of creating new instances.
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
        
        # Pool statistics
        self.pool_stats = {
            'created_count': 0,
            'reused_count': 0,
            'returned_count': 0,
            'destroyed_count': 0,
            'peak_active': 0,
            'peak_available': 0
        }
        
        # Pre-allocate minimum robots
        self._initialize_pool()
        
        print(f"🏊 RobotMemoryPool initialized: {self.min_pool_size}-{self.max_pool_size} robots")
    
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
    
    def acquire_robot(self, 
                     position: Tuple[float, float] = (0, 10),
                     physical_params: Optional[PhysicalParameters] = None,
                     parent_lineage: Optional[List[str]] = None) -> EvolutionaryCrawlingAgent:
        """
        Acquire a robot from the pool, reusing existing robot if available.
        
        Args:
            position: Position to place the robot
            physical_params: Physical parameters (random if None)
            parent_lineage: Parent lineage for evolution tracking
            
        Returns:
            EvolutionaryCrawlingAgent: Ready-to-use robot
        """
        try:
            # Try to reuse an existing robot
            if self.available_robots:
                robot = self.available_robots.popleft()
                self._reset_robot(robot, position, physical_params, parent_lineage)
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
    
    def return_robot(self, robot: EvolutionaryCrawlingAgent):
        """
        Return a robot to the pool for reuse.
        
        Args:
            robot: Robot to return to pool
        """
        try:
            robot_id = robot.id
            
            # Remove from active robots
            if robot_id in self.active_robots:
                del self.active_robots[robot_id]
            
            # Add to available robots if pool not full
            if len(self.available_robots) < self.max_pool_size:
                # Mark as available for reuse
                robot._destroyed = False  # Reset destruction flag
                self.available_robots.append(robot)
                self.pool_stats['returned_count'] += 1
                print(f"🔄 Returned robot {robot_id} to pool ({len(self.available_robots)} available)")
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
                    parent_lineage: Optional[List[str]] = None):
        """Reset a robot for reuse by updating all its attributes."""
        # Generate new ID for the reused robot
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
        
        # Reset Q-learning state
        if hasattr(robot.q_table, 'reset'):
            robot.q_table.reset()
        else:
            robot.q_table.q_values.clear()
            if hasattr(robot.q_table, 'visit_counts'):
                robot.q_table.visit_counts.clear()
        
        # Reset learning parameters
        robot.learning_rate = robot.physical_params.learning_rate
        robot.epsilon = robot.physical_params.epsilon
        
        # Reset physics state
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
    
    def set_learning_manager(self, learning_manager):
        """Set the learning manager for knowledge transfer."""
        self.learning_manager = learning_manager
        print(f"🔗 Learning manager connected to memory pool") 