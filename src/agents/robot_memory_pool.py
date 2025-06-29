"""
Simplified Robot Memory Pool System

Basic memory management for robot objects using object pooling.
Agents are now standalone and handle their own learning.
"""

import time
import uuid
from typing import Dict, Any, Optional, Tuple
from collections import deque
import Box2D as b2

from .crawling_agent import CrawlingAgent
from .physical_parameters import PhysicalParameters


class RobotMemoryPool:
    """
    Simplified memory pool for efficient robot object reuse.
    Agents are now standalone and handle their own learning.
    """
    
    def __init__(self, world: b2.b2World, min_pool_size: int = 10, max_pool_size: int = 100,
                 category_bits: int = 0x0002, mask_bits: int = 0x0001):
        """Initialize the robot memory pool."""
        self.world = world
        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size
        self.category_bits = category_bits
        self.mask_bits = mask_bits
        
        # Active robots currently in use  
        self.active_robots: Dict[str, CrawlingAgent] = {}
        
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
    
    def _create_new_robot(self, position: Tuple[float, float] = (0, 10)) -> CrawlingAgent:
        """Create a new robot instance."""
        random_params = PhysicalParameters.random_parameters()
        return CrawlingAgent(
            world=self.world,
            agent_id=None,  # Generate new UUID
            position=position,
            category_bits=self.category_bits,
            mask_bits=self.mask_bits,
            physical_params=random_params
        )
    
    def acquire_robot(self, 
                     position: Tuple[float, float] = (0, 10),
                     physical_params: Optional[PhysicalParameters] = None) -> CrawlingAgent:
        """Acquire a robot from the pool."""
        try:
            # Try to reuse an existing robot
            if self.available_robots:
                robot = self.available_robots.popleft()
                self._reset_robot(robot, position, physical_params)
                self.pool_stats['reused_count'] += 1
                print(f"♻️ Reused robot {robot.id} (pool: {len(self.available_robots)} available)")
            else:
                # Create new robot if pool is empty
                if physical_params is None:
                    physical_params = PhysicalParameters.random_parameters()
                
                robot = CrawlingAgent(
                    world=self.world,
                    agent_id=None,
                    position=position,
                    category_bits=self.category_bits,
                    mask_bits=self.mask_bits,
                    physical_params=physical_params
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
    
    def return_robot(self, robot: CrawlingAgent):
        """Return a robot to the pool for reuse."""
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

    def _reset_robot(self, robot: CrawlingAgent, 
                    position: Tuple[float, float],
                    physical_params: Optional[PhysicalParameters] = None):
        """Reset a robot for reuse."""
        # Generate new ID for the reused robot
        robot.id = str(uuid.uuid4())[:8]
        
        # Update physical parameters if provided
        if physical_params is not None:
            robot.physical_params = physical_params.validate_and_repair()
        
        # Reset basic state
        robot.total_reward = 0.0
        robot.steps = 0
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
    
    def _destroy_robot_permanently(self, robot: CrawlingAgent):
        """Permanently destroy a robot's physics body."""
        try:
            robot._destroyed = True
            
            # Destroy physics bodies (simplified)
            if hasattr(robot, 'body') and robot.body:
                try:
                    self.world.DestroyBody(robot.body)
                except:
                    pass
            
            # Clear references
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