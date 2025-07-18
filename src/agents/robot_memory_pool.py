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

from .physical_parameters import PhysicalParameters


class RobotMemoryPool:
    """
    Simplified memory pool for efficient robot object reuse.
    Agents are now standalone and handle their own learning.
    """
    
    def __init__(self, world: b2.b2World, min_pool_size: int = 10, max_pool_size: int = 100,
                 category_bits: int = 0x0002, mask_bits: int = 0x0001, learning_manager=None):
        """Initialize the robot memory pool."""
        self.world = world
        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size
        self.category_bits = category_bits
        self.mask_bits = mask_bits
        self.learning_manager = learning_manager
        
        # Active robots currently in use  
        self.active_robots: Dict[str, Any] = {}
        
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
    
    def _create_new_robot(self, position: Tuple[float, float] = (0, 10)):
        """Create a new robot instance."""
        from .evolutionary_crawling_agent import EvolutionaryCrawlingAgent
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
                     apply_size_mutations: bool = True):
        """Acquire a robot from the pool with optional size mutations."""
        try:
            # Try to reuse an existing robot
            if self.available_robots:
                robot = self.available_robots.popleft()
                self._reset_robot(robot, position, physical_params, apply_size_mutations)
                self.pool_stats['reused_count'] += 1
                print(f"♻️ Reused robot {robot.id} (pool: {len(self.available_robots)} available)")
            else:
                # Create new robot if pool is empty
                if physical_params is None:
                    physical_params = PhysicalParameters.random_parameters()
                
                from .evolutionary_crawling_agent import EvolutionaryCrawlingAgent
                robot = EvolutionaryCrawlingAgent(
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
    
    def return_robot(self, robot):
        """Return a robot to the pool for reuse, preserving ALL state including neural networks."""
        try:
            robot_id = robot.id
            
            # Remove from active robots
            if robot_id in self.active_robots:
                del self.active_robots[robot_id]
            
            # Add to available robots if pool not full - KEEP NETWORK INTACT
            if len(self.available_robots) < self.max_pool_size:
                # Mark as available for reuse (keep neural network and all learned state)
                setattr(robot, '_destroyed', False)  # Reset destruction flag
                self.available_robots.append(robot)
                self.pool_stats['returned_count'] += 1
                
                # Log what type of learning state is being preserved
                network_info = "no network"
                if hasattr(robot, '_learning_system') and robot._learning_system:
                    network_info = "with learned network"
                
                print(f"🔄 Returned robot {robot_id} to pool ({network_info}) - {len(self.available_robots)} available")
            else:
                # Pool is full, actually destroy the robot
                self._destroy_robot_permanently(robot)
                self.pool_stats['destroyed_count'] += 1
                print(f"💥 Destroyed robot {robot_id} (pool full)")
                
        except Exception as e:
            print(f"❌ Error returning robot {robot_id}: {e}")

    def _reset_robot(self, robot, 
                    position: Tuple[float, float],
                    physical_params: Optional[PhysicalParameters] = None,
                    apply_size_mutations: bool = True):
        """Reset a robot for reuse - handle morphology changes by releasing incompatible networks."""
        robot_id = robot.id  # KEEP the same ID to preserve learning continuity
        
        # Store original morphology for comparison
        original_num_arms = getattr(robot.physical_params, 'num_arms', 1) if hasattr(robot, 'physical_params') else 1
        original_segments_per_limb = getattr(robot.physical_params, 'segments_per_limb', 2) if hasattr(robot, 'physical_params') else 2
        original_action_size = getattr(robot, 'action_size', 9) if hasattr(robot, 'action_size') else 9
        
        # Apply size mutations or new physical parameters
        if apply_size_mutations and hasattr(robot, 'physical_params') and robot.physical_params:
            # Apply full mutations INCLUDING morphology changes
            mutated_params = robot.physical_params.mutate(mutation_rate=0.12)
            robot.physical_params = mutated_params
            print(f"🧬 Applied full mutations to pooled robot {robot_id} (allowing morphology changes)")
        elif physical_params is not None:
            robot.physical_params = physical_params.validate_and_repair()
            print(f"🔧 Applied new physical parameters to pooled robot {robot_id}")
        
        # Check if morphology changed (affecting action space)
        new_num_arms = getattr(robot.physical_params, 'num_arms', 1) if hasattr(robot, 'physical_params') else 1
        new_segments_per_limb = getattr(robot.physical_params, 'segments_per_limb', 2) if hasattr(robot, 'physical_params') else 2
        
        morphology_changed = (new_num_arms != original_num_arms or new_segments_per_limb != original_segments_per_limb)
        
        if morphology_changed:
            # CRITICAL FIX: Release old network and let robot get a new one from Learning Manager
            if self.learning_manager and hasattr(robot, 'id') and hasattr(robot, '_learning_system') and robot._learning_system:
                try:
                    # Calculate performance score for the old network
                    performance_score = self._calculate_performance_score(robot)
                    
                    # Release the old network back to Learning Manager
                    self.learning_manager.release_agent_network(robot.id, performance_score)
                    
                    # Clear the network reference so robot will get a new one
                    robot._learning_system = None
                    
                    print(f"🔄 Released incompatible network from pooled robot {robot_id} due to morphology change ({original_num_arms}×{original_segments_per_limb} → {new_num_arms}×{new_segments_per_limb})")
                    print(f"🆕 Robot {robot_id} will acquire new network with correct action space when reinitialized")
                    
                except Exception as e:
                    print(f"⚠️ Failed to release network during morphology change: {e}")
                    # Clear network anyway to prevent dimension mismatch
                    robot._learning_system = None
            else:
                print(f"🔄 Morphology changed for robot {robot_id} - clearing network reference")
                robot._learning_system = None
        else:
            # Morphology unchanged - can preserve network
            network_info = "preserving learned network (compatible morphology)" if hasattr(robot, '_learning_system') and robot._learning_system else "no network"
            print(f"✅ Morphology preserved for robot {robot_id} - {network_info}")
        
        # Reset ONLY basic state - preserve or clear network as determined above
        robot.total_reward = 0.0
        robot.steps = 0
        setattr(robot, '_destroyed', False)
        robot.initial_position = position
        
        # Reset position and physics
        if robot.body:
            robot.body.position = position
            robot.body.angle = 0
            robot.body.linearVelocity = (0, 0)
            robot.body.angularVelocity = 0
            robot.body.awake = True
        
        # Force robot to reinitialize its action space and get appropriate network
        if hasattr(robot, '_generate_dynamic_action_space'):
            try:
                robot.actions = robot._generate_dynamic_action_space()
                robot.action_size = len(robot.actions)
                print(f"🎯 Recalculated action space for robot {robot_id}: {robot.action_size} actions")
            except Exception as e:
                print(f"⚠️ Error recalculating action space for robot {robot_id}: {e}")
        
        print(f"🔄 Reset pooled robot {robot_id} at ({position[0]:.1f}, {position[1]:.1f}) - morphology: {new_num_arms} limbs × {new_segments_per_limb} segments")
    
    def _calculate_performance_score(self, robot) -> float:
        """Calculate performance score for neural network preservation."""
        try:
            # Basic performance metrics
            reward_score = max(0.0, robot.total_reward / max(1, robot.steps * 0.1))  # Reward per step
            survival_score = min(1.0, robot.steps / 1000.0)  # Survival bonus
            
            # Normalize to 0-1 range
            performance_score = (reward_score * 0.7 + survival_score * 0.3)
            return min(1.0, max(0.0, performance_score))
            
        except Exception as e:
            print(f"⚠️ Error calculating performance score: {e}")
            return 0.0
    
    def _destroy_robot_permanently(self, robot):
        """Permanently destroy a robot's physics body."""
        try:
            setattr(robot, '_destroyed', True)
            
            # CRITICAL FIX: Release neural network back to Learning Manager before destroying robot
            if self.learning_manager and hasattr(robot, 'id'):
                try:
                    # Calculate performance score for network quality assessment
                    performance_score = self._calculate_performance_score(robot)
                    # Release network back to pool for reuse
                    self.learning_manager.release_agent_network(robot.id, performance_score)
                    print(f"🧠 Released network from destroyed robot {robot.id} (performance: {performance_score:.3f})")
                except Exception as e:
                    print(f"⚠️ Failed to release network from robot {robot.id}: {e}")
            
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