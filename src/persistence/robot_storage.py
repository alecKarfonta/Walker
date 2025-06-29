"""
Robot Storage System

Provides comprehensive robot state persistence including:
- Physical parameters 
- Q-learning state (Q-table, epsilon, learning rate, etc.)
- Performance metrics (distance, speed, rewards, etc.)
- Learning history and metadata
"""

import json
import pickle
import gzip
import numpy as np
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from copy import deepcopy
import Box2D as b2

from src.agents.physical_parameters import PhysicalParameters
from src.agents.crawling_agent import CrawlingAgent


@dataclass
class PerformanceHistory:
    """Complete performance history for a robot."""
    robot_id: str
    creation_timestamp: float
    last_updated: float
    
    # Performance metrics over time
    reward_history: List[float] = field(default_factory=list)
    distance_history: List[float] = field(default_factory=list)
    speed_history: List[float] = field(default_factory=list)
    stability_history: List[float] = field(default_factory=list)
    
    # Learning progress
    convergence_history: List[float] = field(default_factory=list)
    epsilon_history: List[float] = field(default_factory=list)
    q_table_size_history: List[int] = field(default_factory=list)
    
    # Aggregate statistics
    total_reward: float = 0.0
    max_distance: float = 0.0
    max_speed: float = 0.0
    average_stability: float = 0.0
    total_steps: int = 0
    episodes_completed: int = 0
    
    # Evolution history
    generation: int = 0
    parent_lineage: List[str] = field(default_factory=list)
    mutation_count: int = 0
    crossover_count: int = 0
    
    def update_from_agent(self, agent: CrawlingAgent):
        """Update performance history from current agent state."""
        self.last_updated = time.time()
        
        # Add current performance
        current_reward = getattr(agent, 'total_reward', 0.0)
        current_distance = agent.body.position.x - agent.initial_position[0] if agent.body else 0.0
        current_speed = getattr(agent, 'max_speed', 0.0)
        current_stability = max(0, 1.0 - abs(agent.body.angle)) if agent.body else 0.0
        
        self.reward_history.append(current_reward)
        self.distance_history.append(current_distance)
        self.speed_history.append(current_speed)
        self.stability_history.append(current_stability)
        
        # Learning metrics
        convergence = getattr(agent, 'total_reward', 0.0) / max(1, getattr(agent, 'steps', 1))  # Use reward/step as convergence metric
        epsilon = getattr(agent._learning_system, 'epsilon', 0.0) if hasattr(agent, '_learning_system') and agent._learning_system else 0.0
        q_size = len(agent._learning_system.memory.buffer) if (hasattr(agent, '_learning_system') and agent._learning_system and hasattr(agent._learning_system, 'memory') and agent._learning_system.memory and hasattr(agent._learning_system.memory, 'buffer')) else 0
        
        self.convergence_history.append(convergence)
        self.epsilon_history.append(epsilon)
        self.q_table_size_history.append(q_size)
        
        # Update aggregates
        self.total_reward = current_reward
        self.max_distance = max(self.max_distance, current_distance)
        self.max_speed = max(self.max_speed, current_speed)
        if self.stability_history:
            self.average_stability = float(np.mean(self.stability_history))
        
        self.total_steps = getattr(agent, 'steps', 0)
        
        # Evolution info
        self.generation = getattr(agent, 'generation', 0)
        self.parent_lineage = getattr(agent, 'parent_lineage', [])
        self.mutation_count = getattr(agent, 'mutation_count', 0)
        self.crossover_count = getattr(agent, 'crossover_count', 0)


@dataclass
class RobotState:
    """Complete serializable robot state."""
    robot_id: str
    save_timestamp: float
    
    # Physical parameters
    physical_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Neural network state  
    neural_network_data: Dict[str, Any] = field(default_factory=dict)
    learning_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    performance_history: Optional[PerformanceHistory] = None
    
    # Robot configuration
    position: Tuple[float, float] = (10.0, 20.0)
    category_bits: int = 0x0001
    mask_bits: int = 0xFFFF
    
    # Metadata
    version: str = "1.0"
    save_method: str = "manual"
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        
        # Handle performance history separately
        if self.performance_history:
            data['performance_history'] = asdict(self.performance_history)
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RobotState':
        """Create from dictionary."""
        # Handle performance history
        perf_history = None
        if 'performance_history' in data and data['performance_history']:
            perf_history = PerformanceHistory(**data['performance_history'])
            
        # Remove performance_history from main data to avoid double assignment
        data_copy = deepcopy(data)
        data_copy.pop('performance_history', None)
        
        state = cls(**data_copy)
        state.performance_history = perf_history
        return state


class RobotStorage:
    """
    Comprehensive robot storage and restoration system.
    Handles all aspects of robot persistence including Q-tables, parameters, and metrics.
    """
    
    def __init__(self, storage_directory: str = "robot_storage"):
        """Initialize robot storage system."""
        self.storage_dir = Path(storage_directory)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.storage_dir / "robots").mkdir(exist_ok=True)
        (self.storage_dir / "snapshots").mkdir(exist_ok=True)
        (self.storage_dir / "history").mkdir(exist_ok=True)
        (self.storage_dir / "backups").mkdir(exist_ok=True)
        
        # Storage format options
        self.use_compression = True
        
        print(f"🗄️  Robot storage initialized at: {self.storage_dir}")
    
    def save_robot(self, agent: CrawlingAgent, notes: str = "", save_method: str = "manual") -> str:
        """Save complete robot state to persistent storage."""
        from .storage_helpers import (
            extract_neural_network_data, extract_learning_parameters, 
            extract_performance_metrics, create_performance_history
        )
        
        try:
            robot_id = str(agent.id)
            timestamp = time.time()
            
            # Create robot state
            state = RobotState(
                robot_id=robot_id,
                save_timestamp=timestamp,
                save_method=save_method,
                notes=notes,
                position=agent.initial_position,
                category_bits=getattr(agent, 'category_bits', 0x0001),
                mask_bits=getattr(agent, 'mask_bits', 0xFFFF)
            )
            
            # Extract all data using helper functions
            if hasattr(agent, 'physical_params'):
                state.physical_parameters = agent.physical_params.to_dict()
            
            state.neural_network_data = extract_neural_network_data(agent)
            state.learning_parameters = extract_learning_parameters(agent)
            state.performance_metrics = extract_performance_metrics(agent)
            
            # Get existing history if available
            existing_history = self.get_performance_history(robot_id)
            state.performance_history = create_performance_history(agent, existing_history)
            
            # Save to file
            filename = self._save_robot_state(state)
            
            print(f"💾 Saved robot {robot_id} to {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Error saving robot {getattr(agent, 'id', 'unknown')}: {e}")
            raise
    
    def load_robot(self, robot_id_or_filename: str, world: b2.b2World, 
                                        position: Optional[Tuple[float, float]] = None) -> CrawlingAgent:
        """Load robot from persistent storage and recreate exact state."""
        from .storage_helpers import (
            restore_neural_network_data, restore_learning_parameters, restore_performance_metrics
        )
        
        try:
            # Load robot state
            state = self._load_robot_state(robot_id_or_filename)
            
            # Determine position
            spawn_position = position if position else state.position
            
            # Recreate physical parameters
            physical_params = PhysicalParameters.from_dict(state.physical_parameters)
            
            # Create new robot with saved parameters (use None for agent_id to let it generate UUID)
            robot = CrawlingAgent(
                world=world,
                agent_id=None,  # Let it generate new UUID, then override
                position=spawn_position,
                category_bits=state.category_bits,
                mask_bits=state.mask_bits,
                physical_params=physical_params,
                parent_lineage=state.performance_history.parent_lineage if state.performance_history else []
            )
            
            # Override the generated ID with the saved one
            robot.id = state.robot_id
            
            # Restore all state using helper functions
            restore_neural_network_data(robot, state.neural_network_data)
            restore_learning_parameters(robot, state.learning_parameters)
            restore_performance_metrics(robot, state.performance_metrics)
            
            # Restore evolution info
            if state.performance_history:
                robot.generation = state.performance_history.generation
                robot.mutation_count = state.performance_history.mutation_count
                robot.crossover_count = state.performance_history.crossover_count
            
            print(f"📂 Loaded robot {state.robot_id} from storage")
            return robot
            
        except Exception as e:
            print(f"❌ Error loading robot {robot_id_or_filename}: {e}")
            raise
    
    def list_saved_robots(self) -> List[Dict[str, Any]]:
        """List all saved robots with their metadata."""
        robots = []
        
        for robot_file in (self.storage_dir / "robots").glob("*.json*"):
            try:
                state = self._load_robot_state(robot_file.stem.replace('.json', ''))
                
                robot_info = {
                    'robot_id': state.robot_id,
                    'filename': robot_file.name,
                    'save_timestamp': state.save_timestamp,
                    'save_date': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(state.save_timestamp)),
                    'save_method': state.save_method,
                    'notes': state.notes,
                    'generation': state.performance_history.generation if state.performance_history else 0,
                    'total_reward': state.performance_history.total_reward if state.performance_history else 0.0,
                    'max_distance': state.performance_history.max_distance if state.performance_history else 0.0,
                    'max_speed': state.performance_history.max_speed if state.performance_history else 0.0,
                    'total_steps': state.performance_history.total_steps if state.performance_history else 0
                }
                robots.append(robot_info)
                
            except Exception as e:
                print(f"⚠️  Error reading robot file {robot_file}: {e}")
                continue
        
        robots.sort(key=lambda x: x['save_timestamp'], reverse=True)
        return robots
    
    def delete_robot(self, robot_id_or_filename: str) -> bool:
        """Delete a saved robot."""
        from .storage_helpers import find_robot_file
        
        try:
            robot_file = find_robot_file(self.storage_dir, robot_id_or_filename)
            if robot_file:
                robot_file.unlink()
                print(f"🗑️  Deleted {robot_file.name}")
                return True
            else:
                print(f"❌ Robot {robot_id_or_filename} not found")
                return False
                
        except Exception as e:
            print(f"❌ Error deleting robot {robot_id_or_filename}: {e}")
            return False
    
    def create_snapshot(self, agents: List[CrawlingAgent], snapshot_name: Optional[str] = None) -> str:
        """Save a snapshot of multiple robots."""
        from .storage_helpers import save_state_to_file
        
        if not snapshot_name:
            snapshot_name = f"snapshot_{int(time.time())}"
        
        snapshot_dir = self.storage_dir / "snapshots" / snapshot_name
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        snapshot_info = {
            'snapshot_name': snapshot_name,
            'timestamp': time.time(),
            'robot_count': len(agents),
            'robots': []
        }
        
        # Save each robot
        for agent in agents:
            try:
                # Create robot state for snapshot
                state = self._create_robot_state(agent, save_method="snapshot")
                robot_filename = f"{agent.id}.json"
                
                # Save robot state
                robot_path = snapshot_dir / robot_filename
                save_state_to_file(state, robot_path, self.use_compression)
                
                snapshot_info['robots'].append({
                    'robot_id': str(agent.id),
                    'filename': robot_filename,
                    'total_reward': getattr(agent, 'total_reward', 0.0),
                    'generation': getattr(agent, 'generation', 0)
                })
                
            except Exception as e:
                print(f"⚠️  Error saving robot {getattr(agent, 'id', 'unknown')} in snapshot: {e}")
                continue
        
        # Save snapshot metadata
        snapshot_info_path = snapshot_dir / "snapshot_info.json"
        with open(snapshot_info_path, 'w') as f:
            json.dump(snapshot_info, f, indent=2)
        
        print(f"📸 Created snapshot '{snapshot_name}' with {len(snapshot_info['robots'])} robots")
        return snapshot_name
    
    def get_performance_history(self, robot_id: str) -> Optional[PerformanceHistory]:
        """Get performance history for a robot."""
        try:
            state = self._load_robot_state(robot_id)
            return state.performance_history
        except:
            return None
    
    def _save_robot_state(self, state: RobotState) -> str:
        """Save robot state to file."""
        from .storage_helpers import save_state_to_file
        
        timestamp = int(state.save_timestamp)
        filename = f"{state.robot_id}_{timestamp}"
        filepath = self.storage_dir / "robots" / f"{filename}.json"
        
        save_state_to_file(state, filepath, self.use_compression)
        return filename
    
    def _load_robot_state(self, robot_id_or_filename: str) -> RobotState:
        """Load robot state from file."""
        from .storage_helpers import find_robot_file, load_state_from_file
        
        robot_file = find_robot_file(self.storage_dir, robot_id_or_filename)
        if not robot_file:
            raise FileNotFoundError(f"Robot {robot_id_or_filename} not found in storage")
        
        return load_state_from_file(robot_file)
    
    def _create_robot_state(self, agent: CrawlingAgent, save_method: str = "manual") -> RobotState:
        """Create robot state from agent."""
        from .storage_helpers import (
            extract_neural_network_data, extract_learning_parameters, 
            extract_performance_metrics, create_performance_history
        )
        
        robot_id = str(agent.id)
        
        state = RobotState(
            robot_id=robot_id,
            save_timestamp=time.time(),
            save_method=save_method,
            position=agent.initial_position,
            category_bits=getattr(agent, 'category_bits', 0x0001),
            mask_bits=getattr(agent, 'mask_bits', 0xFFFF)
        )
        
        # Extract all data
        if hasattr(agent, 'physical_params'):
            state.physical_parameters = agent.physical_params.to_dict()
        
        state.neural_network_data = extract_neural_network_data(agent)
        state.learning_parameters = extract_learning_parameters(agent)
        state.performance_metrics = extract_performance_metrics(agent)
        state.performance_history = create_performance_history(agent)
        
        return state 