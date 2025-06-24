"""
Elite Robot Management System

Automatically preserves elite robots during evolution with database limits and restoration capabilities.
Integrates with the training pipeline to save top performers every generation.
"""

import time
import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from .robot_storage import RobotStorage, RobotState
from .storage_manager import StorageManager


@dataclass
class EliteRobotRecord:
    """Record of an elite robot with preservation metadata."""
    robot_id: str
    generation: int
    fitness: float
    rank: int  # Rank within generation (1 = best)
    preservation_timestamp: float
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'robot_id': self.robot_id,
            'generation': self.generation,
            'fitness': self.fitness,
            'rank': self.rank,
            'preservation_timestamp': self.preservation_timestamp,
            'notes': self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EliteRobotRecord':
        """Create from dictionary."""
        return cls(**data)


class EliteManager:
    """
    Manages elite robot preservation during evolution.
    
    Features:
    - Automatic elite preservation every generation
    - Database size limits with smart cleanup
    - Elite restoration for training resumption
    - Performance tracking and analysis
    """
    
    def __init__(self, 
                 storage_directory: str = "robot_storage",
                 elite_per_generation: int = 3,
                 max_elite_storage: int = 100,
                 min_fitness_threshold: float = 0.0):
        """
        Initialize elite management system.
        
        Args:
            storage_directory: Directory for robot storage
            elite_per_generation: Number of elites to preserve per generation
            max_elite_storage: Maximum number of elite robots to store
            min_fitness_threshold: Minimum fitness required for elite preservation
        """
        self.storage_directory = Path(storage_directory)
        self.elite_per_generation = elite_per_generation
        self.max_elite_storage = max_elite_storage
        self.min_fitness_threshold = min_fitness_threshold
        
        # Initialize storage systems
        self.robot_storage = RobotStorage(storage_directory)
        self.storage_manager = StorageManager(storage_directory)
        
        # Create elite database
        self.elite_db_path = self.storage_directory / "elite_robots.db"
        self._initialize_elite_database()
        
        # Elite tracking
        self.current_generation_elites: List[EliteRobotRecord] = []
        self.preservation_history: List[Dict[str, Any]] = []
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        print(f"🏆 Elite Manager initialized:")
        print(f"   • {self.elite_per_generation} elites per generation")
        print(f"   • Max {self.max_elite_storage} total elite storage")
        print(f"   • Min fitness threshold: {self.min_fitness_threshold}")
    
    def _initialize_elite_database(self):
        """Initialize the elite robots database."""
        with sqlite3.connect(self.elite_db_path) as conn:
            cursor = conn.cursor()
            
            # Create elite records table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS elite_robots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    robot_id TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    fitness REAL NOT NULL,
                    rank INTEGER NOT NULL,
                    preservation_timestamp REAL NOT NULL,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(robot_id, generation)
                )
            ''')
            
            # Create performance history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS preservation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    generation INTEGER NOT NULL,
                    elites_preserved INTEGER NOT NULL,
                    best_fitness REAL NOT NULL,
                    avg_elite_fitness REAL NOT NULL,
                    total_elites_stored INTEGER NOT NULL,
                    cleanup_performed BOOLEAN DEFAULT FALSE,
                    cleanup_count INTEGER DEFAULT 0,
                    timestamp REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_generation ON elite_robots(generation)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_fitness ON elite_robots(fitness DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_robot_id ON elite_robots(robot_id)')
            
            conn.commit()
    
    def preserve_generation_elites(self, agents: List, generation: int) -> Dict[str, Any]:
        """
        Preserve elite robots from current generation.
        
        Args:
            agents: List of agents from current generation
            generation: Current generation number
            
        Returns:
            Dictionary with preservation results
        """
        preservation_start = time.time()
        
        try:
            # Filter and rank agents by fitness
            valid_agents = [agent for agent in agents if not getattr(agent, '_destroyed', False)]
            
            if not valid_agents:
                self.logger.warning(f"No valid agents to preserve in generation {generation}")
                return self._create_preservation_result(generation, 0, [], preservation_start)
            
            # Sort by fitness (descending)
            sorted_agents = sorted(valid_agents, 
                                 key=lambda a: getattr(a, 'total_reward', 0.0), 
                                 reverse=True)
            
            # Select elite agents
            num_elites = min(self.elite_per_generation, len(sorted_agents))
            elite_agents = sorted_agents[:num_elites]
            
            # Filter by minimum fitness threshold
            qualified_elites = [
                agent for agent in elite_agents 
                if getattr(agent, 'total_reward', 0.0) >= self.min_fitness_threshold
            ]
            
            if not qualified_elites:
                self.logger.info(f"No agents met minimum fitness threshold ({self.min_fitness_threshold}) in generation {generation}")
                return self._create_preservation_result(generation, 0, [], preservation_start)
            
            # Preserve each elite robot
            preserved_records = []
            
            for rank, agent in enumerate(qualified_elites, 1):
                try:
                    # Save robot with elite-specific notes
                    elite_notes = f"Elite #{rank} from generation {generation} (fitness: {getattr(agent, 'total_reward', 0.0):.3f})"
                    
                    filename = self.robot_storage.save_robot(
                        agent,
                        notes=elite_notes,
                        save_method="elite_preservation"
                    )
                    
                    # Create elite record
                    elite_record = EliteRobotRecord(
                        robot_id=str(agent.id),
                        generation=generation,
                        fitness=getattr(agent, 'total_reward', 0.0),
                        rank=rank,
                        preservation_timestamp=time.time(),
                        notes=elite_notes
                    )
                    
                    # Store in database
                    self._store_elite_record(elite_record)
                    preserved_records.append(elite_record)
                    
                    self.logger.info(f"🏆 Preserved elite robot {agent.id} (rank {rank}, fitness: {elite_record.fitness:.3f})")
                    
                except Exception as e:
                    self.logger.error(f"Failed to preserve elite robot {getattr(agent, 'id', 'unknown')}: {e}")
                    continue
            
            # Update current generation tracking
            self.current_generation_elites = preserved_records
            
            # Perform cleanup if needed
            cleanup_count = self._cleanup_excess_elites()
            
            # Record preservation history
            self._record_preservation_history(generation, preserved_records, cleanup_count)
            
            preservation_time = time.time() - preservation_start
            
            result = self._create_preservation_result(generation, len(preserved_records), preserved_records, preservation_start)
            result.update({
                'cleanup_performed': cleanup_count > 0,
                'cleanup_count': cleanup_count,
                'preservation_time': preservation_time
            })
            
            print(f"🏆 Generation {generation}: Preserved {len(preserved_records)} elite robots in {preservation_time:.2f}s")
            if cleanup_count > 0:
                print(f"🧹 Cleaned up {cleanup_count} old elite robots to maintain storage limits")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to preserve elites for generation {generation}: {e}")
            return self._create_preservation_result(generation, 0, [], preservation_start, error=str(e))
    
    def _store_elite_record(self, record: EliteRobotRecord):
        """Store elite record in database."""
        with sqlite3.connect(self.elite_db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO elite_robots 
                (robot_id, generation, fitness, rank, preservation_timestamp, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                record.robot_id,
                record.generation,
                record.fitness,
                record.rank,
                record.preservation_timestamp,
                record.notes
            ))
            
            conn.commit()
    
    def _cleanup_excess_elites(self) -> int:
        """Clean up excess elite robots to maintain storage limits."""
        with sqlite3.connect(self.elite_db_path) as conn:
            cursor = conn.cursor()
            
            # Count current elite robots
            cursor.execute('SELECT COUNT(*) FROM elite_robots')
            current_count = cursor.fetchone()[0]
            
            if current_count <= self.max_elite_storage:
                return 0
            
            # Calculate how many to remove
            excess_count = current_count - self.max_elite_storage
            
            # Strategy: Remove oldest elites with lowest fitness
            # But always keep at least one elite from recent generations
            cursor.execute('''
                SELECT robot_id, generation, fitness, preservation_timestamp 
                FROM elite_robots 
                ORDER BY generation ASC, fitness ASC
                LIMIT ?
            ''', (excess_count,))
            
            robots_to_remove = cursor.fetchall()
            
            removed_count = 0
            for robot_id, generation, fitness, timestamp in robots_to_remove:
                try:
                    # Remove from database
                    cursor.execute('DELETE FROM elite_robots WHERE robot_id = ?', (robot_id,))
                    
                    # Remove robot file
                    self.robot_storage.delete_robot(robot_id)
                    removed_count += 1
                    
                    self.logger.info(f"🗑️  Cleaned up elite robot {robot_id} (gen {generation}, fitness {fitness:.3f})")
                    
                except Exception as e:
                    self.logger.error(f"Failed to clean up elite robot {robot_id}: {e}")
                    continue
            
            conn.commit()
            return removed_count
    
    def _record_preservation_history(self, generation: int, preserved_records: List[EliteRobotRecord], cleanup_count: int):
        """Record preservation history in database."""
        if not preserved_records:
            return
        
        with sqlite3.connect(self.elite_db_path) as conn:
            cursor = conn.cursor()
            
            # Calculate statistics
            fitnesses = [record.fitness for record in preserved_records]
            best_fitness = max(fitnesses)
            avg_fitness = np.mean(fitnesses)
            
            # Get total elite count
            cursor.execute('SELECT COUNT(*) FROM elite_robots')
            total_elites = cursor.fetchone()[0]
            
            cursor.execute('''
                INSERT INTO preservation_history 
                (generation, elites_preserved, best_fitness, avg_elite_fitness, 
                 total_elites_stored, cleanup_performed, cleanup_count, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                generation,
                len(preserved_records),
                best_fitness,
                avg_fitness,
                total_elites,
                cleanup_count > 0,
                cleanup_count,
                time.time()
            ))
            
            conn.commit()
    
    def restore_elite_robots(self, world, count: Optional[int] = None, min_generation: int = 0) -> List:
        """
        Restore elite robots for training resumption.
        
        Args:
            world: Physics world to create robots in
            count: Number of elites to restore (None = all available)
            min_generation: Minimum generation to consider
            
        Returns:
            List of restored robot agents
        """
        try:
            # Get elite records from database
            with sqlite3.connect(self.elite_db_path) as conn:
                cursor = conn.cursor()
                
                query = '''
                    SELECT robot_id, generation, fitness, rank, preservation_timestamp, notes
                    FROM elite_robots 
                    WHERE generation >= ?
                    ORDER BY fitness DESC, generation DESC
                '''
                
                if count:
                    query += f' LIMIT {count}'
                
                cursor.execute(query, (min_generation,))
                elite_records = cursor.fetchall()
            
            if not elite_records:
                self.logger.warning(f"No elite robots found for restoration (min_generation: {min_generation})")
                return []
            
            # Restore robots
            restored_robots = []
            
            for robot_id, generation, fitness, rank, timestamp, notes in elite_records:
                try:
                    # Load robot from storage
                    robot = self.robot_storage.load_robot(robot_id, world)
                    restored_robots.append(robot)
                    
                    self.logger.info(f"🔄 Restored elite robot {robot_id} (gen {generation}, fitness {fitness:.3f})")
                    
                except Exception as e:
                    self.logger.error(f"Failed to restore elite robot {robot_id}: {e}")
                    continue
            
            print(f"🔄 Restored {len(restored_robots)} elite robots from storage")
            return restored_robots
            
        except Exception as e:
            self.logger.error(f"Failed to restore elite robots: {e}")
            return []
    
    def get_elite_statistics(self) -> Dict[str, Any]:
        """Get comprehensive elite preservation statistics."""
        try:
            with sqlite3.connect(self.elite_db_path) as conn:
                cursor = conn.cursor()
                
                # Basic counts
                cursor.execute('SELECT COUNT(*) FROM elite_robots')
                total_elites = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(DISTINCT generation) FROM elite_robots')
                generations_with_elites = cursor.fetchone()[0]
                
                # Fitness statistics
                cursor.execute('SELECT MAX(fitness), AVG(fitness), MIN(fitness) FROM elite_robots')
                max_fitness, avg_fitness, min_fitness = cursor.fetchone()
                
                # Recent preservation activity
                cursor.execute('''
                    SELECT generation, elites_preserved, best_fitness 
                    FROM preservation_history 
                    ORDER BY generation DESC 
                    LIMIT 10
                ''')
                recent_activity = cursor.fetchall()
                
                # Generation distribution
                cursor.execute('''
                    SELECT generation, COUNT(*) as count, MAX(fitness) as best_fitness
                    FROM elite_robots 
                    GROUP BY generation 
                    ORDER BY generation DESC 
                    LIMIT 20
                ''')
                generation_distribution = cursor.fetchall()
                
                return {
                    'total_elites_stored': total_elites,
                    'generations_with_elites': generations_with_elites,
                    'max_elite_storage': self.max_elite_storage,
                    'storage_utilization': total_elites / self.max_elite_storage if self.max_elite_storage > 0 else 0,
                    'fitness_statistics': {
                        'max': float(max_fitness) if max_fitness else 0.0,
                        'avg': float(avg_fitness) if avg_fitness else 0.0,
                        'min': float(min_fitness) if min_fitness else 0.0
                    },
                    'recent_preservation_activity': [
                        {
                            'generation': gen,
                            'elites_preserved': count,
                            'best_fitness': fitness
                        }
                        for gen, count, fitness in recent_activity
                    ],
                    'generation_distribution': [
                        {
                            'generation': gen,
                            'elite_count': count,
                            'best_fitness': fitness
                        }
                        for gen, count, fitness in generation_distribution
                    ]
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get elite statistics: {e}")
            return {}
    
    def get_top_elites(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get top performing elite robots across all generations."""
        try:
            with sqlite3.connect(self.elite_db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT robot_id, generation, fitness, rank, preservation_timestamp, notes
                    FROM elite_robots 
                    ORDER BY fitness DESC 
                    LIMIT ?
                ''', (count,))
                
                results = cursor.fetchall()
                
                return [
                    {
                        'robot_id': robot_id,
                        'generation': generation,
                        'fitness': fitness,
                        'rank': rank,
                        'preservation_date': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp)),
                        'notes': notes
                    }
                    for robot_id, generation, fitness, rank, timestamp, notes in results
                ]
                
        except Exception as e:
            self.logger.error(f"Failed to get top elites: {e}")
            return []
    
    def _create_preservation_result(self, generation: int, preserved_count: int, 
                                  records: List[EliteRobotRecord], start_time: float,
                                  error: Optional[str] = None) -> Dict[str, Any]:
        """Create standardized preservation result."""
        return {
            'generation': generation,
            'elites_preserved': preserved_count,
            'preservation_time': time.time() - start_time,
            'elite_records': [record.to_dict() for record in records],
            'success': error is None,
            'error': error
        } 