"""
Storage Manager

High-level storage management for robot persistence system.
Provides utilities for managing robot storage, backups, and batch operations.
"""

import time
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import json
import numpy as np

from .robot_storage import RobotStorage, RobotState, PerformanceHistory


@dataclass
class StorageStats:
    """Storage system statistics."""
    total_robots: int = 0
    total_snapshots: int = 0
    storage_size_mb: float = 0.0
    oldest_robot_date: Optional[str] = None
    newest_robot_date: Optional[str] = None
    backup_count: int = 0


class StorageManager:
    """
    High-level storage management system.
    Provides utilities for managing robot storage, backups, and batch operations.
    """
    
    def __init__(self, storage_directory: str = "robot_storage"):
        """Initialize storage manager."""
        self.robot_storage = RobotStorage(storage_directory)
        self.storage_dir = Path(storage_directory)
        
        # Auto-save settings
        self.auto_save_enabled = False
        self.auto_save_interval = 300  # 5 minutes
        self.last_auto_save = time.time()
        
        # Backup settings
        self.backup_enabled = True
        self.max_backups = 10
        
        print(f"🗂️  Storage manager initialized")
    
    def enable_auto_save(self, interval_seconds: int = 300):
        """Enable automatic saving of robots."""
        self.auto_save_enabled = True
        self.auto_save_interval = interval_seconds
        self.last_auto_save = time.time()
        print(f"💾 Auto-save enabled (interval: {interval_seconds}s)")
    
    def disable_auto_save(self):
        """Disable automatic saving."""
        self.auto_save_enabled = False
        print("💾 Auto-save disabled")
    
    def check_auto_save(self, agents: List) -> bool:
        """Check if auto-save should trigger and save if needed."""
        if not self.auto_save_enabled:
            return False
        
        current_time = time.time()
        if current_time - self.last_auto_save >= self.auto_save_interval:
            self.save_population_checkpoint(agents, auto_save=True)
            self.last_auto_save = current_time
            return True
        
        return False
    
    def save_population_checkpoint(self, agents: List, auto_save: bool = False) -> str:
        """Save a checkpoint of the entire population."""
        checkpoint_name = f"checkpoint_{int(time.time())}"
        if auto_save:
            checkpoint_name = f"auto_save_{int(time.time())}"
        
        try:
            # Save as snapshot
            snapshot_name = self.robot_storage.create_snapshot(agents, checkpoint_name)
            
            # Create backup if enabled
            if self.backup_enabled and not auto_save:
                self._create_backup()
            
            print(f"✅ Population checkpoint saved: {snapshot_name}")
            return snapshot_name
            
        except Exception as e:
            print(f"❌ Error saving population checkpoint: {e}")
            raise
    
    def save_elite_robots(self, agents: List, top_n: int = 5) -> str:
        """Save only the top performing robots."""
        # Sort agents by total reward
        sorted_agents = sorted(agents, 
                             key=lambda a: getattr(a, 'total_reward', 0.0), 
                             reverse=True)
        
        elite_agents = sorted_agents[:top_n]
        
        elite_name = f"elite_top_{top_n}_{int(time.time())}"
        snapshot_name = self.robot_storage.create_snapshot(elite_agents, elite_name)
        
        print(f"🏆 Saved top {len(elite_agents)} elite robots: {snapshot_name}")
        return snapshot_name
    
    def load_best_robots(self, count: int = 5, world=None) -> List:
        """Load the best performing robots from storage."""
        if not world:
            raise ValueError("World parameter required for loading robots")
        
        # Get all saved robots
        saved_robots = self.robot_storage.list_saved_robots()
        
        # Sort by total reward
        sorted_robots = sorted(saved_robots, 
                             key=lambda r: r['total_reward'], 
                             reverse=True)
        
        # Load top performers
        loaded_robots = []
        for robot_info in sorted_robots[:count]:
            try:
                robot = self.robot_storage.load_robot(robot_info['robot_id'], world)
                loaded_robots.append(robot)
            except Exception as e:
                print(f"⚠️  Failed to load robot {robot_info['robot_id']}: {e}")
                continue
        
        print(f"📂 Loaded {len(loaded_robots)} best robots from storage")
        return loaded_robots
    
    def cleanup_old_saves(self, keep_days: int = 30):
        """Clean up old robot saves."""
        cutoff_time = time.time() - (keep_days * 24 * 60 * 60)
        cleaned = 0
        
        saved_robots = self.robot_storage.list_saved_robots()
        
        for robot_info in saved_robots:
            if robot_info['save_timestamp'] < cutoff_time:
                # Keep elite performers even if old
                if robot_info['total_reward'] < 1.0:  # Threshold for "elite"
                    self.robot_storage.delete_robot(robot_info['robot_id'])
                    cleaned += 1
        
        print(f"🧹 Cleaned up {cleaned} old robot saves (older than {keep_days} days)")
        return cleaned
    
    def get_storage_stats(self) -> StorageStats:
        """Get comprehensive storage statistics."""
        stats = StorageStats()
        
        # Count robots
        saved_robots = self.robot_storage.list_saved_robots()
        stats.total_robots = len(saved_robots)
        
        if saved_robots:
            # Find oldest and newest
            sorted_by_date = sorted(saved_robots, key=lambda r: r['save_timestamp'])
            stats.oldest_robot_date = sorted_by_date[0]['save_date']
            stats.newest_robot_date = sorted_by_date[-1]['save_date']
        
        # Count snapshots
        snapshots_dir = self.storage_dir / "snapshots"
        if snapshots_dir.exists():
            stats.total_snapshots = len(list(snapshots_dir.iterdir()))
        
        # Count backups
        backups_dir = self.storage_dir / "backups"
        if backups_dir.exists():
            stats.backup_count = len(list(backups_dir.iterdir()))
        
        # Calculate storage size
        if self.storage_dir.exists():
            total_size = sum(f.stat().st_size for f in self.storage_dir.rglob('*') if f.is_file())
            stats.storage_size_mb = total_size / (1024 * 1024)
        
        return stats
    
    def export_population_analysis(self, export_path: Optional[str] = None) -> str:
        """Export comprehensive analysis of saved robot population."""
        if not export_path:
            export_path = str(self.storage_dir / f"population_analysis_{int(time.time())}.json")
        
        saved_robots = self.robot_storage.list_saved_robots()
        
        # Calculate statistics
        rewards = [r['total_reward'] for r in saved_robots]
        distances = [r['max_distance'] for r in saved_robots]
        speeds = [r['max_speed'] for r in saved_robots]
        generations = [r['generation'] for r in saved_robots]
        
        analysis = {
            'analysis_timestamp': time.time(),
            'total_robots': len(saved_robots),
            'statistics': {
                'reward': {
                    'mean': float(np.mean(rewards)) if rewards else 0.0,
                    'std': float(np.std(rewards)) if rewards else 0.0,
                    'min': float(min(rewards)) if rewards else 0.0,
                    'max': float(max(rewards)) if rewards else 0.0
                },
                'distance': {
                    'mean': float(np.mean(distances)) if distances else 0.0,
                    'std': float(np.std(distances)) if distances else 0.0,
                    'min': float(min(distances)) if distances else 0.0,
                    'max': float(max(distances)) if distances else 0.0
                },
                'speed': {
                    'mean': float(np.mean(speeds)) if speeds else 0.0,
                    'std': float(np.std(speeds)) if speeds else 0.0,
                    'min': float(min(speeds)) if speeds else 0.0,
                    'max': float(max(speeds)) if speeds else 0.0
                },
                'generation': {
                    'mean': float(np.mean(generations)) if generations else 0.0,
                    'std': float(np.std(generations)) if generations else 0.0,
                    'min': int(min(generations)) if generations else 0,
                    'max': int(max(generations)) if generations else 0
                }
            },
            'top_performers': sorted(saved_robots, key=lambda r: r['total_reward'], reverse=True)[:10],
            'generation_distribution': self._analyze_generation_distribution(saved_robots),
            'performance_trends': self._analyze_performance_trends(saved_robots)
        }
        
        with open(export_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"📊 Population analysis exported to {export_path}")
        return export_path
    
    def find_similar_robots(self, target_robot_id: str, similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Find robots with similar characteristics to the target robot."""
        try:
            target_state = self.robot_storage._load_robot_state(target_robot_id)
            target_params = target_state.physical_parameters
            
            similar_robots = []
            saved_robots = self.robot_storage.list_saved_robots()
            
            for robot_info in saved_robots:
                if robot_info['robot_id'] == target_robot_id:
                    continue
                
                try:
                    robot_state = self.robot_storage._load_robot_state(robot_info['robot_id'])
                    similarity = self._calculate_parameter_similarity(target_params, robot_state.physical_parameters)
                    
                    if similarity >= similarity_threshold:
                        similar_robots.append({
                            'robot_id': robot_info['robot_id'],
                            'similarity': similarity,
                            'total_reward': robot_info['total_reward'],
                            'generation': robot_info['generation']
                        })
                except:
                    continue
            
            # Sort by similarity
            similar_robots.sort(key=lambda r: r['similarity'], reverse=True)
            
            print(f"🔍 Found {len(similar_robots)} robots similar to {target_robot_id}")
            return similar_robots
            
        except Exception as e:
            print(f"❌ Error finding similar robots: {e}")
            return []
    
    def _create_backup(self):
        """Create a backup of the entire storage system."""
        backup_name = f"backup_{int(time.time())}"
        backup_path = self.storage_dir / "backups" / backup_name
        
        try:
            # Copy entire storage directory
            shutil.copytree(self.storage_dir / "robots", backup_path / "robots")
            shutil.copytree(self.storage_dir / "snapshots", backup_path / "snapshots")
            
            print(f"💼 Created backup: {backup_name}")
            
            # Clean up old backups
            self._cleanup_old_backups()
            
        except Exception as e:
            print(f"⚠️  Error creating backup: {e}")
    
    def _cleanup_old_backups(self):
        """Remove old backups to stay within limit."""
        backups_dir = self.storage_dir / "backups"
        if not backups_dir.exists():
            return
        
        backups = sorted(backups_dir.iterdir(), key=lambda p: p.stat().st_mtime)
        
        while len(backups) > self.max_backups:
            old_backup = backups.pop(0)
            shutil.rmtree(old_backup)
            print(f"🗑️  Removed old backup: {old_backup.name}")
    
    def _analyze_generation_distribution(self, robots: List[Dict]) -> Dict[str, int]:
        """Analyze generation distribution."""
        generations = [r['generation'] for r in robots]
        
        distribution = {}
        for gen in set(generations):
            distribution[f"generation_{gen}"] = generations.count(gen)
        
        return distribution
    
    def _analyze_performance_trends(self, robots: List[Dict]) -> Dict[str, Any]:
        """Analyze performance trends across generations."""
        if not robots:
            return {}
        
        # Group by generation
        gen_performance = {}
        for robot in robots:
            gen = robot['generation']
            if gen not in gen_performance:
                gen_performance[gen] = []
            gen_performance[gen].append(robot['total_reward'])
        
        # Calculate trends
        trends = {}
        for gen, rewards in gen_performance.items():
            trends[f"gen_{gen}"] = {
                'mean_reward': float(np.mean(rewards)),
                'max_reward': float(max(rewards)),
                'robot_count': len(rewards)
            }
        
        return trends
    
    def _calculate_parameter_similarity(self, params1: Dict, params2: Dict) -> float:
        """Calculate similarity between two parameter sets."""
        if not params1 or not params2:
            return 0.0
        
        # Compare key parameters
        key_params = ['body_width', 'body_height', 'motor_torque', 'motor_speed', 
                     'learning_rate', 'epsilon', 'discount_factor']
        
        similarities = []
        for param in key_params:
            if param in params1 and param in params2:
                val1, val2 = params1[param], params2[param]
                if val1 == 0 and val2 == 0:
                    similarity = 1.0
                elif val1 == 0 or val2 == 0:
                    similarity = 0.0
                else:
                    similarity = 1.0 - abs(val1 - val2) / max(abs(val1), abs(val2))
                similarities.append(similarity)
        
        return float(np.mean(similarities)) if similarities else 0.0 