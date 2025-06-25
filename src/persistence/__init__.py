"""
Robot Persistence System

This module provides complete robot state storage and restoration capabilities.
Supports saving and loading all robot data including physical parameters, 
Q-tables, and performance metrics.
"""

# Import persistence classes when needed to avoid circular imports
try:
    from .robot_storage import RobotStorage, RobotState, PerformanceHistory
    from .storage_manager import StorageManager
    from .elite_manager import EliteManager, EliteRobotRecord
except ImportError:
    # Fallback for environments where Box2D is not available
    RobotStorage = None
    RobotState = None
    PerformanceHistory = None
    StorageManager = None
    EliteManager = None
    EliteRobotRecord = None

__all__ = [
    'RobotStorage',
    'RobotState', 
    'PerformanceHistory',
    'StorageManager',
    'EliteManager',
    'EliteRobotRecord'
] 