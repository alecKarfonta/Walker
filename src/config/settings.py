"""
Game preferences and configuration management.
"""

import json
import os
from typing import Dict, Any
from . import constants


class GamePreferences:
    """Singleton class for managing game preferences and settings."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GamePreferences, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # World Properties
        self.timestep = constants.DEFAULT_TIMESTEP
        self.gravity = constants.GRAVITY[1]  # Y component
        self.update_timer = 0.1
        self.position_iterations = constants.DEFAULT_POSITION_ITERATIONS
        self.velocity_iterations = constants.DEFAULT_VELOCITY_ITERATIONS
        
        # Player Properties
        self.arm_range = 60
        self.wrist_range = 180
        self.arm_speed = 1.0
        self.wrist_speed = 3.0
        self.arm_torque = 2000.0
        self.wrist_torque = 4000.0
        self.suspension = 10.0
        self.density = constants.ROBOT_DENSITY
        self.friction = constants.ROBOT_FRICTION
        self.linear_dampening = constants.ROBOT_LINEAR_DAMPING
        
        # Learning Properties
        self.randomness = constants.DEFAULT_EXPLORATION_RATE
        self.min_randomness = 0.001
        self.max_randomness = 0.2
        self.learning_rate = constants.DEFAULT_LEARNING_RATE
        self.min_learning_rate = 0.001
        self.max_learning_rate = 1.0
        self.future_discount = constants.DEFAULT_FUTURE_DISCOUNT
        self.exploration_bonus = 100.0
        self.impatience = 0.0001
        self.speed_value_weight = 1.0
        self.mutation_rate = constants.DEFAULT_MUTATION_RATE
        
        # Game Properties
        self.is_showing_stats = False
        self.sound = True
        self.music = True
        self.use_accelerometer = True
        self.vol_sound = 0.5
        self.vol_music = 0.5
        self.show_fps_counter = False
        self.use_monochrome_shader = False
        
        # UI Properties
        self.padding = 10
        self.slide_width = 400
        
        self._initialized = True
        self._prefs_file = "default.prefs"
        self.load()
    
    def load(self):
        """Load preferences from file."""
        try:
            if os.path.exists(self._prefs_file):
                with open(self._prefs_file, 'r') as f:
                    data = json.load(f)
                    for key, value in data.items():
                        if hasattr(self, key):
                            setattr(self, key, value)
        except Exception as e:
            print(f"Error loading preferences: {e}")
    
    def save(self):
        """Save preferences to file."""
        try:
            data = {}
            for key, value in self.__dict__.items():
                if not key.startswith('_'):
                    data[key] = value
            
            with open(self._prefs_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving preferences: {e}")
    
    def clear(self):
        """Clear all preferences and reset to defaults."""
        if os.path.exists(self._prefs_file):
            os.remove(self._prefs_file)
        self.__init__()


# Global instance
instance = GamePreferences() 