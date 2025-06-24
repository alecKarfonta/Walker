"""
Environmental Challenges System for Enhanced Evolution

This module provides dynamic environmental challenges that create
selection pressures and make evolution more interesting.
"""

import random
import time
import math
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np


class WeatherCondition(Enum):
    """Different weather conditions that affect agent performance."""
    NORMAL = "normal"
    WINDY = "windy"
    ICY = "icy"
    HOT = "hot"
    STORMY = "stormy"
    FOGGY = "foggy"


class ObstacleType(Enum):
    """Types of obstacles that can appear in the environment."""
    BOULDER = "boulder"
    PIT = "pit"
    WALL = "wall"
    MOVING_PLATFORM = "moving_platform"
    SPIKE_TRAP = "spike_trap"
    QUICKSAND = "quicksand"
    ICE_PATCH = "ice_patch"
    FIRE_HAZARD = "fire_hazard"


@dataclass
class EnvironmentalObstacle:
    """Represents an obstacle in the environment."""
    type: ObstacleType
    position: Tuple[float, float]
    size: float
    active: bool = True
    movement_pattern: Optional[str] = None
    movement_speed: float = 0.0
    danger_level: float = 1.0
    creation_time: float = 0.0
    
    def __post_init__(self):
        """Initialize creation time."""
        self.creation_time = time.time()


@dataclass
class TerrainModification:
    """Represents a modification to the terrain."""
    type: str  # "elevation", "friction", "texture"
    area: Tuple[float, float, float, float]  # x, y, width, height
    intensity: float
    duration: float
    created_at: float = 0.0
    
    def __post_init__(self):
        """Initialize creation time."""
        self.created_at = time.time()


class EnvironmentalChallengeSystem:
    """
    Manages dynamic environmental challenges that create evolutionary pressure.
    
    Features:
    - Dynamic obstacle spawning and removal
    - Weather system with effects on movement
    - Terrain modifications (elevation, friction, hazards)
    - Seasonal environmental changes
    - Catastrophic events
    - Resource distribution changes
    """
    
    def __init__(self, world_bounds: Tuple[float, float, float, float] = (-100, -10, 200, 50)):
        """Initialize the environmental challenge system."""
        self.world_bounds = world_bounds  # (min_x, min_y, max_x, max_y)
        
        # Current environmental state
        self.obstacles: List[EnvironmentalObstacle] = []
        self.terrain_modifications: List[TerrainModification] = []
        self.current_weather = WeatherCondition.NORMAL
        self.weather_intensity = 1.0
        self.base_terrain_difficulty = 1.0
        
        # Environmental history
        self.weather_history: List[Tuple[float, WeatherCondition]] = []
        self.obstacle_history: List[Dict] = []
        
        # Challenge parameters
        self.max_obstacles = 20
        self.obstacle_spawn_rate = 0.08
        self.obstacle_decay_rate = 0.02
        self.weather_change_probability = 0.05
        self.terrain_change_probability = 0.03
        
        # Seasonal effects
        self.season_multipliers = {
            'spring': {'growth': 1.2, 'obstacles': 0.8, 'weather_stability': 0.9},
            'summer': {'growth': 1.0, 'obstacles': 1.0, 'weather_stability': 1.0},
            'autumn': {'growth': 0.8, 'obstacles': 1.3, 'weather_stability': 0.7},
            'winter': {'growth': 0.6, 'obstacles': 1.1, 'weather_stability': 0.5}
        }
        
        # Event system
        self.event_log: List[Dict] = []
        self.catastrophe_cooldown = 0
        
    def update(self, generation: int, season: str = 'summer') -> Dict[str, Any]:
        """Update environmental challenges for the current generation."""
        updates = {
            'obstacles_added': 0,
            'obstacles_removed': 0,
            'weather_changed': False,
            'terrain_modified': False,
            'events': []
        }
        
        # Apply seasonal multipliers
        season_effects = self.season_multipliers.get(season, self.season_multipliers['summer'])
        
        # Update obstacles
        obstacle_updates = self._update_obstacles(season_effects)
        updates.update(obstacle_updates)
        
        # Update weather
        weather_updates = self._update_weather(season_effects)
        updates.update(weather_updates)
        
        # Update terrain
        terrain_updates = self._update_terrain(season_effects)
        updates.update(terrain_updates)
        
        # Check for catastrophic events
        catastrophe_updates = self._check_catastrophic_events(generation, season_effects)
        updates.update(catastrophe_updates)
        
        # Log environmental state
        self._log_environmental_state(generation)
        
        return updates
    
    def _update_obstacles(self, season_effects: Dict) -> Dict[str, Any]:
        """Update obstacle spawning and removal."""
        updates = {'obstacles_added': 0, 'obstacles_removed': 0}
        
        # Remove expired obstacles
        initial_count = len(self.obstacles)
        self.obstacles = [obs for obs in self.obstacles if self._should_keep_obstacle(obs)]
        removed_count = initial_count - len(self.obstacles)
        updates['obstacles_removed'] = removed_count
        
        # Spawn new obstacles
        spawn_rate = self.obstacle_spawn_rate * season_effects.get('obstacles', 1.0)
        
        while len(self.obstacles) < self.max_obstacles and random.random() < spawn_rate:
            obstacle = self._create_random_obstacle()
            self.obstacles.append(obstacle)
            updates['obstacles_added'] += 1
            
            self._log_event('obstacle_spawned', {
                'type': obstacle.type.value,
                'position': obstacle.position,
                'size': obstacle.size
            })
        
        # Update moving obstacles
        for obstacle in self.obstacles:
            if obstacle.movement_pattern:
                self._update_obstacle_movement(obstacle)
        
        return updates
    
    def _update_weather(self, season_effects: Dict) -> Dict[str, Any]:
        """Update weather conditions."""
        updates = {'weather_changed': False}
        
        # Weather change probability affected by season stability
        change_prob = self.weather_change_probability / season_effects.get('weather_stability', 1.0)
        
        if random.random() < change_prob:
            old_weather = self.current_weather
            self.current_weather = self._select_new_weather()
            self.weather_intensity = random.uniform(0.5, 2.0)
            
            if old_weather != self.current_weather:
                updates['weather_changed'] = True
                self.weather_history.append((time.time(), self.current_weather))
                
                self._log_event('weather_change', {
                    'old_weather': old_weather.value,
                    'new_weather': self.current_weather.value,
                    'intensity': self.weather_intensity
                })
        
        return updates
    
    def _update_terrain(self, season_effects: Dict) -> Dict[str, Any]:
        """Update terrain modifications."""
        updates = {'terrain_modified': False}
        
        # Remove expired terrain modifications
        current_time = time.time()
        self.terrain_modifications = [
            mod for mod in self.terrain_modifications 
            if current_time - mod.created_at < mod.duration
        ]
        
        # Add new terrain modifications
        if random.random() < self.terrain_change_probability:
            modification = self._create_terrain_modification()
            self.terrain_modifications.append(modification)
            updates['terrain_modified'] = True
            
            self._log_event('terrain_modified', {
                'type': modification.type,
                'area': modification.area,
                'intensity': modification.intensity
            })
        
        return updates
    
    def _check_catastrophic_events(self, generation: int, season_effects: Dict) -> Dict[str, Any]:
        """Check for and trigger catastrophic events."""
        updates = {'catastrophic_event': None}
        
        if self.catastrophe_cooldown > 0:
            self.catastrophe_cooldown -= 1
            return updates
        
        # Base catastrophe probability (very low)
        catastrophe_prob = 0.005  # 0.5% per generation
        
        # Increased probability in unstable seasons
        if season_effects.get('weather_stability', 1.0) < 0.8:
            catastrophe_prob *= 2.0
        
        if random.random() < catastrophe_prob:
            event_type = random.choice([
                'earthquake', 'flood', 'wildfire', 'blizzard', 
                'meteor_shower', 'volcanic_eruption'
            ])
            
            catastrophe = self._trigger_catastrophic_event(event_type)
            updates['catastrophic_event'] = catastrophe
            
            # Set cooldown to prevent frequent catastrophes
            self.catastrophe_cooldown = random.randint(20, 50)
        
        return updates
    
    def _should_keep_obstacle(self, obstacle: EnvironmentalObstacle) -> bool:
        """Determine if an obstacle should be kept."""
        # Age-based removal
        age = time.time() - obstacle.creation_time
        max_age = random.uniform(300, 1200)  # 5-20 minutes
        
        if age > max_age:
            return False
        
        # Random decay
        if random.random() < self.obstacle_decay_rate:
            return False
        
        # Distance-based removal (remove obstacles that are too far)
        x, y = obstacle.position
        if (x < self.world_bounds[0] - 50 or x > self.world_bounds[2] + 50 or
            y < self.world_bounds[1] - 10 or y > self.world_bounds[3] + 10):
            return False
        
        return True
    
    def _create_random_obstacle(self) -> EnvironmentalObstacle:
        """Create a random obstacle within world bounds."""
        obstacle_type = random.choice(list(ObstacleType))
        
        # Position within expanded world bounds
        x = random.uniform(self.world_bounds[0] - 20, self.world_bounds[2] + 20)
        y = random.uniform(self.world_bounds[1], self.world_bounds[3])
        
        # Size based on obstacle type
        size_ranges = {
            ObstacleType.BOULDER: (2, 8),
            ObstacleType.PIT: (3, 12),
            ObstacleType.WALL: (1, 20),
            ObstacleType.MOVING_PLATFORM: (4, 10),
            ObstacleType.SPIKE_TRAP: (2, 5),
            ObstacleType.QUICKSAND: (5, 15),
            ObstacleType.ICE_PATCH: (3, 10),
            ObstacleType.FIRE_HAZARD: (2, 6)
        }
        
        size_range = size_ranges.get(obstacle_type, (2, 8))
        size = random.uniform(*size_range)
        
        # Movement for certain obstacles
        movement_pattern = None
        movement_speed = 0.0
        
        if obstacle_type == ObstacleType.MOVING_PLATFORM:
            movement_pattern = random.choice(['horizontal', 'vertical', 'circular'])
            movement_speed = random.uniform(0.5, 2.0)
        
        # Danger level
        danger_levels = {
            ObstacleType.BOULDER: 0.3,
            ObstacleType.PIT: 0.8,
            ObstacleType.WALL: 0.1,
            ObstacleType.MOVING_PLATFORM: 0.2,
            ObstacleType.SPIKE_TRAP: 1.0,
            ObstacleType.QUICKSAND: 0.9,
            ObstacleType.ICE_PATCH: 0.4,
            ObstacleType.FIRE_HAZARD: 0.7
        }
        
        danger_level = danger_levels.get(obstacle_type, 0.5)
        
        return EnvironmentalObstacle(
            type=obstacle_type,
            position=(x, y),
            size=size,
            movement_pattern=movement_pattern,
            movement_speed=movement_speed,
            danger_level=danger_level
        )
    
    def _update_obstacle_movement(self, obstacle: EnvironmentalObstacle):
        """Update position of moving obstacles."""
        if not obstacle.movement_pattern:
            return
        
        x, y = obstacle.position
        speed = obstacle.movement_speed
        time_factor = time.time() * 0.1  # Slow down movement
        
        if obstacle.movement_pattern == 'horizontal':
            new_x = x + math.sin(time_factor) * speed
            obstacle.position = (new_x, y)
        elif obstacle.movement_pattern == 'vertical':
            new_y = y + math.sin(time_factor) * speed
            obstacle.position = (x, new_y)
        elif obstacle.movement_pattern == 'circular':
            radius = 5.0
            new_x = x + math.cos(time_factor) * radius
            new_y = y + math.sin(time_factor) * radius
            obstacle.position = (new_x, new_y)
    
    def _select_new_weather(self) -> WeatherCondition:
        """Select a new weather condition based on current conditions."""
        # Weather transition probabilities
        transitions = {
            WeatherCondition.NORMAL: [
                (WeatherCondition.WINDY, 0.3),
                (WeatherCondition.HOT, 0.2),
                (WeatherCondition.FOGGY, 0.2),
                (WeatherCondition.NORMAL, 0.3)
            ],
            WeatherCondition.WINDY: [
                (WeatherCondition.STORMY, 0.4),
                (WeatherCondition.NORMAL, 0.4),
                (WeatherCondition.FOGGY, 0.2)
            ],
            WeatherCondition.ICY: [
                (WeatherCondition.NORMAL, 0.5),
                (WeatherCondition.FOGGY, 0.3),
                (WeatherCondition.WINDY, 0.2)
            ],
            WeatherCondition.HOT: [
                (WeatherCondition.NORMAL, 0.4),
                (WeatherCondition.WINDY, 0.3),
                (WeatherCondition.STORMY, 0.3)
            ],
            WeatherCondition.STORMY: [
                (WeatherCondition.WINDY, 0.4),
                (WeatherCondition.NORMAL, 0.4),
                (WeatherCondition.ICY, 0.2)
            ],
            WeatherCondition.FOGGY: [
                (WeatherCondition.NORMAL, 0.6),
                (WeatherCondition.WINDY, 0.2),
                (WeatherCondition.HOT, 0.2)
            ]
        }
        
        current_transitions = transitions.get(self.current_weather, 
                                            [(WeatherCondition.NORMAL, 1.0)])
        
        # Select based on probabilities
        rand = random.random()
        cumulative = 0.0
        
        for weather, prob in current_transitions:
            cumulative += prob
            if rand <= cumulative:
                return weather
        
        return WeatherCondition.NORMAL
    
    def _create_terrain_modification(self) -> TerrainModification:
        """Create a random terrain modification."""
        mod_type = random.choice(['elevation', 'friction', 'texture'])
        
        # Random area within world bounds
        x = random.uniform(self.world_bounds[0], self.world_bounds[2])
        y = random.uniform(self.world_bounds[1], self.world_bounds[3])
        width = random.uniform(5, 25)
        height = random.uniform(5, 15)
        
        # Intensity based on type
        intensity_ranges = {
            'elevation': (-5.0, 5.0),    # Height change
            'friction': (0.1, 3.0),     # Friction multiplier
            'texture': (0.5, 2.0)       # Surface roughness
        }
        
        intensity_range = intensity_ranges.get(mod_type, (0.5, 2.0))
        intensity = random.uniform(*intensity_range)
        
        # Duration (in seconds)
        duration = random.uniform(120, 600)  # 2-10 minutes
        
        return TerrainModification(
            type=mod_type,
            area=(x, y, width, height),
            intensity=intensity,
            duration=duration
        )
    
    def _trigger_catastrophic_event(self, event_type: str) -> Dict[str, Any]:
        """Trigger a catastrophic environmental event."""
        event = {
            'type': event_type,
            'timestamp': time.time(),
            'effects': {}
        }
        
        if event_type == 'earthquake':
            # Remove random obstacles, add rubble
            obstacles_removed = min(5, len(self.obstacles))
            for _ in range(obstacles_removed):
                if self.obstacles:
                    self.obstacles.pop(random.randint(0, len(self.obstacles) - 1))
            
            # Add rubble obstacles
            for _ in range(obstacles_removed * 2):
                rubble = self._create_random_obstacle()
                rubble.type = ObstacleType.BOULDER
                rubble.size *= 0.7  # Smaller rubble
                self.obstacles.append(rubble)
            
            event['effects']['obstacles_removed'] = obstacles_removed
            event['effects']['rubble_added'] = obstacles_removed * 2
            
        elif event_type == 'flood':
            # Change weather to stormy, add water hazards
            self.current_weather = WeatherCondition.STORMY
            self.weather_intensity = 2.0
            
            # Add water hazards (quicksand represents water)
            for _ in range(3):
                water = self._create_random_obstacle()
                water.type = ObstacleType.QUICKSAND
                water.size *= 1.5
                self.obstacles.append(water)
            
            event['effects']['weather_change'] = 'stormy'
            event['effects']['water_hazards_added'] = 3
            
        elif event_type == 'wildfire':
            # Add fire hazards, change weather to hot
            self.current_weather = WeatherCondition.HOT
            self.weather_intensity = 1.8
            
            for _ in range(4):
                fire = self._create_random_obstacle()
                fire.type = ObstacleType.FIRE_HAZARD
                self.obstacles.append(fire)
            
            event['effects']['fire_hazards_added'] = 4
            
        elif event_type == 'blizzard':
            # Change weather to icy, add ice patches
            self.current_weather = WeatherCondition.ICY
            self.weather_intensity = 1.9
            
            for _ in range(6):
                ice = self._create_random_obstacle()
                ice.type = ObstacleType.ICE_PATCH
                self.obstacles.append(ice)
            
            event['effects']['ice_patches_added'] = 6
            
        elif event_type == 'meteor_shower':
            # Add spike traps (representing meteor impacts)
            for _ in range(8):
                crater = self._create_random_obstacle()
                crater.type = ObstacleType.SPIKE_TRAP
                crater.size *= 0.8
                self.obstacles.append(crater)
            
            event['effects']['meteor_craters_added'] = 8
            
        elif event_type == 'volcanic_eruption':
            # Add multiple fire hazards and elevation changes
            for _ in range(5):
                lava = self._create_random_obstacle()
                lava.type = ObstacleType.FIRE_HAZARD
                lava.size *= 1.2
                self.obstacles.append(lava)
            
            # Add terrain elevation changes
            for _ in range(3):
                elevation = self._create_terrain_modification()
                elevation.type = 'elevation'
                elevation.intensity = random.uniform(3.0, 8.0)
                self.terrain_modifications.append(elevation)
            
            event['effects']['lava_hazards_added'] = 5
            event['effects']['elevation_changes_added'] = 3
        
        self._log_event('catastrophic_event', event)
        return event
    
    def _log_event(self, event_type: str, details: Dict):
        """Log an environmental event."""
        event = {
            'timestamp': time.time(),
            'type': event_type,
            'details': details
        }
        self.event_log.append(event)
        
        # Keep only recent events
        if len(self.event_log) > 100:
            self.event_log = self.event_log[-50:]
    
    def _log_environmental_state(self, generation: int):
        """Log current environmental state."""
        state = {
            'generation': generation,
            'timestamp': time.time(),
            'obstacles': len(self.obstacles),
            'weather': self.current_weather.value,
            'weather_intensity': self.weather_intensity,
            'terrain_modifications': len(self.terrain_modifications),
            'recent_events': len([e for e in self.event_log 
                                if time.time() - e['timestamp'] < 60])  # Last minute
        }
        
        print(f"🌍 Environmental State Gen {generation}: "
              f"{state['obstacles']} obstacles, {state['weather']} weather "
              f"(intensity {state['weather_intensity']:.1f}), "
              f"{state['terrain_modifications']} terrain mods")
    
    def get_environmental_effects(self, agent_position: Tuple[float, float]) -> Dict[str, float]:
        """Get environmental effects at a specific position."""
        x, y = agent_position
        effects = {
            'movement_multiplier': 1.0,
            'energy_drain': 0.0,
            'stability_penalty': 0.0,
            'danger_level': 0.0
        }
        
        # Weather effects
        weather_effects = {
            WeatherCondition.NORMAL: {'movement_multiplier': 1.0},
            WeatherCondition.WINDY: {'movement_multiplier': 0.85, 'stability_penalty': 0.1},
            WeatherCondition.ICY: {'movement_multiplier': 0.7, 'stability_penalty': 0.2},
            WeatherCondition.HOT: {'energy_drain': 0.1, 'movement_multiplier': 0.9},
            WeatherCondition.STORMY: {'movement_multiplier': 0.6, 'stability_penalty': 0.3, 'energy_drain': 0.05},
            WeatherCondition.FOGGY: {'movement_multiplier': 0.8, 'stability_penalty': 0.05}
        }
        
        weather_effect = weather_effects.get(self.current_weather, {})
        for effect, value in weather_effect.items():
            effects[effect] = effects.get(effect, 0.0) + value * self.weather_intensity
        
        # Obstacle effects (proximity-based)
        for obstacle in self.obstacles:
            ox, oy = obstacle.position
            distance = math.sqrt((x - ox)**2 + (y - oy)**2)
            
            if distance < obstacle.size * 2:  # Within influence range
                proximity_factor = max(0, 1 - distance / (obstacle.size * 2))
                
                # Different obstacles affect agents differently
                if obstacle.type == ObstacleType.QUICKSAND:
                    effects['movement_multiplier'] *= (1 - 0.5 * proximity_factor)
                    effects['energy_drain'] += 0.2 * proximity_factor
                elif obstacle.type == ObstacleType.ICE_PATCH:
                    effects['movement_multiplier'] *= (1 - 0.3 * proximity_factor)
                    effects['stability_penalty'] += 0.3 * proximity_factor
                elif obstacle.type == ObstacleType.FIRE_HAZARD:
                    effects['energy_drain'] += 0.3 * proximity_factor
                    effects['danger_level'] += 0.5 * proximity_factor
                elif obstacle.type == ObstacleType.SPIKE_TRAP:
                    effects['danger_level'] += 0.8 * proximity_factor
                
                effects['danger_level'] += obstacle.danger_level * proximity_factor * 0.1
        
        # Terrain modification effects
        for mod in self.terrain_modifications:
            mx, my, mw, mh = mod.area
            if mx <= x <= mx + mw and my <= y <= my + mh:
                if mod.type == 'friction':
                    effects['movement_multiplier'] *= mod.intensity
                elif mod.type == 'elevation':
                    # Higher elevation = more energy to climb
                    if mod.intensity > 0:
                        effects['energy_drain'] += abs(mod.intensity) * 0.02
                elif mod.type == 'texture':
                    effects['stability_penalty'] += (mod.intensity - 1.0) * 0.1
        
        # Clamp values to reasonable ranges
        effects['movement_multiplier'] = max(0.1, min(2.0, effects['movement_multiplier']))
        effects['energy_drain'] = max(0.0, min(1.0, effects['energy_drain']))
        effects['stability_penalty'] = max(0.0, min(1.0, effects['stability_penalty']))
        effects['danger_level'] = max(0.0, min(2.0, effects['danger_level']))
        
        return effects
    
    def get_challenge_summary(self) -> Dict[str, Any]:
        """Get a summary of current environmental challenges."""
        return {
            'obstacles': {
                'count': len(self.obstacles),
                'types': {obs_type.value: len([o for o in self.obstacles if o.type == obs_type])
                         for obs_type in ObstacleType},
                'moving': len([o for o in self.obstacles if o.movement_pattern])
            },
            'weather': {
                'condition': self.current_weather.value,
                'intensity': self.weather_intensity
            },
            'terrain': {
                'modifications': len(self.terrain_modifications),
                'types': {mod.type: len([m for m in self.terrain_modifications if m.type == mod.type])
                         for mod in self.terrain_modifications}
            },
            'recent_events': [
                {'type': event['type'], 'time_ago': time.time() - event['timestamp']}
                for event in self.event_log[-5:]
            ],
            'danger_zones': len([o for o in self.obstacles if o.danger_level > 0.5])
        } 