"""Environmental Challenges for Enhanced Evolution"""

import random
import time
from typing import List, Dict, Tuple
from enum import Enum

class WeatherType(Enum):
    NORMAL = "normal"
    WINDY = "windy" 
    ICY = "icy"
    HOT = "hot"

class EnvironmentalSystem:
    """Simple environmental challenge system"""
    
    def __init__(self):
        self.weather = WeatherType.NORMAL
        self.obstacles = []
        self.generation = 0
    
    def update_environment(self, generation: int):
        """Update environmental conditions"""
        self.generation = generation
        
        # Weather changes every 10 generations
        if generation % 10 == 0:
            old_weather = self.weather
            self.weather = random.choice(list(WeatherType))
            if old_weather != self.weather:
                print(f"🌦️ Weather changed to {self.weather.value}")
        
        # Spawn obstacles randomly
        if random.random() < 0.1:  # 10% chance
            obstacle = {
                'type': random.choice(['boulder', 'pit', 'wall']),
                'position': (random.uniform(-50, 50), random.uniform(0, 20)),
                'created': generation
            }
            self.obstacles.append(obstacle)
            print(f"🗿 New {obstacle['type']} obstacle spawned")
        
        # Remove old obstacles
        self.obstacles = [obs for obs in self.obstacles if generation - obs['created'] < 20]
    
    def get_effects(self, position: Tuple[float, float]) -> Dict[str, float]:
        """Get environmental effects at position"""
        effects = {'movement_penalty': 0.0, 'energy_cost': 0.0}
        
        # Weather effects
        if self.weather == WeatherType.WINDY:
            effects['movement_penalty'] = 0.15
        elif self.weather == WeatherType.ICY:
            effects['movement_penalty'] = 0.3
        elif self.weather == WeatherType.HOT:
            effects['energy_cost'] = 0.1
        
        return effects
    
    def get_status(self) -> Dict:
        """Get current environmental status"""
        return {
            'weather': self.weather.value,
            'obstacles': len(self.obstacles),
            'generation': self.generation
        } 