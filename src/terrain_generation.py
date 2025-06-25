"""
Robot-Scale Terrain Generation System
Creates navigable terrain features appropriate for 1.5m robots.
Features include gentle slopes, small hills, ramps, and obstacles that robots can actually navigate.
"""

import random
import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

# Robot-scale terrain uses simple mathematical noise generation
# No external noise library needed


class TerrainType(Enum):
    """Types of robot-scale terrain features."""
    SMALL_HILL = "small_hill"           # 0.5-2m tall, navigable slopes
    GENTLE_SLOPE = "gentle_slope"       # Ramps and inclines  
    OBSTACLE = "obstacle"               # Small barriers 0.3-1m tall
    RAMP = "ramp"                      # Designed pathways
    ROUGH_PATCH = "rough_patch"        # Uneven ground
    DEPRESSION = "depression"          # Small dips 0.2-0.8m deep
    RIDGE = "ridge"                    # Linear elevated features


@dataclass
class RobotScaleFeature:
    """Represents a robot-scale terrain feature."""
    type: TerrainType
    center: Tuple[float, float]
    size: float                 # Width/radius in meters
    height: float              # Height in meters (+ up, - down)
    slope_angle: float = 0.0   # For directional features
    friction: float = 0.7
    roughness: float = 0.5
    navigable: bool = True     # Whether robots can traverse this
    properties: Optional[Dict] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


class RobotScaleTerrainMesh:
    """High-resolution terrain mesh for robot navigation."""
    
    def __init__(self, bounds: Tuple[float, float, float, float], resolution: float = 0.25):
        """
        Create terrain mesh optimized for robots.
        
        Args:
            bounds: (x_min, y_min, x_max, y_max) world bounds
            resolution: Grid resolution in meters (0.25m = 25cm for fine detail)
        """
        self.bounds = bounds
        self.resolution = resolution
        
        x_min, y_min, x_max, y_max = bounds
        self.width = int((x_max - x_min) / resolution) + 1
        self.height = int((y_max - y_min) / resolution) + 1
        
        # Initialize flat terrain
        self.elevation = np.zeros((self.height, self.width))
        self.friction = np.full((self.height, self.width), 0.7)
        self.roughness = np.full((self.height, self.width), 0.3)
        
        # Collision margin to prevent embedding (10cm buffer)
        self.collision_margin = 0.1
        
        print(f"🤖 Robot-scale terrain mesh: {self.width}x{self.height} points, {resolution}m resolution")
    
    def world_to_grid(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        x_min, y_min, _, _ = self.bounds
        grid_x = int((world_x - x_min) / self.resolution)
        grid_y = int((world_y - y_min) / self.resolution)
        
        # Clamp to valid range
        grid_x = max(0, min(self.width - 1, grid_x))
        grid_y = max(0, min(self.height - 1, grid_y))
        
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates."""
        x_min, y_min, _, _ = self.bounds
        world_x = x_min + grid_x * self.resolution
        world_y = y_min + grid_y * self.resolution
        return world_x, world_y
    
    def get_elevation(self, world_x: float, world_y: float) -> float:
        """Get elevation at world coordinates with bilinear interpolation."""
        x_min, y_min, _, _ = self.bounds
        
        # Convert to grid space (floating point)
        grid_x = (world_x - x_min) / self.resolution
        grid_y = (world_y - y_min) / self.resolution
        
        # Get integer parts and fractional parts
        x0 = int(grid_x)
        y0 = int(grid_y)
        fx = grid_x - x0
        fy = grid_y - y0
        
        # Clamp to valid range
        x0 = max(0, min(self.width - 2, x0))
        y0 = max(0, min(self.height - 2, y0))
        x1 = x0 + 1
        y1 = y0 + 1
        
        # Bilinear interpolation
        h00 = self.elevation[y0, x0]
        h10 = self.elevation[y0, x1]
        h01 = self.elevation[y1, x0]
        h11 = self.elevation[y1, x1]
        
        h0 = h00 * (1 - fx) + h10 * fx
        h1 = h01 * (1 - fx) + h11 * fx
        
        return h0 * (1 - fy) + h1 * fy
    
    def apply_robot_scale_feature(self, feature: RobotScaleFeature):
        """Apply a robot-scale terrain feature to the mesh."""
        center_x, center_y = feature.center
        center_grid_x, center_grid_y = self.world_to_grid(center_x, center_y)
        
        # Calculate influence radius in grid units
        influence_radius = int(feature.size / self.resolution) + 2
        
        for dy in range(-influence_radius, influence_radius + 1):
            for dx in range(-influence_radius, influence_radius + 1):
                grid_x = center_grid_x + dx
                grid_y = center_grid_y + dy
                
                if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                    # Calculate distance from feature center
                    world_x, world_y = self.grid_to_world(grid_x, grid_y)
                    distance = math.sqrt((world_x - center_x)**2 + (world_y - center_y)**2)
                    
                    if distance <= feature.size:
                        # Calculate influence based on distance and feature type
                        influence = self._calculate_robot_scale_influence(feature, distance, world_x, world_y)
                        
                        if influence > 0:
                            # Apply elevation change
                            self.elevation[grid_y, grid_x] += feature.height * influence
                            
                            # Apply surface properties
                            self.friction[grid_y, grid_x] = (
                                self.friction[grid_y, grid_x] * (1 - influence) +
                                feature.friction * influence
                            )
                            self.roughness[grid_y, grid_x] = (
                                self.roughness[grid_y, grid_x] * (1 - influence) +
                                feature.roughness * influence
                            )
    
    def _calculate_robot_scale_influence(self, feature: RobotScaleFeature, distance: float, 
                                       world_x: float, world_y: float) -> float:
        """Calculate influence for robot-scale features with gentle transitions."""
        if distance >= feature.size:
            return 0.0
        
        # Normalized distance (0 at center, 1 at edge)
        normalized_distance = distance / feature.size
        
        if feature.type == TerrainType.SMALL_HILL:
            # Gentle hill profile - easy to climb
            return (1 - normalized_distance**2) ** 0.8
            
        elif feature.type == TerrainType.GENTLE_SLOPE:
            # Directional slope - forms ramps
            center_x, center_y = feature.center
            dx = world_x - center_x
            dy = world_y - center_y
            
            # Calculate direction relative to slope angle
            slope_dir = math.atan2(dy, dx) - feature.slope_angle
            slope_factor = math.cos(slope_dir)  # 1 in slope direction, -1 opposite
            
            base_influence = 1 - normalized_distance
            return base_influence * (0.5 + 0.5 * slope_factor)
            
        elif feature.type == TerrainType.OBSTACLE:
            # Small obstacles with steep sides but still climbable
            if normalized_distance < 0.4:
                return 1.0
            else:
                return max(0, (1 - normalized_distance) / 0.6)
                
        elif feature.type == TerrainType.RAMP:
            # Designed pathways with gentle grades
            return 1 - normalized_distance**1.5
            
        elif feature.type == TerrainType.ROUGH_PATCH:
            # Uneven surface with random variation
            base = 1 - normalized_distance
            noise_factor = 0.3 * (random.random() - 0.5)
            return base * (1 + noise_factor)
            
        elif feature.type == TerrainType.DEPRESSION:
            # Small dips and holes
            return (1 - normalized_distance**2) ** 0.6
            
        elif feature.type == TerrainType.RIDGE:
            # Linear elevated features
            return max(0, 1 - normalized_distance**1.2)
        
        else:
            # Default smooth profile
            return 1 - normalized_distance**2
    
    def generate_robot_collision_bodies(self) -> List[Dict]:
        """Generate collision bodies optimized for robot physics."""
        bodies = []
        
        # Use much finer sampling for robot-scale features
        sample_step = max(1, int(0.5 / self.resolution))  # Sample every 0.5 meters
        
        for y in range(0, self.height, sample_step):
            for x in range(0, self.width, sample_step):
                elevation = self.elevation[y, x]
                
                # Create collision bodies for any elevation above ground margin
                if elevation > 0.05:  # 5cm threshold instead of 1m
                    world_x, world_y = self.grid_to_world(x, y)
                    
                    # Calculate appropriate body size
                    body_size = self.resolution * sample_step
                    
                    # Position body with collision margin to prevent embedding
                    # Place bottom at ground level, extend upward
                    body_position_y = elevation / 2 + self.collision_margin
                    
                    bodies.append({
                        'type': 'terrain_segment',
                        'position': (world_x, body_position_y),
                        'size': body_size,
                        'height': elevation,
                        'friction': float(self.friction[y, x]),
                        'restitution': 0.2,  # Slight bounce to prevent sticking
                        'properties': {
                            'terrain_type': 'robot_scale',
                            'roughness': float(self.roughness[y, x]),
                            'navigable': elevation < 2.0  # Mark if robot can traverse
                        }
                    })
        
        # Filter out bodies that are too close together to improve performance
        bodies = self._optimize_collision_bodies(bodies)
        
        print(f"🏗️ Generated {len(bodies)} robot-scale collision bodies")
        return bodies
    
    def _optimize_collision_bodies(self, bodies: List[Dict]) -> List[Dict]:
        """Optimize collision bodies by merging nearby ones."""
        if len(bodies) < 2:
            return bodies
        
        optimized = []
        used = set()
        
        for i, body in enumerate(bodies):
            if i in used:
                continue
                
            # Check if we can merge with nearby bodies
            merged = False
            for j, other in enumerate(optimized):
                if self._can_merge_bodies(body, other):
                    # Merge bodies
                    other['size'] = max(other['size'], body['size'])
                    other['height'] = max(other['height'], body['height'])
                    merged = True
                    break
            
            if not merged:
                optimized.append(body.copy())
        
        return optimized
    
    def _can_merge_bodies(self, body1: Dict, body2: Dict) -> bool:
        """Check if two collision bodies can be merged."""
        pos1 = body1['position']
        pos2 = body2['position']
        
        # Distance between centers
        distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        
        # Can merge if they're very close and similar height
        size_threshold = max(body1['size'], body2['size'])
        height_diff = abs(body1['height'] - body2['height'])
        
        return distance < size_threshold * 0.8 and height_diff < 0.2


class RobotScaleTerrainGenerator:
    """Generates robot-navigable terrain with appropriate scale and features."""
    
    def __init__(self, bounds: Tuple[float, float, float, float], resolution: float = 0.25):
        self.bounds = bounds
        self.resolution = resolution
        self.terrain_mesh = RobotScaleTerrainMesh(bounds, resolution)
    
    def generate_robot_terrain(self, style: str = 'mixed') -> Tuple[RobotScaleTerrainMesh, List[Dict]]:
        """Generate robot-scale terrain based on the specified style."""
        
        print(f"🤖 Generating robot-scale {style} terrain...")
        
        # Generate base terrain features appropriate for robots
        if style == 'flat':
            features = self._generate_mostly_flat_terrain()
        elif style == 'gentle_hills':
            features = self._generate_gentle_hills_terrain()
        elif style == 'obstacle_course':
            features = self._generate_obstacle_course_terrain()
        elif style == 'slopes_and_ramps':
            features = self._generate_slopes_and_ramps_terrain()
        elif style == 'rough_terrain':
            features = self._generate_rough_terrain()
        elif style == 'varied':
            features = self._generate_varied_terrain()
        else:  # mixed - balanced for training
            features = self._generate_mixed_robot_terrain()
        
        # Apply all features to the terrain mesh
        for feature in features:
            self.terrain_mesh.apply_robot_scale_feature(feature)
        
        # Add subtle noise for natural variation
        self._add_robot_scale_noise()
        
        # Generate collision bodies for physics
        collision_bodies = self.terrain_mesh.generate_robot_collision_bodies()
        
        print(f"✅ Robot-scale terrain generation complete:")
        print(f"   🗻 {len(features)} terrain features")
        print(f"   🏗️  {len(collision_bodies)} collision bodies")
        print(f"   📏 Resolution: {self.resolution}m")
        
        return self.terrain_mesh, collision_bodies
    
    def _generate_mostly_flat_terrain(self) -> List[RobotScaleFeature]:
        """Generate mostly flat terrain with occasional small features."""
        features = []
        x_min, y_min, x_max, y_max = self.bounds
        
        # Very few small obstacles
        num_obstacles = random.randint(2, 5)
        for _ in range(num_obstacles):
            features.append(RobotScaleFeature(
                type=TerrainType.OBSTACLE,
                center=(random.uniform(x_min, x_max), random.uniform(y_min, y_max)),
                size=random.uniform(1.0, 2.5),        # 1-2.5m wide
                height=random.uniform(0.8, 1.8),      # 0.8-1.8m tall - challenging but passable
                friction=0.7,
                roughness=0.3
            ))
        
        return features
    
    def _generate_gentle_hills_terrain(self) -> List[RobotScaleFeature]:
        """Generate terrain with small, navigable hills."""
        features = []
        x_min, y_min, x_max, y_max = self.bounds
        
        # Small hills that robots can climb
        num_hills = random.randint(8, 12)
        for _ in range(num_hills):
            features.append(RobotScaleFeature(
                type=TerrainType.SMALL_HILL,
                center=(random.uniform(x_min, x_max), random.uniform(y_min, y_max)),
                size=random.uniform(3.0, 6.0),        # 3-6m wide
                height=random.uniform(0.5, 1.5),      # 0.5-1.5m tall (robot height to 2x)
                friction=random.uniform(0.6, 0.8),
                roughness=random.uniform(0.3, 0.5)
            ))
        
        # Some gentle slopes for variety
        num_slopes = random.randint(3, 6)
        for _ in range(num_slopes):
            features.append(RobotScaleFeature(
                type=TerrainType.GENTLE_SLOPE,
                center=(random.uniform(x_min, x_max), random.uniform(y_min, y_max)),
                size=random.uniform(4.0, 8.0),
                height=random.uniform(0.3, 1.0),
                slope_angle=random.uniform(0, 2 * math.pi),
                friction=0.7,
                roughness=0.4
            ))
        
        return features
    
    def _generate_obstacle_course_terrain(self) -> List[RobotScaleFeature]:
        """Generate terrain like an obstacle course."""
        features = []
        x_min, y_min, x_max, y_max = self.bounds
        
        # Various sized obstacles
        num_small_obstacles = random.randint(8, 15)
        for _ in range(num_small_obstacles):
            features.append(RobotScaleFeature(
                type=TerrainType.OBSTACLE,
                center=(random.uniform(x_min, x_max), random.uniform(y_min, y_max)),
                size=random.uniform(0.8, 2.0),        # Small obstacles
                height=random.uniform(1.2, 2.5),      # 1.2-2.5m tall - meaningful obstacles for 1.5m robots
                friction=random.uniform(0.5, 0.8),
                roughness=random.uniform(0.4, 0.7)
            ))
        
        # Medium challenges
        num_medium = random.randint(4, 8)
        for _ in range(num_medium):
            features.append(RobotScaleFeature(
                type=random.choice([TerrainType.SMALL_HILL, TerrainType.OBSTACLE]),
                center=(random.uniform(x_min, x_max), random.uniform(y_min, y_max)),
                size=random.uniform(2.0, 4.0),
                height=random.uniform(1.0, 2.0),      # Challenging obstacles for 1.5m robots
                friction=0.6,
                roughness=0.5
            ))
        
        # A few ramps to help navigation
        num_ramps = random.randint(2, 4)
        for _ in range(num_ramps):
            features.append(RobotScaleFeature(
                type=TerrainType.RAMP,
                center=(random.uniform(x_min, x_max), random.uniform(y_min, y_max)),
                size=random.uniform(3.0, 5.0),
                height=random.uniform(0.4, 0.8),
                friction=0.8,  # Good grip on ramps
                roughness=0.2
            ))
        
        return features
    
    def _generate_slopes_and_ramps_terrain(self) -> List[RobotScaleFeature]:
        """Generate terrain focused on slopes and ramps."""
        features = []
        x_min, y_min, x_max, y_max = self.bounds
        
        # Many gentle slopes
        num_slopes = random.randint(10, 15)
        for _ in range(num_slopes):
            features.append(RobotScaleFeature(
                type=TerrainType.GENTLE_SLOPE,
                center=(random.uniform(x_min, x_max), random.uniform(y_min, y_max)),
                size=random.uniform(3.0, 7.0),
                height=random.uniform(0.3, 1.0),
                slope_angle=random.uniform(0, 2 * math.pi),
                friction=random.uniform(0.7, 0.9),
                roughness=random.uniform(0.2, 0.4)
            ))
        
        # Designed ramps
        num_ramps = random.randint(5, 8)
        for _ in range(num_ramps):
            features.append(RobotScaleFeature(
                type=TerrainType.RAMP,
                center=(random.uniform(x_min, x_max), random.uniform(y_min, y_max)),
                size=random.uniform(4.0, 8.0),
                height=random.uniform(0.5, 1.2),
                friction=0.8,
                roughness=0.2
            ))
        
        return features
    
    def _generate_rough_terrain(self) -> List[RobotScaleFeature]:
        """Generate rough, uneven terrain."""
        features = []
        x_min, y_min, x_max, y_max = self.bounds
        
        # Many small rough patches
        num_rough = random.randint(15, 25)
        for _ in range(num_rough):
            features.append(RobotScaleFeature(
                type=TerrainType.ROUGH_PATCH,
                center=(random.uniform(x_min, x_max), random.uniform(y_min, y_max)),
                size=random.uniform(1.5, 3.5),
                height=random.uniform(0.5, 1.8),      # Meaningful elevation changes for robot navigation
                friction=random.uniform(0.4, 0.7),
                roughness=random.uniform(0.6, 0.9)
            ))
        
        # Some small depressions
        num_depressions = random.randint(5, 10)
        for _ in range(num_depressions):
            features.append(RobotScaleFeature(
                type=TerrainType.DEPRESSION,
                center=(random.uniform(x_min, x_max), random.uniform(y_min, y_max)),
                size=random.uniform(2.0, 4.0),
                height=random.uniform(-0.3, -0.1),    # Small dips
                friction=0.6,
                roughness=0.5
            ))
        
        return features
    
    def _generate_varied_terrain(self) -> List[RobotScaleFeature]:
        """Generate varied terrain with all feature types."""
        features = []
        x_min, y_min, x_max, y_max = self.bounds
        
        # Mix of all terrain types
        terrain_types = [
            (TerrainType.SMALL_HILL, 0.3, 1.2),
            (TerrainType.GENTLE_SLOPE, 0.2, 0.8),
            (TerrainType.OBSTACLE, 0.4, 0.7),
            (TerrainType.RAMP, 0.5, 1.0),
            (TerrainType.ROUGH_PATCH, 0.1, 0.3),
            (TerrainType.DEPRESSION, -0.2, -0.1)
        ]
        
        for terrain_type, min_height, max_height in terrain_types:
            num_features = random.randint(3, 6)
            for _ in range(num_features):
                height_mult = 1 if max_height > 0 else -1
                features.append(RobotScaleFeature(
                    type=terrain_type,
                    center=(random.uniform(x_min, x_max), random.uniform(y_min, y_max)),
                    size=random.uniform(2.0, 5.0),
                    height=random.uniform(abs(min_height), abs(max_height)) * height_mult,
                    slope_angle=random.uniform(0, 2 * math.pi) if terrain_type == TerrainType.GENTLE_SLOPE else 0.0,
                    friction=random.uniform(0.5, 0.8),
                    roughness=random.uniform(0.3, 0.6)
                ))
        
        return features
    
    def _generate_mixed_robot_terrain(self) -> List[RobotScaleFeature]:
        """Generate balanced mixed terrain good for robot training."""
        features = []
        x_min, y_min, x_max, y_max = self.bounds
        
        # Balanced mix focusing on navigable features
        
        # Some small hills (easy challenges)
        num_hills = random.randint(4, 6)
        for _ in range(num_hills):
            features.append(RobotScaleFeature(
                type=TerrainType.SMALL_HILL,
                center=(random.uniform(x_min, x_max), random.uniform(y_min, y_max)),
                size=random.uniform(3.0, 5.0),
                height=random.uniform(1.5, 2.5),      # Meaningful mixed terrain hills
                friction=0.7,
                roughness=0.4
            ))
        
        # Some obstacles (moderate challenges)
        num_obstacles = random.randint(5, 8)
        for _ in range(num_obstacles):
            features.append(RobotScaleFeature(
                type=TerrainType.OBSTACLE,
                center=(random.uniform(x_min, x_max), random.uniform(y_min, y_max)),
                size=random.uniform(1.5, 3.0),
                height=random.uniform(1.2, 2.2),      # Meaningful obstacles for 1.5m robots
                friction=0.6,
                roughness=0.5
            ))
        
        # Some ramps (help navigation)
        num_ramps = random.randint(3, 5)
        for _ in range(num_ramps):
            features.append(RobotScaleFeature(
                type=TerrainType.RAMP,
                center=(random.uniform(x_min, x_max), random.uniform(y_min, y_max)),
                size=random.uniform(4.0, 6.0),
                height=random.uniform(0.5, 0.9),
                friction=0.8,
                roughness=0.3
            ))
        
        # Some gentle slopes
        num_slopes = random.randint(3, 5)
        for _ in range(num_slopes):
            features.append(RobotScaleFeature(
                type=TerrainType.GENTLE_SLOPE,
                center=(random.uniform(x_min, x_max), random.uniform(y_min, y_max)),
                size=random.uniform(3.0, 6.0),
                height=random.uniform(0.3, 0.7),
                slope_angle=random.uniform(0, 2 * math.pi),
                friction=0.7,
                roughness=0.4
            ))
        
        return features
    
    def _add_robot_scale_noise(self):
        """Add subtle noise appropriate for robot-scale terrain."""
        noise_amplitude = 0.05  # 5cm noise - much smaller than before
        
        for y in range(self.terrain_mesh.height):
            for x in range(self.terrain_mesh.width):
                # Add very subtle random variation
                noise_value = (random.random() - 0.5) * noise_amplitude
                self.terrain_mesh.elevation[y, x] += noise_value
                
                # Add some surface roughness variation
                roughness_noise = (random.random() - 0.5) * 0.1
                self.terrain_mesh.roughness[y, x] = max(0.1, min(0.9, 
                    self.terrain_mesh.roughness[y, x] + roughness_noise))


def generate_robot_scale_terrain(style: str = 'mixed', 
                               bounds: Tuple[float, float, float, float] = (-50, 0, 50, 30),
                               resolution: float = 0.25) -> Tuple[RobotScaleTerrainMesh, List[Dict]]:
    """
    Generate robot-scale terrain with appropriate features and collision bodies.
    
    Args:
        style: Terrain style ('flat', 'gentle_hills', 'obstacle_course', 'slopes_and_ramps', 
               'rough_terrain', 'varied', 'mixed')
        bounds: World bounds (x_min, y_min, x_max, y_max)
        resolution: Grid resolution in meters (0.25m recommended for robots)
    
    Returns:
        Tuple of (terrain_mesh, collision_bodies)
    """
    print(f"🌍 Generating robot-scale terrain: {style}")
    print(f"   📐 Bounds: {bounds}")
    print(f"   📏 Resolution: {resolution}m")
    
    generator = RobotScaleTerrainGenerator(bounds, resolution)
    terrain_mesh, collision_bodies = generator.generate_robot_terrain(style)
    
    return terrain_mesh, collision_bodies


# Backwards compatibility function
def generate_realistic_terrain(style: str = 'mixed', 
                             bounds: Tuple[float, float, float, float] = (-50, 0, 50, 30),
                             resolution: float = 0.25) -> Tuple[RobotScaleTerrainMesh, List[Dict]]:
    """Legacy function - now generates robot-scale terrain."""
    print("⚠️  Using legacy function - automatically generating robot-scale terrain")
    return generate_robot_scale_terrain(style, bounds, resolution) 