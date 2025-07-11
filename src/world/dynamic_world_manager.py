"""
Dynamic World Generation System

Creates an expanding world that generates new tiles to the right as robots progress.
Implements a sliding window system that removes old tiles from the left.
"""

import random
import time
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import Box2D as b2
from src.ecosystem_dynamics import FoodSource, FoodZone


class BiomeType(Enum):
    """Different biome types for world tiles."""
    PLAINS = "plains"
    FOREST = "forest"
    DESERT = "desert"
    ROCKY = "rocky"
    WETLANDS = "wetlands"
    VOLCANIC = "volcanic"


@dataclass
class WorldTile:
    """Represents a chunk of the dynamic world."""
    id: int
    x_start: float
    x_end: float
    biome: BiomeType
    ground_bodies: List[Any]  # Box2D bodies for ground
    obstacle_bodies: List[Any]  # Box2D bodies for obstacles
    food_sources: List[FoodSource]  # Food sources in this tile
    food_zones: List[FoodZone]  # Food zones in this tile
    created_time: float
    active: bool = True


class DynamicWorldManager:
    """
    Manages dynamic world generation that expands to the right and creates sliding window effect.
    
    Features:
    - Progressive world generation to the right
    - Left wall barrier to prevent backtracking
    - Random biomes with different obstacles and food
    - Sliding window cleanup system
    - Integration with strategic food zones
    """
    
    def __init__(self, box2d_world: Any, ecosystem_dynamics: Any):
        self.world = box2d_world
        self.ecosystem = ecosystem_dynamics
        
        # World tile configuration
        self.tile_width = 100.0  # Width of each world tile in meters
        self.tile_height = 60.0  # Height of each world tile
        self.max_active_tiles = 8  # Maximum number of active tiles before cleanup
        
        # Current world state
        self.tiles: Dict[int, WorldTile] = {}
        self.next_tile_id = 0
        self.current_right_edge = 200.0  # Start from current strategic food zone edge
        self.left_wall_position = -250.0  # Wall position (left of spawn area)
        
        # Generation triggers
        self.generation_trigger_distance = 150.0  # Generate new tile when robots get within this distance
        self.robot_progress_threshold = 50.0  # How far right robots need to progress
        
        # Left wall barrier
        self.left_wall_body = None
        self.left_wall_height = 100.0
        
        # Biome generation
        self.biome_weights = {
            BiomeType.PLAINS: 0.25,
            BiomeType.FOREST: 0.20,
            BiomeType.ROCKY: 0.20,
            BiomeType.DESERT: 0.15,
            BiomeType.WETLANDS: 0.15,
            BiomeType.VOLCANIC: 0.05
        }
        
        # Statistics
        self.tiles_generated = 0
        self.tiles_cleaned_up = 0
        self.last_generation_time = time.time()
        
        # 🎯 PROGRESSIVE DIFFICULTY SYSTEM
        self.robot_performance_tracker = {
            'max_distances': [],  # Track furthest robot positions
            'average_distances': [],  # Track average robot progression
            'performance_samples': 20,  # Keep last 20 samples
            'baseline_distance': 50.0,  # Starting difficulty baseline
            'difficulty_multiplier': 0.5,  # Start with 50% obstacle sizes
        }
        
        # Create initial setup
        self._create_left_wall_barrier()
        self._generate_initial_tiles()
        
        print("🌍 Dynamic World Manager initialized!")
        print(f"   📏 Tile size: {self.tile_width}m × {self.tile_height}m")
        print(f"   🧱 Max active tiles: {self.max_active_tiles}")
        print(f"   🚧 Left wall at: {self.left_wall_position}m")
        print(f"   ➡️ Starting right edge: {self.current_right_edge}m")

    def _create_left_wall_barrier(self):
        """Create a wall on the left side to prevent robots from going too far left."""
        try:
            # Create static body for the wall
            self.left_wall_body = self.world.CreateStaticBody(
                position=(self.left_wall_position, self.left_wall_height / 2)
            )
            
            # Create wall fixture
            wall_fixture = self.left_wall_body.CreateFixture(
                shape=b2.b2PolygonShape(box=(2.0, self.left_wall_height / 2)),  # 4m thick wall
                density=0.0,
                friction=0.8,
                restitution=0.1,
                filter=b2.b2Filter(
                    categoryBits=0x0004,  # OBSTACLE_CATEGORY
                    maskBits=0x0002  # AGENT_CATEGORY - only collide with agents
                )
            )
            
            # Mark as barrier wall
            self.left_wall_body.userData = {'type': 'barrier_wall', 'dynamic_world': True}
            
            print(f"🚧 Created left barrier wall at x={self.left_wall_position}m")
            
        except Exception as e:
            print(f"❌ Error creating left wall barrier: {e}")

    def _generate_initial_tiles(self):
        """Generate initial world tiles to establish the base world."""
        try:
            # Generate 3 initial tiles to the right of current world
            for i in range(3):
                tile_start = self.current_right_edge + (i * self.tile_width)
                self._generate_world_tile(tile_start)
            
            print(f"🌍 Generated {len(self.tiles)} initial world tiles")
            
        except Exception as e:
            print(f"❌ Error generating initial tiles: {e}")

    def update(self, robot_positions: List[Tuple[str, Tuple[float, float]]]):
        """Update the dynamic world based on robot positions."""
        if not robot_positions:
            return
        
        # CRITICAL SAFEGUARD: Rate limiting to prevent runaway generation
        current_time = time.time()
        if hasattr(self, 'last_update_time'):
            if current_time - self.last_update_time < 1.0:  # Back to 1 second between updates
                return
        self.last_update_time = current_time
        
        # CRITICAL SAFEGUARD: Maximum tiles per session
        MAX_TILES_EVER = 50  # Hard limit to prevent infinite generation
        if self.tiles_generated >= MAX_TILES_EVER:
            print(f"🛑 SAFETY LIMIT: Max tiles ({MAX_TILES_EVER}) reached, stopping generation")
            return
        
        # Find the rightmost robot position with proper error handling
        try:
            # Extract robot positions - simplified filtering
            valid_positions = []
            for agent_id, pos in robot_positions:
                try:
                    if pos and len(pos) >= 2:
                        x_pos = float(pos[0])
                        valid_positions.append(x_pos)
                except (TypeError, ValueError, IndexError):
                    continue  # Skip invalid positions
            
            if not valid_positions:  # No valid positions found
                return
                
            rightmost_x = max(valid_positions)
            average_x = sum(valid_positions) / len(valid_positions)
            
            # 🎯 UPDATE PROGRESSIVE DIFFICULTY SYSTEM
            self._update_robot_performance(rightmost_x, average_x)
            
        except Exception as e:
            print(f"❌ Error processing robot positions: {e}")
            return
        
        # CRITICAL SAFEGUARD: Prevent excessive tile generation
        if len(self.tiles) >= 20:  # Emergency brake before max_active_tiles
            print(f"⚠️ WARNING: {len(self.tiles)} active tiles, forcing cleanup")
            self._cleanup_old_tiles()
            return
        
        # Check if we need to generate new tiles
        distance_to_edge = self.current_right_edge - rightmost_x
        
        if distance_to_edge <= self.generation_trigger_distance:
            # ADDITIONAL SAFEGUARD: Minimum distance between tiles
            if hasattr(self, 'last_generation_time'):
                if current_time - self.last_generation_time < 5.0:  # 5 second cooldown
                    return
            
            print(f"🆕 Generating tile #{self.tiles_generated + 1} at distance {distance_to_edge:.1f}m from rightmost robot")
            self._generate_next_tile()
            self.last_generation_time = current_time
            
        # Check if we need to clean up old tiles
        if len(self.tiles) > self.max_active_tiles:
            self._cleanup_old_tiles()
        
        # Update left wall if needed (push it forward if robots are getting too spread out)
        self._update_left_wall_position(robot_positions)

    def _update_robot_performance(self, max_distance: float, avg_distance: float):
        """Update robot performance tracking and adjust difficulty accordingly."""
        try:
            tracker = self.robot_performance_tracker
            
            # Add new performance samples
            tracker['max_distances'].append(max_distance)
            tracker['average_distances'].append(avg_distance)
            
            # Keep only recent samples
            if len(tracker['max_distances']) > tracker['performance_samples']:
                tracker['max_distances'].pop(0)
            if len(tracker['average_distances']) > tracker['performance_samples']:
                tracker['average_distances'].pop(0)
            
            # Calculate difficulty multiplier based on robot progress
            if len(tracker['max_distances']) >= 5:  # Need at least 5 samples
                avg_max_distance = sum(tracker['max_distances']) / len(tracker['max_distances'])
                
                # Calculate progress beyond baseline
                progress_ratio = max(0.0, (avg_max_distance - tracker['baseline_distance']) / 200.0)
                
                # Scale difficulty from 0.3 (easy) to 1.2 (hard) based on progress
                new_difficulty = 0.3 + (progress_ratio * 0.9)
                new_difficulty = min(1.2, max(0.3, new_difficulty))  # Clamp to reasonable range
                
                # Smooth transition (only change by 10% at a time)
                old_difficulty = tracker['difficulty_multiplier']
                tracker['difficulty_multiplier'] = old_difficulty * 0.9 + new_difficulty * 0.1
                
                # Log difficulty changes
                if abs(tracker['difficulty_multiplier'] - old_difficulty) > 0.05:
                    print(f"🎯 Difficulty adjusted: {old_difficulty:.2f} → {tracker['difficulty_multiplier']:.2f} "
                          f"(progress: {avg_max_distance:.1f}m)")
            
        except Exception as e:
            print(f"❌ Error updating robot performance: {e}")

    def _generate_next_tile(self):
        """Generate the next world tile to the right."""
        try:
            tile_start = self.current_right_edge
            new_tile = self._generate_world_tile(tile_start)
            
            if new_tile:
                self.current_right_edge += self.tile_width
                self.tiles_generated += 1
                
                print(f"🆕 Generated tile #{new_tile.id} ({new_tile.biome.value}) at x={tile_start:.0f}-{tile_start + self.tile_width:.0f}m")
                return new_tile
                
        except Exception as e:
            print(f"❌ Error generating next tile: {e}")
        
        return None

    def _generate_world_tile(self, x_start: float) -> Optional[WorldTile]:
        """Generate a single world tile with random biome and features."""
        try:
            tile_id = self.next_tile_id
            self.next_tile_id += 1
            
            x_end = x_start + self.tile_width
            
            # Select random biome
            biome = self._select_random_biome()
            
            # Create tile
            tile = WorldTile(
                id=tile_id,
                x_start=x_start,
                x_end=x_end,
                biome=biome,
                ground_bodies=[],
                obstacle_bodies=[],
                food_sources=[],
                food_zones=[],
                created_time=time.time(),
                active=True
            )
            
            # Generate tile content
            self._generate_tile_ground(tile)
            self._generate_tile_obstacles(tile)
            self._generate_tile_food_sources(tile)
            
            # Add to tiles dictionary
            self.tiles[tile_id] = tile
            
            # Update ecosystem bounds to include new tile
            self._update_ecosystem_bounds()
            
            return tile
            
        except Exception as e:
            print(f"❌ Error generating world tile: {e}")
            return None

    def _select_random_biome(self) -> BiomeType:
        """Select a random biome based on weights."""
        rand = random.random()
        cumulative = 0.0
        
        for biome, weight in self.biome_weights.items():
            cumulative += weight
            if rand <= cumulative:
                return biome
        
        return BiomeType.PLAINS  # Fallback

    def _generate_tile_ground(self, tile: WorldTile):
        """Generate ground terrain for a tile based on its biome."""
        try:
            biome_config = self._get_biome_config(tile.biome)
            
            # Create base ground
            ground_body = self.world.CreateStaticBody(
                position=(tile.x_start + self.tile_width / 2, -1)
            )
            
            ground_fixture = ground_body.CreateFixture(
                shape=b2.b2PolygonShape(box=(self.tile_width / 2, 1)),
                density=0.0,
                friction=biome_config['ground_friction'],
                restitution=0.1,
                filter=b2.b2Filter(
                    categoryBits=0x0001,  # GROUND_CATEGORY
                    maskBits=0x0002  # AGENT_CATEGORY
                )
            )
            
            ground_body.userData = {
                'type': 'dynamic_ground',
                'tile_id': tile.id,
                'biome': tile.biome.value
            }
            
            tile.ground_bodies.append(ground_body)
            
            # Generate terrain features based on biome
            feature_count = random.randint(biome_config['min_features'], biome_config['max_features'])
            
            for _ in range(feature_count):
                self._generate_terrain_feature(tile, biome_config)
                
        except Exception as e:
            print(f"❌ Error generating tile ground: {e}")

    def _generate_terrain_feature(self, tile: WorldTile, biome_config: Dict):
        """Generate a single terrain feature (hill, pit, etc.)."""
        try:
            feature_type = random.choice(biome_config['terrain_features'])
            
            # Random position within tile
            x = random.uniform(tile.x_start + 5, tile.x_end - 5)
            
            if feature_type == 'hill':
                # 🤖 ROBOT-SCALE: PASSABLE hills for 1.5m robots
                y = random.uniform(0.3, 0.8)  # MUCH LOWER: Ground level position
                width = random.uniform(2.0, 4.0)  # SMALLER: Reduced to 2-4m width
                height = random.uniform(0.4, 1.0)  # PASSABLE: Max 1.0m tall (was 2.5m)
                
                hill_body = self.world.CreateStaticBody(position=(x, y))
                hill_fixture = hill_body.CreateFixture(
                    shape=b2.b2PolygonShape(box=(width / 2, height / 2)),
                    density=0.0,
                    friction=0.7,
                    restitution=0.2,
                    filter=b2.b2Filter(
                        categoryBits=0x0001,  # GROUND_CATEGORY  
                        maskBits=0x0002  # AGENT_CATEGORY
                    )
                )
                
                hill_body.userData = {
                    'type': 'terrain_hill',
                    'tile_id': tile.id,
                    'biome': tile.biome.value,
                    'color': biome_config['color']
                }
                
                tile.ground_bodies.append(hill_body)
                
            elif feature_type == 'platform':
                # 🤖 ROBOT-SCALE: PASSABLE platforms for 1.5m robots
                y = 0  # LOWER: Accessible platform height
                width = random.uniform(3.0, 5.0)  # SMALLER: Reduced to 3-5m width
                height = random.uniform(0.3, 0.8)  # PASSABLE: Max 0.8m tall (was 1.5m)
                
                platform_body = self.world.CreateStaticBody(position=(x, y))
                platform_fixture = platform_body.CreateFixture(
                    shape=b2.b2PolygonShape(box=(width / 2, height / 2)),
                    density=0.0,
                    friction=0.8,
                    restitution=0.1,
                    filter=b2.b2Filter(
                        categoryBits=0x0001,  # GROUND_CATEGORY
                        maskBits=0x0002  # AGENT_CATEGORY
                    )
                )
                
                platform_body.userData = {
                    'type': 'terrain_platform',
                    'tile_id': tile.id,
                    'biome': tile.biome.value,
                    'color': biome_config['color']
                }
                
                tile.ground_bodies.append(platform_body)
                
        except Exception as e:
            print(f"❌ Error generating terrain feature: {e}")

    def _generate_tile_obstacles(self, tile: WorldTile):
        """Generate obstacles for a tile based on its biome and progressive difficulty."""
        try:
            biome_config = self._get_biome_config(tile.biome)
            obstacle_count = random.randint(biome_config['min_obstacles'], biome_config['max_obstacles'])
            
            # CRITICAL SAFEGUARD: Limit obstacles per tile to prevent physics overload
            MAX_OBSTACLES_PER_TILE = 3  # Reduced from 5 to 3
            obstacle_count = min(obstacle_count, MAX_OBSTACLES_PER_TILE)
            
            print(f"🎯 Generating {obstacle_count} obstacles for tile {tile.id} ({tile.biome.value})")
            
            # 🤖 ROBOT-SCALE: Small passable obstacles for 1.5m robots
            for i in range(obstacle_count):
                # Random position within tile
                x = random.uniform(tile.x_start + 8, tile.x_end - 8)
                y = random.uniform(0.2, 1.0)  # PASSABLE: Very low to ground position
                
                # 🔺 PASSABLE TRIANGLES: Small, low obstacles robots can climb over
                size = random.uniform(0.3, 0.8)  # SMALLER: 0.3m to 0.8m width (was 0.8-2.5m)
                height = random.uniform(0.2, 0.6)  # LOWER: 0.2m to 0.6m tall (was 0.8-3.0m)
                
                print(f"   Creating obstacle {i+1}: size={size:.1f}m, height={height:.1f}m at ({x:.1f}, {y:.1f})")
                
                # Create obstacle body
                try:
                    obstacle_body = self.world.CreateStaticBody(position=(x, y))
                    
                    # 🔺 TRIANGULAR OBSTACLES: Create upward-pointing triangular obstacles instead of rectangles
                    triangle_vertices = [
                        (-size / 2, -height / 2),  # Bottom left
                        (size / 2, -height / 2),   # Bottom right  
                        (0, height / 2)            # Top center (pointing up)
                    ]
                    
                    obstacle_fixture = obstacle_body.CreateFixture(
                        shape=b2.b2PolygonShape(vertices=triangle_vertices),
                        density=0.0,
                        friction=biome_config['obstacle_friction'],
                        restitution=0.3,
                        filter=b2.b2Filter(
                            categoryBits=0x0004,  # OBSTACLE_CATEGORY
                            maskBits=0x0002  # AGENT_CATEGORY
                        )
                    )
                    
                    obstacle_body.userData = {
                        'type': 'dynamic_obstacle',
                        'tile_id': tile.id,
                        'biome': tile.biome.value,
                        'obstacle_type': random.choice(biome_config['obstacle_types']),
                        'color': biome_config['color']
                    }
                    
                    tile.obstacle_bodies.append(obstacle_body)
                    print(f"   ✅ Successfully created obstacle {i+1}")
                    
                except Exception as obstacle_error:
                    print(f"⚠️ Failed to create obstacle {i} for tile {tile.id}: {obstacle_error}")
                    import traceback
                    traceback.print_exc()
                    continue
                
        except Exception as e:
            print(f"❌ Error generating tile obstacles: {e}")
            import traceback
            traceback.print_exc()

    def _generate_tile_food_sources(self, tile: WorldTile):
        """Generate food sources for a tile based on its biome."""
        try:
            biome_config = self._get_biome_config(tile.biome)
            
            # Create a food zone for this tile
            zone_center_x = tile.x_start + self.tile_width / 2
            zone_center_y = 4.0  # FIXED: Lower center height to match robot scale (ground at Y=-1)
            zone_radius = self.tile_width * 0.4  # Zone covers 80% of tile width
            
            food_zone = FoodZone(
                center_position=(zone_center_x, zone_center_y),
                radius=zone_radius,
                zone_type=tile.biome.value,
                food_types=biome_config['food_types'],
                max_food_sources=biome_config['max_food_sources'],
                regeneration_multiplier=biome_config['food_regen_multiplier'],
                active=True
            )
            
            tile.food_zones.append(food_zone)
            
            # Generate food sources within the zone
            food_count = random.randint(biome_config['min_food_sources'], biome_config['max_food_sources'])
            
            for _ in range(food_count):
                # Random position within zone
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(0, zone_radius * 0.8)
                
                food_x = zone_center_x + distance * math.cos(angle)
                food_y = zone_center_y + distance * math.sin(angle)
                
                # Keep within tile bounds and proper ground height
                food_x = max(tile.x_start + 5, min(tile.x_end - 5, food_x))
                food_y = max(0.5, min(8.0, food_y))  # FIXED: Ground at Y=-1, so food between Y=0.5 to Y=8.0
                
                # Select food type based on biome
                food_type = random.choice(biome_config['food_types'])
                
                # Create longer-lasting food source
                base_amount = random.uniform(40.0, 80.0)
                max_capacity = base_amount * 2.0
                regen_rate = random.uniform(0.3, 0.6) * biome_config['food_regen_multiplier']
                
                food_source = FoodSource(
                    position=(food_x, food_y),
                    food_type=food_type,
                    amount=base_amount,
                    regeneration_rate=regen_rate,
                    max_capacity=max_capacity,
                    source="dynamic_world"  # 🌍 Mark as dynamic world food source
                )
                
                tile.food_sources.append(food_source)
            
            # Add food sources to ecosystem
            self.ecosystem.food_sources.extend(tile.food_sources)
            self.ecosystem.food_zones.extend(tile.food_zones)
            
        except Exception as e:
            print(f"❌ Error generating tile food sources: {e}")

    def _get_biome_config(self, biome: BiomeType) -> Dict:
        """Get configuration parameters for a specific biome."""
        configs = {
            BiomeType.PLAINS: {
                'ground_friction': 0.7,
                'min_features': 0, 'max_features': 2,  # Reduced features
                'terrain_features': ['hill', 'platform'],
                'min_obstacles': 1, 'max_obstacles': 2,  # Reduced obstacles
                'obstacle_size_min': 0.3, 'obstacle_size_max': 0.8,  # 🤖 PASSABLE: Very small obstacles
                'obstacle_friction': 0.6,
                'obstacle_types': ['rock', 'bush', 'tree'],
                'min_food_sources': 3, 'max_food_sources': 6,
                'food_types': ['plants', 'seeds', 'insects'],
                'food_regen_multiplier': 1.2,
                'color': (0.5, 0.8, 0.3)  # Light green
            },
            BiomeType.FOREST: {
                'ground_friction': 0.8,
                'min_features': 1, 'max_features': 2,  # Reduced features
                'terrain_features': ['hill', 'platform'],
                'min_obstacles': 1, 'max_obstacles': 3,  # Reduced obstacles
                'obstacle_size_min': 0.4, 'obstacle_size_max': 1.0,  # 🤖 PASSABLE: Very small obstacles
                'obstacle_friction': 0.8,
                'obstacle_types': ['tree', 'log', 'boulder'],
                'min_food_sources': 4, 'max_food_sources': 8,
                'food_types': ['plants', 'seeds', 'insects', 'meat'],
                'food_regen_multiplier': 1.5,
                'color': (0.2, 0.6, 0.2)  # Dark green
            },
            BiomeType.DESERT: {
                'ground_friction': 0.5,
                'min_features': 0, 'max_features': 1,  # Reduced features
                'terrain_features': ['hill'],
                'min_obstacles': 0, 'max_obstacles': 1,  # Reduced obstacles
                'obstacle_size_min': 0.3, 'obstacle_size_max': 0.7,  # 🤖 PASSABLE: Very small obstacles
                'obstacle_friction': 0.4,
                'obstacle_types': ['cactus', 'rock', 'dune'],
                'min_food_sources': 1, 'max_food_sources': 3,
                'food_types': ['insects', 'seeds'],
                'food_regen_multiplier': 0.7,
                'color': (0.9, 0.8, 0.4)  # Sandy yellow
            },
            BiomeType.ROCKY: {
                'ground_friction': 0.9,
                'min_features': 1, 'max_features': 3,  # Reduced features
                'terrain_features': ['hill', 'platform'],
                'min_obstacles': 1, 'max_obstacles': 2,  # Reduced obstacles
                'obstacle_size_min': 0.4, 'obstacle_size_max': 1.2,  # 🤖 PASSABLE: Very small obstacles
                'obstacle_friction': 0.9,
                'obstacle_types': ['boulder', 'cliff', 'rock'],
                'min_food_sources': 2, 'max_food_sources': 4,
                'food_types': ['insects', 'meat'],
                'food_regen_multiplier': 0.8,
                'color': (0.6, 0.5, 0.4)  # Gray-brown
            },
            BiomeType.WETLANDS: {
                'ground_friction': 0.4,
                'min_features': 0, 'max_features': 2,  # Reduced features
                'terrain_features': ['platform'],
                'min_obstacles': 1, 'max_obstacles': 2,  # Reduced obstacles
                'obstacle_size_min': 0.3, 'obstacle_size_max': 0.9,  # 🤖 PASSABLE: Very small obstacles
                'obstacle_friction': 0.3,
                'obstacle_types': ['marsh', 'reed', 'mud'],
                'min_food_sources': 5, 'max_food_sources': 9,
                'food_types': ['plants', 'insects', 'seeds'],
                'food_regen_multiplier': 1.8,
                'color': (0.3, 0.7, 0.6)  # Teal
            },
            BiomeType.VOLCANIC: {
                'ground_friction': 0.6,
                'min_features': 1, 'max_features': 2,  # Reduced features
                'terrain_features': ['hill'],
                'min_obstacles': 1, 'max_obstacles': 2,  # Reduced obstacles
                'obstacle_size_min': 0.4, 'obstacle_size_max': 1.1,  # 🤖 PASSABLE: Very small obstacles
                'obstacle_friction': 0.7,
                'obstacle_types': ['lava_rock', 'crater', 'ash'],
                'min_food_sources': 1, 'max_food_sources': 2,
                'food_types': ['meat', 'insects'],
                'food_regen_multiplier': 0.5,
                'color': (0.8, 0.3, 0.2)  # Red-orange
            }
        }
        
        return configs.get(biome, configs[BiomeType.PLAINS])

    def _cleanup_old_tiles(self):
        """Remove the oldest tiles when we exceed the maximum."""
        try:
            if len(self.tiles) <= self.max_active_tiles:
                return
            
            # Sort tiles by creation time (oldest first)
            sorted_tiles = sorted(self.tiles.values(), key=lambda t: t.created_time)
            
            # Remove oldest tiles until we're at the limit
            tiles_to_remove = len(self.tiles) - self.max_active_tiles + 1
            
            for i in range(tiles_to_remove):
                old_tile = sorted_tiles[i]
                self._destroy_tile(old_tile)
                
        except Exception as e:
            print(f"❌ Error cleaning up old tiles: {e}")

    def _destroy_tile(self, tile: WorldTile):
        """Completely destroy a tile and clean up all its resources."""
        try:
            tile.active = False
            
            # Remove food sources from ecosystem
            for food_source in tile.food_sources:
                if food_source in self.ecosystem.food_sources:
                    self.ecosystem.food_sources.remove(food_source)
            
            # Remove food zones from ecosystem
            for food_zone in tile.food_zones:
                if food_zone in self.ecosystem.food_zones:
                    self.ecosystem.food_zones.remove(food_zone)
            
            # Destroy ground bodies
            for body in tile.ground_bodies:
                try:
                    self.world.DestroyBody(body)
                except Exception as e:
                    pass  # Body might already be destroyed
            
            # Destroy obstacle bodies
            for body in tile.obstacle_bodies:
                try:
                    self.world.DestroyBody(body)
                except Exception as e:
                    pass  # Body might already be destroyed
            
            # Remove from tiles dictionary
            if tile.id in self.tiles:
                del self.tiles[tile.id]
            
            self.tiles_cleaned_up += 1
            
            print(f"🗑️ Cleaned up old tile #{tile.id} ({tile.biome.value}) from x={tile.x_start:.0f}-{tile.x_end:.0f}m")
            
        except Exception as e:
            print(f"❌ Error destroying tile {tile.id}: {e}")

    def _update_left_wall_position(self, robot_positions: List[Tuple[str, Tuple[float, float]]]):
        """Update left wall position to maintain sliding window effect."""
        try:
            if not robot_positions or not self.tiles:
                return
            
            # Find leftmost robot with proper error handling
            try:
                # Filter valid robot positions first
                valid_positions = [pos[1][0] for _, pos in robot_positions if isinstance(pos[1], tuple) and len(pos[1]) >= 2]
                
                if not valid_positions:  # No valid positions found
                    return
                    
                leftmost_x = min(valid_positions)
                
            except (ValueError, IndexError, TypeError) as e:
                print(f"❌ Error accessing leftmost robot positions: {e}")
                return
            
            # Find leftmost active tile
            leftmost_tile_x = min(tile.x_start for tile in self.tiles.values() if tile.active)
            
            # Move wall if robots are getting too close to the left edge
            desired_wall_position = leftmost_tile_x - 20.0  # Keep wall 20m left of leftmost tile
            
            if desired_wall_position > self.left_wall_position + 10.0:  # Only move forward
                # Update wall position
                old_position = self.left_wall_position
                self.left_wall_position = desired_wall_position
                
                # Update wall body position
                if self.left_wall_body:
                    self.left_wall_body.position = (self.left_wall_position, self.left_wall_height / 2)
                
                print(f"🚧 Moved left wall from {old_position:.0f}m to {self.left_wall_position:.0f}m")
                
        except Exception as e:
            print(f"❌ Error updating left wall position: {e}")

    def _update_ecosystem_bounds(self):
        """Update ecosystem bounds to include all active tiles."""
        try:
            if not self.tiles:
                return
            
            active_tiles = [tile for tile in self.tiles.values() if tile.active]
            if not active_tiles:
                return
            
            # Calculate new bounds
            min_x = min(tile.x_start for tile in active_tiles)
            max_x = max(tile.x_end for tile in active_tiles)
            
            # Update ecosystem bounds
            self.ecosystem.world_bounds['min_x'] = min_x - 50.0  # Add buffer
            self.ecosystem.world_bounds['max_x'] = max_x + 50.0  # Add buffer
            
        except Exception as e:
            print(f"❌ Error updating ecosystem bounds: {e}")

    def get_world_status(self) -> Dict[str, Any]:
        """Get status information about the dynamic world."""
        active_tiles = [tile for tile in self.tiles.values() if tile.active]
        
        biome_counts = {}
        biome_colors = {}
        biome_configs = {}
        
        for tile in active_tiles:
            biome_name = tile.biome.value
            biome_counts[biome_name] = biome_counts.get(biome_name, 0) + 1
            
            # Get biome color and config info
            if biome_name not in biome_colors:
                config = self._get_biome_config(tile.biome)
                biome_colors[biome_name] = config.get('color', (0.5, 0.5, 0.5))
                biome_configs[biome_name] = {
                    'obstacles': f"{config['min_obstacles']}-{config['max_obstacles']}",
                    'food_sources': f"{config['min_food_sources']}-{config['max_food_sources']}",
                    'food_regen_multiplier': config['food_regen_multiplier']
                }
        
        total_food_sources = sum(len(tile.food_sources) for tile in active_tiles)
        total_obstacles = sum(len(tile.obstacle_bodies) for tile in active_tiles)
        
        return {
            'active_tiles': len(active_tiles),
            'tiles_generated': self.tiles_generated,
            'tiles_cleaned_up': self.tiles_cleaned_up,
            'current_right_edge': self.current_right_edge,
            'left_wall_position': self.left_wall_position,
            'world_span': self.current_right_edge - self.left_wall_position,
            'biome_distribution': biome_counts,
            'biome_colors': biome_colors,  # RGB colors for each biome
            'biome_configs': biome_configs,  # Configuration details for each biome
            'total_food_sources': total_food_sources,
            'total_obstacles': total_obstacles,
            'last_generation_time': self.last_generation_time,
            # 🎯 PROGRESSIVE DIFFICULTY STATUS
            'difficulty_multiplier': self.robot_performance_tracker['difficulty_multiplier'],
            'robot_max_distance': self.robot_performance_tracker['max_distances'][-1] if self.robot_performance_tracker['max_distances'] else 0.0,
            'performance_samples': len(self.robot_performance_tracker['max_distances'])
        }

    def get_tile_at_position(self, x: float) -> Optional[WorldTile]:
        """Get the world tile that contains the given x position."""
        for tile in self.tiles.values():
            if tile.active and tile.x_start <= x <= tile.x_end:
                return tile
        return None

    def force_generate_tile(self) -> Optional[WorldTile]:
        """Force generation of the next tile (for testing/debugging)."""
        return self._generate_next_tile()

    def cleanup_all_tiles(self):
        """Clean up all tiles (for shutdown)."""
        try:
            tiles_to_destroy = list(self.tiles.values())
            for tile in tiles_to_destroy:
                self._destroy_tile(tile)
            
            # Destroy left wall
            if self.left_wall_body:
                try:
                    self.world.DestroyBody(self.left_wall_body)
                except Exception:
                    pass
                self.left_wall_body = None
            
            print(f"🧹 Cleaned up all dynamic world tiles")
            
        except Exception as e:
            print(f"❌ Error during world cleanup: {e}")
