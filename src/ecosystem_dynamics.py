"""Ecosystem Dynamics for Enhanced Evolution"""

import random
import time
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import math

class EcosystemRole(Enum):
    HERBIVORE = "herbivore"
    CARNIVORE = "carnivore"  
    OMNIVORE = "omnivore"
    SCAVENGER = "scavenger"
    SYMBIONT = "symbiont"

@dataclass 
class FoodSource:
    """Represents available food in the ecosystem"""
    position: Tuple[float, float]
    food_type: str  # "plants", "meat", "insects", "seeds"
    amount: float
    regeneration_rate: float
    max_capacity: float
    source: str = "strategic"  # "strategic" or "dynamic_world" - identifies food source origin
    
@dataclass
class FoodZone:
    """Strategic food zone with fixed location and specialized food types"""
    center_position: Tuple[float, float]
    radius: float
    zone_type: str  # "forest", "grassland", "water", "mountain", "desert"
    food_types: List[str]
    max_food_sources: int
    regeneration_multiplier: float
    active: bool = True

class EcosystemDynamics:
    """
    REDESIGNED: Strategic ecosystem with fixed food zones instead of random spawning.
    
    Creates meaningful long-term goals by:
    - Fixed food zone locations that robots must learn to navigate to
    - Minimum travel distances to prevent instant rewards 
    - Larger world size for proper exploration challenges
    - Stable food sources that don't randomly appear near robots
    """
    
    def __init__(self):
        self.food_sources: List[FoodSource] = []
        self.agent_roles: Dict[str, EcosystemRole] = {}
        
        # EXPANDED WORLD: 4x larger than before for meaningful travel distances
        self.world_bounds = {
            'min_x': -200.0,  # Expanded from -60 to -200
            'max_x': 200.0,   # Expanded from 60 to 200  
            'min_y': -10.0,   # Slightly expanded ground level
            'max_y': 50.0     # Expanded from 35 to 50
        }
        
        # Strategic Food Zones - Fixed locations that create long-term goals
        self.food_zones = self._create_strategic_food_zones()
        
        # Zone management
        self.last_zone_update = time.time()
        self.zone_update_interval = 120.0  # Update zones every 2 minutes
        
        # Minimum distances to prevent instant rewards
        self.min_distance_from_agents = 25.0  # INCREASED from 6m to 25m minimum distance
        self.min_distance_between_food = 15.0  # INCREASED from 12m to 15m between food sources
        
        # Ecosystem parameters
        self.carrying_capacity = 150  # Increased for larger world
        self.resource_scarcity = 1.0
        self.cooperation_bonus = 1.2
        
        # Dynamic events
        self.migration_pressure = 0.0
        self.population_pressure = 0.0
        self.seasonal_resource_modifier = 1.0
        
        # Initialize strategic food zones with initial food sources
        self._populate_food_zones()
        
        print("🌍 Strategic Ecosystem initialized!")
        print(f"   🗺️ World size: {self.world_bounds['max_x'] - self.world_bounds['min_x']}m × {self.world_bounds['max_y'] - self.world_bounds['min_y']}m")
        print(f"   🎯 Food zones: {len(self.food_zones)} strategic locations")
        print(f"   📏 Minimum travel distance: {self.min_distance_from_agents}m")
        print(f"   🍽️ Initial food sources: {len(self.food_sources)}")
    
    def _create_strategic_food_zones(self) -> List[FoodZone]:
        """Create fixed food zones at strategic locations across the expanded world."""
        zones = []
        
        # Forest zones (plant-rich areas) - 4 corners of the world
        zones.extend([
            FoodZone((-150, 30), 25.0, "forest", ["plants", "seeds", "insects"], 8, 1.5),
            FoodZone((150, 30), 25.0, "forest", ["plants", "seeds", "insects"], 8, 1.5),
            FoodZone((-150, -5), 25.0, "forest", ["plants", "seeds"], 6, 1.2),
            FoodZone((150, -5), 25.0, "forest", ["plants", "seeds"], 6, 1.2),
        ])
        
        # Grassland zones (balanced food) - middle areas
        zones.extend([
            FoodZone((-75, 20), 20.0, "grassland", ["plants", "insects", "seeds"], 6, 1.3),
            FoodZone((75, 20), 20.0, "grassland", ["plants", "insects", "seeds"], 6, 1.3),
            FoodZone((-75, 5), 20.0, "grassland", ["seeds", "insects"], 5, 1.1),
            FoodZone((75, 5), 20.0, "grassland", ["seeds", "insects"], 5, 1.1),
        ])
        
        # Central oasis (high-value mixed food) - center of world
        zones.append(
            FoodZone((0, 25), 15.0, "oasis", ["plants", "seeds", "insects", "meat"], 10, 2.0)
        )
        
        # Mountain zones (sparse but valuable) - extreme edges
        zones.extend([
            FoodZone((-180, 40), 15.0, "mountain", ["insects", "meat"], 4, 0.8),
            FoodZone((180, 40), 15.0, "mountain", ["insects", "meat"], 4, 0.8),
        ])
        
        # Water zones (specialized food) - along middle axis
        zones.extend([
            FoodZone((-100, 15), 12.0, "water", ["insects", "plants"], 5, 1.4),
            FoodZone((100, 15), 12.0, "water", ["insects", "plants"], 5, 1.4),
            FoodZone((0, 5), 12.0, "water", ["insects"], 4, 1.2),
        ])
        
        return zones
    
    def _populate_food_zones(self):
        """Populate each food zone with initial longer-lasting food sources."""
        total_created = 0
        
        for zone in self.food_zones:
            if not zone.active:
                continue
                
            # Create food sources within each zone
            for _ in range(zone.max_food_sources):
                # Try to add food to zone using the standard method
                new_food = self._add_food_to_zone(zone)
                if new_food:
                    self.food_sources.append(new_food)
                    total_created += 1
        
        print(f"🌱 Populated {len(self.food_zones)} food zones with {total_created} longer-lasting strategic food sources")
        print(f"   🍽️ Food persistence: 2-5x longer lasting than before")
        print(f"   🔄 Strategic respawning: Food respawns in different locations within same zones")

    def update_ecosystem(self, generation: int, population_size: int, season: str = "summer"):
        """Update ecosystem dynamics for the current generation"""
        
        # Update population pressure
        self.population_pressure = population_size / self.carrying_capacity
        
        # Seasonal effects on resources
        seasonal_modifiers = {
            "spring": 1.3,    # Abundant resources
            "summer": 1.0,    # Normal resources  
            "autumn": 0.8,    # Declining resources
            "winter": 0.5     # Scarce resources
        }
        self.seasonal_resource_modifier = seasonal_modifiers.get(season, 1.0)
        
        # Update strategic zones instead of random spawning
        self._update_strategic_zones()
        
        # Environmental pressures
        if self.population_pressure > 1.2:
            self._trigger_resource_competition()
        
        # Migration events (may deactivate some zones temporarily)
        if random.random() < 0.03:  # 3% chance
            self._trigger_migration_event()
        
        print(f"🌍 Strategic Ecosystem Gen {generation}: Population pressure {self.population_pressure:.2f}")
    
    def assign_ecosystem_role(self, agent_id: str, fitness_traits: Dict[str, float]):
        """Assign an ecosystem role based on agent characteristics"""
        
        # Use fitness traits to determine role
        speed = fitness_traits.get('speed', 0.5)
        strength = fitness_traits.get('strength', 0.5) 
        cooperation = fitness_traits.get('cooperation', 0.5)
        
        # Balanced role distribution with proper variety
        role_weights = {
            EcosystemRole.HERBIVORE: 0.25 + cooperation * 0.4 + (1 - speed) * 0.2,  # Cooperative, peaceful = herbivore
            EcosystemRole.CARNIVORE: 0.20 + speed * 0.3 + strength * 0.3,  # High speed+strength = carnivore
            EcosystemRole.OMNIVORE: 0.30 + (speed + strength + cooperation) * 0.1,  # Balanced traits = omnivore (most common)
            EcosystemRole.SCAVENGER: 0.15 + (1 - speed) * 0.2 + (1 - cooperation) * 0.2,  # Low speed/cooperation = scavenger
            EcosystemRole.SYMBIONT: 0.10 + cooperation * 0.5  # High cooperation = symbiont
        }
        
        # Select role based on weights with randomness
        role = max(role_weights.keys(), key=lambda r: role_weights[r] + random.uniform(-0.15, 0.15))
        self.agent_roles[agent_id] = role
        
        print(f"🦎 Agent {str(agent_id)[:8]} assigned role: {role.value}")
        return role
    
    def get_ecosystem_effects(self, agent_id: str, position: Tuple[float, float]) -> Dict[str, float]:
        """Get ecosystem effects for an agent at a specific position"""
        
        effects = {
            'fitness_multiplier': 1.0,
            'resource_access': 1.0,
            'competition_penalty': 0.0
        }
        
        # Role-based effects
        role = self.agent_roles.get(agent_id, EcosystemRole.OMNIVORE)
        role_multipliers = {
            EcosystemRole.HERBIVORE: 1.0,
            EcosystemRole.CARNIVORE: 1.1,
            EcosystemRole.OMNIVORE: 0.95,
            EcosystemRole.SCAVENGER: 0.9,
            EcosystemRole.SYMBIONT: 0.8  # No alliance bonus, so base value is lower
        }
        effects['fitness_multiplier'] *= role_multipliers[role]
        
        # Resource scarcity effects
        effects['resource_access'] *= (2.0 - self.resource_scarcity) / self.seasonal_resource_modifier
        
        # Population pressure effects
        if self.population_pressure > 1.0:
            competition_penalty = (self.population_pressure - 1.0) * 0.2
            effects['competition_penalty'] = competition_penalty
            effects['fitness_multiplier'] *= (1 - competition_penalty)
        
        return effects
    
    def _update_strategic_zones(self):
        """Update food zones with longer-lasting but renewable food sources."""
        current_time = time.time()
        if current_time - self.last_zone_update < self.zone_update_interval:
            return
        
        self.last_zone_update = current_time
        
        # Regenerate existing food sources within zones - SLOWER regeneration for longer goals
        for food in self.food_sources:
            if food.amount < food.max_capacity:
                # REDUCED regeneration rate for longer-lasting goals
                regen_amount = food.regeneration_rate * self.seasonal_resource_modifier * 0.3  # 70% slower
                food.amount = min(food.max_capacity, food.amount + regen_amount)
        
        # Track depleted food for strategic respawning
        depleted_food = []
        depleted_zones = {}  # Track which zones lost food
        
        # Remove completely depleted food sources (lower threshold for longer persistence)
        depletion_threshold = 0.05  # Lower threshold means food persists longer
        initial_count = len(self.food_sources)
        
        for food in self.food_sources[:]:  # Copy to avoid modification during iteration
            if food.amount <= depletion_threshold:
                # Track which zone this food was in for strategic respawning
                depleted_zone = self._find_zone_for_position(food.position)
                if depleted_zone:
                    if depleted_zone not in depleted_zones:
                        depleted_zones[depleted_zone] = []
                    depleted_zones[depleted_zone].append(food)
                
                depleted_food.append(food)
                self.food_sources.remove(food)
        
        removed_count = len(depleted_food)
        
        # STRATEGIC RESPAWNING: Replace depleted food in different locations within same zones
        respawned_count = 0
        for zone, lost_food_list in depleted_zones.items():
            for lost_food in lost_food_list:
                # Try to respawn food in a different location within the same zone
                new_food = self._respawn_food_in_zone(zone, lost_food)
                if new_food:
                    self.food_sources.append(new_food)
                    respawned_count += 1
        
        # Maintain zone capacity - add food to zones below minimum
        for zone in self.food_zones:
            if not zone.active:
                continue
                
            current_food_count = self._count_zone_food(zone)
            min_food_per_zone = max(2, zone.max_food_sources // 2)  # At least 2 or half capacity
            
            if current_food_count < min_food_per_zone:
                deficit = min_food_per_zone - current_food_count
                for _ in range(min(deficit, 1)):  # Add at most 1 per update for stability
                    new_food = self._add_food_to_zone(zone)
                    if new_food:
                        self.food_sources.append(new_food)
                        respawned_count += 1
        
        # Log strategic food management
        if removed_count > 0 or respawned_count > 0:
            print(f"🔄 Strategic food management: {removed_count} depleted → {respawned_count} respawned in zones")
            if removed_count > respawned_count:
                print(f"   📉 Net food reduction: {removed_count - respawned_count} (creates scarcity pressure)")
            elif respawned_count > removed_count:
                print(f"   📈 Net food increase: {respawned_count - removed_count} (maintains sustainability)")

    def _count_zone_food(self, zone: FoodZone) -> int:
        """Count food sources within a zone."""
        count = 0
        for food in self.food_sources:
            distance = math.sqrt((food.position[0] - zone.center_position[0])**2 + 
                               (food.position[1] - zone.center_position[1])**2)
            if distance <= zone.radius:
                count += 1
        return count

    def _add_food_to_zone(self, zone: FoodZone) -> Optional[FoodSource]:
        """Add a new food source to a specific zone."""
        # Find valid position within zone that's not too close to existing food
        for attempt in range(20):  # Try up to 20 positions
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(zone.radius * 0.2, zone.radius * 0.9)  # Not at center or edge
            
            food_x = zone.center_position[0] + distance * math.cos(angle)
            food_y = zone.center_position[1] + distance * math.sin(angle)
            
            # Keep within world bounds
            food_x = max(self.world_bounds['min_x'], min(self.world_bounds['max_x'], food_x))
            food_y = max(self.world_bounds['min_y'], min(self.world_bounds['max_y'], food_y))
            
            position = (food_x, food_y)
            
            # Check minimum distance from existing food
            too_close = False
            for existing_food in self.food_sources:
                distance_to_existing = math.sqrt((position[0] - existing_food.position[0])**2 + 
                                                (position[1] - existing_food.position[1])**2)
                if distance_to_existing < self.min_distance_between_food:
                    too_close = True
                    break
            
            if not too_close:
                # Create appropriate food source for this zone - LONGER LASTING
                food_type = random.choice(zone.food_types)
                base_amount = 60.0 + random.uniform(0, 40.0)  # INCREASED: 60-100 units vs 35-60
                max_capacity = base_amount * 2.0  # INCREASED: 2x capacity vs 1.4x
                regen_rate = (0.4 + random.uniform(0, 0.2)) * zone.regeneration_multiplier  # REDUCED: slower regen
                
                food_source = FoodSource(
                    position=position,
                    food_type=food_type,
                    amount=base_amount,
                    regeneration_rate=regen_rate,
                    max_capacity=max_capacity,
                    source="strategic"  # 🎯 Mark as strategic zone food source
                )
                return food_source
        
        return None  # Failed to find valid position

    def _find_zone_for_position(self, position: Tuple[float, float]) -> Optional[FoodZone]:
        """Find which zone a position belongs to."""
        for zone in self.food_zones:
            distance = math.sqrt((position[0] - zone.center_position[0])**2 + 
                               (position[1] - zone.center_position[1])**2)
            if distance <= zone.radius:
                return zone
        return None

    def _respawn_food_in_zone(self, zone: FoodZone, depleted_food: FoodSource) -> Optional[FoodSource]:
        """Respawn food in a different location within the same zone, maintaining food type."""
        # Try to find a new position that's different from the depleted food location
        min_distance_from_old = 10.0  # Must be at least 10m from old position
        
        for attempt in range(25):  # More attempts since we have additional constraints
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(zone.radius * 0.1, zone.radius * 0.95)
            
            food_x = zone.center_position[0] + distance * math.cos(angle)
            food_y = zone.center_position[1] + distance * math.sin(angle)
            
            # Keep within world bounds
            food_x = max(self.world_bounds['min_x'], min(self.world_bounds['max_x'], food_x))
            food_y = max(self.world_bounds['min_y'], min(self.world_bounds['max_y'], food_y))
            
            new_position = (food_x, food_y)
            
            # Check distance from depleted food position
            distance_from_old = math.sqrt((new_position[0] - depleted_food.position[0])**2 + 
                                        (new_position[1] - depleted_food.position[1])**2)
            if distance_from_old < min_distance_from_old:
                continue  # Too close to old position
            
            # Check minimum distance from existing food
            too_close_to_existing = False
            for existing_food in self.food_sources:
                distance_to_existing = math.sqrt((new_position[0] - existing_food.position[0])**2 + 
                                                (new_position[1] - existing_food.position[1])**2)
                if distance_to_existing < self.min_distance_between_food:
                    too_close_to_existing = True
                    break
            
            if not too_close_to_existing:
                # Create respawned food source - preserve food type but make it longer lasting
                base_amount = 50.0 + random.uniform(0, 30.0)  # Fresh, substantial food
                max_capacity = base_amount * 2.5  # Higher capacity than original
                regen_rate = (0.3 + random.uniform(0, 0.2)) * zone.regeneration_multiplier  # Slower regen
                
                respawned_food = FoodSource(
                    position=new_position,
                    food_type=depleted_food.food_type,  # PRESERVE food type for consistency
                    amount=base_amount,
                    regeneration_rate=regen_rate,
                    max_capacity=max_capacity,
                    source="strategic"  # 🎯 Mark as strategic zone food source
                )
                
                return respawned_food
        
        return None  # Failed to find valid respawn position

    def generate_resources_between_agents(self, agent_positions: List[Tuple[str, Tuple[float, float]]]):
        """
        STRATEGIC REPLACEMENT: Instead of generating random food, ensure zones are properly maintained
        and agents are far enough from food sources to require meaningful travel.
        
        ⚠️ IMPORTANT: This method now preserves dynamic world food sources!
        """
        if len(agent_positions) < 2:
            return
        
        # Remove any food that's too close to agents (prevents instant rewards)
        # BUT PRESERVE DYNAMIC WORLD FOOD SOURCES
        removed_food = []
        for food in self.food_sources[:]:  # Copy list to avoid modification during iteration
            # 🌍 SKIP DYNAMIC WORLD FOOD SOURCES - Don't remove them!
            if hasattr(food, 'source') and food.source == 'dynamic_world':
                continue  # Preserve dynamic world food sources
            
            too_close_to_agent = False
            for agent_id, agent_pos in agent_positions:
                distance_to_agent = math.sqrt((food.position[0] - agent_pos[0])**2 + 
                                            (food.position[1] - agent_pos[1])**2)
                if distance_to_agent < self.min_distance_from_agents:
                    too_close_to_agent = True
                    break
            
            if too_close_to_agent:
                self.food_sources.remove(food)
                removed_food.append(food)
        
        # Log removal of food that was too close to agents
        if removed_food:
            print(f"🚫 Removed {len(removed_food)} strategic food sources too close to agents (<{self.min_distance_from_agents}m)")
            print(f"   🌍 Dynamic world food sources preserved for continuous exploration")
        
        # Ensure all zones have adequate food (but not too close to agents)
        self._maintain_zone_integrity(agent_positions)

    def _maintain_zone_integrity(self, agent_positions: List[Tuple[str, Tuple[float, float]]]):
        """Ensure food zones maintain their intended food density while respecting agent distances."""
        zones_replenished = 0
        
        for zone in self.food_zones:
            if not zone.active:
                continue
                
            current_food_count = self._count_zone_food(zone)
            if current_food_count < zone.max_food_sources // 2:  # If zone is less than half capacity
                # Try to add food to zone (but far from agents)
                attempts = 0
                while attempts < 10 and self._count_zone_food(zone) < zone.max_food_sources:
                    if self._try_add_strategic_food_to_zone(zone, agent_positions):
                        zones_replenished += 1
                    attempts += 1
        
        if zones_replenished > 0:
            print(f"🔄 Replenished {zones_replenished} strategic food sources in zones")

    def _try_add_strategic_food_to_zone(self, zone: FoodZone, agent_positions: List[Tuple[str, Tuple[float, float]]]) -> bool:
        """Try to add food to a zone while maintaining minimum distance from all agents."""
        for attempt in range(15):
            # Random position within zone
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(zone.radius * 0.3, zone.radius * 0.8)
            
            food_x = zone.center_position[0] + distance * math.cos(angle)
            food_y = zone.center_position[1] + distance * math.sin(angle)
            
            # Keep within world bounds
            food_x = max(self.world_bounds['min_x'], min(self.world_bounds['max_x'], food_x))
            food_y = max(self.world_bounds['min_y'], min(self.world_bounds['max_y'], food_y))
            
            position = (food_x, food_y)
            
            # Check minimum distance from ALL agents
            far_enough_from_agents = True
            for agent_id, agent_pos in agent_positions:
                distance_to_agent = math.sqrt((position[0] - agent_pos[0])**2 + 
                                            (position[1] - agent_pos[1])**2)
                if distance_to_agent < self.min_distance_from_agents:
                    far_enough_from_agents = False
                    break
            
            # Check minimum distance from existing food
            far_enough_from_food = True
            for existing_food in self.food_sources:
                distance_to_food = math.sqrt((position[0] - existing_food.position[0])**2 + 
                                           (position[1] - existing_food.position[1])**2)
                if distance_to_food < self.min_distance_between_food:
                    far_enough_from_food = False
                    break
            
            if far_enough_from_agents and far_enough_from_food:
                # Create strategic food source
                food_type = random.choice(zone.food_types)
                base_amount = 40.0 + random.uniform(0, 30.0)
                max_capacity = base_amount * 1.5
                regen_rate = (0.8 + random.uniform(0, 0.4)) * zone.regeneration_multiplier
                
                food_source = FoodSource(
                    position=position,
                    food_type=food_type,
                    amount=base_amount,
                    regeneration_rate=regen_rate,
                    max_capacity=max_capacity,
                    source="strategic"  # 🎯 Mark as strategic zone food source
                )
                self.food_sources.append(food_source)
                return True
        
        return False  # Failed to find valid position after all attempts

    def get_strategic_zone_info(self) -> Dict[str, Any]:
        """Get information about the strategic food zones for debugging/visualization."""
        zone_info = {}
        for i, zone in enumerate(self.food_zones):
            food_count = self._count_zone_food(zone)
            zone_info[f"zone_{i}_{zone.zone_type}"] = {
                'center': zone.center_position,
                'radius': zone.radius,
                'type': zone.zone_type,
                'food_types': zone.food_types,
                'current_food': food_count,
                'max_food': zone.max_food_sources,
                'active': zone.active
            }
        return zone_info
    
    def _trigger_resource_competition(self):
        """Trigger increased competition for resources"""
        
        self.resource_scarcity *= 1.1
        print(f"🍂 Resource competition intensified! Scarcity: {self.resource_scarcity:.2f}")
        
        # Some agents may lose territories
        if self.food_sources and random.random() < 0.3:
            lost_food = random.choice(self.food_sources)
            self.food_sources.remove(lost_food)
            print(f"🏴 Food source abandoned due to resource pressure")
    
    def _trigger_migration_event(self):
        """Trigger a migration event"""
        
        self.migration_pressure = random.uniform(0.5, 1.5)
        print(f"🦅 Migration event! Pressure: {self.migration_pressure:.2f}")
        
        # Reset migration pressure gradually
        self.migration_pressure *= 0.9
    
    def get_ecosystem_status(self) -> Dict[str, Any]:
        """Get comprehensive ecosystem status"""
        
        role_counts = {}
        for role in EcosystemRole:
            role_counts[role.value] = len([r for r in self.agent_roles.values() if r == role])
        
        return {
            'population_pressure': self.population_pressure,
            'resource_scarcity': self.resource_scarcity,
            'food_sources': len(self.food_sources),
            'role_distribution': role_counts,
            'migration_pressure': self.migration_pressure
        } 

    def _get_consumption_efficiency(self, role: EcosystemRole, food_type: str) -> float:
        """Get consumption efficiency based on agent role and food type - CARNIVORES AND SCAVENGERS ARE PURE ROBOT CONSUMERS"""
        efficiency_matrix = {
            EcosystemRole.HERBIVORE: {"plants": 1.0, "seeds": 0.9, "insects": 0.3, "meat": 0.8},  # Can eat meat at good efficiency
            EcosystemRole.CARNIVORE: {"meat": 0.0, "insects": 0.0, "plants": 0.0, "seeds": 0.0},  # PURE PREDATORS - NO environmental food
            EcosystemRole.OMNIVORE: {"plants": 0.8, "insects": 0.8, "seeds": 0.7, "meat": 0.8},  # Good at everything
            EcosystemRole.SCAVENGER: {"meat": 0.0, "insects": 0.0, "plants": 0.0, "seeds": 0.0},  # PURE SCAVENGERS - NO environmental food
            EcosystemRole.SYMBIONT: {"plants": 1.0, "seeds": 0.9, "insects": 0.6, "meat": 0.4}   # Plant specialists but flexible
        }
        
        return efficiency_matrix.get(role, {}).get(food_type, 0.5)
    
    def get_nearby_resources(self, position: Tuple[float, float], radius: float = 5.0, 
                           agent_role: Optional[EcosystemRole] = None) -> List[Dict[str, Any]]:
        """Get resources near a given position, filtered by agent role"""
        nearby = []
        for food in self.food_sources:
            distance = math.sqrt((position[0] - food.position[0])**2 + 
                               (position[1] - food.position[1])**2)
            if distance <= radius:
                # Check if this agent role can consume this food type
                if agent_role and agent_role in [EcosystemRole.CARNIVORE, EcosystemRole.SCAVENGER]:
                    # Carnivores and Scavengers cannot consume ANY environmental food - skip all environmental resources
                    continue
                
                # Check consumption efficiency for other roles
                if agent_role:
                    efficiency = self._get_consumption_efficiency(agent_role, food.food_type)
                    if efficiency <= 0:
                        continue  # Skip food this role cannot consume
                
                nearby.append({
                    'position': food.position,
                    'type': food.food_type,
                    'amount': food.amount,
                    'max_capacity': food.max_capacity,
                    'distance': distance,
                    'regeneration_rate': food.regeneration_rate
                })
        return nearby

    def consume_resource(self, agent_id: str, agent_position: Tuple[float, float], consumption_rate: float = 10.0) -> Tuple[float, str, Optional[Tuple[float, float]]]:
        """Agent consumes nearby resources and gains energy"""
        energy_gained = 0.0
        consumption_distance = 5.0  # Distance within which agents can consume food
        consumed_food_type = "none"
        consumed_food_position = None
        
        # Get agent role
        agent_role = self.agent_roles.get(agent_id, EcosystemRole.OMNIVORE)
        
        # Find nearest consumable resource
        best_food = None
        best_distance = float('inf')
        
        for food in self.food_sources:
            if food.amount <= 0:
                continue
                
            distance = math.sqrt((agent_position[0] - food.position[0])**2 + 
                               (agent_position[1] - food.position[1])**2)
            
            if distance <= consumption_distance:
                # Check if this agent can consume this food type
                consumption_efficiency = self._get_consumption_efficiency(agent_role, food.food_type)
                if consumption_efficiency > 0 and distance < best_distance:
                    best_food = food
                    best_distance = distance
        
        # Consume from best available food source
        if best_food:
            consumption_efficiency = self._get_consumption_efficiency(agent_role, best_food.food_type)
            
            # Calculate consumption amount based on efficiency and hunger
            base_consumption = consumption_rate * consumption_efficiency
            consumed = min(best_food.amount, base_consumption)
            
            # Remove food from source
            best_food.amount -= consumed
            
            # Calculate energy gained (simplified system)
            energy_per_unit = 0.1  # Base energy per unit consumed
            energy_gained = consumed * energy_per_unit
            
            # Track consumption details
            consumed_food_type = best_food.food_type
            consumed_food_position = best_food.position
            
            # Log consumption
            if energy_gained > 0.01:
                efficiency_str = f"{consumption_efficiency*100:.0f}%"
                print(f"🍽️ {str(agent_id)[:8]} consumed {consumed:.1f} {best_food.food_type} "
                      f"(efficiency: {efficiency_str}, energy: +{energy_gained:.2f})")
        
        return energy_gained, consumed_food_type, consumed_food_position

    def consume_robot(self, predator_id: str, predator_position: Tuple[float, float], 
                     all_agents: List[Any], agent_energy_levels: Dict[str, float], 
                     agent_health: Dict[str, Dict]) -> Tuple[float, str, Optional[Tuple[float, float]]]:
        """Implement robot consumption for carnivores and scavengers"""
        
        predator_role = self.agent_roles.get(predator_id, EcosystemRole.OMNIVORE)
        
        # Only certain roles can consume robots
        if predator_role not in [EcosystemRole.CARNIVORE, EcosystemRole.OMNIVORE, EcosystemRole.SCAVENGER]:
            return 0.0, "none", None
        
        consumption_distance = 5.0  # Same as food consumption
        best_prey = None
        best_distance = float('inf')
        
        # Find suitable prey
        for prey_agent in all_agents:
            if (getattr(prey_agent, '_destroyed', False) or not prey_agent.body or 
                prey_agent.id == predator_id):
                continue
            
            prey_position = (prey_agent.body.position.x, prey_agent.body.position.y)
            distance = math.sqrt((predator_position[0] - prey_position[0])**2 + 
                               (predator_position[1] - prey_position[1])**2)
            
            if distance <= consumption_distance:
                prey_energy = agent_energy_levels.get(prey_agent.id, 1.0)
                prey_health = agent_health.get(prey_agent.id, {'health': 1.0})['health']
                prey_role = self.agent_roles.get(prey_agent.id, EcosystemRole.OMNIVORE)
                
                # Role-specific hunting rules
                can_hunt = False
                
                if predator_role == EcosystemRole.CARNIVORE:
                    # Carnivores can hunt herbivores, scavengers, and omnivores
                    valid_prey_roles = [EcosystemRole.HERBIVORE, EcosystemRole.SCAVENGER, EcosystemRole.OMNIVORE]
                    can_hunt = (prey_role in valid_prey_roles and 
                               prey_health > 0.2 and prey_energy > 0.1)
                elif predator_role == EcosystemRole.OMNIVORE:
                    # Omnivores hunt weakened prey
                    can_hunt = (prey_energy < 0.5 or prey_health < 0.7)
                elif predator_role == EcosystemRole.SCAVENGER:
                    # Scavengers can consume weakened robots
                    can_hunt = (prey_energy < 0.3 and prey_health < 0.5)
                
                if can_hunt and distance < best_distance:
                    best_prey = prey_agent
                    best_distance = distance
        
        # Consume from best prey
        if best_prey:
            prey_id = best_prey.id
            prey_energy = agent_energy_levels.get(prey_id, 1.0)
            prey_health = agent_health.get(prey_id, {'health': 1.0})['health']
            
            # Consumption efficiency by predator role - REDUCED for longer survival
            consumption_rates = {
                EcosystemRole.CARNIVORE: 0.015,   # Reduced for longer survival
                EcosystemRole.OMNIVORE: 0.008,    # Reduced for longer survival  
                EcosystemRole.SCAVENGER: 0.012    # Reduced for longer survival
            }
            consumption_rate = consumption_rates.get(predator_role, 0.05)
            
            # Damage prey
            health_damage = min(prey_health, consumption_rate)
            energy_damage = min(prey_energy, consumption_rate * 0.5)
            
            # Apply damage to prey
            agent_health[prey_id]['health'] = max(0.0, prey_health - health_damage)
            agent_energy_levels[prey_id] = max(0.0, prey_energy - energy_damage)
            
            # Predator gains energy from consumed health/energy
            energy_gained = (health_damage + energy_damage) * 0.8  # Good conversion rate
            
            prey_position = (best_prey.body.position.x, best_prey.body.position.y)
            
            # Log predation
            predation_type = {
                EcosystemRole.CARNIVORE: "hunting",
                EcosystemRole.OMNIVORE: "consuming", 
                EcosystemRole.SCAVENGER: "scavenging"
            }.get(predator_role, "consuming")
            
            print(f"🍖 {str(predator_id)[:8]} is {predation_type} {str(prey_id)[:8]} "
                  f"(energy: +{energy_gained:.2f}, prey health: {agent_health[prey_id]['health']:.2f})")
            
            return energy_gained, prey_id, prey_position
        
        return 0.0, "none", None 