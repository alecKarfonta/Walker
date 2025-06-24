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

class TerritoryType(Enum):
    FEEDING_GROUND = "feeding_ground"
    NESTING_AREA = "nesting_area"
    WATER_SOURCE = "water_source"
    SHELTER = "shelter"

@dataclass
class Territory:
    """Represents a territory in the ecosystem"""
    territory_type: TerritoryType
    position: Tuple[float, float]
    size: float
    resource_value: float
    owner_id: Optional[str] = None
    contested: bool = False
    creation_time: float = 0.0
    
    def __post_init__(self):
        self.creation_time = time.time()

@dataclass 
class FoodSource:
    """Represents available food in the ecosystem"""
    position: Tuple[float, float]
    food_type: str  # "plants", "meat", "insects", "seeds"
    amount: float
    regeneration_rate: float
    max_capacity: float
    
class EcosystemDynamics:
    """Advanced ecosystem with territorial behavior, food webs, and cooperation"""
    
    def __init__(self):
        self.territories: List[Territory] = []
        self.food_sources: List[FoodSource] = []
        self.agent_roles: Dict[str, EcosystemRole] = {}
        self.alliances: Dict[str, Set[str]] = {}  # agent_id -> set of ally ids
        self.rivalries: Dict[str, Set[str]] = {}  # agent_id -> set of rival ids
        self.pack_formations: List[Set[str]] = []  # Groups of cooperating agents
        
        # Ecosystem parameters
        self.carrying_capacity = 100
        self.resource_scarcity = 1.0  # 1.0 = normal, >1.0 = scarce, <1.0 = abundant
        self.cooperation_bonus = 1.2
        self.territory_defense_bonus = 1.15
        
        # Dynamic events
        self.migration_pressure = 0.0
        self.population_pressure = 0.0
        self.seasonal_resource_modifier = 1.0
        
        print("🌿 Ecosystem dynamics initialized!")
    
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
        
        # Update territories and resources
        self._update_territories()
        self._update_food_sources()
        self._manage_territorial_conflicts()
        self._update_pack_dynamics()
        
        # Environmental pressures
        if self.population_pressure > 1.2:
            self._trigger_resource_competition()
        
        # Migration events
        if random.random() < 0.03:  # 3% chance
            self._trigger_migration_event()
        
        print(f"🌿 Ecosystem Gen {generation}: Population pressure {self.population_pressure:.2f}, "
              f"{len(self.territories)} territories, {len(self.pack_formations)} packs")
    
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
        
        print(f"🦎 Agent {agent_id[:8]} assigned role: {role.value}")
        return role
    
    def form_alliance(self, agent1_id: str, agent2_id: str, cooperation_score: float):
        """Form an alliance between two agents"""
        
        # Higher cooperation scores make alliances more likely
        if cooperation_score > 0.6 and random.random() < cooperation_score:
            
            if agent1_id not in self.alliances:
                self.alliances[agent1_id] = set()
            if agent2_id not in self.alliances:
                self.alliances[agent2_id] = set()
            
            self.alliances[agent1_id].add(agent2_id)
            self.alliances[agent2_id].add(agent1_id)
            
            print(f"🤝 Alliance formed between {agent1_id[:8]} and {agent2_id[:8]}")
            
            # Check if they should form or join a pack
            self._check_pack_formation([agent1_id, agent2_id])
            
            return True
        return False
    
    def create_rivalry(self, agent1_id: str, agent2_id: str, competition_intensity: float):
        """Create a rivalry between competing agents"""
        
        if competition_intensity > 0.7 and random.random() < 0.3:
            
            if agent1_id not in self.rivalries:
                self.rivalries[agent1_id] = set()
            if agent2_id not in self.rivalries:
                self.rivalries[agent2_id] = set()
            
            self.rivalries[agent1_id].add(agent2_id)
            self.rivalries[agent2_id].add(agent1_id)
            
            print(f"⚔️ Rivalry formed between {agent1_id[:8]} and {agent2_id[:8]}")
            return True
        return False
    
    def claim_territory(self, agent_id: str, position: Tuple[float, float]) -> Optional[Territory]:
        """Attempt to claim territory at a given position"""
        
        territory_type = random.choice(list(TerritoryType))
        size = random.uniform(5, 20)
        resource_value = random.uniform(0.5, 2.0)
        
        # Check for conflicts with existing territories
        for existing in self.territories:
            distance = math.sqrt((position[0] - existing.position[0])**2 + 
                               (position[1] - existing.position[1])**2)
            if distance < (size + existing.size) / 2:
                # Territory overlap - mark as contested
                existing.contested = True
                print(f"🏴 Territory conflict at {position}")
                return None
        
        territory = Territory(
            territory_type=territory_type,
            position=position,
            size=size,
            resource_value=resource_value,
            owner_id=agent_id
        )
        
        self.territories.append(territory)
        print(f"🏴 {agent_id[:8]} claimed {territory_type.value} territory")
        return territory
    
    def get_ecosystem_effects(self, agent_id: str, position: Tuple[float, float]) -> Dict[str, float]:
        """Get ecosystem effects for an agent at a specific position"""
        
        effects = {
            'fitness_multiplier': 1.0,
            'resource_access': 1.0,
            'cooperation_bonus': 0.0,
            'territory_bonus': 0.0,
            'competition_penalty': 0.0
        }
        
        # Role-based effects
        role = self.agent_roles.get(agent_id, EcosystemRole.OMNIVORE)
        role_multipliers = {
            EcosystemRole.HERBIVORE: 1.0,
            EcosystemRole.CARNIVORE: 1.1,
            EcosystemRole.OMNIVORE: 0.95,
            EcosystemRole.SCAVENGER: 0.9,
            EcosystemRole.SYMBIONT: 1.2 if agent_id in self.alliances else 0.8
        }
        effects['fitness_multiplier'] *= role_multipliers[role]
        
        # Alliance benefits
        if agent_id in self.alliances and self.alliances[agent_id]:
            ally_count = len(self.alliances[agent_id])
            effects['cooperation_bonus'] = min(0.3, ally_count * 0.1)
            effects['fitness_multiplier'] *= (1 + effects['cooperation_bonus'])
        
        # Pack benefits
        for pack in self.pack_formations:
            if agent_id in pack:
                pack_size = len(pack)
                pack_bonus = min(0.4, pack_size * 0.08)
                effects['fitness_multiplier'] *= (1 + pack_bonus)
                # Pack bonus applied silently
                break
        
        # Territory effects
        for territory in self.territories:
            if territory.owner_id == agent_id:
                distance = math.sqrt((position[0] - territory.position[0])**2 + 
                                   (position[1] - territory.position[1])**2)
                if distance < territory.size:
                    effects['territory_bonus'] = territory.resource_value * 0.1
                    effects['fitness_multiplier'] *= self.territory_defense_bonus
        
        # Resource scarcity effects
        effects['resource_access'] *= (2.0 - self.resource_scarcity) / self.seasonal_resource_modifier
        
        # Population pressure effects
        if self.population_pressure > 1.0:
            competition_penalty = (self.population_pressure - 1.0) * 0.2
            effects['competition_penalty'] = competition_penalty
            effects['fitness_multiplier'] *= (1 - competition_penalty)
        
        return effects
    
    def _update_territories(self):
        """Update territory states and resolve conflicts"""
        
        # Remove abandoned territories (older than 10 minutes)
        current_time = time.time()
        self.territories = [t for t in self.territories 
                          if current_time - t.creation_time < 600]
        
        # Spawn new territories randomly
        if len(self.territories) < 15 and random.random() < 0.1:
            position = (random.uniform(-50, 50), random.uniform(0, 30))
            territory_type = random.choice(list(TerritoryType))
            
            unclaimed_territory = Territory(
                territory_type=territory_type,
                position=position,
                size=random.uniform(8, 25),
                resource_value=random.uniform(0.8, 1.5)
            )
            self.territories.append(unclaimed_territory)
    
    def _update_food_sources(self):
        """Update food source availability"""
        
        # Clean up existing meat food sources - carnivores must hunt robots instead
        self.food_sources = [food for food in self.food_sources if food.food_type != "meat"]
        
        # Regenerate existing food sources
        for food in self.food_sources:
            if food.amount < food.max_capacity:
                food.amount = min(food.max_capacity, 
                                food.amount + food.regeneration_rate * self.seasonal_resource_modifier)
        
        # Remove effectively depleted food sources (using small threshold for floating point precision)
        depletion_threshold = 0.05  # Food sources with less than 0.05 are considered depleted
        initial_count = len(self.food_sources)
        self.food_sources = [food for food in self.food_sources if food.amount > depletion_threshold]
        removed_count = initial_count - len(self.food_sources)
        
        # Silently remove depleted food sources
        
        # Maintain minimum food population - ensure adequate resources for survival
        min_food_sources = max(15, int(self.carrying_capacity * 0.2))  # At least 15 or 20% of carrying capacity
        current_food_count = len(self.food_sources)
        
        # Enhanced food generation to maintain minimum population
        food_spawn_chance = 0.15 * self.seasonal_resource_modifier
        
        # Increase spawn chance if below minimum threshold
        if current_food_count < min_food_sources:
            shortage_factor = (min_food_sources - current_food_count) / min_food_sources
            food_spawn_chance += shortage_factor * 0.3  # Up to 30% additional chance when short
        
        # Add new food sources based on need and season
        if random.random() < food_spawn_chance:
            position = (random.uniform(-60, 60), random.uniform(-5, 35))
            food_type = random.choice(["plants", "insects", "seeds"])  # Removed meat - carnivores must hunt other robots
            
            # Generate more substantial food sources when population is low
            if current_food_count < min_food_sources:
                amount = random.uniform(20, 60)  # Larger food sources when needed
                max_capacity = random.uniform(30, 100)
                regeneration_rate = random.uniform(0.8, 2.5)  # Faster regeneration
            else:
                amount = random.uniform(10, 50)
                max_capacity = random.uniform(20, 80)
                regeneration_rate = random.uniform(0.5, 2.0)
            
            food_source = FoodSource(
                position=position,
                food_type=food_type,
                amount=amount,
                regeneration_rate=regeneration_rate,
                max_capacity=max_capacity
            )
            self.food_sources.append(food_source)
        
        # Emergency food generation if critically low
        if current_food_count < min_food_sources // 2:  # Less than half minimum
            emergency_spawns = min(3, min_food_sources - current_food_count)  # Spawn up to 3 at once
            for _ in range(emergency_spawns):
                position = (random.uniform(-50, 50), random.uniform(-5, 30))
                food_type = random.choice(["plants", "insects", "seeds"])  # Prefer easier food types
                
                food_source = FoodSource(
                    position=position,
                    food_type=food_type,
                    amount=random.uniform(25, 70),  # Generous emergency food
                    regeneration_rate=random.uniform(1.0, 3.0),  # Fast regeneration
                    max_capacity=random.uniform(40, 120)
                )
                self.food_sources.append(food_source)
                print(f"🚨 Emergency food spawn: {food_type} at ({position[0]:.1f}, {position[1]:.1f})")
    
    def generate_resources_between_agents(self, agent_positions: List[Tuple[str, Tuple[float, float]]]):
        """Generate resources strategically between agents"""
        if len(agent_positions) < 2:
            return
        
        # Clear existing food sources to regenerate fresh ones
        if len(self.food_sources) > 50:  # Don't let resources accumulate too much
            self.food_sources = self.food_sources[-30:]  # Keep only 30 most recent
        
        # Generate resources between adjacent agents
        for i in range(len(agent_positions) - 1):
            agent1_id, pos1 = agent_positions[i]
            agent2_id, pos2 = agent_positions[i + 1]
            
            # Calculate midpoint with some randomness
            mid_x = (pos1[0] + pos2[0]) / 2 + random.uniform(-5, 5)
            mid_y = (pos1[1] + pos2[1]) / 2 + random.uniform(-2, 8)
            
            # Ensure resource is above ground
            mid_y = max(mid_y, 2.0)
            
            # Determine resource type based on nearby agent roles
            agent1_role = self.agent_roles.get(agent1_id, EcosystemRole.OMNIVORE)
            agent2_role = self.agent_roles.get(agent2_id, EcosystemRole.OMNIVORE)
            
            food_type = self._determine_resource_type(agent1_role, agent2_role)
            
            # Create resource if there isn't one too close already
            if not self._resource_nearby((mid_x, mid_y), min_distance=8.0):
                food_source = FoodSource(
                    position=(mid_x, mid_y),
                    food_type=food_type,
                    amount=random.uniform(15, 40),
                    regeneration_rate=random.uniform(0.3, 1.5),
                    max_capacity=random.uniform(25, 60)
                )
                self.food_sources.append(food_source)
    
    def _determine_resource_type(self, role1: EcosystemRole, role2: EcosystemRole) -> str:
        """Determine resource type based on nearby agent roles"""
        role_preferences = {
            EcosystemRole.HERBIVORE: ["plants", "seeds"],
            EcosystemRole.CARNIVORE: ["insects"],  # Removed meat - carnivores must hunt for meat
            EcosystemRole.OMNIVORE: ["plants", "insects", "seeds"],
            EcosystemRole.SCAVENGER: ["insects"],  # Removed meat - scavengers must find robot remains
            EcosystemRole.SYMBIONT: ["plants", "seeds"]
        }
        
        # Combine preferences from both roles
        combined_preferences = []
        combined_preferences.extend(role_preferences.get(role1, ["plants"]))
        combined_preferences.extend(role_preferences.get(role2, ["plants"]))
        
        return random.choice(combined_preferences)
    
    def _resource_nearby(self, position: Tuple[float, float], min_distance: float) -> bool:
        """Check if there's already a resource too close to the given position"""
        for food in self.food_sources:
            distance = math.sqrt((position[0] - food.position[0])**2 + 
                               (position[1] - food.position[1])**2)
            if distance < min_distance:
                return True
        return False
    
    def consume_resource(self, agent_id: str, agent_position: Tuple[float, float], consumption_rate: float = 4.0) -> float:
        """Agent consumes nearby resources and gains energy"""
        energy_gained = 0.0
        consumption_distance = 3.0  # Distance within which agent can consume resources
        
        for food in self.food_sources:
            distance = math.sqrt((agent_position[0] - food.position[0])**2 + 
                               (agent_position[1] - food.position[1])**2)
            
            if distance <= consumption_distance and food.amount > 0:
                # Calculate consumption based on agent role and food type
                agent_role = self.agent_roles.get(agent_id, EcosystemRole.OMNIVORE)
                consumption_efficiency = self._get_consumption_efficiency(agent_role, food.food_type)
                
                # Amount consumed this frame
                consumed = min(food.amount, consumption_rate * consumption_efficiency)
                food.amount -= consumed
                
                # Energy gained (substantial restoration from eating)
                energy_gained += consumed * consumption_efficiency * 0.2  # Doubled base energy gain
                
                # Consumption happens silently
                
                # Break after consuming from one resource per frame
                break
        
        return min(energy_gained, 1.0)  # Allow full energy restoration from eating
    
    def attempt_predation(self, predator_id: str, predator_position: Tuple[float, float], 
                         available_agents: List[Tuple[str, Tuple[float, float], str, float]]) -> Tuple[float, Optional[str]]:
        """Carnivore attempts to hunt and consume another agent for energy.
        
        Args:
            predator_id: ID of the hunting agent
            predator_position: Position of the predator
            available_agents: List of (agent_id, position, role, energy) tuples for potential prey
            
        Returns:
            Tuple of (energy_gained, victim_id) - victim_id is None if no successful predation
        """
        predator_role = self.agent_roles.get(predator_id, EcosystemRole.OMNIVORE)
        
        # Only carnivores and omnivores can hunt
        if predator_role not in [EcosystemRole.CARNIVORE, EcosystemRole.OMNIVORE]:
            return 0.0, None
            
        hunting_range = 5.0  # Distance within which predation can occur
        energy_gained = 0.0
        victim_id = None
        
        # Find potential prey within hunting range
        potential_prey = []
        for agent_id, position, role, energy in available_agents:
            if agent_id == predator_id:  # Can't hunt yourself
                continue
                
            distance = math.sqrt((predator_position[0] - position[0])**2 + 
                               (predator_position[1] - position[1])**2)
            
            if distance <= hunting_range:
                # Calculate hunting success probability  
                prey_role_enum = self.agent_roles.get(agent_id, EcosystemRole.OMNIVORE)
                success_probability = self._calculate_hunting_success(predator_role, prey_role_enum, energy, distance)
                potential_prey.append((agent_id, role, energy, distance, success_probability))
        
        if not potential_prey:
            return 0.0, None
            
        # Sort by hunting success probability (highest first)
        potential_prey.sort(key=lambda x: x[4], reverse=True)
        
        # Attempt to hunt the most viable prey
        for prey_id, prey_role, prey_energy, distance, success_prob in potential_prey:
            if random.random() < success_prob:
                # Successful predation!
                prey_role_enum = self.agent_roles.get(prey_id, EcosystemRole.OMNIVORE)
                energy_gained = self._calculate_predation_energy_gain(predator_role, prey_role_enum, prey_energy)
                victim_id = prey_id
                
                print(f"🦁 {predator_id[:8]} hunted {prey_id[:8]}")
                break
                
        return energy_gained, victim_id
    
    def _calculate_hunting_success(self, predator_role: EcosystemRole, prey_role: EcosystemRole, 
                                  prey_energy: float, distance: float) -> float:
        """Calculate the probability of successful predation"""
        
        # Base hunting success rates by predator role
        base_success = {
            EcosystemRole.CARNIVORE: 0.3,  # 30% base success for specialists
            EcosystemRole.OMNIVORE: 0.15,  # 15% base success for generalists
        }.get(predator_role, 0.0)
        
        # Prey vulnerability by role
        prey_vulnerability = {
            EcosystemRole.HERBIVORE: 1.0,   # Most vulnerable
            EcosystemRole.OMNIVORE: 0.7,    # Moderately vulnerable  
            EcosystemRole.SYMBIONT: 0.8,    # Somewhat vulnerable
            EcosystemRole.SCAVENGER: 0.5,   # Less vulnerable (cautious)
            EcosystemRole.CARNIVORE: 0.2,   # Very difficult to hunt
        }.get(prey_role, 0.5)
        
        # Distance factor (closer = higher success)
        distance_factor = max(0.1, 1.0 - (distance / 5.0))
        
        # Energy factor (weaker prey easier to catch)
        energy_factor = max(0.5, 2.0 - prey_energy * 2.0)  # Lower energy = higher vulnerability
        
        # Calculate final success probability
        success_probability = base_success * prey_vulnerability * distance_factor * energy_factor
        
        return min(0.8, success_probability)  # Cap at 80% max success rate
    
    def _calculate_predation_energy_gain(self, predator_role: EcosystemRole, prey_role: EcosystemRole, 
                                       prey_energy: float) -> float:
        """Calculate energy gained from successful predation"""
        
        # Base energy gain from consuming another robot
        base_energy_gain = 0.4  # Substantial energy from predation
        
        # Predator efficiency
        predator_efficiency = {
            EcosystemRole.CARNIVORE: 1.2,  # Carnivores are efficient hunters
            EcosystemRole.OMNIVORE: 0.9,   # Omnivores are less efficient
        }.get(predator_role, 0.5)
        
        # Prey nutritional value
        prey_nutrition = {
            EcosystemRole.HERBIVORE: 1.0,   # Standard nutrition
            EcosystemRole.OMNIVORE: 1.1,    # Slightly more nutritious
            EcosystemRole.SYMBIONT: 0.9,    # Slightly less nutritious
            EcosystemRole.SCAVENGER: 0.8,   # Less nutritious
            EcosystemRole.CARNIVORE: 1.3,   # Most nutritious but hardest to catch
        }.get(prey_role, 1.0)
        
        # Energy factor (healthier prey = more energy gain)
        energy_multiplier = 0.5 + prey_energy * 0.5  # 0.5 to 1.0 multiplier
        
        total_energy_gain = base_energy_gain * predator_efficiency * prey_nutrition * energy_multiplier
        
        return min(1.0, total_energy_gain)  # Cap at full energy restoration
    
    def _get_consumption_efficiency(self, role: EcosystemRole, food_type: str) -> float:
        """Get consumption efficiency based on agent role and food type"""
        efficiency_matrix = {
            EcosystemRole.HERBIVORE: {"plants": 1.0, "seeds": 0.8, "insects": 0.0, "meat": 0.0},  # Herbivores can ONLY eat plants and seeds
            EcosystemRole.CARNIVORE: {"meat": 0.0, "insects": 0.7, "plants": 0.0, "seeds": 0.0},  # Carnivores can't eat environmental meat - must hunt robots
            EcosystemRole.OMNIVORE: {"plants": 0.8, "insects": 0.8, "seeds": 0.7, "meat": 0.0},  # Omnivores can't eat environmental meat either
            EcosystemRole.SCAVENGER: {"meat": 0.0, "insects": 0.8, "plants": 0.3, "seeds": 0.2},  # Scavengers can't eat environmental meat
            EcosystemRole.SYMBIONT: {"plants": 0.9, "seeds": 0.8, "insects": 0.5, "meat": 0.0}   # Symbionts can't eat meat
        }
        
        return efficiency_matrix.get(role, {}).get(food_type, 0.5)
    
    def get_nearby_resources(self, position: Tuple[float, float], radius: float = 5.0) -> List[Dict[str, Any]]:
        """Get resources near a given position"""
        nearby = []
        for food in self.food_sources:
            distance = math.sqrt((position[0] - food.position[0])**2 + 
                               (position[1] - food.position[1])**2)
            if distance <= radius:
                nearby.append({
                    'position': food.position,
                    'type': food.food_type,
                    'amount': food.amount,
                    'max_capacity': food.max_capacity,
                    'distance': distance,
                    'regeneration_rate': food.regeneration_rate
                })
        return nearby
    
    def _manage_territorial_conflicts(self):
        """Resolve territorial conflicts"""
        
        contested_territories = [t for t in self.territories if t.contested]
        
        for territory in contested_territories:
            # 50% chance to resolve conflict each update
            if random.random() < 0.5:
                territory.contested = False
                # Winner keeps territory, loser may form rivalry
                print(f"🏴 Territorial conflict resolved at {territory.position}")
    
    def _update_pack_dynamics(self):
        """Update pack formations and dynamics"""
        
        # Dissolve old packs occasionally
        self.pack_formations = [pack for pack in self.pack_formations 
                               if len(pack) >= 2 and random.random() > 0.05]
        
        # Form new packs from strong alliances
        alliance_groups = {}
        for agent_id, allies in self.alliances.items():
            if len(allies) >= 2:
                # Create potential pack from strong alliances
                potential_pack = {agent_id} | allies
                if len(potential_pack) >= 3:
                    # Check if this group should form a pack
                    if random.random() < 0.3:  # 30% chance
                        self.pack_formations.append(potential_pack)
                        print(f"🐺 New pack formed with {len(potential_pack)} members")
    
    def _check_pack_formation(self, agent_ids: List[str]):
        """Check if agents should form or join a pack"""
        
        # Look for existing packs that these agents could join
        for pack in self.pack_formations:
            if any(agent_id in pack for agent_id in agent_ids):
                # Add other agents to existing pack
                for agent_id in agent_ids:
                    pack.add(agent_id)
                print(f"🐺 Agents joined existing pack (now {len(pack)} members)")
                return
        
        # Form new pack if conditions are right
        if len(agent_ids) >= 2 and random.random() < 0.4:
            new_pack = set(agent_ids)
            self.pack_formations.append(new_pack)
            print(f"🐺 New pack formed with {len(new_pack)} founding members")
    
    def _trigger_resource_competition(self):
        """Trigger increased competition for resources"""
        
        self.resource_scarcity *= 1.1
        print(f"🍂 Resource competition intensified! Scarcity: {self.resource_scarcity:.2f}")
        
        # Some agents may lose territories
        if self.territories and random.random() < 0.3:
            lost_territory = random.choice(self.territories)
            self.territories.remove(lost_territory)
            print(f"🏴 Territory abandoned due to resource pressure")
    
    def _trigger_migration_event(self):
        """Trigger a migration event that affects alliances and territories"""
        
        self.migration_pressure = random.uniform(0.5, 1.5)
        print(f"🦅 Migration event! Pressure: {self.migration_pressure:.2f}")
        
        # Some alliances may break due to migration
        if self.alliances and random.random() < 0.4:
            agent_id = random.choice(list(self.alliances.keys()))
            if self.alliances[agent_id]:
                former_ally = random.choice(list(self.alliances[agent_id]))
                self.alliances[agent_id].discard(former_ally)
                if former_ally in self.alliances:
                    self.alliances[former_ally].discard(agent_id)
                print(f"💔 Alliance broken due to migration")
        
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
            'territories': len(self.territories),
            'contested_territories': len([t for t in self.territories if t.contested]),
            'food_sources': len(self.food_sources),
            'alliances': len(self.alliances),
            'rivalries': len(self.rivalries),
            'packs': len(self.pack_formations),
            'role_distribution': role_counts,
            'migration_pressure': self.migration_pressure
        } 