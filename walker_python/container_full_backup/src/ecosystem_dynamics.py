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
        self.predation_events: List[Dict[str, Any]] = []  # Track predation events
        
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
        
        role_weights = {
            EcosystemRole.HERBIVORE: 0.4 + cooperation * 0.4,  # Increased base and cooperation weight
            EcosystemRole.CARNIVORE: 0.3 + speed * 0.3 + strength * 0.2,  # Reduced speed bonus
            EcosystemRole.OMNIVORE: 0.35 + (speed + strength + cooperation) * 0.08,  # Reduced base and bonus
            EcosystemRole.SCAVENGER: 0.2 + (1 - speed) * 0.3,  # Increased weight for slow agents
            EcosystemRole.SYMBIONT: 0.1 + cooperation * 0.6  # Increased cooperation dependency
        }
        
        # Ensure balanced distribution with some forced diversity
        role_counts = {}
        for r in EcosystemRole:
            role_counts[r] = len([x for x in self.agent_roles.values() if x == r])
        
        total_agents = len(self.agent_roles)
        
        # Force minimum representation of each role (at least 15% of population)
        min_per_role = max(1, total_agents // 6)  # About 16% minimum per role
        
        # Check if any role is underrepresented
        underrepresented_roles = [r for r, count in role_counts.items() if count < min_per_role]
        
        if underrepresented_roles and total_agents >= 5:
            # Force assign one of the underrepresented roles
            role = random.choice(underrepresented_roles)
            print(f"🎯 Force-assigned {role.value} for diversity (currently {role_counts[role]} of {total_agents})")
        else:
            # Normal weighted selection with reduced randomness
            role = max(role_weights.keys(), key=lambda r: role_weights[r] + random.uniform(-0.05, 0.05))
        
        self.agent_roles[agent_id] = role
        
        # Enhanced logging with weight details
        final_weight = role_weights[role]
        print(f"🦎 Agent {agent_id[:8]} assigned role: {role.value} (weight: {final_weight:.2f})")
        
        # Log role distribution periodically
        if len(self.agent_roles) % 5 == 0:
            updated_counts = {}
            for r in EcosystemRole:
                updated_counts[r.value] = len([x for x in self.agent_roles.values() if x == r])
            print(f"📊 Current role distribution: {updated_counts}")
        
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
                print(f"🐺 Pack bonus {pack_bonus:.2f} for {agent_id[:8]}")
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
        """Update food source availability - no regeneration, spawn new when depleted"""
        
        # Remove depleted food sources (anything below consumption threshold)
        initial_count = len(self.food_sources)
        depleted_sources = [food for food in self.food_sources if food.amount <= 0.15]
        self.food_sources = [food for food in self.food_sources if food.amount > 0.15]
        removed_count = initial_count - len(self.food_sources)
        
        if removed_count > 0:
            print(f"🗑️ Removed {removed_count} depleted food sources, {len(self.food_sources)} remaining")
            
            # Only replace some depleted resources to prevent resource explosion
            replacement_count = min(removed_count, max(1, removed_count // 2))  # Replace at most half
            for _ in range(replacement_count):
                if len(self.food_sources) < 25:  # Cap total resources at 25
                    self._spawn_new_food_source()
        
        # Only spawn new resources if we're low on total resources (reduced frequency)
        if len(self.food_sources) < 20 and random.random() < 0.05:  # Much lower spawn rate, higher target
            self._spawn_new_food_source()
    
    def _spawn_new_food_source(self):
        """Spawn a new food source in a fresh location"""
        # Find a location away from existing resources
        max_attempts = 10
        for _ in range(max_attempts):
            position = (random.uniform(-60, 60), random.uniform(-5, 35))
            
            # Check if location is far enough from existing resources
            if not self._resource_nearby(position, min_distance=8.0):
                food_type = random.choice(["plants", "insects", "seeds"])  # No meat spawns
                
                food_source = FoodSource(
                    position=position,
                    food_type=food_type,
                    amount=random.uniform(10, 25),  # Fresh resources start with good amounts
                    regeneration_rate=0.0,  # No regeneration - when it's gone, it's gone
                    max_capacity=random.uniform(15, 40)
                )
                self.food_sources.append(food_source)
                print(f"🍃 Spawned new {food_type} resource at ({position[0]:.1f}, {position[1]:.1f})")
                break
    
    def generate_resources_between_agents(self, agent_positions: List[Tuple[str, Tuple[float, float]]]):
        """Generate resources strategically between agents"""
        if len(agent_positions) < 2:
            return
        
        # Limit total resources to prevent performance issues
        if len(self.food_sources) > 25:  # Cap at 25 total resources
            return  # Don't generate more if we're at the limit
        
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
                    amount=random.uniform(12, 25),  # Good starting amounts
                    regeneration_rate=0.0,  # No regeneration
                    max_capacity=random.uniform(15, 35)
                )
                self.food_sources.append(food_source)
                print(f"🍃 Generated {food_type} resource at ({mid_x:.1f}, {mid_y:.1f})")
    
    def _determine_resource_type(self, role1: EcosystemRole, role2: EcosystemRole) -> str:
        """Determine resource type based on nearby agent roles (NO MEAT - carnivores hunt)"""
        role_preferences = {
            EcosystemRole.HERBIVORE: ["plants", "seeds"],
            EcosystemRole.CARNIVORE: ["insects"],  # Carnivores only get insects from environment, must hunt for meat
            EcosystemRole.OMNIVORE: ["plants", "insects", "seeds"],
            EcosystemRole.SCAVENGER: ["insects"],  # Scavengers eat insects and hunt weak agents
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
    
    def consume_resource(self, agent_id: str, agent_position: Tuple[float, float], consumption_rate: float = 0.1) -> Dict[str, Any]:
        """Agent consumes nearby resources and gains energy"""
        consumption_result = {
            'energy_gained': 0.0,
            'consumed': False,
            'food_type': None,
            'food_position': None,
            'amount_consumed': 0.0
        }
        
        consumption_distance = 8.0  # Distance within which agent can consume resources (increased for testing)
        
        for food in self.food_sources:
            distance = math.sqrt((agent_position[0] - food.position[0])**2 + 
                               (agent_position[1] - food.position[1])**2)
            
            if distance <= consumption_distance and food.amount > 0.1:  # Stricter check - don't consume from nearly empty resources
                # Calculate consumption based on agent role and food type
                agent_role = self.agent_roles.get(agent_id, EcosystemRole.OMNIVORE)
                consumption_efficiency = self._get_consumption_efficiency(agent_role, food.food_type)
                
                # Special handling for herbivores encountering non-plant foods
                if agent_role == EcosystemRole.HERBIVORE and food.food_type in ["insects", "meat"]:
                    # Herbivores cannot consume insects or meat at all
                    continue
                
                # Amount consumed this frame
                consumed = min(food.amount, consumption_rate * consumption_efficiency * 0.25)
                food.amount -= consumed
                
                # Energy gained (with better balance for slower consumption rate)
                energy_gained = consumed * consumption_efficiency * 0.2
                
                #if consumed > 0:
                #    print(f"🍽️ {agent_id[:8]} consumed {consumed:.1f} {food.food_type} (energy +{energy_gained:.2f}) at ({food.position[0]:.1f}, {food.position[1]:.1f})")
                
                if consumed > 0:
                    # Update consumption result
                    consumption_result.update({
                        'energy_gained': min(energy_gained, 0.5),  # Cap energy gain per frame
                        'consumed': True,
                        'food_type': food.food_type,
                        'food_position': food.position,
                        'amount_consumed': consumed
                    })
                
                # Break after consuming from one resource per frame
                break
        
        return consumption_result
    
    def _get_consumption_efficiency(self, role: EcosystemRole, food_type: str) -> float:
        """Get consumption efficiency based on agent role and food type (NO ENVIRONMENTAL MEAT)"""
        efficiency_matrix = {
            EcosystemRole.HERBIVORE: {"plants": 1.0, "seeds": 0.8, "insects": 0.0, "meat": 0.0},
            EcosystemRole.CARNIVORE: {"plants": 0.1, "seeds": 0.05, "insects": 0.6, "meat": 0.0},  # No environmental meat
            EcosystemRole.OMNIVORE: {"plants": 0.8, "insects": 0.8, "seeds": 0.7, "meat": 0.0},    # No environmental meat
            EcosystemRole.SCAVENGER: {"plants": 0.2, "seeds": 0.1, "insects": 0.7, "meat": 0.0},   # No environmental meat
            EcosystemRole.SYMBIONT: {"plants": 0.9, "seeds": 0.8, "insects": 0.5, "meat": 0.0}
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
    
    def attempt_predation(self, predator_id: str, predator_position: Tuple[float, float], 
                         potential_prey: List[Tuple[str, Tuple[float, float], float]]) -> Dict[str, Any]:
        """Carnivores/scavengers attempt to hunt other agents for meat"""
        predation_result = {
            'energy_gained': 0.0,
            'consumed': False,
            'prey_id': None,
            'hunt_successful': False
        }
        
        predator_role = self.agent_roles.get(predator_id, EcosystemRole.OMNIVORE)
        
        # Only carnivores and scavengers can hunt
        if predator_role not in [EcosystemRole.CARNIVORE, EcosystemRole.SCAVENGER]:
            return predation_result
        
        hunting_distance = 6.0  # Distance within which predation can occur
        
        for prey_id, prey_position, prey_energy in potential_prey:
            if prey_id == predator_id:  # Can't hunt yourself
                continue
                
            distance = math.sqrt((predator_position[0] - prey_position[0])**2 + 
                               (predator_position[1] - prey_position[1])**2)
            
            if distance <= hunting_distance:
                prey_role = self.agent_roles.get(prey_id, EcosystemRole.OMNIVORE)
                
                # Calculate hunting success probability
                base_success_rate = 0.15  # Base 15% success rate per attempt
                
                # Scavengers prefer weak prey, carnivores hunt any prey
                if predator_role == EcosystemRole.SCAVENGER:
                    # Scavengers more likely to succeed against weak prey
                    if prey_energy < 0.3:  # Very weak prey
                        success_rate = base_success_rate * 3.0  # 45% chance
                    elif prey_energy < 0.5:  # Somewhat weak prey
                        success_rate = base_success_rate * 2.0  # 30% chance
                    else:
                        success_rate = base_success_rate * 0.5  # Only 7.5% chance against healthy prey
                else:  # Carnivore
                    # Carnivores have consistent hunting ability
                    success_rate = base_success_rate * 1.5  # 22.5% base chance
                    
                    # Bonus against herbivores (natural prey)
                    if prey_role == EcosystemRole.HERBIVORE:
                        success_rate *= 1.4  # 31.5% chance
                
                # Attempt the hunt
                if random.random() < success_rate:
                    # Hunt successful! 
                    meat_gained = min(prey_energy * 0.6, 0.4)  # Gain up to 40% energy, based on prey health
                    
                    predation_result.update({
                        'energy_gained': meat_gained,
                        'consumed': True,
                        'prey_id': prey_id,
                        'hunt_successful': True
                    })
                    
                    print(f"🥩 {predator_role.value.title()} {predator_id[:8]} hunted {prey_role.value} {prey_id[:8]} (+{meat_gained:.2f} energy)")
                    
                    # Track predation event for ecosystem dynamics
                    self.predation_events.append({
                        'predator_id': predator_id,
                        'prey_id': prey_id,
                        'timestamp': time.time(),
                        'energy_transferred': meat_gained
                    })
                    
                    return predation_result  # Only one successful hunt per frame
        
        return predation_result
    
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