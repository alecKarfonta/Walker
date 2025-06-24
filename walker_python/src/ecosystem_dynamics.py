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
        
        role_weights = {
            EcosystemRole.HERBIVORE: 0.3 + cooperation * 0.3,
            EcosystemRole.CARNIVORE: 0.2 + speed * 0.4 + strength * 0.2,
            EcosystemRole.OMNIVORE: 0.4 + (speed + strength + cooperation) * 0.1,
            EcosystemRole.SCAVENGER: 0.1 + (1 - speed) * 0.2,
            EcosystemRole.SYMBIONT: cooperation * 0.5
        }
        
        # Select role based on weights
        role = max(role_weights.keys(), key=lambda r: role_weights[r] + random.uniform(-0.1, 0.1))
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
        """Update food source availability"""
        
        # Regenerate existing food sources
        for food in self.food_sources:
            if food.amount < food.max_capacity:
                food.amount = min(food.max_capacity, 
                                food.amount + food.regeneration_rate * self.seasonal_resource_modifier)
        
        # Add new food sources based on season and territory
        if random.random() < 0.15 * self.seasonal_resource_modifier:
            position = (random.uniform(-60, 60), random.uniform(-5, 35))
            food_type = random.choice(["plants", "insects", "seeds", "meat"])
            
            food_source = FoodSource(
                position=position,
                food_type=food_type,
                amount=random.uniform(10, 50),
                regeneration_rate=random.uniform(0.5, 2.0),
                max_capacity=random.uniform(20, 80)
            )
            self.food_sources.append(food_source)
    
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