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
    
class EcosystemDynamics:
    """Advanced ecosystem with food webs and cooperation"""
    
    def __init__(self):
        self.food_sources: List[FoodSource] = []
        self.agent_roles: Dict[str, EcosystemRole] = {}
        self.rivalries: Dict[str, Set[str]] = {}  # agent_id -> set of rival ids
        
        # Ecosystem parameters
        self.carrying_capacity = 100
        self.resource_scarcity = 1.0  # 1.0 = normal, >1.0 = scarce, <1.0 = abundant
        self.cooperation_bonus = 1.2
        
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
        
        # Update resources
        self._update_food_sources()
        
        # Environmental pressures
        if self.population_pressure > 1.2:
            self._trigger_resource_competition()
        
        # Migration events
        if random.random() < 0.03:  # 3% chance
            self._trigger_migration_event()
        
        print(f"🌿 Ecosystem Gen {generation}: Population pressure {self.population_pressure:.2f}")
    
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
    
    def _update_food_sources(self):
        """Update food source availability"""
        
        # Keep all food sources including meat - all roles can consume meat at different efficiencies
        
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
        
        # Balanced food generation to maintain minimum population
        food_spawn_chance = 0.08 * self.seasonal_resource_modifier  # Reduced from 15% to 8%
        
        # Increase spawn chance if below minimum threshold
        if current_food_count < min_food_sources:
            shortage_factor = (min_food_sources - current_food_count) / min_food_sources
            food_spawn_chance += shortage_factor * 0.15  # Reduced from 30% to 15% additional chance when short
        
        # Add new food sources based on need and season
        if random.random() < food_spawn_chance:
            position = (random.uniform(-60, 60), random.uniform(-5, 35))
            food_type = random.choice(["plants", "insects", "seeds", "meat"])  # All food types available
            
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
        
        # Emergency food generation if critically low (much less aggressive)
        if current_food_count < min_food_sources // 3:  # Less than one third minimum (was half)
            emergency_spawns = min(2, min_food_sources - current_food_count)  # Spawn up to 2 at once (was 3)
            for _ in range(emergency_spawns):
                position = (random.uniform(-50, 50), random.uniform(-5, 30))
                food_type = random.choice(["plants", "insects", "seeds", "meat"])  # All food types for survival
                
                food_source = FoodSource(
                    position=position,
                    food_type=food_type,
                    amount=random.uniform(25, 70),  # Generous emergency food
                    regeneration_rate=random.uniform(1.0, 3.0),  # Fast regeneration
                    max_capacity=random.uniform(40, 120)
                )
                self.food_sources.append(food_source)
                # Silently create emergency food to reduce spam
                pass
    
    def generate_resources_between_agents(self, agent_positions: List[Tuple[str, Tuple[float, float]]]):
        """Generate resources strategically between agents with stable placement and minimum distance from robots"""
        if len(agent_positions) < 2:
            return
        
        # REDUCED: Only trim food if significantly over capacity (was 35, now 50)
        if len(self.food_sources) > 50:  # Less aggressive trimming
            # Keep more food sources for stability (was 25, now 35)
            self.food_sources = self.food_sources[-35:]
            print(f"🍂 Trimmed food sources to maintain performance (kept {len(self.food_sources)})")
        
        # Track original food count
        initial_food_count = len(self.food_sources)
        
        # Generate resources between adjacent agents with MINIMUM DISTANCE enforcement
        resources_created = 0
        for i in range(len(agent_positions) - 1):
            agent1_id, pos1 = agent_positions[i]
            agent2_id, pos2 = agent_positions[i + 1]
            
            # Calculate midpoint with moderate randomness
            base_mid_x = (pos1[0] + pos2[0]) / 2 + random.uniform(-8, 8)  # Increased spread
            base_mid_y = (pos1[1] + pos2[1]) / 2 + random.uniform(-3, 10)  # Increased spread
            
            # Ensure resource is above ground
            base_mid_y = max(base_mid_y, 3.0)  # Higher minimum ground clearance
            
            # CRITICAL: Ensure minimum distance from ALL agents (not just the two adjacent ones)
            min_distance_from_agents = 6.0  # Must be at least 6m from any agent
            valid_position = None
            
            # Try multiple positions to find one that's far enough from all agents
            for attempt in range(10):  # Up to 10 attempts to find good position
                test_x = base_mid_x + random.uniform(-5, 5)
                test_y = base_mid_y + random.uniform(-3, 3)
                test_position = (test_x, test_y)
                
                # Check distance from ALL agents
                too_close = False
                for agent_id, agent_pos in agent_positions:
                    distance_to_agent = math.sqrt((test_x - agent_pos[0])**2 + (test_y - agent_pos[1])**2)
                    if distance_to_agent < min_distance_from_agents:
                        too_close = True
                        break
                
                if not too_close:
                    valid_position = test_position
                    break
            
            # Only create resource if we found a valid position far from agents
            if valid_position is None:
                continue  # Skip this resource - couldn't find safe distance
            
            # Check if there's already a resource too close to this position
            if self._resource_nearby(valid_position, min_distance=12.0):  # Increased from 8.0 to 12.0
                continue  # Skip if too close to existing food
            
            # Determine resource type based on nearby agent roles
            agent1_role = self.agent_roles.get(agent1_id, EcosystemRole.OMNIVORE)
            agent2_role = self.agent_roles.get(agent2_id, EcosystemRole.OMNIVORE)
            
            food_type = self._determine_resource_type(agent1_role, agent2_role)
            
            # Create resource with longer-lasting properties for stability
            food_source = FoodSource(
                position=valid_position,
                food_type=food_type,
                amount=random.uniform(25, 50),  # Larger initial amounts for longer stability
                regeneration_rate=random.uniform(0.5, 1.2),  # Slower regeneration for stability
                max_capacity=random.uniform(40, 80)  # Higher capacity for longer lasting food
            )
            self.food_sources.append(food_source)
            resources_created += 1
        
        # Log resource generation for transparency
        if resources_created > 0:
            print(f"🌱 Created {resources_created} new food sources (was {initial_food_count}, now {len(self.food_sources)})")
            print(f"   📍 All food placed >6m from agents for stable rewards")
    
    def _determine_resource_type(self, role1: EcosystemRole, role2: EcosystemRole) -> str:
        """Determine resource type based on nearby agent roles"""
        role_preferences = {
            EcosystemRole.HERBIVORE: ["plants", "seeds", "meat"],  # Can consume meat efficiently
            EcosystemRole.CARNIVORE: [],  # PURE PREDATORS - NO environmental food preferences
            EcosystemRole.OMNIVORE: ["plants", "insects", "seeds", "meat"],  # Can eat everything
            EcosystemRole.SCAVENGER: [],  # PURE SCAVENGERS - NO environmental food preferences
            EcosystemRole.SYMBIONT: ["plants", "seeds", "insects"]  # Plant focused but flexible
        }
        
        # Combine preferences from both roles
        combined_preferences = []
        combined_preferences.extend(role_preferences.get(role1, ["plants"]))
        combined_preferences.extend(role_preferences.get(role2, ["plants"]))
        
        # If no preferences (e.g., carnivores and scavengers), default to herbivore food
        if not combined_preferences:
            combined_preferences = ["plants", "seeds", "insects", "meat"]  # Default food types
        
        return random.choice(combined_preferences)
    
    def _resource_nearby(self, position: Tuple[float, float], min_distance: float) -> bool:
        """Check if there's already a resource too close to the given position"""
        for food in self.food_sources:
            distance = math.sqrt((position[0] - food.position[0])**2 + 
                               (position[1] - food.position[1])**2)
            if distance < min_distance:
                return True
        return False
    
    def consume_resource(self, agent_id: str, agent_position: Tuple[float, float], consumption_rate: float = 10.0) -> Tuple[float, str, Optional[Tuple[float, float]]]:
        """Agent consumes nearby resources and gains energy - COMPLETELY REWRITTEN"""
        energy_gained = 0.0
        consumption_distance = 5.0  # Increased to match resource placement distance
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
                print(f"🍽️ {agent_id[:8]} consumed {consumed:.1f} {best_food.food_type} "
                      f"(efficiency: {efficiency_str}, energy: +{energy_gained:.2f})")
        
        return energy_gained, consumed_food_type, consumed_food_position

    def consume_robot(self, predator_id: str, predator_position: Tuple[float, float], 
                     all_agents: List[Any], agent_energy_levels: Dict[str, float], 
                     agent_health: Dict[str, Dict]) -> Tuple[float, str, Optional[Tuple[float, float]]]:
        """FIXED: Properly implement robot consumption for carnivores and scavengers"""
        
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
                    # Carnivores can ONLY hunt herbivores, scavengers, and omnivores (not other carnivores or symbionts)
                    valid_prey_roles = [EcosystemRole.HERBIVORE, EcosystemRole.SCAVENGER, EcosystemRole.OMNIVORE]
                    can_hunt = (prey_role in valid_prey_roles and 
                               prey_health > 0.2 and prey_energy > 0.1)
                elif predator_role == EcosystemRole.OMNIVORE:
                    # Omnivores hunt weakened prey
                    can_hunt = (prey_energy < 0.5 or prey_health < 0.7)
                elif predator_role == EcosystemRole.SCAVENGER:
                    # Scavengers can consume ANY weakened robots (any role when low energy/health)
                    can_hunt = (prey_energy < 0.3 and prey_health < 0.5)
                
                if can_hunt and distance < best_distance:
                    best_prey = prey_agent
                    best_distance = distance
        
        # Consume from best prey
        if best_prey:
            prey_id = best_prey.id
            prey_energy = agent_energy_levels.get(prey_id, 1.0)
            prey_health = agent_health.get(prey_id, {'health': 1.0})['health']
            
            # Consumption efficiency by predator role
            consumption_rates = {
                EcosystemRole.CARNIVORE: 0.15,   # Fast consumption
                EcosystemRole.OMNIVORE: 0.08,    # Moderate consumption  
                EcosystemRole.SCAVENGER: 0.12    # Good at scavenging
            }
            consumption_rate = consumption_rates.get(predator_role, 0.05)
            
            # Damage prey (no regeneration - predation should be effective)
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
            
            print(f"🍖 {predator_id[:8]} is {predation_type} {prey_id[:8]} "
                  f"(energy: +{energy_gained:.2f}, prey health: {agent_health[prey_id]['health']:.2f})")
            
            return energy_gained, prey_id, prey_position
        
        return 0.0, "none", None
    
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
        """Trigger a migration event that affects rivalries"""
        
        self.migration_pressure = random.uniform(0.5, 1.5)
        print(f"🦅 Migration event! Pressure: {self.migration_pressure:.2f}")
        
        # Some rivalries may break due to migration
        if self.rivalries and random.random() < 0.4:
            agent_id = random.choice(list(self.rivalries.keys()))
            if self.rivalries[agent_id]:
                former_rival = random.choice(list(self.rivalries[agent_id]))
                self.rivalries[agent_id].discard(former_rival)
                if former_rival in self.rivalries:
                    self.rivalries[former_rival].discard(agent_id)
                print(f"💔 Rivalry broken due to migration")
        
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
            'rivalries': len(self.rivalries),
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