"""
Enhanced Evolution Engine for Physical Parameter Evolution.

This module provides a comprehensive evolutionary algorithm implementation
inspired by the Java CrawlingCrate evolution system, with robust selection
strategies, diversity maintenance, and advanced genetic operators.
"""

import numpy as np
import random
from typing import List, Dict, Any, Optional, Callable, Tuple, Union, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import defaultdict
import math
from copy import deepcopy
import time
import logging

from src.agents.base_agent import BaseAgent
from src.agents.evolutionary_crawling_agent import EvolutionaryCrawlingAgent
from src.agents.physical_parameters import PhysicalParameters, PhysicalParameterSpace
from src.population.population_controller import PopulationController, AgentRecord
# from src.physics.world import WorldController  # Avoid import issues


@dataclass
class EvolutionConfig:
    """Configuration for evolutionary algorithm parameters."""
    
    # Population management
    population_size: int = 50
    elite_size: int = 5
    max_generations: int = 100
    
    # Selection parameters
    tournament_size: int = 3
    selection_pressure: float = 1.5
    diversity_weight: float = 0.3
    
    # Reproduction parameters
    crossover_rate: float = 0.7
    mutation_rate: float = 0.1
    clone_rate: float = 0.1
    immigration_rate: float = 0.05  # Rate of introducing random individuals
    
    # Diversity maintenance
    target_diversity: float = 0.3
    diversity_check_interval: int = 5  # Generations between diversity checks
    niche_size: float = 0.2  # Size of fitness niches for speciation
    
    # Adaptive parameters
    adaptive_mutation: bool = True
    adaptive_selection: bool = True
    fitness_stagnation_threshold: int = 10  # Generations without improvement
    
    # Advanced features
    use_speciation: bool = True
    use_co_evolution: bool = False
    use_hall_of_fame: bool = True
    hall_of_fame_size: int = 10
    
    # 🌟 NEW EVOLUTIONARY FEATURES 🌟
    
    # Environmental Challenges
    enable_environmental_challenges: bool = True
    obstacle_spawn_rate: float = 0.1  # Chance per generation
    terrain_change_rate: float = 0.05  # Chance of terrain modification
    weather_change_interval: int = 20  # Generations between weather changes
    
    # Multi-Objective Evolution
    enable_multi_objective: bool = True
    fitness_objectives: Optional[List[str]] = None  # Will be set in __post_init__
    objective_weights: Optional[Dict[str, float]] = None  # Dynamic weights
    
    # Seasonal Evolution
    enable_seasonal_evolution: bool = True
    season_length: int = 25  # Generations per season
    seasonal_pressure_intensity: float = 0.3  # How much seasons affect evolution
    
    # Mutation Events
    enable_mutation_storms: bool = True
    mutation_storm_probability: float = 0.02  # Per generation
    storm_mutation_multiplier: float = 5.0  # Mutation rate during storms
    
    # Social Learning
    enable_social_learning: bool = True
    social_learning_rate: float = 0.1  # Rate of knowledge transfer
    teacher_selection_pressure: float = 1.5  # Bias toward learning from better agents
    
    # Resource Competition
    enable_resource_competition: bool = True
    total_resources: float = 1000.0  # Total energy available per generation
    resource_decay_rate: float = 0.02  # Resources decrease over time
    
    # Predator-Prey Dynamics
    enable_predator_prey: bool = True
    predator_ratio: float = 0.1  # Fraction of population that becomes predators
    predator_advantage: float = 1.2  # Fitness bonus for successful predation
    
    # Sexual Selection & Mating Preferences
    enable_sexual_selection: bool = True
    mating_preference_strength: float = 0.3  # How much preferences matter
    beauty_traits: Optional[List[str]] = None  # Traits that affect attractiveness
    
    # Catastrophic Events
    enable_catastrophes: bool = True
    catastrophe_probability: float = 0.01  # Per generation
    survival_rate_during_catastrophe: float = 0.3  # Fraction that survives
    
    def __post_init__(self):
        """Initialize default values that depend on other fields."""
        if self.fitness_objectives is None:
            self.fitness_objectives = ['speed', 'efficiency', 'stability', 'exploration', 'cooperation']
        
        if self.objective_weights is None:
            self.objective_weights = {
                'speed': 0.3,
                'efficiency': 0.25,
                'stability': 0.2,
                'exploration': 0.15,
                'cooperation': 0.1
            }
        
        if self.beauty_traits is None:
            self.beauty_traits = ['symmetry', 'size', 'color_pattern', 'movement_grace']


class SelectionStrategy(ABC):
    """Abstract base class for selection strategies."""
    
    @abstractmethod
    def select(self, population: List[EvolutionaryCrawlingAgent], 
              num_parents: int, config: EvolutionConfig) -> List[EvolutionaryCrawlingAgent]:
        """Select parents for reproduction."""
        pass


class TournamentSelection(SelectionStrategy):
    """Tournament selection with optional diversity consideration."""
    
    def select(self, population: List[EvolutionaryCrawlingAgent], 
              num_parents: int, config: EvolutionConfig) -> List[EvolutionaryCrawlingAgent]:
        """Select parents using tournament selection."""
        parents = []
        
        for _ in range(num_parents):
            # Create tournament
            tournament_size = min(config.tournament_size, len(population))
            tournament = random.sample(population, tournament_size)
            
            # Select best from tournament
            if config.diversity_weight > 0:
                # Consider both fitness and diversity
                best = self._select_with_diversity(tournament, population, config)
            else:
                # Pure fitness-based selection
                best = max(tournament, key=lambda agent: agent.get_evolutionary_fitness())
            
            parents.append(best)
        
        return parents
    
    def _select_with_diversity(self, tournament: List[EvolutionaryCrawlingAgent],
                             full_population: List[EvolutionaryCrawlingAgent],
                             config: EvolutionConfig) -> EvolutionaryCrawlingAgent:
        """Select considering both fitness and diversity."""
        best_score = -float('inf')
        best_agent = tournament[0]
        
        for agent in tournament:
            fitness = agent.get_evolutionary_fitness()
            diversity = self._calculate_diversity_score(agent, full_population)
            
            # Combined score
            score = fitness + config.diversity_weight * diversity
            
            if score > best_score:
                best_score = score
                best_agent = agent
        
        return best_agent
    
    def _calculate_diversity_score(self, agent: EvolutionaryCrawlingAgent,
                                 population: List[EvolutionaryCrawlingAgent]) -> float:
        """Calculate diversity score for an agent relative to population."""
        agent_metrics = agent.get_diversity_metrics()
        
        # Calculate average distance to other agents
        distances = []
        for other in population:
            if other.id != agent.id:
                other_metrics = other.get_diversity_metrics()
                distance = self._calculate_metric_distance(agent_metrics, other_metrics)
                distances.append(distance)
        
        return np.mean(distances) if distances else 0.0
    
    def _calculate_metric_distance(self, metrics1: Dict[str, float], 
                                 metrics2: Dict[str, float]) -> float:
        """Calculate Euclidean distance between metric dictionaries."""
        distance = 0.0
        for key in metrics1:
            if key in metrics2:
                distance += (metrics1[key] - metrics2[key]) ** 2
        return math.sqrt(distance)


class FitnessProportionateSelection(SelectionStrategy):
    """Roulette wheel selection based on fitness."""
    
    def select(self, population: List[EvolutionaryCrawlingAgent], 
              num_parents: int, config: EvolutionConfig) -> List[EvolutionaryCrawlingAgent]:
        """Select parents using fitness-proportionate selection."""
        # Get fitness values
        fitnesses = [agent.get_evolutionary_fitness() for agent in population]
        
        # Handle negative fitnesses by shifting
        min_fitness = min(fitnesses)
        if min_fitness < 0:
            fitnesses = [f - min_fitness + 0.1 for f in fitnesses]
        
        # Calculate selection probabilities
        total_fitness = sum(fitnesses)
        if total_fitness == 0:
            # Random selection if all fitnesses are zero
            return random.sample(population, num_parents)
        
        probabilities = [f / total_fitness for f in fitnesses]
        
        # Select parents
        parents = []
        for _ in range(num_parents):
            selected_idx = np.random.choice(len(population), p=probabilities)
            parents.append(population[selected_idx])
        
        return parents


class RankSelection(SelectionStrategy):
    """Rank-based selection to avoid dominance by super-fit individuals."""
    
    def select(self, population: List[EvolutionaryCrawlingAgent], 
              num_parents: int, config: EvolutionConfig) -> List[EvolutionaryCrawlingAgent]:
        """Select parents using rank-based selection."""
        # Sort population by fitness
        sorted_pop = sorted(population, key=lambda agent: agent.get_evolutionary_fitness())
        
        # Assign ranks (higher rank = better fitness)
        ranks = list(range(1, len(population) + 1))
        
        # Apply selection pressure
        adjusted_ranks = [rank ** config.selection_pressure for rank in ranks]
        
        # Calculate selection probabilities
        total_rank = sum(adjusted_ranks)
        probabilities = [rank / total_rank for rank in adjusted_ranks]
        
        # Select parents
        parents = []
        for _ in range(num_parents):
            selected_idx = np.random.choice(len(population), p=probabilities)
            parents.append(sorted_pop[selected_idx])
        
        return parents


class StochasticUniversalSampling(SelectionStrategy):
    """Stochastic Universal Sampling for fair selection."""
    
    def select(self, population: List[EvolutionaryCrawlingAgent], 
              num_parents: int, config: EvolutionConfig) -> List[EvolutionaryCrawlingAgent]:
        """Select parents using stochastic universal sampling."""
        # For now, use fitness proportionate as fallback
        fallback = FitnessProportionateSelection()
        return fallback.select(population, num_parents, config)


class SpeciationManager:
    """Manages species formation and evolution within the population."""
    
    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.species: List[List[EvolutionaryCrawlingAgent]] = []
        self.species_representatives: List[EvolutionaryCrawlingAgent] = []
        self.species_fitness_history: List[List[float]] = []
    
    def organize_species(self, population: List[EvolutionaryCrawlingAgent]) -> List[List[EvolutionaryCrawlingAgent]]:
        """Organize population into species based on similarity."""
        self.species = []
        self.species_representatives = []
        
        for agent in population:
            placed = False
            
            # Try to place in existing species
            for i, representative in enumerate(self.species_representatives):
                if self._are_compatible(agent, representative):
                    self.species[i].append(agent)
                    placed = True
                    break
            
            # Create new species if not placed
            if not placed:
                self.species.append([agent])
                self.species_representatives.append(agent)
        
        # Update fitness history
        self._update_species_fitness()
        
        return self.species
    
    def _are_compatible(self, agent1: EvolutionaryCrawlingAgent, 
                       agent2: EvolutionaryCrawlingAgent) -> bool:
        """Check if two agents are compatible (same species)."""
        metrics1 = agent1.get_diversity_metrics()
        metrics2 = agent2.get_diversity_metrics()
        
        # Calculate compatibility distance
        distance = 0.0
        for key in metrics1:
            if key in metrics2:
                distance += abs(metrics1[key] - metrics2[key])
        
        return distance < self.config.niche_size
    
    def _update_species_fitness(self):
        """Update fitness history for each species."""
        current_fitness = []
        
        for species in self.species:
            if species:
                avg_fitness = np.mean([agent.get_evolutionary_fitness() for agent in species])
                current_fitness.append(avg_fitness)
            else:
                current_fitness.append(0.0)
        
        self.species_fitness_history.append(current_fitness)
        
        # Keep only recent history
        if len(self.species_fitness_history) > 20:
            self.species_fitness_history.pop(0)
    
    def allocate_offspring(self, total_offspring: int) -> List[int]:
        """Allocate number of offspring per species based on fitness."""
        if not self.species:
            return []
        
        # Calculate adjusted fitness for each species
        adjusted_fitness = []
        for i, species in enumerate(self.species):
            if species:
                # Calculate species fitness with sharing
                species_fitness = np.mean([agent.get_evolutionary_fitness() for agent in species])
                shared_fitness = species_fitness / len(species)  # Fitness sharing
                adjusted_fitness.append(max(0.1, shared_fitness))  # Minimum allocation
            else:
                adjusted_fitness.append(0.1)
        
        # Allocate offspring proportionally
        total_adjusted = sum(adjusted_fitness)
        allocations = []
        
        for fitness in adjusted_fitness:
            allocation = int((fitness / total_adjusted) * total_offspring)
            allocations.append(max(1, allocation))  # At least 1 offspring per species
        
        # Adjust for exact total
        current_total = sum(allocations)
        diff = total_offspring - current_total
        
        if diff > 0:
            # Add to best species
            best_species_idx = np.argmax(adjusted_fitness)
            allocations[best_species_idx] += diff
        elif diff < 0:
            # Remove from worst species
            worst_species_idx = np.argmin(adjusted_fitness)
            allocations[worst_species_idx] = max(1, allocations[worst_species_idx] + diff)
        
        return allocations


class HallOfFame:
    """Maintains a hall of fame of best individuals ever seen."""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.members: List[EvolutionaryCrawlingAgent] = []
        self.fitness_history: List[float] = []
    
    def update(self, population: List[EvolutionaryCrawlingAgent]):
        """Update hall of fame with current population."""
        for agent in population:
            self._consider_agent(agent)
    
    def _consider_agent(self, agent: EvolutionaryCrawlingAgent):
        """Consider adding an agent to the hall of fame."""
        fitness = agent.get_evolutionary_fitness()
        
        if len(self.members) < self.max_size:
            # Add if there's space
            self.members.append(agent)
            self.fitness_history.append(fitness)
        else:
            # Replace worst member if this agent is better
            worst_idx = np.argmin(self.fitness_history)
            if fitness > self.fitness_history[worst_idx]:
                self.members[worst_idx] = agent
                self.fitness_history[worst_idx] = fitness
    
    def get_best(self) -> Optional[EvolutionaryCrawlingAgent]:
        """Get the best agent from hall of fame."""
        if not self.members:
            return None
        
        best_idx = np.argmax(self.fitness_history)
        return self.members[best_idx]
    
    def get_diverse_representatives(self, count: int) -> List[EvolutionaryCrawlingAgent]:
        """Get diverse representatives from hall of fame."""
        if len(self.members) <= count:
            return self.members.copy()
        
        # Select diverse subset
        selected = [self.get_best()]  # Always include best
        remaining = [agent for agent in self.members if agent != selected[0]]
        
        while len(selected) < count and remaining:
            # Find most diverse agent from selected ones
            best_diversity = -1
            best_agent = None
            
            for candidate in remaining:
                diversity = self._calculate_diversity_from_set(candidate, selected)
                if diversity > best_diversity:
                    best_diversity = diversity
                    best_agent = candidate
            
            if best_agent:
                selected.append(best_agent)
                remaining.remove(best_agent)
        
        return selected
    
    def _calculate_diversity_from_set(self, agent: EvolutionaryCrawlingAgent,
                                    agent_set: List[EvolutionaryCrawlingAgent]) -> float:
        """Calculate average diversity of agent from a set of agents."""
        if not agent_set:
            return 0.0
        
        agent_metrics = agent.get_diversity_metrics()
        distances = []
        
        for other in agent_set:
            other_metrics = other.get_diversity_metrics()
            distance = sum(abs(agent_metrics[key] - other_metrics.get(key, 0)) 
                         for key in agent_metrics)
            distances.append(distance)
        
        return np.mean(distances)


class EnhancedEvolutionEngine:
    """
    Comprehensive evolution engine with advanced features.
    
    Features:
    - Multiple selection strategies
    - Speciation and niching
    - Adaptive parameters
    - Diversity maintenance
    - Hall of fame
    - Immigration
    """
    
    def __init__(self, 
                 world: Any,
                 config: Optional[EvolutionConfig] = None,
                 mlflow_integration=None,
                 logger=None):
        """Initialize the enhanced evolution engine."""
        self.world = world
        self.config = config or EvolutionConfig()
        self.logger = logger or logging.getLogger(__name__)
        self.mlflow_integration = mlflow_integration
        
        # Core evolutionary state
        self.generation = 0
        self.population: List[EvolutionaryCrawlingAgent] = []
        self.species: List[Any] = []  # Species objects
        self.hall_of_fame: Any = None  # HallOfFame object
        
        # Fitness tracking
        self.fitness_history: List[float] = []
        self.diversity_history: List[float] = []
        self.best_fitness_history: List[float] = []
        self.best_fitness: float = float('-inf')
        self.stagnation_count: int = 0
        self.stagnation_counter: int = 0
        self.last_best_fitness: float = float('-inf')
        
        # Selection strategies
        self.selection_strategies = {
            'tournament': TournamentSelection(),
            'fitness_proportionate': FitnessProportionateSelection(),
            'rank': RankSelection(),
            'sus': StochasticUniversalSampling()
        }
        self.current_selection_strategy = 'tournament'
        
        # Population and diversity management
        self.target_species_count = max(2, self.config.population_size // 10)
        self.species_id_counter = 0
        self.parameter_space = PhysicalParameterSpace()
        self.speciation_manager = SpeciationManager(self.config) if self.config.use_speciation else None
        self.hall_of_fame = HallOfFame(self.config.hall_of_fame_size) if self.config.use_hall_of_fame else None
        self.selection_strategy = self.selection_strategies[self.current_selection_strategy]
        
        # 🌟 NEW EVOLUTIONARY STATE 🌟
        
        # Environmental challenges
        self.environment_obstacles: List[Dict] = []
        self.current_terrain_difficulty = 1.0
        self.weather_condition = "normal"  # normal, windy, icy, hot
        
        # Multi-objective evolution
        self.pareto_front: List[EvolutionaryCrawlingAgent] = []
        self.objective_history: Dict[str, List[float]] = {}
        
        # Seasonal evolution
        self.current_season = 0  # 0=spring, 1=summer, 2=autumn, 3=winter
        self.season_counter = 0
        
        # Mutation events
        self.mutation_storm_active = False
        self.storm_duration = 0
        
        # Social learning
        self.knowledge_network: Dict[str, Dict] = {}  # Agent ID -> learned behaviors
        self.teaching_relationships: List[Tuple[str, str]] = []  # (teacher, student)
        
        # Resource competition
        self.available_resources = self.config.total_resources
        self.resource_allocation: Dict[str, float] = {}
        
        # Predator-prey dynamics
        self.predators: Set[str] = set()  # Agent IDs that are predators
        self.prey_relationships: List[Tuple[str, str]] = []  # (predator, prey)
        
        # Sexual selection
        self.mating_preferences: Dict[str, Dict[str, float]] = {}  # Agent preferences
        self.beauty_scores: Dict[str, Dict[str, float]] = {}  # Beauty trait scores
        
        # Event logging
        self.event_log: List[Dict] = []
        
        # Initialize objective tracking
        if self.config.enable_multi_objective:
            for obj in self.config.fitness_objectives:
                self.objective_history[obj] = []
    
    def _log_evolution_event(self, event_type: str, details: Dict):
        """Log significant evolutionary events."""
        event = {
            'generation': self.generation,
            'timestamp': time.time(),
            'event_type': event_type,
            'details': details
        }
        self.event_log.append(event)
        
        if self.logger:
            self.logger.info(f"🌟 Evolution Event: {event_type} - {details}")
    
    def _update_environmental_challenges(self):
        """Update environmental obstacles and terrain."""
        if not self.config.enable_environmental_challenges:
            return
        
        # DISABLED: Dynamic obstacle spawning replaced by static world generation
        # Obstacles are now generated at launch and persist throughout the simulation
        
        # Change terrain difficulty
        if random.random() < self.config.terrain_change_rate:
            old_difficulty = self.current_terrain_difficulty
            self.current_terrain_difficulty = max(0.5, min(3.0, 
                self.current_terrain_difficulty + random.uniform(-0.3, 0.3)))
            self._log_evolution_event('terrain_change', {
                'old_difficulty': old_difficulty,
                'new_difficulty': self.current_terrain_difficulty
            })
        
        # Weather changes
        if self.generation % self.config.weather_change_interval == 0:
            old_weather = self.weather_condition
            self.weather_condition = random.choice(['normal', 'windy', 'icy', 'hot'])
            if old_weather != self.weather_condition:
                self._log_evolution_event('weather_change', {
                    'old_weather': old_weather,
                    'new_weather': self.weather_condition
                })
    
    def _update_seasonal_evolution(self):
        """Update seasonal pressures and conditions."""
        if not self.config.enable_seasonal_evolution:
            return
        
        self.season_counter += 1
        
        if self.season_counter >= self.config.season_length:
            old_season = self.current_season
            self.current_season = (self.current_season + 1) % 4
            self.season_counter = 0
            
            season_names = ['Spring', 'Summer', 'Autumn', 'Winter']
            self._log_evolution_event('season_change', {
                'old_season': season_names[old_season],
                'new_season': season_names[self.current_season]
            })
    
    def _check_mutation_storm(self):
        """Check for and handle mutation storms."""
        if not self.config.enable_mutation_storms:
            return
        
        if self.mutation_storm_active:
            self.storm_duration -= 1
            if self.storm_duration <= 0:
                self.mutation_storm_active = False
                self._log_evolution_event('mutation_storm_ended', {})
        else:
            if random.random() < self.config.mutation_storm_probability:
                self.mutation_storm_active = True
                self.storm_duration = random.randint(3, 8)  # 3-8 generations
                self._log_evolution_event('mutation_storm_started', {
                    'duration': self.storm_duration,
                    'intensity': self.config.storm_mutation_multiplier
                })
    
    def _apply_social_learning(self):
        """Apply social learning between agents."""
        if not self.config.enable_social_learning or len(self.population) < 2:
            return
        
        # Clear old teaching relationships
        self.teaching_relationships.clear()
        
        # Select teachers and students
        sorted_agents = sorted(self.population, key=lambda a: a.get_evolutionary_fitness(), reverse=True)
        top_performers = sorted_agents[:len(sorted_agents)//3]  # Top third as potential teachers
        
        for student in self.population:
            if random.random() < self.config.social_learning_rate:
                # Select teacher with bias toward high-fitness agents
                weights = [agent.get_evolutionary_fitness() + 1.0 for agent in top_performers]  # +1 to avoid zero weights
                teacher = random.choices(top_performers, weights=weights)[0]
                
                if teacher.id != student.id:
                    self._transfer_knowledge(teacher, student)
                    self.teaching_relationships.append((teacher.id, student.id))
        
        if self.teaching_relationships:
            self._log_evolution_event('social_learning', {
                'relationships': len(self.teaching_relationships)
            })
    
    def _transfer_knowledge(self, teacher: EvolutionaryCrawlingAgent, student: EvolutionaryCrawlingAgent):
        """Transfer knowledge from teacher to student."""
        # Transfer some Q-table values (if using Q-learning)
        if hasattr(teacher, 'q_table') and hasattr(student, 'q_table'):
            # Sample some states from teacher's experience
            if hasattr(teacher.q_table, 'q_values') and teacher.q_table.q_values:
                try:
                    sample_states = random.sample(
                        list(teacher.q_table.q_values.keys()),
                        min(10, len(teacher.q_table.q_values))
                    )
                    for state in sample_states:
                        if state in teacher.q_table.q_values:
                            teacher_state_values = teacher.q_table.q_values[state]
                            
                            # Handle different Q-table structures
                            if isinstance(teacher_state_values, dict):
                                # Dictionary structure: {action: value}
                                if state in student.q_table.q_values:
                                    student_state_values = student.q_table.q_values[state]
                                    if isinstance(student_state_values, dict):
                                        for action, value in teacher_state_values.items():
                                            if action in student_state_values:
                                                blend_factor = 0.3  # How much to learn from teacher
                                                student_state_values[action] = (
                                                    (1 - blend_factor) * student_state_values[action] +
                                                    blend_factor * value
                                                )
                                            else:
                                                student_state_values[action] = value * 0.5
                                else:
                                    student.q_table.q_values[state] = {
                                        action: value * 0.5 for action, value in teacher_state_values.items()
                                    }
                            elif isinstance(teacher_state_values, (list, tuple)):
                                # Array/list structure: [value1, value2, ...]
                                if state in student.q_table.q_values:
                                    student_state_values = student.q_table.q_values[state]
                                    if isinstance(student_state_values, list):
                                        # Blend array values (only for lists, not tuples)
                                        blend_factor = 0.3
                                        for i in range(min(len(teacher_state_values), len(student_state_values))):
                                            try:
                                                student_state_values[i] = (
                                                    (1 - blend_factor) * student_state_values[i] +
                                                    blend_factor * teacher_state_values[i]
                                                )
                                            except (TypeError, IndexError):
                                                pass  # Skip if values aren't numeric or index issues
                                else:
                                    # Copy teacher's values with reduced strength
                                    student.q_table.q_values[state] = [
                                        v * 0.5 if isinstance(v, (int, float)) else v 
                                        for v in teacher_state_values
                                    ]
                except (AttributeError, KeyError, TypeError, ValueError):
                    # If Q-table structure is incompatible, skip knowledge transfer
                    pass
        
        # Transfer some physical parameters
        learning_factor = 0.2  # How much to adapt teacher's traits
        if hasattr(teacher, 'physical_params') and hasattr(student, 'physical_params'):
            for param_name in ['motor_speed', 'joint_strength', 'sensor_range']:
                if hasattr(teacher.physical_params, param_name) and hasattr(student.physical_params, param_name):
                    teacher_value = getattr(teacher.physical_params, param_name)
                    student_value = getattr(student.physical_params, param_name)
                    new_value = student_value + learning_factor * (teacher_value - student_value)
                    setattr(student.physical_params, param_name, new_value)
    
    def _manage_resource_competition(self):
        """Manage resource allocation and competition."""
        if not self.config.enable_resource_competition:
            return
        
        # Decay available resources
        self.available_resources *= (1 - self.config.resource_decay_rate)
        self.available_resources = max(100.0, self.available_resources)  # Minimum resources
        
        # Allocate resources based on fitness
        total_fitness = sum(max(0.1, agent.get_evolutionary_fitness()) for agent in self.population)
        
        self.resource_allocation.clear()
        for agent in self.population:
            agent_fitness = agent.get_evolutionary_fitness()
            fitness_share = max(0.1, agent_fitness) / total_fitness
            allocated = self.available_resources * fitness_share
            self.resource_allocation[agent.id] = allocated
        
        # Apply resource effects to agents
        for agent in self.population:
            resources = self.resource_allocation.get(agent.id, 0)
            
            # Resource scarcity affects mutation rate and energy
            resource_ratio = resources / (self.available_resources / len(self.population))
            
            if resource_ratio < 0.5:  # Resource stressed
                # Apply effects through available attributes
                if hasattr(agent, 'mutation_rate'):
                    agent.mutation_rate *= 1.5  # Higher mutation when stressed
                # Note effects in agent state if possible
                if hasattr(agent, 'resource_stress'):
                    agent.resource_stress = 0.1
            elif resource_ratio > 1.5:  # Resource abundant
                if hasattr(agent, 'mutation_rate'):
                    agent.mutation_rate *= 0.8  # Lower mutation when well-fed  
                if hasattr(agent, 'resource_bonus'):
                    agent.resource_bonus = 0.1
    
    def _manage_predator_prey_dynamics(self):
        """Manage predator-prey relationships."""
        if not self.config.enable_predator_prey:
            return
        
        # Designate predators (top performers become predators)
        sorted_agents = sorted(self.population, key=lambda a: a.get_evolutionary_fitness(), reverse=True)
        predator_count = max(1, int(len(self.population) * self.config.predator_ratio))
        
        new_predators = set(agent.id for agent in sorted_agents[:predator_count])
        
        if new_predators != self.predators:
            self.predators = new_predators
            self._log_evolution_event('predator_designation', {
                'predator_count': len(self.predators),
                'predator_ids': list(self.predators)
            })
        
        # Simulate predation events
        self.prey_relationships.clear()
        for predator_id in self.predators:
            predator_agent = next((a for a in self.population if a.id == predator_id), None)
            if predator_agent:
                # Predators try to catch nearby prey
                prey_candidates = [a for a in self.population if a.id not in self.predators]
                if prey_candidates:
                    # Success based on relative fitness
                    for prey in random.sample(prey_candidates, min(3, len(prey_candidates))):
                        predator_fitness = predator_agent.get_evolutionary_fitness()
                        prey_fitness = prey.get_evolutionary_fitness()
                        catch_probability = 0.1 * (predator_fitness / (prey_fitness + 1.0))
                        if random.random() < catch_probability:
                            self.prey_relationships.append((predator_id, prey.id))
                            # Note: We can't directly modify fitness since it's calculated
                            # Instead, we could modify agent attributes that affect fitness
                            print(f"🦁 Predation: {predator_id[:8]} caught {prey.id[:8]}")
                            # Apply effects through attributes if available
                            if hasattr(predator_agent, 'predation_bonus'):
                                predator_agent.predation_bonus = getattr(predator_agent, 'predation_bonus', 0) + 0.1
                            if hasattr(prey, 'predation_penalty'):
                                prey.predation_penalty = getattr(prey, 'predation_penalty', 0) + 0.1
        
        if self.prey_relationships:
            self._log_evolution_event('predation_events', {
                'events': len(self.prey_relationships)
            })
    
    def _apply_sexual_selection(self):
        """Apply sexual selection and mating preferences."""
        if not self.config.enable_sexual_selection:
            return
        
        # Calculate beauty scores for all agents
        self.beauty_scores.clear()
        for agent in self.population:
            beauty_score = {}
            
            # Symmetry: based on physical parameter balance
            if hasattr(agent, 'physical_params'):
                symmetry = 1.0 - abs(
                    getattr(agent.physical_params, 'left_motor_bias', 0) - 
                    getattr(agent.physical_params, 'right_motor_bias', 0)
                )
                beauty_score['symmetry'] = max(0, symmetry)
            
            # Size: normalized body size
            if hasattr(agent, 'physical_params'):
                size_score = min(1.0, getattr(agent.physical_params, 'body_scale', 1.0))
                beauty_score['size'] = size_score
            
            # Movement grace: based on stability and smoothness
            agent_fitness = agent.get_evolutionary_fitness()
            grace_score = min(1.0, agent_fitness / (max(fit for fit in self.fitness_history[-10:]) if self.fitness_history else 1.0))
            beauty_score['movement_grace'] = grace_score
            
            # Color pattern: randomized aesthetic trait
            beauty_score['color_pattern'] = random.random()
            
            self.beauty_scores[agent.id] = beauty_score
        
        # Update mating preferences
        self.mating_preferences.clear()
        for agent in self.population:
            preferences = {}
            for trait in self.config.beauty_traits:
                # Agents develop preferences based on their own traits and random variation
                own_score = self.beauty_scores.get(agent.id, {}).get(trait, 0.5)
                preference = own_score + random.uniform(-0.3, 0.3)
                preferences[trait] = max(0, min(1, preference))
            self.mating_preferences[agent.id] = preferences
    
    def _check_catastrophic_events(self):
        """Check for and handle catastrophic events."""
        if not self.config.enable_catastrophes:
            return
        
        if random.random() < self.config.catastrophe_probability:
            # Catastrophe occurs!
            survivors_count = int(len(self.population) * self.config.survival_rate_during_catastrophe)
            
            # Biased survival (fittest more likely to survive, but randomness)
            survival_weights = [agent.get_evolutionary_fitness() + random.uniform(0, 2) for agent in self.population]
            survivors = random.choices(self.population, weights=survival_weights, k=survivors_count)
            
            eliminated = len(self.population) - len(survivors)
            
            self.population = survivors
            
            # Fill population back up with new random agents
            while len(self.population) < self.config.population_size:
                new_agent = self._create_random_agent()
                self.population.append(new_agent)
            
            self._log_evolution_event('catastrophic_event', {
                'eliminated': eliminated,
                'survivors': len(survivors),
                'repopulated': self.config.population_size - len(survivors)
            })
    
    def _calculate_multi_objective_fitness(self, agent: EvolutionaryCrawlingAgent) -> Dict[str, float]:
        """Calculate fitness for multiple objectives."""
        objectives = {}
        
        # Get agent's primary fitness value
        agent_fitness = agent.get_evolutionary_fitness()
        
        # Speed objective
        objectives['speed'] = agent_fitness  # Primary fitness is speed-based
        
        # Efficiency objective (distance per energy unit)
        objectives['efficiency'] = agent_fitness / max(1.0, getattr(agent, 'energy_consumed', 1.0))
        
        # Stability objective (consistency of movement)
        objectives['stability'] = 1.0 - getattr(agent, 'movement_variance', 0.5)
        
        # Exploration objective (area covered)
        objectives['exploration'] = getattr(agent, 'exploration_score', 0.0)
        
        # Cooperation objective (social learning participation)
        cooperation_score = 0.0
        if agent.id in [rel[1] for rel in self.teaching_relationships]:  # Is a student
            cooperation_score += 0.5
        if agent.id in [rel[0] for rel in self.teaching_relationships]:  # Is a teacher
            cooperation_score += 0.5
        objectives['cooperation'] = cooperation_score
        
        return objectives
        
    def _update_pareto_front(self):
        """Update the Pareto front for multi-objective optimization."""
        if not self.config.enable_multi_objective:
            return
        
        # Calculate objectives for all agents
        agent_objectives = []
        for agent in self.population:
            objectives = self._calculate_multi_objective_fitness(agent)
            agent_objectives.append((agent, objectives))
        
        # Find Pareto front
        pareto_front = []
        for i, (agent1, obj1) in enumerate(agent_objectives):
            is_dominated = False
            for j, (agent2, obj2) in enumerate(agent_objectives):
                if i != j and self._dominates(obj2, obj1):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_front.append(agent1)
        
        self.pareto_front = pareto_front
        
        # Update objective history
        for obj_name in self.config.fitness_objectives:
            obj_values = [objectives[obj_name] for _, objectives in agent_objectives]
            self.objective_history[obj_name].append(np.mean(obj_values))
    
    def _dominates(self, obj1: Dict[str, float], obj2: Dict[str, float]) -> bool:
        """Check if obj1 dominates obj2 (Pareto dominance)."""
        better_in_all = all(obj1.get(key, 0) >= obj2.get(key, 0) for key in obj1.keys())
        better_in_at_least_one = any(obj1.get(key, 0) > obj2.get(key, 0) for key in obj1.keys())
        return better_in_all and better_in_at_least_one
    
    def get_current_mutation_rate(self) -> float:
        """Get the current mutation rate, accounting for storms and other factors."""
        base_rate = self.config.mutation_rate
        
        # Mutation storm effect
        if self.mutation_storm_active:
            base_rate *= self.config.storm_mutation_multiplier
        
        # Seasonal effects
        if self.config.enable_seasonal_evolution:
            seasonal_multipliers = [1.2, 1.0, 1.3, 0.8]  # Spring, Summer, Autumn, Winter
            base_rate *= seasonal_multipliers[self.current_season]
        
        # Environmental pressure
        if self.current_terrain_difficulty > 1.5:
            base_rate *= 1.2  # Higher mutation in difficult terrain
        
        return min(0.8, base_rate)  # Cap at 80%
    
    def get_evolution_status(self) -> Dict:
        """Get comprehensive evolution status including new features."""
        status = {
            'generation': self.generation,
            'population_size': len(self.population),
            'best_fitness': self.best_fitness,
            'average_fitness': np.mean([agent.get_evolutionary_fitness() for agent in self.population]) if self.population else 0,
            'diversity': self.diversity_history[-1] if self.diversity_history else 0,
            'species_count': len(self.species),
            'stagnation_count': self.stagnation_count,
            
            # New feature status
            'environmental_challenges': {
                'active_obstacles': len([obs for obs in self.environment_obstacles if obs.get('active', False)]),
                'terrain_difficulty': self.current_terrain_difficulty,
                'weather_condition': self.weather_condition
            },
            
            'seasonal_evolution': {
                'current_season': ['Spring', 'Summer', 'Autumn', 'Winter'][self.current_season],
                'season_progress': self.season_counter / self.config.season_length
            },
            
            'mutation_events': {
                'storm_active': self.mutation_storm_active,
                'storm_duration_remaining': self.storm_duration,
                'current_mutation_rate': self.get_current_mutation_rate()
            },
            
            'social_dynamics': {
                'teaching_relationships': len(self.teaching_relationships),
                'knowledge_transfers': len(self.knowledge_network)
            },
            
            'resource_competition': {
                'available_resources': self.available_resources,
                'resource_pressure': 1.0 - (self.available_resources / self.config.total_resources)
            },
            
            'predator_prey': {
                'predator_count': len(self.predators),
                'recent_predation_events': len(self.prey_relationships)
            },
            
            'multi_objective': {
                'pareto_front_size': len(self.pareto_front),
                'objective_averages': {
                    obj: (self.objective_history[obj][-1] if self.objective_history[obj] else 0)
                    for obj in self.config.fitness_objectives
                }
            },
            
            'recent_events': self.event_log[-5:] if self.event_log else []
        }
        
        return status
    
    def initialize_population(self) -> List[EvolutionaryCrawlingAgent]:
        """Create initial population with diverse physical parameters."""
        population = []
        
        print(f"🐣 Creating initial population of {self.config.population_size} agents...")
        
        for i in range(self.config.population_size):
            # Create random physical parameters for diversity
            if i < 5:  # First few agents use default parameters
                physical_params = PhysicalParameters()
            else:
                # Create diverse variants
                physical_params = PhysicalParameters.random_parameters()
            
            # Calculate spacing to avoid overlaps
            spacing = 8 if self.config.population_size > 20 else 15
            position = (i * spacing, 6)
            
            agent = EvolutionaryCrawlingAgent(
                world=self.world,
                agent_id=i,
                position=position,
                category_bits=0x0002,  # Agent category
                mask_bits=0x0001,     # Collide with ground only
                physical_params=physical_params
            )
            
            population.append(agent)
        
        self.population = population
        self._update_evolution_stats()
        
        print(f"✅ Initial population created with diversity score: {self.diversity_history[-1]:.3f}")
        return population
    
    def evolve_generation(self) -> Tuple[List[EvolutionaryCrawlingAgent], List[EvolutionaryCrawlingAgent]]:
        """Evolve one generation of the population with advanced dynamics.
        
        Returns:
            Tuple of (new_population, agents_to_destroy)
        """
        self.generation += 1
        print(f"\n🧬 === GENERATION {self.generation} === 🌟")
        
        # 🌟 PRE-EVOLUTION PHASE: Environmental & Temporal Updates
        self._update_environmental_challenges()
        self._update_seasonal_evolution()
        self._check_mutation_storm()
        self._check_catastrophic_events()
        
        # Update fitness and statistics
        self._update_evolution_stats()
        
        # 🌟 MULTI-OBJECTIVE EVOLUTION: Update Pareto front
        if self.config.enable_multi_objective:
            self._update_pareto_front()
        
        # Update hall of fame
        if self.hall_of_fame:
            self.hall_of_fame.update(self.population)
        
        # 🌟 SOCIAL DYNAMICS PHASE: Learning and cooperation
        self._apply_social_learning()
        self._manage_resource_competition()
        self._manage_predator_prey_dynamics()
        self._apply_sexual_selection()
        
        # Check for stagnation and adapt parameters
        self._check_stagnation_and_adapt()
        
        # Organize into species if using speciation
        species = None
        if self.speciation_manager:
            species = self.speciation_manager.organize_species(self.population)
            print(f"🔬 Organized into {len(species)} species")
        
        # 🌟 REPRODUCTION PHASE: Create next generation with dynamic mutation rates
        original_mutation_rate = self.config.mutation_rate
        self.config.mutation_rate = self.get_current_mutation_rate()
        
        new_population = self._create_next_generation(species)
        
        # Restore original mutation rate
        self.config.mutation_rate = original_mutation_rate
        
        # Maintain diversity if needed
        if self.generation % self.config.diversity_check_interval == 0:
            new_population, diversity_casualties = self._maintain_diversity(new_population)
        else:
            diversity_casualties = []
        
        # Immigration: introduce random individuals
        immigration_casualties = []
        if self.config.immigration_rate > 0:
            new_population, immigration_casualties = self._apply_immigration(new_population)
        
        # Identify agents to destroy (those not in new population)
        old_population = self.population
        new_agent_ids = {agent.id for agent in new_population}
        agents_to_destroy = [agent for agent in old_population if agent.id not in new_agent_ids]
        agents_to_destroy.extend(diversity_casualties)
        agents_to_destroy.extend(immigration_casualties)
        
        # Update population (don't destroy agents here)
        self.population = new_population
        
        # 🌟 POST-EVOLUTION PHASE: Log enhanced results
        self._log_enhanced_generation_results()
        
        return self.population, agents_to_destroy
    
    def _create_next_generation(self, species: Optional[List[List[EvolutionaryCrawlingAgent]]]) -> List[EvolutionaryCrawlingAgent]:
        """Create the next generation through selection and reproduction."""
        new_population = []
        
        if species and self.config.use_speciation:
            # Speciation-based reproduction
            offspring_allocations = self.speciation_manager.allocate_offspring(
                self.config.population_size - self.config.elite_size
            )
            
            for species_idx, (species_pop, offspring_count) in enumerate(zip(species, offspring_allocations)):
                if not species_pop:
                    continue
                
                print(f"  Species {species_idx}: {len(species_pop)} agents → {offspring_count} offspring")
                
                # Create offspring for this species
                for _ in range(offspring_count):
                    child = self._create_offspring(species_pop)
                    if child:
                        new_population.append(child)
        else:
            # Standard reproduction
            offspring_count = self.config.population_size - self.config.elite_size
            for _ in range(offspring_count):
                child = self._create_offspring(self.population)
                if child:
                    new_population.append(child)
        
        # Add elite individuals
        elite = self._select_elite()
        new_population.extend(elite)
        
        # Ensure population size
        while len(new_population) < self.config.population_size:
            # Fill with random agents if needed
            random_agent = self._create_random_agent(len(new_population))
            new_population.append(random_agent)
        
        # Trim if too many
        new_population = new_population[:self.config.population_size]
        
        # No need to reassign IDs - agents already have unique IDs from cloning/crossover
        # Reassigning breaks leaderboard button references after evolution
        
        return new_population
    
    def _create_offspring(self, population: List[EvolutionaryCrawlingAgent]) -> Optional[EvolutionaryCrawlingAgent]:
        """Create a single offspring through selection and reproduction."""
        if len(population) < 2:
            return None
        
        reproduction_type = random.random()
        
        if reproduction_type < self.config.crossover_rate:
            # Crossover
            parents = self.selection_strategy.select(population, 2, self.config)
            if len(parents) >= 2:
                child = parents[0].evolve_with(parents[1], self.config.mutation_rate)
                return child
        elif reproduction_type < self.config.crossover_rate + self.config.clone_rate:
            # Cloning with mutation
            parents = self.selection_strategy.select(population, 1, self.config)
            if parents:
                child = parents[0].clone_with_mutation(self.config.mutation_rate)
                return child
        
        # Fallback to mutation
        parents = self.selection_strategy.select(population, 1, self.config)
        if parents:
            child = parents[0].clone_with_mutation(self.config.mutation_rate * 1.5)
            return child
        
        return None
    
    def _select_elite(self) -> List[EvolutionaryCrawlingAgent]:
        """Select elite individuals to preserve."""
        sorted_pop = sorted(self.population, 
                          key=lambda agent: agent.get_evolutionary_fitness(), 
                          reverse=True)
        
        elite = []
        for i in range(min(self.config.elite_size, len(sorted_pop))):
            # Clone elite to preserve them
            elite_clone = sorted_pop[i].clone_with_mutation(0.0)  # No mutation for pure elite
            elite.append(elite_clone)
        
        return elite
    
    def _maintain_diversity(self, population: List[EvolutionaryCrawlingAgent]) -> Tuple[List[EvolutionaryCrawlingAgent], List[EvolutionaryCrawlingAgent]]:
        """Maintain population diversity.
        
        Returns:
            Tuple of (updated_population, agents_to_destroy)
        """
        # Extract physical parameters
        param_sets = [agent.physical_params for agent in population]
        
        # Use parameter space to maintain diversity
        enhanced_params = self.parameter_space.maintain_diversity(
            param_sets, self.config.target_diversity
        )
        
        # Track agents that need to be destroyed
        agents_to_destroy = []
        
        # Update agents with enhanced parameters
        for i, (agent, new_params) in enumerate(zip(population, enhanced_params)):
            if new_params != agent.physical_params:
                # Create new agent with diverse parameters
                diverse_agent = EvolutionaryCrawlingAgent(
                    world=self.world,
                    agent_id=None,  # Let agent generate its own UUID
                    position=agent.initial_position,
                    category_bits=agent.category_bits,
                    mask_bits=agent.mask_bits,
                    physical_params=new_params,
                    parent_lineage=agent.parent_lineage
                )
                # Copy learned behavior
                diverse_agent.q_table = agent.q_table.copy()
                population[i] = diverse_agent
                
                # Queue old agent for destruction
                agents_to_destroy.append(agent)
        
        return population, agents_to_destroy
    
    def _apply_immigration(self, population: List[EvolutionaryCrawlingAgent]) -> Tuple[List[EvolutionaryCrawlingAgent], List[EvolutionaryCrawlingAgent]]:
        """Introduce random individuals to maintain diversity.
        
        Returns:
            Tuple of (updated_population, agents_to_destroy)
        """
        immigrant_count = int(len(population) * self.config.immigration_rate)
        agents_to_destroy = []
        
        if immigrant_count > 0:
            print(f"🌍 Introducing {immigrant_count} immigrants")
            
            # Replace worst individuals with immigrants
            sorted_pop = sorted(population, key=lambda agent: agent.get_evolutionary_fitness())
            
            for i in range(immigrant_count):
                if i < len(sorted_pop):
                    # Queue old agent for destruction
                    agents_to_destroy.append(sorted_pop[i])
                    
                    # Create immigrant
                    immigrant = self._create_random_agent(i)
                    sorted_pop[i] = immigrant
        
        return population, agents_to_destroy
    
    def _create_random_agent(self, position_index: Optional[int] = None) -> EvolutionaryCrawlingAgent:
        """Create a random agent with diverse parameters."""
        random_params = PhysicalParameters.random_parameters()
        spacing = 8 if self.config.population_size > 20 else 15
        
        # Use position_index for spacing if provided, otherwise use random position
        if position_index is not None:
            position = (position_index * spacing, 6)
        else:
            position = (random.randint(0, self.config.population_size) * spacing, 6)
        
        return EvolutionaryCrawlingAgent(
            world=self.world,
            agent_id=None,  # Let agent generate its own UUID
            position=position,
            category_bits=0x0002,
            mask_bits=0x0001,
            physical_params=random_params
        )
    
    def _check_stagnation_and_adapt(self):
        """Check for fitness stagnation and adapt parameters."""
        if not self.best_fitness_history:
            return
        
        current_best = max(agent.get_evolutionary_fitness() for agent in self.population)
        
        if current_best <= self.last_best_fitness + 0.01:  # Minimal improvement threshold
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
            self.last_best_fitness = current_best
        
        # Adapt parameters if stagnating
        if self.stagnation_counter >= self.config.fitness_stagnation_threshold:
            print(f"⚠️  Fitness stagnation detected ({self.stagnation_counter} generations)")
            
            if self.config.adaptive_mutation:
                # Increase mutation rate
                self.config.mutation_rate = min(0.3, self.config.mutation_rate * 1.2)
                print(f"📈 Increased mutation rate to {self.config.mutation_rate:.3f}")
            
            if self.config.adaptive_selection:
                # Increase immigration rate
                self.config.immigration_rate = min(0.2, self.config.immigration_rate * 1.5)
                print(f"🌍 Increased immigration rate to {self.config.immigration_rate:.3f}")
            
            self.stagnation_counter = 0  # Reset counter
    
    def _update_evolution_stats(self):
        """Update evolution statistics."""
        if not self.population:
            return
        
        # Fitness statistics
        fitnesses = [agent.get_evolutionary_fitness() for agent in self.population]
        avg_fitness = np.mean(fitnesses)
        best_fitness = max(fitnesses)
        
        self.fitness_history.append(avg_fitness)
        self.best_fitness_history.append(best_fitness)
        
        # Diversity statistics
        param_sets = [agent.physical_params for agent in self.population]
        diversity = self.parameter_space.calculate_population_diversity(param_sets)
        self.diversity_history.append(diversity)
    
    def _log_generation_results(self):
        """Log results of the current generation."""
        if not self.population:
            return
        
        fitnesses = [agent.get_evolutionary_fitness() for agent in self.population]
        avg_fitness = np.mean(fitnesses)
        best_fitness = max(fitnesses)
        worst_fitness = min(fitnesses)
        
        print(f"📊 Generation {self.generation} Results:")
        print(f"   Fitness: avg={avg_fitness:.3f}, best={best_fitness:.3f}, worst={worst_fitness:.3f}")
        print(f"   Diversity: {self.diversity_history[-1]:.3f}")
        print(f"   Mutation rate: {self.config.mutation_rate:.3f}")
        print(f"   Immigration rate: {self.config.immigration_rate:.3f}")
        
        if self.hall_of_fame:
            hall_best = self.hall_of_fame.get_best()
            if hall_best:
                print(f"   Hall of Fame best: {hall_best.get_evolutionary_fitness():.3f}")
    
    def _log_enhanced_generation_results(self):
        """Log comprehensive results including new evolutionary features."""
        self._log_generation_results()  # Call original logging
        
        # Enhanced logging for new features
        print(f"🌟 Enhanced Evolution Status:")
        
        # Environmental challenges
        if self.config.enable_environmental_challenges:
            active_obstacles = len([obs for obs in self.environment_obstacles if obs.get('active', False)])
            print(f"   🌍 Environment: {active_obstacles} obstacles, terrain={self.current_terrain_difficulty:.2f}, weather={self.weather_condition}")
        
        # Seasonal evolution
        if self.config.enable_seasonal_evolution:
            season_names = ['Spring', 'Summer', 'Autumn', 'Winter']
            progress = self.season_counter / self.config.season_length * 100
            print(f"   🌺 Season: {season_names[self.current_season]} ({progress:.1f}% complete)")
        
        # Mutation storms
        if self.config.enable_mutation_storms:
            if self.mutation_storm_active:
                print(f"   ⚡ Mutation Storm: ACTIVE ({self.storm_duration} generations remaining)")
            else:
                print(f"   ⚡ Mutation Storm: inactive")
        
        # Social dynamics
        if self.config.enable_social_learning:
            teaching_count = len(self.teaching_relationships)
            print(f"   🤝 Social Learning: {teaching_count} teaching relationships")
        
        # Resource competition
        if self.config.enable_resource_competition:
            resource_ratio = self.available_resources / self.config.total_resources
            print(f"   💰 Resources: {resource_ratio:.1%} available ({self.available_resources:.0f}/{self.config.total_resources:.0f})")
        
        # Predator-prey dynamics
        if self.config.enable_predator_prey:
            predation_events = len(self.prey_relationships)
            print(f"   🦁 Predator-Prey: {len(self.predators)} predators, {predation_events} recent events")
        
        # Multi-objective evolution
        if self.config.enable_multi_objective:
            pareto_size = len(self.pareto_front)
            print(f"   🎯 Multi-Objective: {pareto_size} agents on Pareto front")
        
        # Recent events
        if self.event_log:
            recent_events = self.event_log[-3:]  # Last 3 events
            print(f"   📝 Recent Events: {', '.join(event['event_type'] for event in recent_events)}")
        
        print(f"   🔄 Current Mutation Rate: {self.get_current_mutation_rate():.4f}")
        print()
    
    def _cleanup_agents(self, agents: List[EvolutionaryCrawlingAgent]):
        """DEPRECATED: Clean up Box2D bodies from old agents.
        
        This method is deprecated. Agents should be queued for destruction
        by the training environment instead of being destroyed directly here.
        """
        print("⚠️  WARNING: _cleanup_agents is deprecated and should not be called")
        # Don't actually destroy anything here to avoid race conditions
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get comprehensive evolution summary."""
        return {
            'generation': self.generation,
            'population_size': len(self.population),
            'average_fitness': self.fitness_history[-1] if self.fitness_history else 0,
            'best_fitness': self.best_fitness_history[-1] if self.best_fitness_history else 0,
            'diversity': self.diversity_history[-1] if self.diversity_history else 0,
            'stagnation_counter': self.stagnation_counter,
            'mutation_rate': self.config.mutation_rate,
            'immigration_rate': self.config.immigration_rate,
            'hall_of_fame_size': len(self.hall_of_fame.members) if self.hall_of_fame else 0,
            'species_count': len(self.speciation_manager.species) if self.speciation_manager else 1,
        }
    
    def get_best_agent(self) -> Optional[EvolutionaryCrawlingAgent]:
        """Get the best agent from current population."""
        if not self.population:
            return None
        
        return max(self.population, key=lambda agent: agent.get_evolutionary_fitness())
    
    def get_diverse_representatives(self, count: int = 5) -> List[EvolutionaryCrawlingAgent]:
        """Get diverse representatives from the population."""
        if len(self.population) <= count:
            return self.population.copy()
        
        # Start with best agent
        best_agent = self.get_best_agent()
        selected = [best_agent] if best_agent else []
        remaining = [agent for agent in self.population if agent != best_agent]
        
        # Select diverse agents
        while len(selected) < count and remaining:
            best_diversity = -1
            best_agent = None
            
            for candidate in remaining:
                diversity = self._calculate_agent_diversity_from_set(candidate, selected)
                if diversity > best_diversity:
                    best_diversity = diversity
                    best_agent = candidate
            
            if best_agent:
                selected.append(best_agent)
                remaining.remove(best_agent)
        
        return selected
    
    def _calculate_agent_diversity_from_set(self, agent: EvolutionaryCrawlingAgent,
                                          agent_set: List[EvolutionaryCrawlingAgent]) -> float:
        """Calculate diversity of agent from a set of agents."""
        if not agent_set:
            return 0.0
        
        agent_metrics = agent.get_diversity_metrics()
        distances = []
        
        for other in agent_set:
            other_metrics = other.get_diversity_metrics()
            distance = sum(abs(agent_metrics[key] - other_metrics.get(key, 0)) 
                         for key in agent_metrics)
            distances.append(distance)
        
        return float(np.mean(distances)) 