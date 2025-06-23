"""
Enhanced Evolution Engine for Physical Parameter Evolution.

This module provides a comprehensive evolutionary algorithm implementation
inspired by the Java CrawlingCrate evolution system, with robust selection
strategies, diversity maintenance, and advanced genetic operators.
"""

import numpy as np
import random
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import defaultdict
import math
from copy import deepcopy

from src.agents.base_agent import BaseAgent
from src.agents.evolutionary_crawling_agent import EvolutionaryCrawlingAgent
from src.agents.physical_parameters import PhysicalParameters, PhysicalParameterSpace
from src.population.population_controller import PopulationController, AgentRecord


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
                 world,
                 config: Optional[EvolutionConfig] = None,
                 selection_strategy: Optional[SelectionStrategy] = None):
        """
        Initialize the enhanced evolution engine.
        
        Args:
            world: Box2D world for creating agents
            config: Evolution configuration
            selection_strategy: Selection strategy to use
        """
        self.world = world
        self.config = config or EvolutionConfig()
        self.selection_strategy = selection_strategy or TournamentSelection()
        
        # Evolution management
        self.generation = 0
        self.population: List[EvolutionaryCrawlingAgent] = []
        self.fitness_history: List[float] = []
        self.diversity_history: List[float] = []
        self.best_fitness_history: List[float] = []
        
        # Advanced features
        self.speciation_manager = SpeciationManager(self.config) if self.config.use_speciation else None
        self.hall_of_fame = HallOfFame(self.config.hall_of_fame_size) if self.config.use_hall_of_fame else None
        self.parameter_space = PhysicalParameterSpace()
        
        # Adaptive evolution tracking
        self.stagnation_counter = 0
        self.last_best_fitness = -float('inf')
        
        print(f"🧬 Enhanced Evolution Engine initialized")
        print(f"   Population size: {self.config.population_size}")
        print(f"   Elite size: {self.config.elite_size}")
        print(f"   Selection: {type(self.selection_strategy).__name__}")
        print(f"   Speciation: {self.config.use_speciation}")
        print(f"   Hall of Fame: {self.config.use_hall_of_fame}")
    
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
        """Evolve one generation of the population.
        
        Returns:
            Tuple of (new_population, agents_to_destroy)
        """
        self.generation += 1
        print(f"\n🧬 === GENERATION {self.generation} ===")
        
        # Update fitness and statistics
        self._update_evolution_stats()
        
        # Update hall of fame
        if self.hall_of_fame:
            self.hall_of_fame.update(self.population)
        
        # Check for stagnation and adapt parameters
        self._check_stagnation_and_adapt()
        
        # Organize into species if using speciation
        species = None
        if self.speciation_manager:
            species = self.speciation_manager.organize_species(self.population)
            print(f"🔬 Organized into {len(species)} species")
        
        # Create next generation
        new_population = self._create_next_generation(species)
        
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
        
        # Log generation results
        self._log_generation_results()
        
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
                    agent_id=agent.id,
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
                    immigrant = self._create_random_agent(sorted_pop[i].id)
                    sorted_pop[i] = immigrant
        
        return population, agents_to_destroy
    
    def _create_random_agent(self, agent_id: int) -> EvolutionaryCrawlingAgent:
        """Create a random agent with diverse parameters."""
        random_params = PhysicalParameters.random_parameters()
        spacing = 8 if self.config.population_size > 20 else 15
        position = (agent_id * spacing, 6)
        
        return EvolutionaryCrawlingAgent(
            world=self.world,
            agent_id=agent_id,
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