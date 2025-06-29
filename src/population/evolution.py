"""
Evolution Engine for implementing genetic algorithms.

This module provides the core evolutionary algorithm functionality including
genetic operators (mutation, crossover), and generation management for neural network-based agents.
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
import numpy as np
import random
from abc import ABC, abstractmethod

from src.agents.base_agent import BaseAgent
from src.agents.crawling_agent import CrawlingAgent
from src.population.population_controller import PopulationController, AgentRecord


class GeneticOperator(ABC):
    """Abstract base class for genetic operators."""
    
    @abstractmethod
    def apply(self, parent: BaseAgent) -> BaseAgent:
        """Apply the genetic operator to create a new agent."""
        pass


class MutationOperator(GeneticOperator):
    """Mutation operator for evolving neural network-based agents."""
    
    def __init__(self, 
                 physical_mutation_rate: float = 0.1,
                 learning_mutation_rate: float = 0.1,
                 mutation_strength: float = 0.1):
        """
        Initialize mutation operator for neural network agents.
        
        Args:
            physical_mutation_rate: Probability of mutating physical parameters
            learning_mutation_rate: Probability of mutating learning parameters  
            mutation_strength: Strength of mutations (0-1)
        """
        self.physical_mutation_rate = physical_mutation_rate
        self.learning_mutation_rate = learning_mutation_rate
        self.mutation_strength = mutation_strength
    
    def apply(self, parent: BaseAgent) -> BaseAgent:
        """
        Create a mutated copy of the parent agent.
        
        Args:
            parent: The parent agent to mutate
            
        Returns:
            A new agent with mutations applied
        """
        if isinstance(parent, CrawlingAgent):
            return self._mutate_crawling_agent(parent)
        else:
            # For other agent types, use their own mutate method if available
            if hasattr(parent, 'clone_with_mutation'):
                return parent.clone_with_mutation(self.mutation_strength)
            else:
                raise ValueError(f"Agent type {type(parent).__name__} does not support mutation")
    
    def _mutate_crawling_agent(self, parent: CrawlingAgent) -> CrawlingAgent:
        """Mutate a CrawlingAgent with neural network-based learning."""
        # Use the agent's built-in mutation capabilities
        return parent.clone_with_mutation(self.mutation_strength)


class CrossoverOperator(GeneticOperator):
    """Crossover operator for combining two parent agents."""
    
    def __init__(self, crossover_rate: float = 0.8):
        """
        Initialize crossover operator.
        
        Args:
            crossover_rate: Probability of performing crossover
        """
        self.crossover_rate = crossover_rate
    
    def apply(self, parent1: BaseAgent, parent2: BaseAgent) -> BaseAgent:
        """
        Create a child agent by crossing over two parents.
        
        Args:
            parent1: First parent agent
            parent2: Second parent agent
            
        Returns:
            A new agent created by crossover
        """
        if isinstance(parent1, CrawlingAgent) and isinstance(parent2, CrawlingAgent):
            return self._crossover_crawling_agents(parent1, parent2)
        else:
            # For other agent types, use their own crossover method if available
            if hasattr(parent1, 'crossover') and hasattr(parent2, 'crossover'):
                if random.random() > self.crossover_rate:
                    # No crossover, return mutated copy of parent1
                    return parent1.clone_with_mutation(0.01)  # Small mutation
                else:
                    # Use the agent's own crossover method
                    return parent1.crossover(parent2)
            else:
                # Fallback: return mutated copy of parent1
                return parent1.clone_with_mutation(0.01)
    
    def _crossover_crawling_agents(self, parent1: CrawlingAgent, parent2: CrawlingAgent) -> CrawlingAgent:
        """Crossover two CrawlingAgent instances."""
        if random.random() > self.crossover_rate:
            # No crossover, return mutated copy of parent1
            return parent1.clone_with_mutation(0.01)
        
        # Use the agent's built-in crossover method
        return parent1.crossover(parent2)


class EvolutionEngine:
    """
    Main evolution engine that coordinates genetic operations.
    
    This class manages the evolutionary process including selection,
    reproduction, and population replacement for neural network-based agents.
    """
    
    def __init__(self,
                 population_controller: PopulationController,
                 mutation_operator: Optional[MutationOperator] = None,
                 crossover_operator: Optional[CrossoverOperator] = None,
                 elite_size: int = 2,
                 tournament_size: int = 3):
        """
        Initialize the evolution engine.
        
        Args:
            population_controller: The population controller to manage
            mutation_operator: Mutation operator (auto-created if None)
            crossover_operator: Crossover operator (auto-created if None)
            elite_size: Number of best agents to preserve
            tournament_size: Size of tournament for selection
        """
        self.population_controller = population_controller
        self.mutation_operator = mutation_operator or MutationOperator()
        self.crossover_operator = crossover_operator or CrossoverOperator()
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        
        # Evolution statistics
        self.generation_stats: List[Dict[str, Any]] = []
    
    def evolve_generation(self) -> List[BaseAgent]:
        """
        Evolve the current population to create the next generation.
        
        Returns:
            List of new agents for the next generation
        """
        # Get current population ranked by fitness
        ranked_agents = self.population_controller.get_ranked_agents()
        
        if len(ranked_agents) < 2:
            raise ValueError("Need at least 2 agents for evolution")
        
        new_population = []
        
        # Preserve elite agents (create copies to avoid reference issues)
        elite_agents = ranked_agents[:self.elite_size]
        for record in elite_agents:
            if hasattr(record.agent, 'clone_with_mutation'):
                # Elite agents get minimal mutation to preserve their performance
                elite_copy = record.agent.clone_with_mutation(0.001)  # Very small mutation
                new_population.append(elite_copy)
            else:
                new_population.append(record.agent)
        
        # Generate remaining agents through selection and reproduction
        while len(new_population) < self.population_controller.population_size:
            # Select parents
            parent1 = self._tournament_selection(ranked_agents)
            parent2 = self._tournament_selection(ranked_agents)
            
            # Create child through crossover and mutation
            if random.random() < 0.7:  # 70% crossover, 30% mutation
                child = self.crossover_operator.apply(parent1, parent2)
                # Apply additional mutation to crossover child
                if hasattr(child, 'clone_with_mutation'):
                    child = child.clone_with_mutation(0.05)  # Small additional mutation
            else:
                # Direct mutation
                child = self.mutation_operator.apply(parent1)
            
            new_population.append(child)
        
        return new_population
    
    def _tournament_selection(self, ranked_agents: List[AgentRecord]) -> BaseAgent:
        """
        Select an agent using tournament selection.
        
        Args:
            ranked_agents: List of agents ranked by fitness
            
        Returns:
            Selected agent
        """
        # Select random tournament participants
        tournament = random.sample(ranked_agents, 
                                 min(self.tournament_size, len(ranked_agents)))
        
        # Return the best agent from the tournament
        best_agent = max(tournament, key=lambda x: x.fitness)
        return best_agent.agent
    
    def evaluate_fitness(self, agent: BaseAgent, 
                        fitness_function: Callable[[BaseAgent], float]) -> float:
        """
        Evaluate the fitness of an agent.
        
        Args:
            agent: The agent to evaluate
            fitness_function: Function that computes fitness
            
        Returns:
            Fitness value
        """
        return fitness_function(agent)
    
    def run_evolution(self, 
                     initial_population: List[BaseAgent],
                     fitness_function: Callable[[BaseAgent], float],
                     max_generations: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the complete evolutionary process.
        
        Args:
            initial_population: Starting population
            fitness_function: Function to evaluate agent fitness
            max_generations: Maximum generations (uses controller default if None)
            
        Returns:
            Evolution results and statistics
        """
        # Set up initial population
        self.population_controller.clear_population()
        for agent in initial_population:
            self.population_controller.add_agent(agent)
        
        max_gen = max_generations or self.population_controller.max_generations
        generation = 0
        
        while generation < max_gen and not self.population_controller.is_evolution_complete():
            # Evaluate all agents in current generation
            self._evaluate_generation(fitness_function)
            
            # Save generation statistics
            self.population_controller.save_generation_stats()
            
            # Check if evolution is complete
            if self.population_controller.is_evolution_complete():
                break
            
            # Evolve to next generation
            new_population = self.evolve_generation()
            
            # Replace population
            self.population_controller.clear_population()
            self.population_controller.advance_generation()
            
            for agent in new_population:
                self.population_controller.add_agent(agent)
            
            generation += 1
        
        return self.population_controller.get_evolution_progress()
    
    def _evaluate_generation(self, fitness_function: Callable[[BaseAgent], float]):
        """Evaluate all agents in the current generation."""
        for record in self.population_controller.agents:
            if record.status.value == "active":
                fitness = self.evaluate_fitness(record.agent, fitness_function)
                self.population_controller.update_agent_fitness(
                    record.agent, fitness
                )
    
    def get_best_agent(self) -> Optional[BaseAgent]:
        """Get the best agent from the current population."""
        best_record = self.population_controller.get_best_agent()
        return best_record.agent if best_record else None
    
    def get_population_diversity(self) -> float:
        """Get the current population diversity."""
        return self.population_controller.get_population_diversity()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the evolution process.
        
        Returns:
            Dictionary with evolution statistics
        """
        population_stats = self.population_controller.get_stats()
        evolution_progress = self.population_controller.get_evolution_progress()
        
        return {
            'population_stats': population_stats,
            'evolution_progress': evolution_progress,
            'elite_size': self.elite_size,
            'tournament_size': self.tournament_size,
            'mutation_rate': self.mutation_operator.physical_mutation_rate,
            'crossover_rate': self.crossover_operator.crossover_rate,
            'generations_completed': len(self.generation_stats),
            'best_agent_id': self.get_best_agent().id if self.get_best_agent() else None
        } 