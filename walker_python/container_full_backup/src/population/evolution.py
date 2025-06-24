"""
Evolution Engine for implementing genetic algorithms.

This module provides the core evolutionary algorithm functionality including
genetic operators (mutation, crossover), fitness evaluation, and generation management.
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
import numpy as np
import random
from abc import ABC, abstractmethod

from src.agents.base_agent import BaseAgent
from src.agents.basic_agent import BasicAgent
from src.agents.q_table import QTable
from src.population.population_controller import PopulationController, AgentRecord


class GeneticOperator(ABC):
    """Abstract base class for genetic operators."""
    
    @abstractmethod
    def apply(self, parent: BaseAgent) -> BaseAgent:
        """Apply the genetic operator to create a new agent."""
        pass


class MutationOperator(GeneticOperator):
    """Mutation operator for evolving agents."""
    
    def __init__(self, 
                 learning_rate_mutation: float = 0.1,
                 epsilon_mutation: float = 0.1,
                 q_value_mutation: float = 0.05,
                 mutation_strength: float = 0.1):
        """
        Initialize mutation operator.
        
        Args:
            learning_rate_mutation: Probability of mutating learning rate
            epsilon_mutation: Probability of mutating epsilon
            q_value_mutation: Probability of mutating individual Q-values
            mutation_strength: Strength of mutations (0-1)
        """
        self.learning_rate_mutation = learning_rate_mutation
        self.epsilon_mutation = epsilon_mutation
        self.q_value_mutation = q_value_mutation
        self.mutation_strength = mutation_strength
    
    def apply(self, parent: BaseAgent) -> BaseAgent:
        """
        Create a mutated copy of the parent agent.
        
        Args:
            parent: The parent agent to mutate
            
        Returns:
            A new agent with mutations applied
        """
        if isinstance(parent, BasicAgent):
            return self._mutate_basic_agent(parent)
        else:
            # For other agent types (like CrawlingCrate), use their own mutate method
            if hasattr(parent, 'mutate') and hasattr(parent, 'copy'):
                child = parent.copy()
                child.mutate(self.mutation_strength)
                return child
            else:
                raise ValueError(f"Agent type {type(parent).__name__} does not support mutation")
    
    def _mutate_basic_agent(self, parent: BasicAgent) -> BasicAgent:
        """Mutate a BasicAgent specifically."""
        # Create a copy of the parent
        child = BasicAgent(
            state_dimensions=parent.state_dimensions,
            action_count=parent.action_count
        )
        
        # Copy Q-table
        child.q_table = parent.q_table.copy()
        
        # Mutate learning parameters
        if random.random() < self.learning_rate_mutation:
            child.learning_rate = self._mutate_value(
                parent.learning_rate, 0.01, 0.5
            )
        
        if random.random() < self.epsilon_mutation:
            child.randomness = self._mutate_value(
                parent.randomness, 0.01, 0.5
            )
        
        # Mutate Q-values
        if random.random() < self.q_value_mutation:
            self._mutate_q_table(child.q_table)
        
        return child
    
    def _mutate_value(self, value: float, min_val: float, max_val: float) -> float:
        """Mutate a single value within bounds."""
        mutation = (random.random() - 0.5) * 2 * self.mutation_strength
        new_value = value + mutation
        return max(min_val, min(max_val, new_value))
    
    def _mutate_q_table(self, q_table: QTable):
        """Mutate Q-values in the Q-table."""
        for state in q_table.table:
            for action in q_table.table[state]:
                if random.random() < self.q_value_mutation:
                    current_value = q_table.table[state][action]
                    mutation = (random.random() - 0.5) * 2 * self.mutation_strength
                    q_table.table[state][action] = current_value + mutation


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
        if isinstance(parent1, BasicAgent) and isinstance(parent2, BasicAgent):
            return self._crossover_basic_agents(parent1, parent2)
        else:
            # For other agent types (like CrawlingCrate), use their own crossover method
            if (hasattr(parent1, 'crossover') and hasattr(parent2, 'crossover') and 
                hasattr(parent1, 'copy') and hasattr(parent2, 'copy')):
                if random.random() > self.crossover_rate:
                    # No crossover, return copy of parent1
                    return parent1.copy()
                else:
                    # Use the agent's own crossover method
                    return parent1.crossover(parent2)
            else:
                raise ValueError(f"Agent type {type(parent1).__name__} does not support crossover")
    
    def _copy_agent(self, agent: BasicAgent) -> BasicAgent:
        """Create a copy of an agent."""
        copy_agent = BasicAgent(
            state_dimensions=agent.state_dimensions,
            action_count=agent.action_count
        )
        copy_agent.q_table = agent.q_table.copy()
        copy_agent.learning_rate = agent.learning_rate
        copy_agent.future_discount = agent.future_discount
        copy_agent.randomness = agent.randomness
        return copy_agent
    
    def _crossover_q_tables(self, q_table1: QTable, q_table2: QTable) -> QTable:
        """Crossover two Q-tables."""
        # Get all unique states
        all_states = set(q_table1.table.keys()) | set(q_table2.table.keys())
        
        # Create new Q-table
        new_q_table = QTable()
        
        for state in all_states:
            new_q_table.table[state] = {}
            
            # Get actions from both parents
            actions1 = q_table1.table.get(state, {})
            actions2 = q_table2.table.get(state, {})
            all_actions = set(actions1.keys()) | set(actions2.keys())
            
            for action in all_actions:
                value1 = actions1.get(action, 0.0)
                value2 = actions2.get(action, 0.0)
                
                # Randomly choose from parent or average
                if random.random() < 0.5:
                    new_q_table.table[state][action] = value1
                else:
                    new_q_table.table[state][action] = value2
        
        return new_q_table
    
    def _crossover_basic_agents(self, parent1: BasicAgent, parent2: BasicAgent) -> BasicAgent:
        """Crossover two BasicAgent instances specifically."""
        if random.random() > self.crossover_rate:
            # No crossover, return copy of parent1
            return self._copy_agent(parent1)
        
        # Create child with average parameters
        child = BasicAgent(
            state_dimensions=parent1.state_dimensions,
            action_count=parent1.action_count
        )
        
        # Average learning parameters
        child.learning_rate = (parent1.learning_rate + parent2.learning_rate) / 2
        child.future_discount = (parent1.future_discount + parent2.future_discount) / 2
        child.randomness = (parent1.randomness + parent2.randomness) / 2
        
        # Crossover Q-table
        child.q_table = self._crossover_q_tables(parent1.q_table, parent2.q_table)
        
        return child


class EvolutionEngine:
    """
    Main evolution engine that coordinates genetic operations.
    
    This class manages the evolutionary process including selection,
    reproduction, and population replacement.
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
        
        # Preserve elite agents
        elite_agents = ranked_agents[:self.elite_size]
        for record in elite_agents:
            new_population.append(record.agent)
        
        # Generate remaining agents through selection and reproduction
        while len(new_population) < self.population_controller.population_size:
            # Select parents
            parent1 = self._tournament_selection(ranked_agents)
            parent2 = self._tournament_selection(ranked_agents)
            
            # Create child through crossover and mutation
            if random.random() < 0.7:  # 70% crossover, 30% mutation
                child = self.crossover_operator.apply(parent1, parent2)
                # Apply mutation to crossover child
                child = self.mutation_operator.apply(child)
            else:
                # Direct mutation
                child = self.mutation_operator.apply(parent1)
            
            new_population.append(child)
        
        # No need to reassign IDs - agents already have unique IDs from cloning/crossover
        # Reassigning breaks leaderboard button references after evolution

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