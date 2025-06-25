"""
Selection strategies for evolutionary algorithms.

This module provides various selection strategies for choosing parents
in genetic algorithms, including tournament selection and roulette wheel selection.
"""

from typing import List, Callable, Optional
import random
import numpy as np
from abc import ABC, abstractmethod

from src.agents.base_agent import BaseAgent
from src.population.population_controller import AgentRecord


class SelectionStrategy(ABC):
    """Abstract base class for selection strategies."""
    
    @abstractmethod
    def select(self, agents: List[AgentRecord], count: int = 1) -> List[BaseAgent]:
        """
        Select agents from the population.
        
        Args:
            agents: List of agent records to select from
            count: Number of agents to select
            
        Returns:
            List of selected agents
        """
        pass


class TournamentSelection(SelectionStrategy):
    """Tournament selection strategy."""
    
    def __init__(self, tournament_size: int = 3):
        """
        Initialize tournament selection.
        
        Args:
            tournament_size: Size of each tournament
        """
        self.tournament_size = tournament_size
    
    def select(self, agents: List[AgentRecord], count: int = 1) -> List[BaseAgent]:
        """
        Select agents using tournament selection.
        
        Args:
            agents: List of agent records to select from
            count: Number of agents to select
            
        Returns:
            List of selected agents
        """
        if not agents:
            return []
        
        selected = []
        for _ in range(count):
            # Select random tournament participants
            tournament = random.sample(agents, min(self.tournament_size, len(agents)))
            
            # Return the best agent from the tournament
            best_agent = max(tournament, key=lambda x: x.fitness)
            selected.append(best_agent.agent)
        
        return selected


class RouletteWheelSelection(SelectionStrategy):
    """Roulette wheel (fitness-proportional) selection strategy."""
    
    def __init__(self, scaling_factor: float = 1.0):
        """
        Initialize roulette wheel selection.
        
        Args:
            scaling_factor: Factor to scale fitness values (helps with negative fitness)
        """
        self.scaling_factor = scaling_factor
    
    def select(self, agents: List[AgentRecord], count: int = 1) -> List[BaseAgent]:
        """
        Select agents using roulette wheel selection.
        
        Args:
            agents: List of agent records to select from
            count: Number of agents to select
            
        Returns:
            List of selected agents
        """
        if not agents:
            return []
        
        # Calculate scaled fitness values
        fitnesses = [agent.fitness for agent in agents]
        min_fitness = min(fitnesses)
        
        # Scale fitness to be positive
        scaled_fitnesses = [f - min_fitness + self.scaling_factor for f in fitnesses]
        
        # Calculate selection probabilities
        total_fitness = sum(scaled_fitnesses)
        if total_fitness == 0:
            # If all fitnesses are equal, use uniform selection
            probabilities = [1.0 / len(agents)] * len(agents)
        else:
            probabilities = [f / total_fitness for f in scaled_fitnesses]
        
        # Select agents
        selected = []
        for _ in range(count):
            # Roulette wheel selection
            r = random.random()
            cumulative_prob = 0.0
            
            for i, prob in enumerate(probabilities):
                cumulative_prob += prob
                if r <= cumulative_prob:
                    selected.append(agents[i].agent)
                    break
        
        return selected


class RankBasedSelection(SelectionStrategy):
    """Rank-based selection strategy."""
    
    def __init__(self, selection_pressure: float = 1.5):
        """
        Initialize rank-based selection.
        
        Args:
            selection_pressure: Selection pressure (higher = more selective)
        """
        self.selection_pressure = selection_pressure
    
    def select(self, agents: List[AgentRecord], count: int = 1) -> List[BaseAgent]:
        """
        Select agents using rank-based selection.
        
        Args:
            agents: List of agent records to select from
            count: Number of agents to select
            
        Returns:
            List of selected agents
        """
        if not agents:
            return []
        
        # Sort agents by fitness (best first)
        ranked_agents = sorted(agents, key=lambda x: x.fitness, reverse=True)
        
        # Calculate rank probabilities
        n = len(ranked_agents)
        probabilities = []
        
        for rank in range(n):
            # Linear ranking: p(rank) = (2 - s) / n + 2 * (s - 1) * rank / (n * (n - 1))
            # where s is selection pressure
            prob = (2 - self.selection_pressure) / n + 2 * (self.selection_pressure - 1) * rank / (n * (n - 1))
            probabilities.append(prob)
        
        # Select agents
        selected = []
        for _ in range(count):
            # Roulette wheel selection based on rank
            r = random.random()
            cumulative_prob = 0.0
            
            for i, prob in enumerate(probabilities):
                cumulative_prob += prob
                if r <= cumulative_prob:
                    selected.append(ranked_agents[i].agent)
                    break
        
        return selected


class ElitismSelection(SelectionStrategy):
    """Elitism selection strategy that preserves the best agents."""
    
    def __init__(self, elite_ratio: float = 0.1):
        """
        Initialize elitism selection.
        
        Args:
            elite_ratio: Ratio of best agents to preserve
        """
        self.elite_ratio = elite_ratio
    
    def select(self, agents: List[AgentRecord], count: int = 1) -> List[BaseAgent]:
        """
        Select the best agents (elitism).
        
        Args:
            agents: List of agent records to select from
            count: Number of agents to select
            
        Returns:
            List of selected agents
        """
        if not agents:
            return []
        
        # Sort agents by fitness (best first)
        ranked_agents = sorted(agents, key=lambda x: x.fitness, reverse=True)
        
        # Calculate number of elite agents
        elite_count = max(1, int(len(agents) * self.elite_ratio))
        
        # Return the best agents
        return [agent.agent for agent in ranked_agents[:elite_count]]


class StochasticUniversalSampling(SelectionStrategy):
    """Stochastic Universal Sampling (SUS) selection strategy."""
    
    def select(self, agents: List[AgentRecord], count: int = 1) -> List[BaseAgent]:
        """
        Select agents using Stochastic Universal Sampling.
        
        Args:
            agents: List of agent records to select from
            count: Number of agents to select
            
        Returns:
            List of selected agents
        """
        if not agents:
            return []
        
        # Calculate fitness values
        fitnesses = [agent.fitness for agent in agents]
        min_fitness = min(fitnesses)
        
        # Scale fitness to be positive
        scaled_fitnesses = [f - min_fitness + 1.0 for f in fitnesses]
        total_fitness = sum(scaled_fitnesses)
        
        if total_fitness == 0:
            # If all fitnesses are equal, use uniform selection
            return random.sample([agent.agent for agent in agents], min(count, len(agents)))
        
        # SUS selection
        selected = []
        pointer_distance = total_fitness / count
        start = random.uniform(0, pointer_distance)
        
        current_sum = 0.0
        agent_index = 0
        
        for i in range(count):
            pointer = start + i * pointer_distance
            
            while current_sum < pointer and agent_index < len(agents):
                current_sum += scaled_fitnesses[agent_index]
                agent_index += 1
            
            if agent_index > 0:
                selected.append(agents[agent_index - 1].agent)
        
        return selected 