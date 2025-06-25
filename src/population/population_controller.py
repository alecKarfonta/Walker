"""
Population Controller for managing multiple agents and their evolution.

This module provides the core population management functionality including
agent spawning, fitness tracking, ranking, and population operations.
"""

from typing import List, Dict, Optional, Callable, Any, Union
import numpy as np
from dataclasses import dataclass, field
from enum import Enum

from src.agents.base_agent import BaseAgent


class AgentStatus(Enum):
    """Status of an agent in the population."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    EVALUATING = "evaluating"
    COMPLETED = "completed"


@dataclass
class AgentRecord:
    """Record for tracking an agent's information and performance."""
    agent: BaseAgent
    status: AgentStatus = AgentStatus.ACTIVE
    fitness: float = 0.0
    generation: int = 0
    evaluation_time: float = 0.0
    distance_traveled: float = 0.0
    energy_used: float = 0.0
    stability_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize metadata with agent info."""
        self.metadata = {
            'agent_id': id(self.agent),
            'agent_type': type(self.agent).__name__,
            'creation_time': self.evaluation_time
        }


class PopulationController:
    """
    Manages a population of agents for evolutionary training.
    
    This class handles agent spawning, fitness tracking, ranking,
    and population-level operations needed for evolutionary algorithms.
    """
    
    def __init__(self, 
                 population_size: int = 50,
                 max_generations: int = 100,
                 evaluation_time: float = 30.0):
        """
        Initialize the population controller.
        
        Args:
            population_size: Maximum number of agents in population
            max_generations: Maximum number of generations to evolve
            evaluation_time: Time in seconds to evaluate each agent
        """
        self.population_size = population_size
        self.max_generations = max_generations
        self.evaluation_time = evaluation_time
        
        # Population storage
        self.agents: List[AgentRecord] = []
        self.current_generation = 0
        
        # Statistics tracking
        self.generation_stats: List[Dict[str, Any]] = []
        self.best_fitness_history: List[float] = []
        self.average_fitness_history: List[float] = []
        
        # Callbacks
        self.on_agent_spawn: Optional[Callable[[BaseAgent], None]] = None
        self.on_agent_remove: Optional[Callable[[BaseAgent], None]] = None
        self.on_generation_complete: Optional[Callable[[int, Dict[str, Any]], None]] = None
    
    def add_agent(self, agent: BaseAgent, generation: int = 0) -> AgentRecord:
        """
        Add an agent to the population.
        
        Args:
            agent: The agent to add
            generation: Generation number for the agent
            
        Returns:
            AgentRecord for the added agent
        """
        if len(self.agents) >= self.population_size:
            raise ValueError(f"Population is full (max {self.population_size})")
        
        record = AgentRecord(agent=agent, generation=generation)
        self.agents.append(record)
        
        if self.on_agent_spawn:
            self.on_agent_spawn(agent)
        
        return record
    
    def remove_agent(self, agent: BaseAgent) -> bool:
        """
        Remove an agent from the population.
        
        Args:
            agent: The agent to remove
            
        Returns:
            True if agent was found and removed, False otherwise
        """
        for i, record in enumerate(self.agents):
            if record.agent == agent:
                removed_record = self.agents.pop(i)
                if self.on_agent_remove:
                    self.on_agent_remove(agent)
                return True
        return False
    
    def get_active_agents(self) -> List[BaseAgent]:
        """Get all active agents in the population."""
        return [record.agent for record in self.agents 
                if record.status == AgentStatus.ACTIVE]
    
    def get_agents_by_status(self, status: AgentStatus) -> List[BaseAgent]:
        """Get agents with a specific status."""
        return [record.agent for record in self.agents 
                if record.status == status]
    
    def update_agent_fitness(self, agent: BaseAgent, fitness: float, 
                           **metrics) -> bool:
        """
        Update fitness and metrics for an agent.
        
        Args:
            agent: The agent to update
            fitness: New fitness value
            **metrics: Additional metrics to update
            
        Returns:
            True if agent was found and updated, False otherwise
        """
        for record in self.agents:
            if record.agent == agent:
                record.fitness = fitness
                record.status = AgentStatus.COMPLETED
                
                # Update additional metrics
                for key, value in metrics.items():
                    if hasattr(record, key):
                        setattr(record, key, value)
                    else:
                        record.metadata[key] = value
                
                return True
        return False
    
    def get_ranked_agents(self) -> List[AgentRecord]:
        """
        Get agents ranked by fitness (best first).
        
        Returns:
            List of AgentRecord objects sorted by fitness
        """
        return sorted(self.agents, key=lambda x: x.fitness, reverse=True)
    
    def get_best_agent(self) -> Optional[AgentRecord]:
        """Get the agent with the highest fitness."""
        ranked = self.get_ranked_agents()
        return ranked[0] if ranked else None
    
    def get_worst_agent(self) -> Optional[AgentRecord]:
        """Get the agent with the lowest fitness."""
        ranked = self.get_ranked_agents()
        return ranked[-1] if ranked else None
    
    def get_fitness_statistics(self) -> Dict[str, float]:
        """
        Calculate fitness statistics for the current population.
        
        Returns:
            Dictionary with fitness statistics
        """
        if not self.agents:
            return {
                'min': 0.0,
                'max': 0.0,
                'mean': 0.0,
                'std': 0.0,
                'median': 0.0
            }
        
        fitnesses = [record.fitness for record in self.agents]
        return {
            'min': float(min(fitnesses)),
            'max': float(max(fitnesses)),
            'mean': float(np.mean(fitnesses)),
            'std': float(np.std(fitnesses)),
            'median': float(np.median(fitnesses))
        }
    
    def clear_population(self):
        """Remove all agents from the population."""
        if self.on_agent_remove:
            for record in self.agents:
                self.on_agent_remove(record.agent)
        
        self.agents.clear()
    
    def advance_generation(self):
        """Advance to the next generation."""
        self.current_generation += 1
        
        # Update generation numbers for all agents
        for record in self.agents:
            record.generation = self.current_generation
    
    def get_generation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current generation.
        
        Returns:
            Dictionary with generation statistics
        """
        stats = self.get_fitness_statistics()
        best_agent = self.get_best_agent()
        
        return {
            'generation': self.current_generation,
            'population_size': len(self.agents),
            'active_agents': len(self.get_active_agents()),
            'completed_agents': len(self.get_agents_by_status(AgentStatus.COMPLETED)),
            'best_fitness': best_agent.fitness if best_agent else 0.0,
            'best_agent_id': best_agent.metadata.get('agent_id') if best_agent else None,
            'fitness_stats': stats
        }
    
    def save_generation_stats(self):
        """Save current generation statistics."""
        summary = self.get_generation_summary()
        self.generation_stats.append(summary)
        
        # Update history
        self.best_fitness_history.append(summary['best_fitness'])
        self.average_fitness_history.append(summary['fitness_stats']['mean'])
        
        if self.on_generation_complete:
            self.on_generation_complete(self.current_generation, summary)
    
    def get_evolution_progress(self) -> Dict[str, Any]:
        """
        Get overall evolution progress.
        
        Returns:
            Dictionary with evolution progress information
        """
        return {
            'current_generation': self.current_generation,
            'max_generations': self.max_generations,
            'generations_completed': len(self.generation_stats),
            'best_fitness_history': self.best_fitness_history,
            'average_fitness_history': self.average_fitness_history,
            'improvement_rate': self._calculate_improvement_rate()
        }
    
    def _calculate_improvement_rate(self) -> float:
        """Calculate the rate of fitness improvement over generations."""
        if len(self.best_fitness_history) < 2:
            return 0.0
        
        recent = self.best_fitness_history[-10:]  # Last 10 generations
        if len(recent) < 2:
            return 0.0
        
        # Linear regression slope
        x = np.arange(len(recent))
        y = np.array(recent)
        
        if np.all(y == y[0]):  # No change
            return 0.0
        
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)
    
    def is_evolution_complete(self) -> bool:
        """Check if evolution has reached completion criteria."""
        return (self.current_generation >= self.max_generations or
                len(self.generation_stats) >= self.max_generations)
    
    def get_population_diversity(self) -> float:
        """
        Calculate population diversity based on fitness variance.
        
        Returns:
            Diversity score (0-1, higher is more diverse)
        """
        if len(self.agents) < 2:
            return 0.0
        
        fitnesses = [record.fitness for record in self.agents]
        variance = np.var(fitnesses)
        max_possible_variance = (max(fitnesses) - min(fitnesses)) ** 2 / 4
        
        if max_possible_variance == 0:
            return 0.0
        
        return float(min(variance / max_possible_variance, 1.0))
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the population.
        
        Returns:
            Dictionary with population statistics
        """
        if not self.agents:
            return {
                'population_size': 0,
                'active_agents': 0,
                'completed_agents': 0,
                'best_fitness': 0.0,
                'average_fitness': 0.0,
                'fitness_std': 0.0,
                'diversity': 0.0,
                'generation': self.current_generation,
                'evolution_progress': 0.0
            }
        
        fitness_stats = self.get_fitness_statistics()
        best_agent = self.get_best_agent()
        
        return {
            'population_size': len(self.agents),
            'active_agents': len(self.get_active_agents()),
            'completed_agents': len(self.get_agents_by_status(AgentStatus.COMPLETED)),
            'best_fitness': best_agent.fitness if best_agent else 0.0,
            'average_fitness': fitness_stats['mean'],
            'fitness_std': fitness_stats['std'],
            'diversity': self.get_population_diversity(),
            'generation': self.current_generation,
            'evolution_progress': len(self.generation_stats) / self.max_generations
        } 