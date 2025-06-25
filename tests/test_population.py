"""
Tests for population management system.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from src.agents.basic_agent import BasicAgent
from src.population.population_controller import PopulationController, AgentStatus
from src.population.evolution import EvolutionEngine, MutationOperator, CrossoverOperator
from src.population.selection import TournamentSelection, RouletteWheelSelection, ElitismSelection


class TestPopulationController:
    """Test the PopulationController class."""
    
    def test_initialization(self):
        """Test population controller initialization."""
        controller = PopulationController(population_size=10, max_generations=50)
        
        assert controller.population_size == 10
        assert controller.max_generations == 50
        assert len(controller.agents) == 0
        assert controller.current_generation == 0
    
    def test_add_agent(self):
        """Test adding agents to population."""
        controller = PopulationController(population_size=5)
        agent = BasicAgent(state_dimensions=[10], action_count=3)
        
        record = controller.add_agent(agent)
        
        assert len(controller.agents) == 1
        assert record.agent == agent
        assert record.status == AgentStatus.ACTIVE
        assert record.fitness == 0.0
    
    def test_population_full(self):
        """Test that population respects size limit."""
        controller = PopulationController(population_size=2)
        
        agent1 = BasicAgent(state_dimensions=[10], action_count=3)
        agent2 = BasicAgent(state_dimensions=[10], action_count=3)
        agent3 = BasicAgent(state_dimensions=[10], action_count=3)
        
        controller.add_agent(agent1)
        controller.add_agent(agent2)
        
        with pytest.raises(ValueError, match="Population is full"):
            controller.add_agent(agent3)
    
    def test_remove_agent(self):
        """Test removing agents from population."""
        controller = PopulationController()
        agent = BasicAgent(state_dimensions=[10], action_count=3)
        
        controller.add_agent(agent)
        assert len(controller.agents) == 1
        
        success = controller.remove_agent(agent)
        assert success is True
        assert len(controller.agents) == 0
        
        # Try to remove non-existent agent
        success = controller.remove_agent(agent)
        assert success is False
    
    def test_update_fitness(self):
        """Test updating agent fitness."""
        controller = PopulationController()
        agent = BasicAgent(state_dimensions=[10], action_count=3)
        
        controller.add_agent(agent)
        success = controller.update_agent_fitness(agent, 42.5, distance_traveled=100.0)
        
        assert success is True
        record = controller.agents[0]
        assert record.fitness == 42.5
        assert record.status == AgentStatus.COMPLETED
        assert record.distance_traveled == 100.0
    
    def test_get_ranked_agents(self):
        """Test getting agents ranked by fitness."""
        controller = PopulationController()
        
        agent1 = BasicAgent(state_dimensions=[10], action_count=3)
        agent2 = BasicAgent(state_dimensions=[10], action_count=3)
        agent3 = BasicAgent(state_dimensions=[10], action_count=3)
        
        controller.add_agent(agent1)
        controller.add_agent(agent2)
        controller.add_agent(agent3)
        
        controller.update_agent_fitness(agent1, 10.0)
        controller.update_agent_fitness(agent2, 30.0)
        controller.update_agent_fitness(agent3, 20.0)
        
        ranked = controller.get_ranked_agents()
        assert len(ranked) == 3
        assert ranked[0].fitness == 30.0  # Best
        assert ranked[1].fitness == 20.0  # Middle
        assert ranked[2].fitness == 10.0  # Worst
    
    def test_fitness_statistics(self):
        """Test fitness statistics calculation."""
        controller = PopulationController()
        
        agent1 = BasicAgent(state_dimensions=[10], action_count=3)
        agent2 = BasicAgent(state_dimensions=[10], action_count=3)
        agent3 = BasicAgent(state_dimensions=[10], action_count=3)
        
        controller.add_agent(agent1)
        controller.add_agent(agent2)
        controller.add_agent(agent3)
        
        controller.update_agent_fitness(agent1, 10.0)
        controller.update_agent_fitness(agent2, 20.0)
        controller.update_agent_fitness(agent3, 30.0)
        
        stats = controller.get_fitness_statistics()
        
        assert stats['min'] == 10.0
        assert stats['max'] == 30.0
        assert stats['mean'] == 20.0
        assert stats['std'] == pytest.approx(8.1649658, rel=1e-6)
        assert stats['median'] == 20.0
    
    def test_empty_population_statistics(self):
        """Test statistics for empty population."""
        controller = PopulationController()
        stats = controller.get_fitness_statistics()
        
        assert stats['min'] == 0.0
        assert stats['max'] == 0.0
        assert stats['mean'] == 0.0
        assert stats['std'] == 0.0
        assert stats['median'] == 0.0
    
    def test_advance_generation(self):
        """Test advancing to next generation."""
        controller = PopulationController()
        agent = BasicAgent(state_dimensions=[10], action_count=3)
        
        controller.add_agent(agent)
        assert controller.current_generation == 0
        assert controller.agents[0].generation == 0
        
        controller.advance_generation()
        assert controller.current_generation == 1
        assert controller.agents[0].generation == 1
    
    def test_population_diversity(self):
        """Test population diversity calculation."""
        controller = PopulationController()
        
        # Same fitness = low diversity
        agent1 = BasicAgent(state_dimensions=[10], action_count=3)
        agent2 = BasicAgent(state_dimensions=[10], action_count=3)
        
        controller.add_agent(agent1)
        controller.add_agent(agent2)
        controller.update_agent_fitness(agent1, 10.0)
        controller.update_agent_fitness(agent2, 10.0)
        
        diversity = controller.get_population_diversity()
        assert diversity == 0.0
        
        # Different fitness = high diversity
        controller.update_agent_fitness(agent2, 20.0)
        diversity = controller.get_population_diversity()
        assert diversity > 0.0


class TestEvolutionEngine:
    """Test the EvolutionEngine class."""
    
    def test_initialization(self):
        """Test evolution engine initialization."""
        controller = PopulationController()
        engine = EvolutionEngine(controller)
        
        assert engine.population_controller == controller
        assert isinstance(engine.mutation_operator, MutationOperator)
        assert isinstance(engine.crossover_operator, CrossoverOperator)
        assert engine.elite_size == 2
        assert engine.tournament_size == 3
    
    def test_tournament_selection(self):
        """Test tournament selection."""
        controller = PopulationController()
        engine = EvolutionEngine(controller)
        
        agent1 = BasicAgent(state_dimensions=[10], action_count=3)
        agent2 = BasicAgent(state_dimensions=[10], action_count=3)
        agent3 = BasicAgent(state_dimensions=[10], action_count=3)
        
        record1 = controller.add_agent(agent1)
        record2 = controller.add_agent(agent2)
        record3 = controller.add_agent(agent3)
        
        controller.update_agent_fitness(agent1, 10.0)
        controller.update_agent_fitness(agent2, 30.0)
        controller.update_agent_fitness(agent3, 20.0)
        
        ranked = controller.get_ranked_agents()
        selected = engine._tournament_selection(ranked)
        
        # Should select one of the agents
        assert selected in [agent1, agent2, agent3]
    
    def test_fitness_evaluation(self):
        """Test fitness evaluation."""
        controller = PopulationController()
        engine = EvolutionEngine(controller)
        
        agent = BasicAgent(state_dimensions=[10], action_count=3)
        
        def fitness_func(agent):
            return 42.0
        
        fitness = engine.evaluate_fitness(agent, fitness_func)
        assert fitness == 42.0


class TestSelectionStrategies:
    """Test selection strategies."""
    
    def test_tournament_selection(self):
        """Test tournament selection strategy."""
        from src.population.population_controller import AgentRecord, AgentStatus
        
        strategy = TournamentSelection(tournament_size=2)
        
        # Create mock agents with different fitness
        agent1 = BasicAgent(state_dimensions=[10], action_count=3)
        agent2 = BasicAgent(state_dimensions=[10], action_count=3)
        agent3 = BasicAgent(state_dimensions=[10], action_count=3)
        
        record1 = AgentRecord(agent=agent1, fitness=10.0)
        record2 = AgentRecord(agent=agent2, fitness=30.0)
        record3 = AgentRecord(agent=agent3, fitness=20.0)
        
        agents = [record1, record2, record3]
        
        selected = strategy.select(agents, count=2)
        assert len(selected) == 2
        assert all(agent in [agent1, agent2, agent3] for agent in selected)
    
    def test_roulette_wheel_selection(self):
        """Test roulette wheel selection strategy."""
        from src.population.population_controller import AgentRecord, AgentStatus
        
        strategy = RouletteWheelSelection()
        
        agent1 = BasicAgent(state_dimensions=[10], action_count=3)
        agent2 = BasicAgent(state_dimensions=[10], action_count=3)
        agent3 = BasicAgent(state_dimensions=[10], action_count=3)
        
        record1 = AgentRecord(agent=agent1, fitness=10.0)
        record2 = AgentRecord(agent=agent2, fitness=20.0)
        record3 = AgentRecord(agent=agent3, fitness=30.0)
        
        agents = [record1, record2, record3]
        
        selected = strategy.select(agents, count=2)
        assert len(selected) == 2
        assert all(agent in [agent1, agent2, agent3] for agent in selected)
    
    def test_elitism_selection(self):
        """Test elitism selection strategy."""
        from src.population.population_controller import AgentRecord, AgentStatus
        
        strategy = ElitismSelection(elite_ratio=0.5)
        
        agent1 = BasicAgent(state_dimensions=[10], action_count=3)
        agent2 = BasicAgent(state_dimensions=[10], action_count=3)
        agent3 = BasicAgent(state_dimensions=[10], action_count=3)
        agent4 = BasicAgent(state_dimensions=[10], action_count=3)
        
        record1 = AgentRecord(agent=agent1, fitness=10.0)
        record2 = AgentRecord(agent=agent2, fitness=20.0)
        record3 = AgentRecord(agent=agent3, fitness=30.0)
        record4 = AgentRecord(agent=agent4, fitness=40.0)
        
        agents = [record1, record2, record3, record4]
        
        selected = strategy.select(agents, count=10)  # Should only return elite
        assert len(selected) == 2  # 50% of 4 agents
        assert agent4 in selected  # Best agent should be selected
        assert agent3 in selected  # Second best should be selected


if __name__ == "__main__":
    pytest.main([__file__]) 