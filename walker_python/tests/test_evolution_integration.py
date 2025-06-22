"""
Integration test for evolution system with physics world.
"""

import pytest
import numpy as np
import pymunk
import time

from src.agents.basic_agent import BasicAgent
from src.physics.world import WorldController
from src.population.population_controller import PopulationController
from src.population.evolution import EvolutionEngine


def create_simple_physics_world():
    """Create a simple physics world for testing."""
    world = WorldController()
    
    # Add a simple ball that agents can control
    ball_body = pymunk.Body(1, pymunk.moment_for_circle(1, 0, 2))
    ball_body.position = (10, 20)
    ball_shape = pymunk.Circle(ball_body, 2)
    world.space.add(ball_body, ball_shape)
    
    return world, ball_body


def simple_fitness_function(agent, world, ball, max_steps=100):
    """
    Simple fitness function that rewards moving the ball to the right.
    
    Args:
        agent: The agent to evaluate
        world: Physics world
        ball: The ball to control
        max_steps: Maximum simulation steps
        
    Returns:
        Fitness score (distance moved right)
    """
    # Reset world and agent
    agent.reset()
    
    # Set initial ball position
    ball.position = (10, 20)
    ball.velocity = (0, 0)
    
    # Track ball position
    initial_x = ball.position.x
    max_x = initial_x
    
    # Run simulation
    for step in range(max_steps):
        # Get current state (simplified: just ball position)
        ball_x = int(ball.position.x)
        ball_y = int(ball.position.y)
        velocity_x = int(ball.velocity.x)
        
        # Discretize state
        state = (ball_x, ball_y, velocity_x)
        agent.set_state(state)
        
        # Agent selects action
        action = agent.select_action(epsilon=0.1)
        
        # Apply action (simplified: just apply force)
        if action == 0:  # Move left
            ball.apply_force_at_local_point((50, 0), (0, 0))
        elif action == 1:  # Move right
            ball.apply_force_at_local_point((-50, 0), (0, 0))
        elif action == 2:  # Jump
            ball.apply_force_at_local_point((0, -100), (0, 0))
        
        # Step physics
        world.space.step(1/60.0)
        
        # Update max position
        max_x = max(max_x, ball.position.x)
        
        # Give reward based on movement
        reward = 0.0
        if ball.position.x > initial_x:
            reward = (ball.position.x - initial_x) * 0.1
        
        agent.set_reward(reward)
        agent.update(1/60.0)
        
        # Stop if ball falls off screen
        if ball.position.y < -10:
            break
    
    # Fitness is the maximum distance moved right
    fitness = max(0, max_x - initial_x)
    return fitness


class TestEvolutionIntegration:
    """Test evolution system integration with physics world."""
    
    def test_simple_evolution(self):
        """Test a simple evolution scenario."""
        # Create population controller
        controller = PopulationController(
            population_size=10,
            max_generations=5,
            evaluation_time=2.0
        )
        
        # Create evolution engine
        engine = EvolutionEngine(controller, elite_size=2)
        
        # Create initial population
        initial_population = []
        for i in range(10):
            agent = BasicAgent(state_dimensions=[20, 20, 10], action_count=3)
            initial_population.append(agent)
        
        # Create physics world
        world, ball = create_simple_physics_world()
        
        # Define fitness function
        def fitness_func(agent):
            return simple_fitness_function(agent, world, ball, max_steps=50)
        
        # Run evolution
        results = engine.run_evolution(
            initial_population=initial_population,
            fitness_function=fitness_func,
            max_generations=3
        )
        
        # Check results
        assert results['generations_completed'] > 0
        assert len(results['best_fitness_history']) > 0
        assert len(results['average_fitness_history']) > 0
        
        # Check that we have a best agent
        best_agent = engine.get_best_agent()
        assert best_agent is not None
        
        print(f"Evolution completed:")
        print(f"  Generations: {results['generations_completed']}")
        print(f"  Best fitness: {results['best_fitness_history'][-1]:.2f}")
        print(f"  Average fitness: {results['average_fitness_history'][-1]:.2f}")
        print(f"  Improvement rate: {results['improvement_rate']:.4f}")
    
    def test_population_diversity_tracking(self):
        """Test that population diversity is tracked correctly."""
        controller = PopulationController(population_size=5)
        engine = EvolutionEngine(controller)
        
        # Create initial population
        initial_population = []
        for i in range(5):
            agent = BasicAgent(state_dimensions=[10, 10, 5], action_count=3)
            initial_population.append(agent)
        
        # Add agents to controller
        for agent in initial_population:
            controller.add_agent(agent)
        
        # Set different fitness values
        for i, agent in enumerate(initial_population):
            controller.update_agent_fitness(agent, float(i + 1))
        
        # Check diversity
        diversity = engine.get_population_diversity()
        assert 0.0 <= diversity <= 1.0
        assert diversity > 0.0  # Should have some diversity with different fitnesses
        
        print(f"Population diversity: {diversity:.4f}")
    
    def test_generation_statistics(self):
        """Test generation statistics tracking."""
        controller = PopulationController(population_size=5)
        
        # Create and add agents
        agents = []
        for i in range(5):
            agent = BasicAgent(state_dimensions=[10, 10, 5], action_count=3)
            agents.append(agent)
            controller.add_agent(agent)
        
        # Set fitness values
        for i, agent in enumerate(agents):
            controller.update_agent_fitness(agent, float(i + 1))
        
        # Save generation stats
        controller.save_generation_stats()
        
        # Check statistics
        assert len(controller.generation_stats) == 1
        stats = controller.generation_stats[0]
        
        assert stats['generation'] == 0
        assert stats['population_size'] == 5
        assert stats['best_fitness'] == 5.0
        assert stats['fitness_stats']['mean'] == 3.0
        
        print(f"Generation stats: {stats}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 