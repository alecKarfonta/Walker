"""
Integration test for CrawlingCrate agent: ensures only crawling (arm actuation) can move the robot.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
import pymunk
import numpy as np
from src.agents.crawling_crate import CrawlingCrate
from src.population.population_controller import PopulationController
from src.population.evolution import EvolutionEngine


def create_flat_world():
    """Create a flat ground in a Pymunk space."""
    space = pymunk.Space()
    space.gravity = (0, -9.8)
    # Flat ground
    static_body = space.static_body
    ground = pymunk.Segment(static_body, (-100, 0), (100, 0), 1.0)
    ground.friction = 1.0
    space.add(ground)
    return space


def crawling_fitness_alternate(agent: CrawlingCrate, max_steps=200) -> float:
    """Fitness: alternate arm crawling pattern."""
    agent.reset()
    prev_x = agent.body.position.x
    total_reward = 0.0
    
    print(f"Starting position: {agent.body.position.x:.2f}")
    
    for step in range(max_steps):
        # More realistic crawling: push with one arm while other is in air
        if step % 60 < 30:
            action = (3.0, 0.0)  # Only left arm pushing
        else:
            action = (0.0, 3.0)  # Only right arm pushing
        
        agent.apply_action(action)
        agent.step(1/60.0)
        reward = agent.get_reward(prev_x)
        total_reward += reward
        prev_x = agent.body.position.x
        
        # Debug output every 50 steps
        if step % 50 == 0:
            debug = agent.get_debug_info()
            print(f"Step {step}: pos=({debug['crate_pos'][0]:.2f}, {debug['crate_pos'][1]:.2f}), "
                  f"vel=({debug['crate_vel'][0]:.2f}, {debug['crate_vel'][1]:.2f}), "
                  f"left_angle={debug['left_arm_angle']:.2f}, right_angle={debug['right_arm_angle']:.2f}")
        
        # End if crate flips or falls
        if agent.body.position.y < -2 or abs(agent.body.angle) > np.pi/2:
            break
    
    final_dist = agent.body.position.x - 10
    print(f"Final position: {agent.body.position.x:.2f}, Distance: {final_dist:.2f}")
    return final_dist


def crawling_fitness_one_arm(agent: CrawlingCrate, max_steps=200) -> float:
    """Fitness: one arm at a time crawling pattern."""
    agent.reset()
    prev_x = agent.body.position.x
    
    for step in range(max_steps):
        # One arm at a time: left arm, then right arm
        if step % 60 < 30:
            action = (10.0, 0.0)  # Only left arm
        else:
            action = (0.0, 10.0)  # Only right arm
        
        agent.apply_action(action)
        agent.step(1/60.0)
        reward = agent.get_reward(prev_x)
        prev_x = agent.body.position.x
        
        # End if crate flips or falls
        if agent.body.position.y < -2 or abs(agent.body.angle) > np.pi/2:
            break
    
    return agent.body.position.x - 10


def crawling_fitness_both_arms(agent: CrawlingCrate, max_steps=200) -> float:
    """Fitness: both arms together crawling pattern."""
    agent.reset()
    prev_x = agent.body.position.x
    
    for step in range(max_steps):
        # Both arms together: forward, then back
        if step % 60 < 30:
            action = (12.0, 12.0)  # Both forward
        else:
            action = (-12.0, -12.0)  # Both back
        
        agent.apply_action(action)
        agent.step(1/60.0)
        reward = agent.get_reward(prev_x)
        prev_x = agent.body.position.x
        
        # End if crate flips or falls
        if agent.body.position.y < -2 or abs(agent.body.angle) > np.pi/2:
            break
    
    return agent.body.position.x - 10


def test_crawling_crate_alternate_pattern():
    """Test that CrawlingCrate can move forward with alternate arm pattern."""
    space = create_flat_world()
    agent = CrawlingCrate(space)
    dist = crawling_fitness_alternate(agent, max_steps=300)
    print(f"CrawlingCrate alternate pattern distance: {dist:.2f}")
    assert dist > 0.5  # Should move forward by crawling


def test_crawling_crate_one_arm_pattern():
    """Test that CrawlingCrate can move forward with one arm at a time."""
    space = create_flat_world()
    agent = CrawlingCrate(space)
    dist = crawling_fitness_one_arm(agent, max_steps=300)
    print(f"CrawlingCrate one arm pattern distance: {dist:.2f}")
    assert dist > 0.5  # Should move forward by crawling


def test_crawling_crate_both_arms_pattern():
    """Test that CrawlingCrate can move forward with both arms together."""
    space = create_flat_world()
    agent = CrawlingCrate(space)
    dist = crawling_fitness_both_arms(agent, max_steps=300)
    print(f"CrawlingCrate both arms pattern distance: {dist:.2f}")
    assert dist > 0.5  # Should move forward by crawling


def test_crawling_crate_cannot_move_without_arms():
    """Test that CrawlingCrate cannot move if arms are not actuated."""
    space = create_flat_world()
    agent = CrawlingCrate(space)
    agent.reset()
    prev_x = agent.body.position.x
    
    print(f"Starting position (no arms): {agent.body.position.x:.2f}")
    
    for step in range(200):
        agent.apply_action((0.0, 0.0))  # No arm movement
        agent.step(1/60.0)
        
        if step % 50 == 0:
            debug = agent.get_debug_info()
            print(f"Step {step}: pos=({debug['crate_pos'][0]:.2f}, {debug['crate_pos'][1]:.2f}), "
                  f"vel=({debug['crate_vel'][0]:.2f}, {debug['crate_vel'][1]:.2f})")
    
    dist = agent.body.position.x - prev_x
    print(f"CrawlingCrate distance without arm movement: {dist:.2f}")
    assert abs(dist) < 0.1  # Should not move without arm actuation


def test_evolution_with_crawling_crate():
    """Test evolution system using CrawlingCrate agents."""
    population_size = 6
    generations = 2
    spaces = [create_flat_world() for _ in range(population_size)]
    initial_population = [CrawlingCrate(spaces[i]) for i in range(population_size)]
    controller = PopulationController(population_size=population_size, max_generations=generations)
    engine = EvolutionEngine(controller, elite_size=2)
    def fitness_func(agent):
        return crawling_fitness_alternate(agent, max_steps=150)
    results = engine.run_evolution(
        initial_population=initial_population,
        fitness_function=fitness_func,
        max_generations=generations
    )
    print(f"Evolution generations: {results['generations_completed']}")
    print(f"Best fitness: {results['best_fitness_history'][-1]:.2f}")
    assert results['generations_completed'] > 0
    assert results['best_fitness_history'][-1] > 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 