#!/usr/bin/env python3
"""
Simple test to verify the training system works.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import pymunk
import numpy as np
from src.agents.evolutionary_crawling_agent import EvolutionaryCrawlingAgent


def test_basic_training():
    """Test basic training functionality."""
    print("🧪 Testing basic training functionality...")
    
    # Create physics world
    space = pymunk.Space()
    space.gravity = (0, -9.8)
    
    # Create ground
    static_body = space.static_body
    ground = pymunk.Segment(static_body, (-100, 0), (100, 0), 1.0)
    ground.friction = 1.0
    space.add(ground)
    
    # Create agent
    agent = EvolutionaryCrawlingAgent(space, position=(10, 20))
    
    print(f"✅ Created agent at position: {agent.body.position}")
    
    # Test basic movement
    print("🔄 Testing agent movement...")
    start_x = agent.body.position.x
    
    for step in range(50):
        # Random action
        action = (np.random.uniform(-2, 2), np.random.uniform(-2, 2))
        agent.apply_action(action)
        agent.step(1/60.0)
        
        if step % 10 == 0:
            debug = agent.get_debug_info()
            print(f"   Step {step}: pos=({debug['crate_pos'][0]:.2f}, {debug['crate_pos'][1]:.2f}), "
                  f"vel=({debug['crate_vel'][0]:.2f}, {debug['crate_vel'][1]:.2f})")
    
    end_x = agent.body.position.x
    distance = end_x - start_x
    print(f"✅ Agent moved {distance:.2f} units")
    
    # Test fitness evaluation
    fitness = agent.get_reward(start_x)
    print(f"✅ Agent fitness: {fitness:.2f}")
    
    print("🎉 Basic training test passed!")
    return True


def test_population():
    """Test population creation."""
    print("\n🧪 Testing population creation...")
    
    # Create physics world
    space = pymunk.Space()
    space.gravity = (0, -9.8)
    
    # Create ground
    static_body = space.static_body
    ground = pymunk.Segment(static_body, (-100, 0), (100, 0), 1.0)
    ground.friction = 1.0
    space.add(ground)
    
    # Create population
    agents = []
    for i in range(4):
        agent = EvolutionaryCrawlingAgent(space, position=(10 + i * 10, 20))
        agents.append(agent)
    
    print(f"✅ Created population of {len(agents)} agents")
    
    # Test fitness evaluation
    fitnesses = []
    for i, agent in enumerate(agents):
        agent.reset()
        start_x = agent.body.position.x
        
        # Run for a few steps
        for step in range(30):
            action = (np.random.uniform(-2, 2), np.random.uniform(-2, 2))
            agent.apply_action(action)
            agent.step(1/60.0)
        
        fitness = agent.body.position.x - start_x
        fitnesses.append(fitness)
        print(f"   Agent {i}: fitness = {fitness:.2f}")
    
    best_fitness = max(fitnesses)
    avg_fitness = np.mean(fitnesses)
    print(f"✅ Best fitness: {best_fitness:.2f}")
    print(f"✅ Average fitness: {avg_fitness:.2f}")
    
    print("🎉 Population test passed!")
    return True


if __name__ == "__main__":
    print("🚀 Starting Walker Robot Training Tests")
    print("=" * 50)
    
    try:
        test_basic_training()
        test_population()
        print("\n🎉 All tests passed! Ready to run training.")
        print("\n📋 To run the full training visualization:")
        print("   python3 train_robots.py")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 