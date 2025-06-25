#!/usr/bin/env python3
"""
Debug script to test Q-learning components.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import numpy as np
import Box2D as b2
from src.agents.crawling_crate_agent import CrawlingCrateAgent

def test_q_learning():
    print("🧪 Testing Q-learning components...")
    
    # Create world
    world = b2.b2World(gravity=(0, -10), doSleep=True)
    
    try:
        # Create agent
        print("Creating CrawlingCrateAgent...")
        agent = CrawlingCrateAgent(world, position=(0, 5))
        print("✅ Agent created successfully")
        
        # Test state discretization
        print("Testing state discretization...")
        state = agent.get_discretized_state()
        print(f"✅ Discretized state: {state}")
        
        # Test action selection
        print("Testing action selection...")
        action_idx = agent.choose_action()
        print(f"✅ Chosen action: {action_idx}")
        
        # Test Q-table access
        print("Testing Q-table...")
        q_value = agent.q_table.get_q_value(state, action_idx)
        print(f"✅ Q-value: {q_value}")
        
        # Test reward calculation
        print("Testing reward calculation...")
        reward = agent.get_reward(0.0)
        print(f"✅ Reward: {reward}")
        
        # Test Q-value update
        print("Testing Q-value update...")
        next_state = agent.get_discretized_state()
        agent.update_q_value(next_state, reward)
        print("✅ Q-value updated successfully")
        
        print("🎉 All Q-learning components working!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_q_learning() 