#!/usr/bin/env python3
"""
Q-Learning Diagnostic Script
Run this to identify specific issues with the Q-learning system.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import numpy as np
import Box2D as b2
from src.agents.crawling_crate_agent import CrawlingCrateAgent

def test_reward_magnitudes():
    """Test if rewards are too small to drive learning."""
    print("🔍 TESTING REWARD MAGNITUDES")
    print("=" * 50)
    
    world = b2.b2World(gravity=(0, -10), doSleep=True)
    agent = CrawlingCrateAgent(world, agent_id=1, position=(0, 5))
    
    # Test reward ranges
    rewards = []
    for i in range(100):
        # Simulate different movement scenarios
        try:
            prev_x = float(agent.body.position[0])  # Safe access to x coordinate
            
            # Step physics first to create natural movement
            world.Step(1/60.0, 8, 3)
            
            # Get reward based on position change
            current_x = float(agent.body.position[0])
            reward = agent.get_reward(prev_x)
            rewards.append(reward)
        except Exception as e:
            print(f"Warning: Error in reward test iteration {i}: {e}")
            rewards.append(0.0)  # Add zero reward for failed iterations
    
    print(f"Reward statistics over 100 steps:")
    print(f"  Min reward: {min(rewards):.6f}")
    print(f"  Max reward: {max(rewards):.6f}")
    print(f"  Mean reward: {np.mean(rewards):.6f}")
    print(f"  Std reward: {np.std(rewards):.6f}")
    print(f"  Non-zero rewards: {sum(1 for r in rewards if abs(r) > 0.001)}/100")
    
    if max(rewards) < 0.1:
        print("❌ ISSUE: Rewards are too small (max < 0.1)")
    else:
        print("✅ Reward magnitudes seem reasonable")
    
    print()

def test_learning_rate_impact():
    """Test if learning rate allows Q-values to change meaningfully."""
    print("🔍 TESTING LEARNING RATE IMPACT")
    print("=" * 50)
    
    world = b2.b2World(gravity=(0, -10), doSleep=True)
    agent = CrawlingCrateAgent(world, agent_id=1, position=(0, 5))
    
    # Get initial state and action
    state = agent.get_discretized_state()
    action = 0
    initial_q = agent.q_table.get_q_value(state, action)
    
    print(f"Initial Q-value: {initial_q:.6f}")
    print(f"Learning rate: {agent.learning_rate}")
    
    # Simulate learning with different reward magnitudes
    for reward_mag in [0.01, 0.05, 0.1, 0.5]:
        # Reset Q-value
        agent.q_table.set_q_value(state, action, 0.0)
        
        # Apply learning update
        next_state = agent.get_discretized_state()
        agent.q_table.update_q_value_enhanced(
            state=state,
            action=action,
            reward=reward_mag,
            next_state=next_state,
            base_learning_rate=agent.learning_rate,
            discount_factor=agent.discount_factor,
            use_adaptive_lr=False
        )
        
        new_q = agent.q_table.get_q_value(state, action)
        change = abs(new_q - 0.0)
        
        print(f"  Reward {reward_mag:.3f} -> Q-change: {change:.6f}")
        
        if change < 0.001:
            print(f"    ❌ ISSUE: Q-value barely changed with reward {reward_mag}")
        elif change > 0.01:
            print(f"    ✅ Good: Meaningful Q-value change")
        else:
            print(f"    ⚠️  Marginal: Small but detectable change")
    
    print()

def test_state_discretization():
    """Test if state discretization is too coarse."""
    print("🔍 TESTING STATE DISCRETIZATION")
    print("=" * 50)
    
    world = b2.b2World(gravity=(0, -10), doSleep=True)
    agent = CrawlingCrateAgent(world, agent_id=1, position=(0, 5))
    
    # Test state space size
    states_seen = set()
    
    # Simulate various arm positions
    for shoulder_angle in np.linspace(-np.pi, np.pi, 100):
        for elbow_angle in np.linspace(-np.pi, np.pi, 20):  # Fewer to keep test fast
            # Set arm angles manually for testing
            if hasattr(agent, 'upper_arm'):
                agent.upper_arm.angle = shoulder_angle
            if hasattr(agent, 'lower_arm'):
                agent.lower_arm.angle = elbow_angle
            
            state = agent.get_discretized_state()
            states_seen.add(state)
    
    total_possible_states = 8 * 8 * 4  # From the discretization bins
    
    print(f"State discretization analysis:")
    print(f"  States seen in test: {len(states_seen)}")
    print(f"  Theoretical max states: {total_possible_states}")
    print(f"  State space utilization: {len(states_seen)/total_possible_states*100:.1f}%")
    
    # Test resolution
    print(f"  Shoulder bins: 8 (45° resolution)")
    print(f"  Elbow bins: 8 (45° resolution)") 
    print(f"  Velocity bins: 4")
    
    if len(states_seen) < 10:
        print("❌ ISSUE: Very few unique states - discretization too coarse")
    elif len(states_seen) > 1000:
        print("⚠️  Warning: Very large state space - may slow learning")
    else:
        print("✅ State space size seems reasonable")
    
    print()

def test_action_timing():
    """Test if action persistence is affecting learning."""
    print("🔍 TESTING ACTION TIMING")
    print("=" * 50)
    
    world = b2.b2World(gravity=(0, -10), doSleep=True)
    agent = CrawlingCrateAgent(world, agent_id=1, position=(0, 5))
    
    print(f"Action persistence duration: {agent.action_persistence_duration}s")
    print(f"At 60 FPS, this means actions change every {agent.action_persistence_duration * 60:.0f} frames")
    
    if agent.action_persistence_duration > 0.1:
        print("❌ ISSUE: Action persistence too long - slows learning")
        print("   Recommendation: Reduce to 0.05s or less")
    else:
        print("✅ Action timing seems reasonable")
    
    print()

def test_epsilon_decay():
    """Test if epsilon decays too slowly."""
    print("🔍 TESTING EPSILON DECAY")
    print("=" * 50)
    
    world = b2.b2World(gravity=(0, -10), doSleep=True)
    agent = CrawlingCrateAgent(world, agent_id=1, position=(0, 5))
    
    initial_epsilon = agent.epsilon
    decay_rate = agent.epsilon_decay
    
    print(f"Initial epsilon: {initial_epsilon:.4f}")
    print(f"Decay rate: {decay_rate}")
    
    # Calculate how long it takes to reach min_epsilon
    epsilon = initial_epsilon
    steps = 0
    while epsilon > agent.min_epsilon and steps < 100000:
        epsilon *= decay_rate
        steps += 1
    
    time_to_min = steps / 60  # Convert to seconds at 60 FPS
    
    print(f"Steps to reach min epsilon: {steps}")
    print(f"Time to reach min epsilon: {time_to_min:.1f} seconds")
    
    if time_to_min > 300:  # More than 5 minutes
        print("❌ ISSUE: Epsilon decays too slowly")
        print("   Recommendation: Use decay rate ~0.995 for faster learning")
    else:
        print("✅ Epsilon decay seems reasonable")
    
    print()

def run_all_diagnostics():
    """Run all diagnostic tests."""
    print("🧪 Q-LEARNING SYSTEM DIAGNOSTICS")
    print("=" * 60)
    print()
    
    try:
        test_reward_magnitudes()
        test_learning_rate_impact()
        test_state_discretization()
        test_action_timing()
        test_epsilon_decay()
        
        print("🏁 DIAGNOSTIC SUMMARY")
        print("=" * 50)
        print("Review the output above for specific issues.")
        print("Key things to look for:")
        print("  ❌ = Critical issue that prevents learning")
        print("  ⚠️  = Warning that may slow learning")
        print("  ✅ = Component working correctly")
        
    except Exception as e:
        print(f"❌ Error running diagnostics: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_diagnostics() 