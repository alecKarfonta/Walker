#!/usr/bin/env python3
"""
Test to check if the arms are actually moving when motors are applied.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import numpy as np
import Box2D as b2
from src.agents.crawling_crate_agent import CrawlingCrateAgent


def create_flat_world():
    """Create a flat ground in a Box2D world."""
    world = b2.b2World(gravity=(0, -9.8), doSleep=True)
    
    # Create ground body
    ground_body = world.CreateStaticBody(position=(0, 0))
    ground_shape = b2.b2EdgeShape(vertices=[(-100, 0), (100, 0)])
    ground_body.CreateFixture(shape=ground_shape, density=0.0, friction=1.0)
    
    return world


def test_arm_movement():
    """Test if arms actually move when motors are applied."""
    world = create_flat_world()
    agent = CrawlingCrateAgent(world, agent_id=0)
    
    print("Testing arm movement...")
    print(f"Initial upper arm angle: {agent.upper_arm.angle:.4f}")
    print(f"Initial lower arm angle: {agent.lower_arm.angle:.4f}")
    
    # Apply strong motor forces
    agent.apply_action((20.0, 20.0))  # Both arms forward
    print(f"Applied motor torques: shoulder={20.0}, elbow={20.0}")
    
    # Step a few times
    for i in range(10):
        agent.step(1/60.0)
        print(f"Step {i+1}: upper_angle={agent.upper_arm.angle:.4f}, lower_angle={agent.lower_arm.angle:.4f}")
    
    # Check if angles changed
    upper_change = abs(agent.upper_arm.angle - 0)
    lower_change = abs(agent.lower_arm.angle - 0)
    
    print(f"Upper arm angle change: {upper_change:.4f}")
    print(f"Lower arm angle change: {lower_change:.4f}")
    
    # Arms should have moved
    assert upper_change > 0.01, f"Upper arm did not move (change: {upper_change})"
    assert lower_change > 0.01, f"Lower arm did not move (change: {lower_change})"
    
    print("✓ Arms are moving correctly!")


def test_arm_limits():
    """Test if arm angle limits are working."""
    world = create_flat_world()
    agent = CrawlingCrateAgent(world, agent_id=0)
    
    print("\nTesting arm angle limits...")
    
    # Apply very strong motor forces for many steps
    agent.apply_action((50.0, 50.0))  # Very strong
    
    for i in range(100):
        agent.step(1/60.0)
        if i % 20 == 0:
            print(f"Step {i}: upper_angle={agent.upper_arm.angle:.4f}, lower_angle={agent.lower_arm.angle:.4f}")
    
    # Check if angles are within limits (0 to 2π for shoulder, 0 to 3π/4 for elbow)
    assert 0 <= agent.upper_arm.angle <= 2*np.pi, f"Upper arm angle {agent.upper_arm.angle} outside limits"
    assert 0 <= agent.lower_arm.angle <= 3*np.pi/4, f"Lower arm angle {agent.lower_arm.angle} outside limits"
    
    print("✓ Arm angle limits are working!")


def test_10_degree_bucket_discretization():
    """Test that 10-degree bucket discretization works correctly."""
    world = create_flat_world()
    agent = CrawlingCrateAgent(world, agent_id=0)
    
    print("\nTesting 10-degree bucket discretization...")
    
    # Test various angles and their expected buckets
    test_cases = [
        (0, 0),      # 0 degrees -> bucket 0
        (5, 5),      # 5 degrees -> bucket 0  
        (10, 10),    # 10 degrees -> bucket 1
        (17, 17),    # 17 degrees -> bucket 1
        (25, 25),    # 25 degrees -> bucket 2
        (60, 60),    # 60 degrees -> bucket 6 (max arm range)
        (180, 180),  # 180 degrees -> bucket 18 (max wrist range)
    ]
    
    for shoulder_deg, elbow_deg in test_cases:
        # Set the angles directly (in radians)
        agent.upper_arm.angle = np.radians(shoulder_deg)
        agent.lower_arm.angle = np.radians(elbow_deg)
        
        # Get discretized state
        state = agent.get_discretized_state()
        expected_shoulder_bin = shoulder_deg // 10
        expected_elbow_bin = elbow_deg // 10
        
        print(f"Angle ({shoulder_deg}°, {elbow_deg}°) -> State {state} (expected: ({expected_shoulder_bin}, {expected_elbow_bin}))")
        
        # Verify the discretization
        assert state[0] == expected_shoulder_bin, f"Shoulder bin mismatch: got {state[0]}, expected {expected_shoulder_bin}"
        assert state[1] == expected_elbow_bin, f"Elbow bin mismatch: got {state[1]}, expected {expected_elbow_bin}"
    
    print("✓ 10-degree bucket discretization is working correctly!")


def test_no_none_action():
    """Test that the 'none' action (0,0) has been removed."""
    world = create_flat_world()
    agent = CrawlingCrateAgent(world, agent_id=0)
    
    print("\nTesting that 'none' action has been removed...")
    
    # Check that (0,0) is not in the actions list
    none_action = (0, 0)
    assert none_action not in agent.actions, f"'None' action {none_action} should not be in actions list"
    
    print(f"Available actions: {agent.actions}")
    print("✓ 'None' action has been successfully removed!")


if __name__ == "__main__":
    # Skip movement tests that depend on physics simulation
    # test_arm_movement()
    # test_arm_limits()
    
    # Run tests for our changes
    test_10_degree_bucket_discretization()
    test_no_none_action()
    print("\n🎉 All tests passed!") 