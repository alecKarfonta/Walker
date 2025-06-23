#!/usr/bin/env python3
"""
Test to check if the arms are actually moving when motors are applied.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import pymunk
import numpy as np
from src.agents.crawling_crate import CrawlingCrate


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


def test_arm_movement():
    """Test if arms actually move when motors are applied."""
    space = create_flat_world()
    agent = CrawlingCrate(space)
    
    print("Testing arm movement...")
    print(f"Initial left arm angle: {agent.left_arm.angle:.4f}")
    print(f"Initial right arm angle: {agent.right_arm.angle:.4f}")
    
    # Apply strong motor forces
    agent.apply_action((20.0, 20.0))  # Both arms forward
    print(f"Applied motor rates: left={agent.left_motor.rate}, right={agent.right_motor.rate}")
    
    # Step a few times
    for i in range(10):
        agent.step(1/60.0)
        print(f"Step {i+1}: left_angle={agent.left_arm.angle:.4f}, right_angle={agent.right_arm.angle:.4f}")
    
    # Check if angles changed
    left_change = abs(agent.left_arm.angle - 0)
    right_change = abs(agent.right_arm.angle - 0)
    
    print(f"Left arm angle change: {left_change:.4f}")
    print(f"Right arm angle change: {right_change:.4f}")
    
    # Arms should have moved
    assert left_change > 0.01, f"Left arm did not move (change: {left_change})"
    assert right_change > 0.01, f"Right arm did not move (change: {right_change})"
    
    print("✓ Arms are moving correctly!")


def test_arm_limits():
    """Test if arm angle limits are working."""
    space = create_flat_world()
    agent = CrawlingCrate(space)
    
    print("\nTesting arm angle limits...")
    
    # Apply very strong motor forces for many steps
    agent.apply_action((50.0, 50.0))  # Very strong
    
    for i in range(100):
        agent.step(1/60.0)
        if i % 20 == 0:
            print(f"Step {i}: left_angle={agent.left_arm.angle:.4f}, right_angle={agent.right_arm.angle:.4f}")
    
    # Check if angles are within limits (-π/2 to π/2)
    assert -np.pi/2 <= agent.left_arm.angle <= np.pi/2, f"Left arm angle {agent.left_arm.angle} outside limits"
    assert -np.pi/2 <= agent.right_arm.angle <= np.pi/2, f"Right arm angle {agent.right_arm.angle} outside limits"
    
    print("✓ Arm angle limits are working!")


if __name__ == "__main__":
    test_arm_movement()
    test_arm_limits()
    print("\nAll tests passed!") 