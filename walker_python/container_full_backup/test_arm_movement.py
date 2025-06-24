#!/usr/bin/env python3
"""
Simple test to verify arm movements are working.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import Box2D as b2
import numpy as np
import time
from src.agents.crawling_crate_agent import CrawlingCrateAgent

def test_arm_movement():
    """Test that arm movements are working correctly."""
    print("🔧 Testing arm movement...")
    
    # Create physics world
    world = b2.b2World(gravity=(0, -10), doSleep=True)
    
    # Create a simple ground
    ground_body = world.CreateStaticBody(position=(0, -1))
    ground_body.CreateFixture(
        shape=b2.b2PolygonShape(box=(50, 1)),
        density=0.0,
        friction=0.9
    )
    
    # Create test agent
    agent = CrawlingCrateAgent(
        world,
        agent_id=0,
        position=(0, 6)
    )
    
    print(f"Initial arm angles: shoulder={np.degrees(agent.upper_arm.angle):.1f}°, elbow={np.degrees(agent.lower_arm.angle):.1f}°")
    
    # Test different actions and see if arms move
    test_actions = [
        (1, 0),    # Shoulder forward
        (-1, 0),   # Shoulder backward
        (0, 1),    # Elbow flex
        (0, -1),   # Elbow extend
        (1, 1),    # Both forward/flex
        (-1, -1),  # Both backward/extend
    ]
    
    dt = 1.0 / 60.0  # 60 FPS
    
    for i, action in enumerate(test_actions):
        print(f"\n--- Test {i+1}: Action {action} ---")
        initial_shoulder = agent.upper_arm.angle
        initial_elbow = agent.lower_arm.angle
        
        # Apply action for multiple steps
        for step in range(60):  # 1 second at 60 FPS
            agent.apply_action(action)
            world.Step(dt, 8, 3)
        
        final_shoulder = agent.upper_arm.angle
        final_elbow = agent.lower_arm.angle
        
        shoulder_change = np.degrees(final_shoulder - initial_shoulder)
        elbow_change = np.degrees(final_elbow - initial_elbow)
        
        print(f"Shoulder angle change: {shoulder_change:.1f}° (from {np.degrees(initial_shoulder):.1f}° to {np.degrees(final_shoulder):.1f}°)")
        print(f"Elbow angle change: {elbow_change:.1f}° (from {np.degrees(initial_elbow):.1f}° to {np.degrees(final_elbow):.1f}°)")
        
        # Check if arms moved significantly
        if abs(shoulder_change) > 1.0 or abs(elbow_change) > 1.0:
            print("✅ Arms moved!")
        else:
            print("❌ Arms barely moved")
        
        # Check for joint limits
        shoulder_limit_hit = abs(agent.upper_arm.angle) >= np.pi/2 * 0.9
        elbow_limit_hit = (agent.lower_arm.angle <= 0.1 or agent.lower_arm.angle >= 3*np.pi/4 * 0.9)
        
        if shoulder_limit_hit:
            print("⚠️  Shoulder joint at limit")
        if elbow_limit_hit:
            print("⚠️  Elbow joint at limit")
    
    print("\n🔧 Arm movement test complete!")

if __name__ == "__main__":
    test_arm_movement() 