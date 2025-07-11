"""
Test script to verify that robots can detect obstacles with their ray sensors.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import Box2D as b2
import numpy as np
from src.agents.crawling_agent import CrawlingAgent
from src.agents.physical_parameters import PhysicalParameters

def create_test_world_with_obstacles():
    """Create a test world with obstacles in front of the robot."""
    world = b2.b2World(gravity=(0, -10))
    
    # Create ground
    ground_body = world.CreateStaticBody(
        position=(0, 0),
        shapes=b2.b2PolygonShape(box=(50, 1))
    )
    
    # Add obstacles directly in the robot's forward-right cone (ray angles: 0°, 22.5°, 45°, 67.5°, 90°)
    obstacles = []
    
    # Close obstacle at 2 meters - positioned to be hit by ray 1 (0°)
    obs1 = world.CreateStaticBody(
        position=(2, 3),
        shapes=b2.b2PolygonShape(box=(0.5, 1))
    )
    obs1.userData = {'type': 'obstacle'}
    obstacles.append(obs1)
    
    # Medium distance obstacle at 4 meters - positioned to be hit by ray 3 (45°)
    obs2 = world.CreateStaticBody(
        position=(3, 6),  # 45 degree angle from (0,3)
        shapes=b2.b2PolygonShape(box=(0.5, 1))
    )
    obs2.userData = {'type': 'obstacle'}
    obstacles.append(obs2)
    
    # Far obstacle at 6 meters - positioned to be hit by ray 5 (90°)
    obs3 = world.CreateStaticBody(
        position=(0, 8),  # 90 degree angle from (0,3)
        shapes=b2.b2PolygonShape(box=(0.5, 1))
    )
    obs3.userData = {'type': 'obstacle'}
    obstacles.append(obs3)
    
    return world, obstacles

def test_obstacle_detection():
    """Test that robots can detect obstacles with their ray sensors."""
    print("🧪 Testing Obstacle Detection with Ray Sensors")
    print("=" * 60)
    
    # Create world with obstacles
    world, obstacles = create_test_world_with_obstacles()
    
    # Create robot at origin
    robot = CrawlingAgent(
        world=world,
        agent_id=None,
        position=(0, 3),
        category_bits=0x0002,
        mask_bits=0x0001,
        physical_params=PhysicalParameters(
            num_arms=1,
            segments_per_limb=2
        )
    )
    
    print(f"🤖 Created robot at position (0, 3)")
    print(f"📊 Obstacles placed at: (2,3), (3,6), (0,8)")
    print()
    
    # Test ray casting
    print("🔍 Performing ray scan with obstacles...")
    ray_results = robot._perform_ray_scan()
    
    print("\nRay Results:")
    for i, result in enumerate(ray_results):
        distance = result[0]  # First element is distance
        obj_type = result[1]  # Second element is object type
        angle = robot.ray_angles[i] * 180 / np.pi
        
        status = "🔴 OBSTACLE" if obj_type == 1 else "🟢 clear"
        print(f"  Ray {i+1}: {angle:5.1f}° | {distance:5.1f}m | {status}")
    
    # Test state representation
    print("\n🧠 Testing Neural Network State with Obstacle Detection...")
    state = robot.get_state_representation()
    ray_data = state[19:]  # Last 10 values are ray data
    
    print(f"State vector size: {len(state)} dimensions")
    print("Ray sensing data (last 10 values):")
    for i in range(5):
        distance = ray_data[i*2]
        obj_type = ray_data[i*2 + 1]
        angle = robot.ray_angles[i] * 180 / np.pi
        
        distance_m = distance * 8.0  # Convert back to meters
        type_str = "obstacle" if obj_type == 1 else "clear"
        print(f"  Ray {i+1}: distance={distance_m:.1f}m, type={type_str}, angle={angle:.1f}°")
    
    # Count detected obstacles
    detected_obstacles = sum(1 for i in range(5) if ray_data[i*2 + 1] == 1)
    print(f"\n📊 Summary: {detected_obstacles}/5 rays detected obstacles")
    
    if detected_obstacles > 0:
        print("✅ SUCCESS: Robot can detect obstacles with ray sensors!")
    else:
        print("❌ ISSUE: Robot did not detect any obstacles")
    
    print("\n" + "=" * 60)
    return detected_obstacles > 0

if __name__ == "__main__":
    test_obstacle_detection() 