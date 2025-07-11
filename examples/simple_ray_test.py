"""
Simple test to verify ray casting works by placing obstacles directly in ray paths.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import Box2D as b2
import numpy as np
from src.agents.crawling_agent import CrawlingAgent
from src.agents.physical_parameters import PhysicalParameters

def test_simple_ray_casting():
    """Test ray casting with a single obstacle directly in front."""
    print("🔬 Simple Ray Casting Test")
    print("=" * 50)
    
    # Create world
    world = b2.b2World(gravity=(0, -10))
    
    # Create ground (should be detected as terrain)
    ground = world.CreateStaticBody(
        position=(0, 0),
        shapes=b2.b2PolygonShape(box=(20, 1))
    )
    print(f"✅ Created ground at y=0")
    
    # Create simple obstacle directly east of robot
    obstacle = world.CreateStaticBody(
        position=(3, 3),  # Same height as robot, 3 meters to the right
        shapes=b2.b2PolygonShape(box=(0.5, 2))  # Tall obstacle
    )
    obstacle.userData = {'type': 'obstacle'}
    print(f"✅ Created obstacle at (3, 3)")
    
    # Create robot
    robot = CrawlingAgent(
        world=world,
        agent_id=None,
        position=(0, 3),
        physical_params=PhysicalParameters(num_arms=1, segments_per_limb=2)
    )
    print(f"🤖 Created robot at (0, 3)")
    
    # Print robot orientation
    robot_angle = robot.body.angle if robot.body else 0.0
    print(f"🧭 Robot angle: {robot_angle * 180 / np.pi:.1f}°")
    
    # Print ray configuration
    print(f"📡 Ray configuration:")
    for i, angle in enumerate(robot.ray_angles):
        angle_deg = angle * 180 / np.pi
        print(f"  Ray {i+1}: {angle_deg:6.1f}° relative to robot")
    
    # Manually test each ray direction
    print(f"\n🔍 Manual ray testing:")
    robot_pos = (robot.body.position.x, robot.body.position.y) if robot.body else (0, 3)
    
    for i, ray_angle in enumerate(robot.ray_angles):
        world_angle = robot_angle + ray_angle
        distance, obj_type = robot._cast_ray(robot_pos, world_angle)
        
        type_str = "clear" if obj_type == 0 else ("obstacle" if obj_type == 1 else "terrain")
        status = "🔴" if obj_type > 0 else "🟢"
        
        print(f"  Ray {i+1}: {world_angle*180/np.pi:6.1f}° | {distance:5.1f}m | {status} {type_str}")
    
    # Test complete ray scan
    print(f"\n🔍 Complete ray scan:")
    ray_results = robot._perform_ray_scan()
    
    for i, (distance, obj_type) in enumerate(ray_results):
        angle_deg = robot.ray_angles[i] * 180 / np.pi
        type_str = "clear" if obj_type == 0 else ("obstacle" if obj_type == 1 else "terrain")
        status = "🔴" if obj_type > 0 else "🟢"
        
        print(f"  Ray {i+1}: {angle_deg:6.1f}° | {distance:5.1f}m | {status} {type_str}")
    
    # Test neural network state
    print(f"\n🧠 Neural network state:")
    state = robot.get_state_representation()
    print(f"State size: {len(state)}")
    print(f"Ray data (indices 19-28):")
    for i in range(5):
        base_idx = 19 + (i * 2)
        distance_norm = state[base_idx]
        obj_type_norm = state[base_idx + 1]
        
        distance_m = distance_norm * 8.0
        obj_type = int(obj_type_norm * 2.0 + 0.5)  # Convert back from normalized
        
        type_str = "clear" if obj_type == 0 else ("obstacle" if obj_type == 1 else "terrain")
        print(f"  Ray {i+1}: {distance_m:.1f}m, {type_str}")
    
    # Count detections
    detected = sum(1 for _, obj_type in ray_results if obj_type > 0)
    print(f"\n📊 Results: {detected}/5 rays detected objects")
    
    if detected > 0:
        print("✅ SUCCESS: Ray casting is working!")
    else:
        print("❌ FAILURE: No objects detected")
    
    return detected > 0

if __name__ == "__main__":
    test_simple_ray_casting() 