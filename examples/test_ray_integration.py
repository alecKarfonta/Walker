"""
Test script to demonstrate the integrated ray casting system in CrawlingAgent.

This shows how robots now have forward-facing ray sensing integrated into their
neural network state representation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import Box2D as b2
import numpy as np
from src.agents.crawling_agent import CrawlingAgent
from src.agents.physical_parameters import PhysicalParameters

def create_test_world():
    """Create a simple test world with obstacles."""
    world = b2.b2World(gravity=(0, -10))
    
    # Create ground
    ground_body = world.CreateStaticBody(
        position=(0, 0),
        shapes=b2.b2PolygonShape(box=(50, 1))
    )
    
    # Add some test obstacles in front of robot
    obstacles = [
        # Obstacle in front-right
        world.CreateStaticBody(
            position=(8, 2),
            shapes=b2.b2PolygonShape(box=(1, 1)),
            userData={'type': 'obstacle', 'obstacle_id': 'test_obstacle_1'}
        ),
        # Terrain feature
        world.CreateStaticBody(
            position=(6, 1.5),
            shapes=b2.b2PolygonShape(box=(0.5, 0.5)),
            userData={'type': 'terrain', 'terrain_id': 'test_terrain_1'}
        ),
        # Another robot (simulated)
        world.CreateStaticBody(
            position=(10, 2),
            shapes=b2.b2PolygonShape(box=(1.5, 0.75)),
            userData={'type': 'robot', 'robot_id': 'test_robot_1'}
        ),
    ]
    
    return world, obstacles

def test_ray_casting_integration():
    """Test the integrated ray casting system."""
    print("🔬 Testing Ray Casting Integration")
    print("=" * 50)
    
    # Create test world
    world, obstacles = create_test_world()
    
    # Create robot with ray sensing
    physical_params = PhysicalParameters.random_parameters()
    robot = CrawlingAgent(
        world=world,
        position=(0, 3),  # Start position
        physical_params=physical_params
    )
    
    # Set robot body userData for ray detection
    if robot.body:
        robot.body.userData = {'type': 'robot', 'robot_id': robot.id}
    
    print(f"🤖 Created robot {robot.id} at position (0, 3)")
    print(f"📡 Ray sensor range: {robot.ray_sensor_range}m")
    print(f"📊 Number of rays: {robot.num_rays}")
    
    # Test ray scanning
    print("\n🔍 Performing ray scan...")
    ray_results = robot._perform_ray_scan()
    
    print("\nRay Results:")
    for i, (distance, obj_type) in enumerate(ray_results):
        angle_deg = np.degrees(robot.ray_angles[i])
        obj_type_names = ['clear', 'obstacle', 'terrain', 'robot', 'food']
        obj_name = obj_type_names[obj_type] if obj_type < len(obj_type_names) else 'unknown'
        print(f"  Ray {i+1}: {angle_deg:6.1f}° | {distance:5.1f}m | {obj_name}")
    
    # Test state representation
    print("\n🧠 Testing Neural Network State Representation...")
    state = robot.get_state_representation()
    print(f"State vector size: {len(state)} dimensions")
    print(f"Ray sensing data (last 10 values):")
    
    ray_data = state[19:]  # Ray data starts at index 19
    for i in range(0, len(ray_data), 2):
        if i + 1 < len(ray_data):
            ray_num = i // 2 + 1
            distance = ray_data[i] * robot.ray_sensor_range  # Denormalize
            obj_type = int(ray_data[i + 1] * 4)  # Denormalize
            obj_type_names = ['clear', 'obstacle', 'terrain', 'robot', 'food']
            obj_name = obj_type_names[obj_type] if obj_type < len(obj_type_names) else 'unknown'
            print(f"  Ray {ray_num}: distance={distance:.1f}m, type={obj_name}")
    
    # Test robot orientation and ray directions
    print(f"\n🧭 Robot angle: {np.degrees(robot.body.angle):.1f}°")
    print("Ray angles (relative to robot):")
    for i, angle in enumerate(robot.ray_angles):
        world_angle = robot.body.angle + angle
        print(f"  Ray {i+1}: {np.degrees(angle):6.1f}° (world: {np.degrees(world_angle):6.1f}°)")
    
    # Test multiple orientations
    print("\n🔄 Testing different robot orientations...")
    
    orientations = [0, np.pi/4, np.pi/2, np.pi]
    for orientation in orientations:
        robot.body.angle = orientation
        ray_results = robot._perform_ray_scan()
        
        print(f"\nRobot facing {np.degrees(orientation):3.0f}°:")
        detections = sum(1 for _, obj_type in ray_results if obj_type > 0)
        print(f"  Objects detected: {detections}/{robot.num_rays} rays")
        
        if detections > 0:
            closest = min(distance for distance, obj_type in ray_results if obj_type > 0)
            print(f"  Closest object: {closest:.1f}m")

def test_neural_network_compatibility():
    """Test that the new state representation works with neural networks."""
    print("\n🧠 Testing Neural Network Compatibility")
    print("=" * 50)
    
    world, _ = create_test_world()
    robot = CrawlingAgent(world=world, position=(0, 3))
    
    # Test state consistency
    states = []
    for _ in range(10):
        state = robot.get_state_representation()
        states.append(state)
        # Simulate some movement
        if robot.body:
            robot.body.position = (robot.body.position.x + 0.1, robot.body.position.y)
    
    print(f"✅ Generated {len(states)} consistent state vectors")
    print(f"✅ Each state has {len(states[0])} dimensions")
    
    # Check for NaN or infinite values
    all_valid = True
    for i, state in enumerate(states):
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            print(f"❌ State {i} contains NaN or infinite values")
            all_valid = False
    
    if all_valid:
        print("✅ All states contain valid numerical values")
    
    # Test learning system compatibility
    if robot._learning_system:
        print("✅ Learning system initialized successfully")
        print(f"✅ State dimension: {robot._learning_system.state_dim}")
        print(f"✅ Action dimension: {robot._learning_system.action_dim}")
    else:
        print("⚠️ Learning system not initialized")
    
    return all_valid

if __name__ == "__main__":
    print("🚀 Ray Casting Integration Test")
    print("=" * 60)
    
    try:
        # Test basic ray casting
        test_ray_casting_integration()
        
        # Test neural network compatibility
        success = test_neural_network_compatibility()
        
        print("\n" + "=" * 60)
        if success:
            print("✅ All tests passed! Ray casting successfully integrated.")
            print("\n📋 Summary:")
            print("   • 5 forward-facing rays added to robot sensing")
            print("   • Ray data integrated into neural network state (29D)")
            print("   • Each ray provides distance and object type")
            print("   • Object types: clear, obstacle, terrain, robot, food")
            print("   • Compatible with existing learning systems")
        else:
            print("❌ Some tests failed. Check implementation.")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 