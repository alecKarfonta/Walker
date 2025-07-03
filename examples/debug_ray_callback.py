"""
Debug script to investigate ray casting callback issues.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import Box2D as b2
import numpy as np

def test_manual_ray_casting():
    """Test Box2D ray casting manually to debug the callback."""
    print("🔬 Manual Ray Casting Debug")
    print("=" * 50)
    
    # Create world
    world = b2.b2World(gravity=(0, -10))
    
    # Create obstacle
    obstacle = world.CreateStaticBody(
        position=(3, 3),
        shapes=b2.b2PolygonShape(box=(0.5, 2))
    )
    obstacle.userData = {'type': 'obstacle'}
    print(f"✅ Created obstacle at (3, 3)")
    print(f"   Box extends from x={3-0.5} to x={3+0.5}, y={3-2} to y={3+2}")
    
    # Test simple ray cast
    start_point = (0, 3)
    end_point = (8, 3)
    
    print(f"\n🎯 Testing ray from {start_point} to {end_point}")
    
    class DebugRayCallback(b2.b2RayCastCallback):
        def __init__(self):
            super().__init__()
            self.hits = []
            self.call_count = 0
            
        def ReportFixture(self, fixture, point, normal, fraction):
            self.call_count += 1
            print(f"  📍 Hit #{self.call_count}: fixture at point {point}, fraction {fraction}")
            print(f"      Body position: {fixture.body.position}")
            print(f"      User data: {fixture.body.userData}")
            
            self.hits.append({
                'point': point,
                'normal': normal,
                'fraction': fraction,
                'body_pos': fixture.body.position,
                'user_data': fixture.body.userData
            })
            
            return fraction  # Continue to find all hits
    
    # Cast the ray
    callback = DebugRayCallback()
    world.RayCast(callback, start_point, end_point)
    
    print(f"\n📊 Ray casting results:")
    print(f"   Total callbacks: {callback.call_count}")
    print(f"   Total hits: {len(callback.hits)}")
    
    if callback.hits:
        for i, hit in enumerate(callback.hits):
            distance = hit['fraction'] * 8.0  # 8m is the total ray length
            print(f"   Hit {i+1}: distance {distance:.2f}m, point {hit['point']}")
            print(f"            userData: {hit['user_data']}")
    else:
        print("   ❌ No hits detected!")
    
    # Test if we can find the body manually
    print(f"\n🔍 Manual body search:")
    print(f"   World has {len(list(world.bodies))} bodies:")
    for i, body in enumerate(world.bodies):
        print(f"   Body {i+1}: position {body.position}, userData: {body.userData}")
        if body.userData and isinstance(body.userData, dict):
            print(f"             type: {body.userData.get('type', 'unknown')}")
    
    return len(callback.hits) > 0

def test_robot_ray_exclusion():
    """Test if robot self-exclusion is working correctly."""
    print("\n🤖 Robot Self-Exclusion Test")
    print("=" * 50)
    
    # Import here to avoid circular imports during investigation
    from src.agents.crawling_agent import CrawlingAgent
    from src.agents.physical_parameters import PhysicalParameters
    
    # Create world
    world = b2.b2World(gravity=(0, -10))
    
    # Create obstacle
    obstacle = world.CreateStaticBody(
        position=(3, 3),
        shapes=b2.b2PolygonShape(box=(0.5, 2))
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
    print(f"✅ Created robot at (0, 3)")
    
    # List all bodies in world after robot creation
    print(f"\n🌍 World contents after robot creation:")
    print(f"   Total bodies: {len(list(world.bodies))}")
    for i, body in enumerate(world.bodies):
        body_type = "static" if body.type == b2.b2_staticBody else "dynamic"
        print(f"   Body {i+1}: {body_type} at {body.position}, userData: {body.userData}")
    
    # Test robot's ray casting with debugging
    print(f"\n🔍 Robot ray casting test:")
    robot_pos = (robot.body.position.x, robot.body.position.y)
    print(f"   Robot position: {robot_pos}")
    
    # Test ray 1 (0 degrees - straight right)
    ray_angle = 0.0
    distance, obj_type = robot._cast_ray(robot_pos, ray_angle)
    print(f"   Ray 0°: distance={distance:.2f}m, obj_type={obj_type}")
    
    return distance < 8.0  # Should hit something before max range

if __name__ == "__main__":
    print("🧪 Debugging Ray Casting Issues")
    print("=" * 60)
    
    # Test 1: Manual ray casting
    manual_works = test_manual_ray_casting()
    
    # Test 2: Robot ray casting
    robot_works = test_robot_ray_exclusion()
    
    print(f"\n📋 Debug Summary:")
    print(f"   Manual ray casting works: {manual_works}")
    print(f"   Robot ray casting works: {robot_works}")
    
    if manual_works and not robot_works:
        print("   🔍 Issue likely in robot ray casting logic")
    elif not manual_works:
        print("   🔍 Issue likely in basic Box2D setup")
    else:
        print("   ✅ Ray casting appears to be working") 