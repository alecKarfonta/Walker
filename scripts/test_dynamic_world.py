#!/usr/bin/env python3
"""
Test script for the Dynamic World Generation system.

This script tests the basic functionality of the dynamic world manager
without running the full training environment.
"""

import time
import random
import Box2D as b2
from src.world.dynamic_world_manager import DynamicWorldManager
from src.ecosystem_dynamics import EcosystemDynamics

def test_dynamic_world():
    """Test the dynamic world generation system."""
    print("🧪 Testing Dynamic World Generation System")
    print("=" * 50)
    
    # Create a simple Box2D world
    world = b2.b2World(gravity=(0, -9.8))
    print("🌍 Created Box2D world")
    
    # Create ecosystem dynamics
    ecosystem = EcosystemDynamics()
    print("🌿 Created ecosystem dynamics")
    
    # Create dynamic world manager
    try:
        dynamic_world = DynamicWorldManager(world, ecosystem)
        print("✅ Dynamic World Manager created successfully!")
        
        # Print initial status
        status = dynamic_world.get_world_status()
        print(f"\n📊 Initial Status:")
        print(f"   Active tiles: {status['active_tiles']}")
        print(f"   World span: {status['world_span']:.1f}m")
        print(f"   Left wall: {status['left_wall_position']:.1f}m")
        print(f"   Right edge: {status['current_right_edge']:.1f}m")
        print(f"   Biomes: {status['biome_distribution']}")
        
        # Simulate robot positions progressing to the right
        print(f"\n🤖 Simulating robot progression...")
        
        robot_positions = [
            ("robot_1", (0.0, 5.0)),
            ("robot_2", (10.0, 5.0)),
            ("robot_3", (-5.0, 5.0))
        ]
        
        for step in range(10):
            print(f"\n--- Step {step + 1} ---")
            
            # Move robots gradually to the right
            for i, (robot_id, pos) in enumerate(robot_positions):
                new_x = pos[0] + random.uniform(5, 15)  # Move 5-15m right
                new_y = pos[1] + random.uniform(-2, 2)   # Small vertical movement
                robot_positions[i] = (robot_id, (new_x, new_y))
            
            # Find rightmost robot
            rightmost_x = max(pos[1][0] for _, pos in robot_positions)
            print(f"Rightmost robot at x={rightmost_x:.1f}m")
            
            # Update dynamic world
            dynamic_world.update(robot_positions)
            
            # Get updated status
            status = dynamic_world.get_world_status()
            print(f"Active tiles: {status['active_tiles']}, "
                  f"Generated: {status['tiles_generated']}, "
                  f"Cleaned: {status['tiles_cleaned_up']}")
            
            if status['tiles_generated'] > 0:
                print(f"Biomes: {status['biome_distribution']}")
                print(f"Food sources: {status['total_food_sources']}")
                print(f"Obstacles: {status['total_obstacles']}")
            
            # Step physics simulation
            world.Step(1.0/60.0, 6, 2)
            
            time.sleep(0.5)  # Brief pause for readability
        
        # Test force generation
        print(f"\n🔬 Testing force generation...")
        new_tile = dynamic_world.force_generate_tile()
        if new_tile:
            print(f"✅ Force generated tile #{new_tile.id} ({new_tile.biome.value})")
        
        # Final status
        final_status = dynamic_world.get_world_status()
        print(f"\n📊 Final Status:")
        print(f"   Active tiles: {final_status['active_tiles']}")
        print(f"   Total generated: {final_status['tiles_generated']}")
        print(f"   Total cleaned: {final_status['tiles_cleaned_up']}")
        print(f"   World span: {final_status['world_span']:.1f}m")
        print(f"   Biome distribution: {final_status['biome_distribution']}")
        print(f"   Total food sources: {final_status['total_food_sources']}")
        print(f"   Total obstacles: {final_status['total_obstacles']}")
        
        # Test cleanup
        print(f"\n🧹 Testing cleanup...")
        dynamic_world.cleanup_all_tiles()
        
        cleanup_status = dynamic_world.get_world_status()
        print(f"✅ After cleanup: {cleanup_status['active_tiles']} active tiles")
        
        print(f"\n🎉 Dynamic World Generation test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_dynamic_world()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Tests failed!")
        exit(1) 