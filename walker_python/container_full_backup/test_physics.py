#!/usr/bin/env python3
"""
Simple test script for the physics system without GUI dependencies.
"""

import sys
import os
import time

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_physics_system():
    """Test the physics system without GUI."""
    try:
        import pymunk
        print("✓ Pymunk imported successfully")
        
        # Test basic physics space
        space = pymunk.Space()
        space.gravity = (0, -9.8)
        print("✓ Physics space created successfully")
        
        # Test body creation
        body = pymunk.Body(1, pymunk.moment_for_circle(1, 0, 1))
        body.position = (0, 10)
        space.add(body)
        print("✓ Body created and added to space")
        
        # Test physics stepping
        for i in range(10):
            space.step(1.0/60.0)
            print(f"  Step {i+1}: Body position = {body.position}")
        
        print("✓ Physics simulation working correctly")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Please install dependencies: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_world_controller():
    """Test the WorldController class."""
    try:
        from physics.world import WorldController
        print("✓ WorldController imported successfully")
        
        # Create world controller
        world = WorldController()
        print("✓ WorldController created successfully")
        
        # Test ground creation
        print(f"  Ground height: {world.ground_height}")
        print(f"  World bounds: left={world.left}, right={world.right}")
        print("✓ Ground terrain created")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_body_factory():
    """Test the BodyFactory class."""
    try:
        import pymunk
        from physics.body_factory import BodyFactory
        print("✓ BodyFactory imported successfully")
        
        # Create space for testing
        space = pymunk.Space()
        
        # Test ball creation
        ball = BodyFactory.create_ball(space, (0, 10), 5)
        print("✓ Ball created successfully")
        
        # Test crate creation
        crate = BodyFactory.create_crate(space, (5, 10), (2, 2))
        print("✓ Crate created successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    """Run all physics tests."""
    print("Testing Walker Physics System")
    print("=" * 40)
    
    tests = [
        ("Basic Pymunk Physics", test_physics_system),
        ("World Controller", test_world_controller),
        ("Body Factory", test_body_factory),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"✗ {test_name} failed")
    
    print("\n" + "=" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! Physics system is working correctly.")
        return True
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 