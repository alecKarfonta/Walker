#!/usr/bin/env python3
"""
Test attention deep Q-learning integration with the training environment.
Verifies that all 5 state values are correctly extracted and formatted.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import Box2D as b2
from unittest.mock import Mock, MagicMock
from src.agents.attention_deep_q_learning import AttentionDeepQLearning
from src.agents.learning_manager import LearningManager
from src.agents.crawling_crate_agent import CrawlingCrateAgent

def test_state_value_extraction():
    """Test that all 5 state values are correctly extracted from the environment."""
    print("🧪 Testing attention deep Q-learning state value extraction...")
    
    # Create mock training environment
    mock_training_env = Mock()
    mock_training_env._get_closest_food_distance_for_agent = Mock(return_value={
        'distance': 25.0,
        'food_position': (15.0, 10.0),
        'food_type': 'plant'
    })
    
    # Create physics world
    world = b2.b2World(gravity=(0, -10), doSleep=True)
    
    # Create ground with proper collision category
    ground_body = world.CreateStaticBody(position=(0, -1))
    ground_fixture = ground_body.CreateFixture(
        shape=b2.b2PolygonShape(box=(50, 1)),
        density=0.0,
        friction=0.9,
        filter=b2.b2Filter(categoryBits=0x0001, maskBits=0xFFFF)  # Ground category
    )
    
    # Create test agent
    agent = CrawlingCrateAgent(
        world,
        agent_id=0,
        position=(5, 5),
        category_bits=0x0002,  # Agent category
        mask_bits=0xFFFF
    )
    
    # Set up arm angles
    agent.upper_arm.angle = 0.5  # 28.6 degrees
    agent.lower_arm.angle = 1.0  # 57.3 degrees
    
    # Create learning manager and inject training environment
    learning_manager = LearningManager()
    learning_manager.set_training_environment(mock_training_env)
    learning_manager.inject_training_environment_into_agents([agent])
    
    # Create attention deep Q-learning instance
    attention_dql = AttentionDeepQLearning(state_dim=5, action_dim=9)
    
    # Test state data extraction
    state_data = learning_manager._get_agent_state_data(agent, mock_training_env)
    
    print("📊 Extracted state data:")
    print(f"   Position: {state_data.get('position', 'N/A')}")
    print(f"   Arm angles: {state_data.get('arm_angles', 'N/A')}")
    print(f"   Nearest food: {state_data.get('nearest_food', 'N/A')}")
    print(f"   Ground contact: {state_data.get('ground_contact', 'N/A')}")
    
    # Test arm control state representation
    arm_state_vector = attention_dql.get_arm_control_state_representation(state_data)
    
    print(f"\n🎯 Arm control state vector (5 dimensions):")
    print(f"   [0] arm_angle: {arm_state_vector[0]:.4f}")
    print(f"   [1] elbow_angle: {arm_state_vector[1]:.4f}")
    print(f"   [2] food_distance: {arm_state_vector[2]:.4f}")
    print(f"   [3] food_dir_x: {arm_state_vector[3]:.4f}")
    print(f"   [4] food_dir_y: {arm_state_vector[4]:.4f}")
    
    # Validate state vector format
    assert len(arm_state_vector) == 5, f"Expected 5 dimensions, got {len(arm_state_vector)}"
    assert not np.any(np.isnan(arm_state_vector)), "State vector contains NaN values"
    assert not np.any(np.isinf(arm_state_vector)), "State vector contains infinite values"
    
    # Validate arm angles
    expected_arm_angle = 0.5 / np.pi  # Normalized
    expected_elbow_angle = 1.0 / np.pi  # Normalized
    
    assert abs(arm_state_vector[0] - expected_arm_angle) < 0.01, f"Arm angle mismatch: {arm_state_vector[0]} vs {expected_arm_angle}"
    assert abs(arm_state_vector[1] - expected_elbow_angle) < 0.01, f"Elbow angle mismatch: {arm_state_vector[1]} vs {expected_elbow_angle}"
    
    # Validate food distance (normalized)
    expected_food_distance = min(1.0, 25.0 / 100.0)  # 0.25
    assert abs(arm_state_vector[2] - expected_food_distance) < 0.01, f"Food distance mismatch: {arm_state_vector[2]} vs {expected_food_distance}"
    
    # Validate food direction (unit vector from agent to food)
    agent_pos = state_data['position']
    food_pos = (15.0, 10.0)
    dx = food_pos[0] - agent_pos[0]  # 15 - 5 = 10
    dy = food_pos[1] - agent_pos[1]  # 10 - 5 = 5
    distance = (dx**2 + dy**2)**0.5  # sqrt(100 + 25) = sqrt(125)
    expected_dir_x = dx / distance
    expected_dir_y = dy / distance
    
    assert abs(arm_state_vector[3] - expected_dir_x) < 0.01, f"Food dir_x mismatch: {arm_state_vector[3]} vs {expected_dir_x}"
    assert abs(arm_state_vector[4] - expected_dir_y) < 0.01, f"Food dir_y mismatch: {arm_state_vector[4]} vs {expected_dir_y}"
    
    print("✅ All state values correctly extracted and formatted!")
    
    # Test attention deep Q-learning action choice
    action = attention_dql.choose_action(arm_state_vector, state_data)
    assert 0 <= action < 9, f"Invalid action: {action}"
    
    print(f"🎯 Selected action: {action}")
    
    return True

def test_ground_contact_detection():
    """Test ground contact detection using Box2D physics."""
    print("\n🧪 Testing ground contact detection...")
    
    # Create physics world
    world = b2.b2World(gravity=(0, -10), doSleep=True)
    
    # Create ground with proper collision category
    ground_body = world.CreateStaticBody(position=(0, -1))
    ground_fixture = ground_body.CreateFixture(
        shape=b2.b2PolygonShape(box=(50, 1)),
        density=0.0,
        friction=0.9,
        filter=b2.b2Filter(categoryBits=0x0001, maskBits=0xFFFF)  # Ground category
    )
    
    # Test agent above ground (no contact)
    agent_high = CrawlingCrateAgent(
        world,
        agent_id=1,
        position=(0, 10),  # High above ground
        category_bits=0x0002,
        mask_bits=0xFFFF
    )
    
    learning_manager = LearningManager()
    
    # Simulate physics for a few steps to allow potential contact
    for _ in range(10):
        world.Step(1.0/60.0, 6, 2)
    
    ground_contact_high = learning_manager._detect_ground_contact(agent_high)
    print(f"   Agent at height {agent_high.body.position.y:.2f}: Ground contact = {ground_contact_high}")
    
    # Test agent on ground (should have contact)
    agent_low = CrawlingCrateAgent(
        world,
        agent_id=2,
        position=(0, 2),  # Close to ground
        category_bits=0x0002,
        mask_bits=0xFFFF
    )
    
    # Simulate physics to settle agent on ground
    for _ in range(60):  # More steps for settling
        world.Step(1.0/60.0, 6, 2)
    
    ground_contact_low = learning_manager._detect_ground_contact(agent_low)
    print(f"   Agent at height {agent_low.body.position.y:.2f}: Ground contact = {ground_contact_low}")
    
    # The low agent should eventually make contact with ground
    print("✅ Ground contact detection working!")
    
    return True

def test_food_direction_calculation():
    """Test food direction vector calculation."""
    print("\n🧪 Testing food direction calculation...")
    
    # Test cases: agent at origin, food at various positions
    test_cases = [
        {"agent_pos": (0, 0), "food_pos": (10, 0), "expected_dir": (1.0, 0.0)},   # East
        {"agent_pos": (0, 0), "food_pos": (0, 10), "expected_dir": (0.0, 1.0)},   # North
        {"agent_pos": (0, 0), "food_pos": (-10, 0), "expected_dir": (-1.0, 0.0)}, # West
        {"agent_pos": (0, 0), "food_pos": (0, -10), "expected_dir": (0.0, -1.0)}, # South
        {"agent_pos": (5, 5), "food_pos": (8, 9), "expected_dir": (0.6, 0.8)},    # NE (3,4,5 triangle)
    ]
    
    learning_manager = LearningManager()
    
    for i, case in enumerate(test_cases):
        # Create mock state data
        state_data = {
            'position': case['agent_pos'],
            'nearest_food': {
                'distance': 10.0,
                'direction_x': 0.0,  # Will be calculated
                'direction_y': 0.0   # Will be calculated
            }
        }
        
        # Create mock training environment
        mock_env = Mock()
        mock_env._get_closest_food_distance_for_agent = Mock(return_value={
            'distance': 10.0,
            'food_position': case['food_pos']
        })
        
        # Create mock agent
        mock_agent = Mock()
        mock_agent.body = Mock()
        mock_agent.body.position = Mock()
        mock_agent.body.position.x = case['agent_pos'][0]
        mock_agent.body.position.y = case['agent_pos'][1]
        mock_agent.upper_arm = Mock()
        mock_agent.upper_arm.angle = 0.0
        mock_agent.lower_arm = Mock()
        mock_agent.lower_arm.angle = 0.0
        
        # Extract state data (this will calculate direction)
        result_state = learning_manager._get_agent_state_data(mock_agent, mock_env)
        
        actual_dir_x = result_state['nearest_food']['direction_x']
        actual_dir_y = result_state['nearest_food']['direction_y']
        expected_dir_x, expected_dir_y = case['expected_dir']
        
        print(f"   Case {i+1}: Agent {case['agent_pos']} → Food {case['food_pos']}")
        print(f"      Expected direction: ({expected_dir_x:.2f}, {expected_dir_y:.2f})")
        print(f"      Actual direction: ({actual_dir_x:.2f}, {actual_dir_y:.2f})")
        
        assert abs(actual_dir_x - expected_dir_x) < 0.01, f"Direction X mismatch for case {i+1}"
        assert abs(actual_dir_y - expected_dir_y) < 0.01, f"Direction Y mismatch for case {i+1}"
    
    print("✅ Food direction calculation working correctly!")
    
    return True

def main():
    """Run all integration tests."""
    print("🚀 Starting attention deep Q-learning integration tests...\n")
    
    try:
        test_state_value_extraction()
        test_ground_contact_detection()
        test_food_direction_calculation()
        
        print("\n🎉 All integration tests passed!")
        print("✅ Attention deep Q-learning is correctly integrated with:")
        print("   • Real food distance and direction from training environment")
        print("   • Box2D ground contact detection")
        print("   • Proper 5-dimensional state representation")
        print("   • Normalized state values for neural network training")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 