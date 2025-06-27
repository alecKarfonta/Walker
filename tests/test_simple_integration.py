#!/usr/bin/env python3
"""
Simplified test for attention deep Q-learning integration.
Tests state data extraction and formatting without requiring Box2D.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from unittest.mock import Mock
from src.agents.attention_deep_q_learning import AttentionDeepQLearning
from src.agents.learning_manager import LearningManager

def test_learning_manager_state_extraction():
    """Test the learning manager's state data extraction logic."""
    print("🧪 Testing learning manager state data extraction...")
    
    # Create mock training environment
    mock_training_env = Mock()
    mock_training_env._get_closest_food_distance_for_agent = Mock(return_value={
        'distance': 25.0,
        'food_position': (15.0, 10.0),
        'food_type': 'plant'
    })
    
    # Create mock agent with proper attributes
    mock_agent = Mock()
    mock_agent.body = Mock()
    mock_agent.body.position = Mock()
    mock_agent.body.position.x = 5.0
    mock_agent.body.position.y = 5.0
    mock_agent.body.linearVelocity = Mock()
    mock_agent.body.linearVelocity.x = 1.0
    mock_agent.body.linearVelocity.y = 0.5
    mock_agent.body.angle = 0.1
    mock_agent.body.contacts = []  # No contacts
    
    # Mock arm angles
    mock_agent.upper_arm = Mock()
    mock_agent.upper_arm.angle = 0.5  # 28.6 degrees
    mock_agent.upper_arm.contacts = []
    mock_agent.lower_arm = Mock()
    mock_agent.lower_arm.angle = 1.0  # 57.3 degrees
    mock_agent.lower_arm.contacts = []
    
    # Mock energy/health
    mock_agent.energy_level = 0.8
    mock_agent.health_level = 0.9
    
    # Create learning manager
    learning_manager = LearningManager()
    
    # Test state data extraction
    state_data = learning_manager._get_agent_state_data(mock_agent, mock_training_env)
    
    print("📊 Extracted state data:")
    for key, value in state_data.items():
        print(f"   {key}: {value}")
    
    # Verify all required keys are present
    required_keys = ['position', 'arm_angles', 'nearest_food', 'ground_contact']
    for key in required_keys:
        assert key in state_data, f"Missing required key: {key}"
    
    # Verify position data
    assert state_data['position'] == (5.0, 5.0), f"Position mismatch: {state_data['position']}"
    
    # Verify arm angles
    arm_angles = state_data['arm_angles']
    assert arm_angles['shoulder'] == 0.5, f"Shoulder angle mismatch: {arm_angles['shoulder']}"
    assert arm_angles['elbow'] == 1.0, f"Elbow angle mismatch: {arm_angles['elbow']}"
    
    # Verify food data extraction and direction calculation
    food_data = state_data['nearest_food']
    assert food_data['distance'] == 25.0, f"Food distance mismatch: {food_data['distance']}"
    
    # Check direction calculation
    # Agent at (5,5), food at (15,10)
    # Direction vector: (10, 5) normalized
    dx, dy = 10.0, 5.0
    distance = (dx**2 + dy**2)**0.5  # sqrt(125) ≈ 11.18
    expected_dir_x = dx / distance  # ≈ 0.894
    expected_dir_y = dy / distance  # ≈ 0.447
    
    assert abs(food_data['direction_x'] - expected_dir_x) < 0.01, f"Direction X mismatch: {food_data['direction_x']} vs {expected_dir_x}"
    assert abs(food_data['direction_y'] - expected_dir_y) < 0.01, f"Direction Y mismatch: {food_data['direction_y']} vs {expected_dir_y}"
    
    # Verify ground contact detection (should be False for mock with no contacts)
    assert state_data['ground_contact'] == False, f"Ground contact should be False for mock agent"
    
    print("✅ State data extraction working correctly!")
    return True

def test_attention_dql_state_representation():
    """Test the attention deep Q-learning state representation conversion."""
    print("\n🧪 Testing attention DQL state representation...")
    
    # Create attention deep Q-learning instance
    attention_dql = AttentionDeepQLearning(state_dim=5, action_dim=9)
    
    # Create test state data
    state_data = {
        'arm_angles': {
            'shoulder': 0.5,  # 28.6 degrees
            'elbow': 1.0      # 57.3 degrees
        },
        'nearest_food': {
            'distance': 25.0,
            'direction_x': 0.894,
            'direction_y': 0.447
        },
        'ground_contact': True,
        'physics_body': None  # Mock
    }
    
    # Test arm control state representation
    arm_state_vector = attention_dql.get_arm_control_state_representation(state_data)
    
    print(f"🎯 Arm control state vector (5 dimensions):")
    print(f"   [0] arm_angle: {arm_state_vector[0]:.4f}")
    print(f"   [1] elbow_angle: {arm_state_vector[1]:.4f}")
    print(f"   [2] food_distance: {arm_state_vector[2]:.4f}")
    print(f"   [3] food_dir_x: {arm_state_vector[3]:.4f}")
    print(f"   [4] ground_contact: {arm_state_vector[4]:.4f}")
    
    # Validate state vector format
    assert len(arm_state_vector) == 5, f"Expected 5 dimensions, got {len(arm_state_vector)}"
    assert not np.any(np.isnan(arm_state_vector)), "State vector contains NaN values"
    assert not np.any(np.isinf(arm_state_vector)), "State vector contains infinite values"
    
    # Validate arm angles (normalized to [-1, 1])
    expected_arm_angle = 0.5 / np.pi  # ≈ 0.159
    expected_elbow_angle = 1.0 / np.pi  # ≈ 0.318
    
    assert abs(arm_state_vector[0] - expected_arm_angle) < 0.01, f"Arm angle mismatch: {arm_state_vector[0]} vs {expected_arm_angle}"
    assert abs(arm_state_vector[1] - expected_elbow_angle) < 0.01, f"Elbow angle mismatch: {arm_state_vector[1]} vs {expected_elbow_angle}"
    
    # Validate food distance (normalized to [0, 1])
    expected_food_distance = min(1.0, 25.0 / 100.0)  # 0.25
    assert abs(arm_state_vector[2] - expected_food_distance) < 0.01, f"Food distance mismatch: {arm_state_vector[2]} vs {expected_food_distance}"
    
    # Validate food direction (already normalized)
    assert abs(arm_state_vector[3] - 0.894) < 0.01, f"Food dir_x mismatch: {arm_state_vector[3]} vs 0.894"
    # Note: We only check direction_x since direction_y goes to index 4, but we have ground_contact there
    
    # Validate ground contact (binary 0 or 1)
    assert arm_state_vector[4] == 1.0, f"Ground contact mismatch: {arm_state_vector[4]} vs 1.0"
    
    print("✅ State representation conversion working correctly!")
    return True

def test_action_selection():
    """Test that attention deep Q-learning can select actions."""
    print("\n🧪 Testing attention DQL action selection...")
    
    # Create attention deep Q-learning instance
    attention_dql = AttentionDeepQLearning(state_dim=5, action_dim=9)
    
    # Create test state vector
    state_vector = np.array([0.159, 0.318, 0.25, 0.894, 1.0], dtype=np.float32)
    
    # Test action selection
    action = attention_dql.choose_action(state_vector)
    
    print(f"🎯 Selected action: {action}")
    print(f"   Action type: {type(action)}")
    print(f"   Action range valid: {0 <= action < 9}")
    
    # Validate action
    assert isinstance(action, (int, np.integer)), f"Action should be integer, got {type(action)}"
    assert 0 <= action < 9, f"Action out of range: {action}"
    
    print("✅ Action selection working correctly!")
    return True

def test_learning_step():
    """Test that the learning step can process experiences."""
    print("\n🧪 Testing attention DQL learning step...")
    
    # Create attention deep Q-learning instance
    attention_dql = AttentionDeepQLearning(state_dim=5, action_dim=9)
    
    # Create test experience
    state = np.array([0.159, 0.318, 0.25, 0.894, 1.0], dtype=np.float32)
    action = 3
    reward = 0.1
    next_state = np.array([0.160, 0.320, 0.24, 0.890, 1.0], dtype=np.float32)
    done = False
    
    # Store experience
    attention_dql.store_experience(state, action, reward, next_state, done)
    
    # Try learning (should work even with minimal data)
    learning_stats = attention_dql.learn()
    
    print(f"📚 Learning stats: {learning_stats}")
    
    # Should return empty dict if not enough data, which is fine
    assert isinstance(learning_stats, dict), f"Learning stats should be dict, got {type(learning_stats)}"
    
    print("✅ Learning step working correctly!")
    return True

def test_attention_analysis():
    """Test attention analysis functionality."""
    print("\n🧪 Testing attention analysis...")
    
    # Create attention deep Q-learning instance
    attention_dql = AttentionDeepQLearning(state_dim=5, action_dim=9)
    
    # Test attention analysis (should work even without history)
    analysis = attention_dql.get_attention_analysis()
    
    print(f"🔍 Attention analysis: {analysis}")
    
    # Should return a dict with the expected structure
    assert isinstance(analysis, dict), f"Analysis should be dict, got {type(analysis)}"
    
    if analysis:  # If there's data
        expected_keys = ['feature_names', 'state_dimensions']
        for key in expected_keys:
            if key in analysis:
                print(f"   {key}: {analysis[key]}")
    
    print("✅ Attention analysis working correctly!")
    return True

def main():
    """Run all simplified integration tests."""
    print("🚀 Starting simplified attention deep Q-learning integration tests...\n")
    
    try:
        test_learning_manager_state_extraction()
        test_attention_dql_state_representation()
        test_action_selection()
        test_learning_step()
        test_attention_analysis()
        
        print("\n🎉 All simplified integration tests passed!")
        print("✅ Attention deep Q-learning integration verified:")
        print("   ✓ State data extraction from training environment")
        print("   ✓ 5-dimensional state representation:")
        print("     • arm_angle (normalized shoulder angle)")
        print("     • elbow_angle (normalized elbow angle)")
        print("     • food_distance (normalized distance to food)")
        print("     • food_direction_x (unit vector X component)")
        print("     • ground_contact (binary contact detection)")
        print("   ✓ Action selection using attention mechanisms")
        print("   ✓ Experience storage and learning")
        print("   ✓ Attention pattern analysis")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 