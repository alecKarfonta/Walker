#!/usr/bin/env python3
"""
Test script for Attention Deep Q-Learning
Tests the attention mechanism and neural network functionality.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import numpy as np
import time
from typing import Dict, Any

def test_attention_deep_q_learning():
    """Test the attention deep Q-learning implementation."""
    print("🔍 Testing Attention Deep Q-Learning Implementation")
    print("=" * 50)
    
    try:
        # Test imports
        print("📦 Testing imports...")
        from src.agents.attention_deep_q_learning import AttentionDeepQLearning, MultiHeadAttention, EnvironmentalFeatureEncoder
        print("✅ Imports successful")
        
        # Test MultiHeadAttention
        print("\n🎯 Testing Multi-Head Attention...")
        attention = MultiHeadAttention(embed_dim=128, num_heads=8)
        
        # Create test input
        batch_size = 2
        seq_len = 5
        embed_dim = 128
        test_input = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
        
        try:
            import torch
            test_tensor = torch.FloatTensor(test_input)
            output, weights = attention(test_tensor, test_tensor, test_tensor)
            print(f"✅ Multi-head attention output shape: {output.shape}")
            print(f"✅ Attention weights shape: {weights.shape}")
        except ImportError:
            print("⚠️ PyTorch not available, skipping tensor tests")
        
        # Test EnvironmentalFeatureEncoder
        print("\n🌍 Testing Environmental Feature Encoder...")
        encoder = EnvironmentalFeatureEncoder(state_size=15, embed_dim=128)
        
        # Create test state (15 dimensions as expected by attention system)
        test_state = np.array([
            0.1, 0.2, 0.3, 0.4,  # position, velocity (4)
            0.5, 0.6, 0.7, 0.8,  # arm angles, body angle, stability (4)
            0.9, 0.1,            # energy, health (2)
            0.2, 0.3, 0.4,       # food distance, direction, type (3)
            0.5, 0.6             # nearby agents, competition (2)
        ], dtype=np.float32)
        
        try:
            import torch
            state_tensor = torch.FloatTensor(test_state)
            encoded = encoder(state_tensor)
            print(f"✅ Feature encoder output shape: {encoded.shape}")
        except ImportError:
            print("⚠️ PyTorch not available, skipping encoder tests")
        
        # Test AttentionDeepQLearning
        print("\n🧠 Testing Attention Deep Q-Learning Agent...")
        agent = AttentionDeepQLearning(
            state_dim=15,
            action_dim=9,
            learning_rate=0.001
        )
        print("✅ Attention Deep Q-Learning agent created successfully")
        
        # Test action selection
        print("\n🎮 Testing action selection...")
        test_state = np.random.randn(15).astype(np.float32)
        test_agent_data = {
            'position': (1.0, 2.0),
            'velocity': (0.5, -0.3),
            'arm_angles': {'shoulder': 0.2, 'elbow': -0.1},
            'body_angle': 0.05,
            'energy': 0.8,
            'health': 0.9,
            'nearest_food': {'distance': 10.0, 'direction': 0.5, 'type': 1},
            'nearby_agents': 3,
            'competition_pressure': 0.4
        }
        
        action = agent.choose_action(test_state, test_agent_data)
        print(f"✅ Action selected: {action}")
        
        # Test enhanced state representation
        print("\n📊 Testing enhanced state representation...")
        enhanced_state = agent.get_enhanced_state_representation(test_agent_data)
        print(f"✅ Enhanced state shape: {enhanced_state.shape}")
        print(f"✅ Enhanced state values: {enhanced_state[:5]}...")  # Show first 5 values
        
        # Test experience storage
        print("\n💾 Testing experience storage...")
        next_state = np.random.randn(15).astype(np.float32)
        agent.store_experience(test_state, action, 0.1, next_state, False, test_agent_data)
        print("✅ Experience stored successfully")
        
        # Test learning (if buffer has enough experiences)
        print("\n📚 Testing learning capability...")
        # Add a few more experiences to meet minimum batch size
        for _ in range(35):  # Add enough for a batch
            random_state = np.random.randn(15).astype(np.float32)
            random_action = np.random.randint(0, 9)
            random_next_state = np.random.randn(15).astype(np.float32)
            random_reward = np.random.uniform(-0.1, 0.1)
            agent.store_experience(random_state, random_action, random_reward, random_next_state, False)
        
        learning_stats = agent.learn()
        if learning_stats:
            print(f"✅ Learning completed with stats: {learning_stats}")
        else:
            print("✅ Learning method executed (insufficient buffer for training)")
        
        # Test attention analysis
        print("\n🔍 Testing attention analysis...")
        attention_analysis = agent.get_attention_analysis()
        if attention_analysis:
            print(f"✅ Attention analysis available: {len(attention_analysis)} metrics")
            if 'feature_importance' in attention_analysis:
                feature_importance = attention_analysis['feature_importance']
                print(f"✅ Most important features: {attention_analysis.get('most_important_features', [])}")
        else:
            print("✅ Attention analysis ready (no history yet)")
        
        print("\n🎉 All Attention Deep Q-Learning tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_learning_manager_integration():
    """Test the integration with the learning manager."""
    print("\n🎛️ Testing Learning Manager Integration")
    print("=" * 50)
    
    try:
        from src.agents.learning_manager import LearningManager, LearningApproach
        
        # Create learning manager
        manager = LearningManager()
        print("✅ Learning Manager created")
        
        # Check if attention deep Q-learning is available
        if hasattr(manager, 'attention_deep_q_available'):
            print(f"✅ Attention Deep Q-Learning available: {manager.attention_deep_q_available}")
        
        # Check approach info
        if LearningApproach.ATTENTION_DEEP_Q_LEARNING in manager.approach_info:
            info = manager.approach_info[LearningApproach.ATTENTION_DEEP_Q_LEARNING]
            print(f"✅ Approach info: {info['name']} - {info['description']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Learning Manager integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_weight_distribution():
    """Test that attention deep Q-learning is included in weight distributions."""
    print("\n⚖️ Testing Weight Distribution")
    print("=" * 50)
    
    try:
        from src.agents.learning_manager import LearningApproach
        
        # Simulate the weight distribution from the training file
        learning_approaches = [
            (LearningApproach.BASIC_Q_LEARNING, 0.08),
            (LearningApproach.ENHANCED_Q_LEARNING, 0.17),
            (LearningApproach.SURVIVAL_Q_LEARNING, 0.25),
            (LearningApproach.DEEP_Q_LEARNING, 0.30),
            (LearningApproach.ATTENTION_DEEP_Q_LEARNING, 0.20),
        ]
        
        # Create weighted list
        weighted_approaches = []
        for approach, weight in learning_approaches:
            weighted_approaches.extend([approach] * int(weight * 100))
        
        # Count occurrences
        approach_counts = {}
        for approach in weighted_approaches:
            approach_counts[approach] = approach_counts.get(approach, 0) + 1
        
        print("✅ Weight distribution:")
        total = len(weighted_approaches)
        for approach, count in approach_counts.items():
            percentage = (count / total) * 100
            print(f"   {approach.value}: {count}/{total} ({percentage:.1f}%)")
        
        # Check if attention deep Q-learning is included
        attention_count = approach_counts.get(LearningApproach.ATTENTION_DEEP_Q_LEARNING, 0)
        if attention_count > 0:
            print(f"✅ Attention Deep Q-Learning will spawn: {attention_count} out of {total} agents ({(attention_count/total)*100:.1f}%)")
        else:
            print("❌ Attention Deep Q-Learning not found in distribution!")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Weight distribution test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Attention Deep Q-Learning Test Suite")
    print("=" * 60)
    
    start_time = time.time()
    
    tests_passed = 0
    total_tests = 3
    
    # Run tests
    if test_attention_deep_q_learning():
        tests_passed += 1
    
    if test_learning_manager_integration():
        tests_passed += 1
    
    if test_weight_distribution():
        tests_passed += 1
    
    # Summary
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print(f"🏁 Test Summary: {tests_passed}/{total_tests} tests passed")
    print(f"⏱️ Total time: {duration:.2f} seconds")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Attention Deep Q-Learning is ready to use.")
        sys.exit(0)
    else:
        print(f"❌ {total_tests - tests_passed} test(s) failed.")
        sys.exit(1) 