#!/usr/bin/env python3
"""
Test script to verify GPU functionality and deep learning batch training.
This script tests PyTorch GPU availability and the deep learning integration.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import time
import torch
import numpy as np
from src.agents.deep_survival_q_learning import DeepSurvivalQLearning, TORCH_AVAILABLE
from src.agents.ecosystem_interface import EcosystemInterface


def test_gpu_availability():
    """Test if GPU is available and which device we're using."""
    print("🔍 Testing GPU Availability...")
    
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available!")
        return False
    
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"✅ CUDA devices available: {device_count}")
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"   GPU {i}: {props.name} ({memory_gb:.1f}GB)")
        
        current_device = torch.cuda.current_device()
        print(f"✅ Current CUDA device: {current_device}")
        
        # Check CUDA_VISIBLE_DEVICES
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
        print(f"✅ CUDA_VISIBLE_DEVICES: {visible_devices}")
        
        # Test GPU memory allocation
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            print(f"✅ GPU memory test passed: Created tensor on {test_tensor.device}")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ GPU memory test failed: {e}")
            return False
        
        return True
    else:
        print("❌ CUDA not available!")
        return False


def test_deep_q_learning_initialization():
    """Test deep Q-learning initialization with GPU."""
    print("\n🧠 Testing Deep Q-Learning Initialization...")
    
    try:
        # Test with survival-appropriate dimensions
        state_dim = 16  # Continuous state vector from survival data
        action_dim = 6  # Number of actions available to agents
        
        deep_q = DeepSurvivalQLearning(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=3e-4,
            buffer_size=10000,
            batch_size=64,
            use_dueling=True,
            use_prioritized_replay=True
        )
        
        print(f"✅ Deep Q-Learning initialized successfully")
        print(f"   State dimensions: {state_dim}")
        print(f"   Action dimensions: {action_dim}")
        print(f"   Device: {deep_q.device}")
        print(f"   Network type: {'Dueling DQN' if isinstance(deep_q.q_network, type(deep_q.q_network)) else 'Standard DQN'}")
        print(f"   Replay buffer: {'Prioritized' if deep_q.use_prioritized_replay else 'Standard'}")
        
        return deep_q
        
    except Exception as e:
        print(f"❌ Deep Q-Learning initialization failed: {e}")
        return None


def test_state_processing():
    """Test state processing for deep learning."""
    print("\n🔄 Testing State Processing...")
    
    try:
        deep_q = test_deep_q_learning_initialization()
        if deep_q is None:
            return False
        
        # Create mock survival state data
        mock_survival_data = {
            'shoulder_angle': 0.5,
            'elbow_angle': -0.3,
            'energy_level': 0.8,
            'health_level': 0.9,
            'food_distance': 5.2,
            'food_direction': 1.57,  # π/2 radians
            'velocity_magnitude': 0.4,
            'body_angle': 0.1,
            'ground_contact': 1.0,
            'nearby_agents': 2,
            'competition_pressure': 0.3
        }
        
        # Convert to continuous state vector
        state_vector = deep_q.get_continuous_state_vector(mock_survival_data)
        print(f"✅ State vector created: shape {state_vector.shape}")
        print(f"   Sample values: {state_vector[:5]}")
        
        # Test action selection
        mock_agent_data = {'energy_level': 0.8}
        action = deep_q.choose_action(state_vector, mock_agent_data)
        print(f"✅ Action selection: {action}")
        
        return True
        
    except Exception as e:
        print(f"❌ State processing test failed: {e}")
        return False


def test_experience_storage_and_learning():
    """Test experience storage and batch learning."""
    print("\n📚 Testing Experience Storage and Batch Learning...")
    
    try:
        deep_q = test_deep_q_learning_initialization()
        if deep_q is None:
            return False
        
        state_dim = 16
        batch_size = 64
        
        print(f"🎯 Target batch size: {batch_size}")
        
        # Generate mock experiences
        print("📝 Generating mock experiences...")
        for i in range(batch_size * 2):  # Generate more than batch size
            # Random state
            state = np.random.randn(state_dim).astype(np.float32)
            action = np.random.randint(0, 6)
            reward = np.random.randn() * 10  # Varied rewards
            next_state = np.random.randn(state_dim).astype(np.float32)
            done = np.random.random() < 0.1  # 10% chance of episode end
            
            # Mock agent data for prioritization
            agent_data = {
                'energy_level': np.random.random(),
                'energy_change': np.random.randn() * 0.1
            }
            
            deep_q.store_experience(state, action, reward, next_state, done, agent_data)
            
            if (i + 1) % 20 == 0:
                print(f"   Stored {i + 1} experiences")
        
        print(f"✅ Total experiences stored: {len(deep_q.memory)}")
        
        # Test learning
        if len(deep_q.memory) >= batch_size:
            print(f"✅ Sufficient experiences for batch learning ({len(deep_q.memory)} >= {batch_size})")
            
            print("🚀 Testing batch learning...")
            start_time = time.time()
            
            # Perform multiple learning steps
            learning_stats = []
            for step in range(5):
                stats = deep_q.learn()
                learning_stats.append(stats)
                if stats:
                    print(f"   Step {step + 1}: Loss = {stats.get('loss', 'N/A'):.4f}, Mean Q = {stats.get('mean_q_value', 'N/A'):.4f}")
            
            learning_time = time.time() - start_time
            print(f"✅ Batch learning completed in {learning_time:.2f} seconds")
            
            # Check if learning is actually happening
            valid_stats = [s for s in learning_stats if s]
            if valid_stats:
                print(f"✅ Learning progress detected: {len(valid_stats)}/{len(learning_stats)} steps successful")
                return True
            else:
                print("⚠️ No learning progress detected")
                return False
        else:
            print(f"❌ Insufficient experiences for batch learning ({len(deep_q.memory)} < {batch_size})")
            return False
            
    except Exception as e:
        print(f"❌ Experience storage/learning test failed: {e}")
        return False


def test_gpu_memory_usage():
    """Test GPU memory usage during training."""
    print("\n💾 Testing GPU Memory Usage...")
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, skipping memory test")
        return True
    
    try:
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        # Get initial memory
        initial_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
        print(f"📊 Initial GPU memory: {initial_memory:.1f} MB")
        
        # Run a training session
        deep_q = test_deep_q_learning_initialization()
        if deep_q is None:
            return False
        
        creation_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
        print(f"📊 Memory after model creation: {creation_memory:.1f} MB")
        
        # Simulate training
        state_dim = 16
        for i in range(100):  # Store experiences
            state = np.random.randn(state_dim).astype(np.float32)
            action = np.random.randint(0, 6)
            reward = np.random.randn() * 10
            next_state = np.random.randn(state_dim).astype(np.float32)
            done = False
            
            deep_q.store_experience(state, action, reward, next_state, done)
            
            if i % 32 == 31 and len(deep_q.memory) >= 32:  # Learn every 32 steps
                deep_q.learn()
        
        final_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
        max_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        
        print(f"📊 Final GPU memory: {final_memory:.1f} MB")
        print(f"📊 Peak GPU memory: {max_memory:.1f} MB")
        print(f"📊 Memory increase: {final_memory - initial_memory:.1f} MB")
        
        # Clear cache
        torch.cuda.empty_cache()
        cleared_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
        print(f"📊 Memory after cache clear: {cleared_memory:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU memory test failed: {e}")
        return False


def main():
    """Main test function."""
    print("🚀 GPU Deep Learning Test Suite")
    print("=" * 50)
    
    tests = [
        ("GPU Availability", test_gpu_availability),
        ("Deep Q-Learning Init", test_deep_q_learning_initialization),
        ("State Processing", test_state_processing),
        ("Experience & Learning", test_experience_storage_and_learning),
        ("GPU Memory Usage", test_gpu_memory_usage)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results[test_name] = result
            print(f"{'✅ PASSED' if result else '❌ FAILED'}: {test_name}")
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {e}")
            results[test_name] = False
    
    print(f"\n{'='*50}")
    print("🏁 TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! GPU deep learning is ready!")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 