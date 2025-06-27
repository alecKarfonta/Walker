#!/usr/bin/env python3
"""
Test script for reward signal quality evaluation system.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import time
import random

# Direct imports to avoid dependency issues
try:
    import numpy as np
except ImportError:
    print("⚠️  NumPy not available, using basic math")
    np = None

# Import directly without going through __init__.py to avoid dependency issues
import importlib.util
import sys

# Load reward signal evaluator directly
spec = importlib.util.spec_from_file_location(
    "reward_signal_evaluator", 
    "src/evaluation/reward_signal_evaluator.py"
)
reward_signal_evaluator = importlib.util.module_from_spec(spec)
sys.modules["reward_signal_evaluator"] = reward_signal_evaluator
spec.loader.exec_module(reward_signal_evaluator)

# Load integration adapter directly
spec2 = importlib.util.spec_from_file_location(
    "reward_signal_integration", 
    "src/evaluation/reward_signal_integration.py"
)
reward_signal_integration = importlib.util.module_from_spec(spec2)
sys.modules["reward_signal_integration"] = reward_signal_integration
spec2.loader.exec_module(reward_signal_integration)

# Extract classes
RewardSignalEvaluator = reward_signal_evaluator.RewardSignalEvaluator
RewardQualityIssue = reward_signal_evaluator.RewardQualityIssue
RewardSignalIntegrationAdapter = reward_signal_integration.RewardSignalIntegrationAdapter


def generate_test_data(evaluator, agent_id, scenario):
    """Generate test reward data for different scenarios."""
    print(f"📊 Testing scenario: {scenario}")
    
    if scenario == "high_quality":
        # Dense, consistent, well-balanced rewards
        for i in range(200):
            state = [random.uniform(0, 1) for _ in range(3)]
            action = random.randint(0, 3)
            # Create consistent reward based on state-action pattern
            base_reward = sum(state) * 0.3 + action * 0.1
            reward = base_reward + random.gauss(0, 0.1)  # Small noise
            evaluator.record_reward(agent_id, state, action, reward)
    
    elif scenario == "sparse_rewards":
        # Very sparse rewards (mostly zeros)
        for i in range(200):
            state = [random.uniform(0, 1) for _ in range(3)]
            action = random.randint(0, 3)
            # Only give reward 5% of the time
            reward = 1.0 if random.random() < 0.05 else 0.0
            evaluator.record_reward(agent_id, state, action, reward)
    
    elif scenario == "noisy_rewards":
        # High noise, low signal
        for i in range(200):
            state = [random.uniform(0, 1) for _ in range(3)]
            action = random.randint(0, 3)
            # Small signal with large noise
            signal = 0.1
            noise = random.gauss(0, 1.0)  # High noise
            reward = signal + noise
            evaluator.record_reward(agent_id, state, action, reward)
    
    elif scenario == "inconsistent_rewards":
        # Same state-action pairs get very different rewards
        for i in range(200):
            state = [0.5, 0.5, 0.5]  # Fixed state
            action = 1  # Fixed action
            # Random reward for same state-action
            reward = random.uniform(-2, 2)
            evaluator.record_reward(agent_id, state, action, reward)
            
    elif scenario == "biased_rewards":
        # Only positive rewards
        for i in range(200):
            state = [random.uniform(0, 1) for _ in range(3)]
            action = random.randint(0, 3)
            reward = random.uniform(0, 1)  # Only positive
            evaluator.record_reward(agent_id, state, action, reward)


def test_basic_functionality():
    """Test basic functionality of reward signal evaluator."""
    print("🧪 Testing basic functionality...")
    
    evaluator = RewardSignalEvaluator(window_size=100, min_samples=20)
    
    # Test with simple reward data
    agent_id = "test_agent_1"
    for i in range(50):
        state = [i / 50.0]
        action = i % 2
        reward = 1.0 if action == 1 else 0.0
        evaluator.record_reward(agent_id, state, action, reward)
    
    # Get metrics
    metrics = evaluator.get_agent_metrics(agent_id)
    assert metrics is not None, "Should have metrics after sufficient samples"
    
    print(f"✅ Basic test passed - Quality score: {metrics.quality_score:.3f}")
    return True


def test_quality_scenarios():
    """Test different reward quality scenarios."""
    print("🧪 Testing quality scenarios...")
    
    evaluator = RewardSignalEvaluator(window_size=300, min_samples=50)
    
    scenarios = {
        "high_quality": "agent_hq",
        "sparse_rewards": "agent_sparse", 
        "noisy_rewards": "agent_noisy",
        "inconsistent_rewards": "agent_inconsistent",
        "biased_rewards": "agent_biased"
    }
    
    # Generate test data for each scenario
    for scenario, agent_id in scenarios.items():
        generate_test_data(evaluator, agent_id, scenario)
    
    # Analyze results
    print("\n📈 Quality Analysis Results:")
    print("=" * 60)
    
    results = {}
    for scenario, agent_id in scenarios.items():
        metrics = evaluator.get_agent_metrics(agent_id)
        if metrics:
            results[scenario] = metrics
            print(f"\n{scenario.upper()}")
            print(f"  Quality Score: {metrics.quality_score:.3f}")
            print(f"  SNR: {metrics.signal_to_noise_ratio:.3f}")
            print(f"  Sparsity: {metrics.reward_sparsity:.3f}")
            print(f"  Consistency: {metrics.reward_consistency:.3f}")
            print(f"  Issues: {[issue.value for issue in metrics.quality_issues]}")
            print(f"  Recommendations: {len(metrics.recommendations)} provided")
    
    # Verify expected behaviors
    assert results["high_quality"].quality_score > 0.5, "High quality should have good score"
    assert RewardQualityIssue.SPARSE_REWARDS in results["sparse_rewards"].quality_issues
    assert RewardQualityIssue.NOISY_REWARDS in results["noisy_rewards"].quality_issues
    assert RewardQualityIssue.INCONSISTENT_REWARDS in results["inconsistent_rewards"].quality_issues
    assert RewardQualityIssue.BIASED_REWARDS in results["biased_rewards"].quality_issues
    
    print("✅ Quality scenarios test passed!")
    return True


def test_integration_adapter():
    """Test the integration adapter functionality."""  
    print("🧪 Testing integration adapter...")
    
    adapter = RewardSignalIntegrationAdapter()
    
    # Register agents
    adapter.register_agent("agent_1", "q_learning", {"lr": 0.01})
    adapter.register_agent("agent_2", "dqn", {"epsilon": 0.1})
    
    # Record reward signals
    for i in range(100):
        state = [random.uniform(0, 1) for _ in range(3)]
        action = random.randint(0, 3)
        reward = random.uniform(-1, 1)
        
        adapter.record_reward_signal("agent_1", state, action, reward)
        if i % 2 == 0:  # Less data for agent_2
            adapter.record_reward_signal("agent_2", state, action, reward * 0.5)
    
    # Test system status
    status = adapter.get_system_status()
    assert status['active'] == True
    assert status['total_agents'] == 2
    assert status['total_rewards_recorded'] >= 150
    
    # Test agent metrics
    metrics_1 = adapter.get_agent_reward_metrics("agent_1")
    assert metrics_1 is not None
    
    # Test diagnostics
    diagnostics = adapter.get_agent_diagnostics("agent_1")
    assert diagnostics['agent_id'] == "agent_1"
    assert 'reward_signal_analysis' in diagnostics
    
    # Test comparative report
    report = adapter.get_reward_comparative_report()
    assert 'total_agents_analyzed' in report
    assert report['total_agents_analyzed'] >= 1
    
    print("✅ Integration adapter test passed!")
    return True


def test_api_format_compatibility():
    """Test that outputs are compatible with API expectations."""
    print("🧪 Testing API format compatibility...")
    
    adapter = RewardSignalIntegrationAdapter()
    adapter.register_agent("api_test_agent", "test_type")
    
    # Add some data
    for i in range(60):
        adapter.record_reward_signal("api_test_agent", [i], i % 3, random.uniform(-1, 1))
    
    # Test JSON serialization
    import json
    
    # Test metrics serialization
    metrics = adapter.get_agent_reward_metrics("api_test_agent")
    if metrics:
        metrics_dict = metrics.to_dict()
        json_str = json.dumps(metrics_dict)  # Should not raise exception
        parsed = json.loads(json_str)
        assert 'agent_id' in parsed
        assert 'basic_statistics' in parsed
        assert 'quality_assessment' in parsed
    
    # Test diagnostics serialization
    diagnostics = adapter.get_agent_diagnostics("api_test_agent")
    json_str = json.dumps(diagnostics)  # Should not raise exception
    
    # Test status serialization  
    status = adapter.get_system_status()
    json_str = json.dumps(status)  # Should not raise exception
    
    print("✅ API format compatibility test passed!")
    return True


def run_performance_test():
    """Test performance characteristics."""
    print("🧪 Testing performance characteristics...")
    
    evaluator = RewardSignalEvaluator(window_size=1000, min_samples=50)
    
    # Test with large number of agents
    num_agents = 10
    samples_per_agent = 500
    
    start_time = time.time()
    
    for agent_idx in range(num_agents):
        agent_id = f"perf_agent_{agent_idx}"
        for sample_idx in range(samples_per_agent):
            state = [random.uniform(0, 1) for _ in range(5)]
            action = random.randint(0, 4)
            reward = random.gauss(0.5, 0.2)
            evaluator.record_reward(agent_id, state, action, reward)
    
    total_time = time.time() - start_time
    total_samples = num_agents * samples_per_agent
    samples_per_second = total_samples / total_time
    
    print(f"  Processed {total_samples} samples in {total_time:.2f}s")
    print(f"  Rate: {samples_per_second:.0f} samples/second")
    
    # Verify all agents have metrics
    all_metrics = evaluator.get_all_metrics()
    assert len(all_metrics) == num_agents, f"Expected {num_agents} agents with metrics"
    
    print("✅ Performance test passed!")
    return True


def main():
    """Run all tests."""
    print("🚀 Starting Reward Signal Evaluation System Tests")
    print("=" * 60)
    
    tests = [
        test_basic_functionality,
        test_quality_scenarios,
        test_integration_adapter,
        test_api_format_compatibility,
        run_performance_test
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            print(f"\n{'-' * 40}")
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"❌ {test_func.__name__} failed")
        except Exception as e:
            failed += 1
            print(f"❌ {test_func.__name__} failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'=' * 60}")
    print(f"🏁 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Reward signal evaluation system is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 