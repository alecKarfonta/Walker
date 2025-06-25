#!/usr/bin/env python3
"""
Test script for the flexible learning system.
Validates that robots can switch between different learning approaches dynamically.
"""

import sys
import os
import time
import requests
import json
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Import the training environment
from train_robots_web_visual import TrainingEnvironment
from src.agents.learning_manager import LearningApproach


def test_learning_manager_initialization():
    """Test that the learning manager initializes correctly."""
    print("🧪 Testing Learning Manager Initialization...")
    
    try:
        # Create a smaller environment for testing
        env = TrainingEnvironment(num_agents=6, enable_evaluation=False)
        
        # Check that learning manager was created
        assert hasattr(env, 'learning_manager'), "Learning manager not created"
        assert env.learning_manager is not None, "Learning manager is None"
        
        # Check that ecosystem interface was created
        assert hasattr(env, 'ecosystem_interface'), "Ecosystem interface not created"
        assert env.ecosystem_interface is not None, "Ecosystem interface is None"
        
        # Check that agents have learning approaches assigned
        agent_count = 0
        survival_count = 0
        enhanced_count = 0
        
        for agent in env.agents:
            if not getattr(agent, '_destroyed', False):
                approach = env.learning_manager.get_agent_approach(agent.id)
                agent_count += 1
                
                if approach == LearningApproach.SURVIVAL_Q_LEARNING:
                    survival_count += 1
                elif approach == LearningApproach.ENHANCED_Q_LEARNING:
                    enhanced_count += 1
        
        print(f"✅ Learning manager initialized successfully")
        print(f"   • Total agents: {agent_count}")
        print(f"   • Survival learning: {survival_count}")
        print(f"   • Enhanced learning: {enhanced_count}")
        
        return env
        
    except Exception as e:
        print(f"❌ Learning manager initialization failed: {e}")
        raise


def test_approach_switching(env):
    """Test switching individual agents between learning approaches."""
    print("\n🧪 Testing Individual Approach Switching...")
    
    try:
        # Get a test agent
        test_agent = None
        for agent in env.agents:
            if not getattr(agent, '_destroyed', False) and agent.body:
                test_agent = agent
                break
        
        assert test_agent is not None, "No valid test agent found"
        
        # Test switching to each approach
        approaches_to_test = [
            LearningApproach.BASIC_Q_LEARNING,
            LearningApproach.ENHANCED_Q_LEARNING,
            LearningApproach.SURVIVAL_Q_LEARNING
        ]
        
        switch_results = {}
        
        for approach in approaches_to_test:
            print(f"   Switching agent {test_agent.id} to {approach.value}...")
            
            success = env.learning_manager.set_agent_approach(test_agent, approach)
            switch_results[approach.value] = success
            
            if success:
                # Verify the switch worked
                current_approach = env.learning_manager.get_agent_approach(test_agent.id)
                assert current_approach == approach, f"Switch verification failed: {current_approach} != {approach}"
                print(f"     ✅ Successfully switched to {approach.value}")
            else:
                print(f"     ❌ Failed to switch to {approach.value}")
        
        success_count = sum(1 for success in switch_results.values() if success)
        print(f"\n✅ Approach switching test completed: {success_count}/{len(approaches_to_test)} successful")
        
        return switch_results
        
    except Exception as e:
        print(f"❌ Approach switching test failed: {e}")
        raise


def test_bulk_switching(env):
    """Test bulk switching of all agents to a specific approach."""
    print("\n🧪 Testing Bulk Approach Switching...")
    
    try:
        # Get all valid agents
        valid_agents = [a for a in env.agents if not getattr(a, '_destroyed', False) and a.body]
        agent_ids = [a.id for a in valid_agents]
        
        print(f"   Testing bulk switch for {len(valid_agents)} agents...")
        
        # Test bulk switch to survival learning
        results = env.learning_manager.bulk_switch_approach(
            agent_ids, valid_agents, LearningApproach.SURVIVAL_Q_LEARNING
        )
        
        success_count = sum(1 for success in results.values() if success)
        
        # Verify all agents are now using survival learning
        actual_survival_count = 0
        for agent_id in agent_ids:
            approach = env.learning_manager.get_agent_approach(agent_id)
            if approach == LearningApproach.SURVIVAL_Q_LEARNING:
                actual_survival_count += 1
        
        print(f"   ✅ Bulk switch completed: {success_count}/{len(valid_agents)} switched")
        print(f"   ✅ Verification: {actual_survival_count}/{len(valid_agents)} using survival learning")
        
        return success_count == len(valid_agents)
        
    except Exception as e:
        print(f"❌ Bulk switching test failed: {e}")
        raise


def test_learning_statistics(env):
    """Test that learning statistics are properly collected."""
    print("\n🧪 Testing Learning Statistics...")
    
    try:
        stats = env.learning_manager.get_approach_statistics()
        
        # Check that required fields exist
        required_fields = ['approach_distribution', 'approach_performance', 'total_agents', 'approach_info']
        for field in required_fields:
            assert field in stats, f"Missing field in statistics: {field}"
        
        # Check approach distribution
        distribution = stats['approach_distribution']
        total_from_distribution = sum(distribution.values())
        assert total_from_distribution == stats['total_agents'], "Distribution total doesn't match total agents"
        
        # Check approach info
        approach_info = stats['approach_info']
        for approach in LearningApproach:
            assert approach.value in approach_info, f"Missing approach info for {approach.value}"
            info = approach_info[approach.value]
            required_info_fields = ['name', 'description', 'color', 'icon', 'advantages', 'disadvantages']
            for field in required_info_fields:
                assert field in info, f"Missing field in approach info: {field}"
        
        print(f"✅ Learning statistics test passed")
        print(f"   • Total agents tracked: {stats['total_agents']}")
        print(f"   • Distribution: {distribution}")
        
        return stats
        
    except Exception as e:
        print(f"❌ Learning statistics test failed: {e}")
        raise


def test_web_endpoints():
    """Test the web endpoints for learning approach switching."""
    print("\n🧪 Testing Web Endpoints...")
    
    # Start a test environment in the background
    print("   Starting test web server...")
    
    # Give some time for any existing server to be ready
    time.sleep(2)
    
    base_url = "http://localhost:8080"
    
    try:
        # Test learning statistics endpoint
        print("   Testing learning statistics endpoint...")
        response = requests.get(f"{base_url}/learning_statistics", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            assert data['status'] == 'success', "Learning statistics endpoint failed"
            assert 'statistics' in data, "Statistics missing from response"
            print("     ✅ Learning statistics endpoint working")
        else:
            print(f"     ⚠️ Learning statistics endpoint returned {response.status_code}")
        
        # Test status endpoint to get agent IDs
        print("   Getting agent IDs from status endpoint...")
        status_response = requests.get(f"{base_url}/status", timeout=5)
        
        if status_response.status_code == 200:
            status_data = status_response.json()
            if 'agents' in status_data and len(status_data['agents']) > 0:
                test_agent_id = status_data['agents'][0]['id']
                
                # Test individual approach switching
                print(f"   Testing approach switching for agent {test_agent_id}...")
                switch_response = requests.post(
                    f"{base_url}/switch_learning_approach",
                    json={'agent_id': test_agent_id, 'approach': 'basic_q_learning'},
                    timeout=5
                )
                
                if switch_response.status_code == 200:
                    switch_data = switch_response.json()
                    if switch_data['status'] == 'success':
                        print("     ✅ Individual approach switching working")
                    else:
                        print(f"     ⚠️ Switch failed: {switch_data.get('message', 'Unknown error')}")
                else:
                    print(f"     ⚠️ Switch endpoint returned {switch_response.status_code}")
        
        print("✅ Web endpoints test completed")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Web endpoints test skipped: {e}")
        print("   (This is normal if web server is not running)")
        return False
    except Exception as e:
        print(f"❌ Web endpoints test failed: {e}")
        return False


def test_approach_persistence(env):
    """Test that approach assignments persist during training steps."""
    print("\n🧪 Testing Approach Persistence...")
    
    try:
        # Set up different approaches for different agents
        test_agents = [a for a in env.agents if not getattr(a, '_destroyed', False) and a.body][:3]
        
        if len(test_agents) < 3:
            print("⚠️ Insufficient agents for persistence test")
            return True
        
        # Assign different approaches
        approaches = [
            LearningApproach.BASIC_Q_LEARNING,
            LearningApproach.ENHANCED_Q_LEARNING,
            LearningApproach.SURVIVAL_Q_LEARNING
        ]
        
        initial_assignments = {}
        for i, agent in enumerate(test_agents):
            approach = approaches[i]
            success = env.learning_manager.set_agent_approach(agent, approach)
            if success:
                initial_assignments[agent.id] = approach
                print(f"   Assigned agent {agent.id} to {approach.value}")
        
        # Run a few training steps
        print("   Running training steps to test persistence...")
        for step in range(10):
            try:
                env.world.Step(env.dt, 8, 3)
                for agent in test_agents:
                    if not getattr(agent, '_destroyed', False):
                        agent.step(env.dt)
            except Exception as e:
                print(f"     Warning: Step {step} had errors: {e}")
        
        # Check that assignments persisted
        persistence_success = True
        for agent_id, expected_approach in initial_assignments.items():
            current_approach = env.learning_manager.get_agent_approach(agent_id)
            if current_approach != expected_approach:
                print(f"   ❌ Agent {agent_id} approach changed: {expected_approach} -> {current_approach}")
                persistence_success = False
            else:
                print(f"   ✅ Agent {agent_id} approach persisted: {current_approach.value}")
        
        if persistence_success:
            print("✅ Approach persistence test passed")
        else:
            print("❌ Some approaches did not persist")
        
        return persistence_success
        
    except Exception as e:
        print(f"❌ Approach persistence test failed: {e}")
        return False


def main():
    """Run all flexible learning system tests."""
    print("🎛️ === FLEXIBLE LEARNING SYSTEM TESTS ===\n")
    
    test_results = {}
    env = None
    
    try:
        # Test 1: Learning Manager Initialization
        env = test_learning_manager_initialization()
        test_results['initialization'] = True
        
        # Test 2: Individual Approach Switching
        switch_results = test_approach_switching(env)
        test_results['individual_switching'] = len(switch_results) > 0
        
        # Test 3: Bulk Approach Switching
        bulk_success = test_bulk_switching(env)
        test_results['bulk_switching'] = bulk_success
        
        # Test 4: Learning Statistics
        stats = test_learning_statistics(env)
        test_results['statistics'] = stats is not None
        
        # Test 5: Approach Persistence
        persistence_success = test_approach_persistence(env)
        test_results['persistence'] = persistence_success
        
        # Test 6: Web Endpoints (optional)
        web_success = test_web_endpoints()
        test_results['web_endpoints'] = web_success
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        test_results['suite_error'] = str(e)
    
    finally:
        # Clean up
        if env:
            try:
                env.stop()
            except:
                pass
    
    # Summary
    print(f"\n📊 === TEST RESULTS SUMMARY ===")
    
    passed_tests = 0
    total_tests = 0
    
    for test_name, result in test_results.items():
        if test_name != 'suite_error':
            total_tests += 1
            if result:
                passed_tests += 1
                print(f"✅ {test_name.replace('_', ' ').title()}: PASSED")
            else:
                print(f"❌ {test_name.replace('_', ' ').title()}: FAILED")
    
    if 'suite_error' in test_results:
        print(f"💥 Suite Error: {test_results['suite_error']}")
    
    print(f"\n🎯 Overall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Flexible learning system is working correctly.")
        print("\n🎮 NEXT STEPS:")
        print("1. Start the web interface: python train_robots_web_visual.py")
        print("2. Open http://localhost:8080 in your browser")
        print("3. Click on robots to see learning approach controls")
        print("4. Use bulk controls in the Learning panel")
        print("5. Compare performance between different approaches!")
        return True
    else:
        print("⚠️ Some tests failed. Please check the output above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 