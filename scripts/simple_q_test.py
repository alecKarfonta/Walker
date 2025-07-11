#!/usr/bin/env python3
"""
Simple test for Q-learning evaluator core functionality.
Tests only the core evaluator without dependencies.
"""

import sys
import time
from typing import Tuple

# Test just the core Q-learning evaluator
try:
    # Import only the core module directly
    sys.path.insert(0, 'src/evaluation')
    from q_learning_evaluator import QLearningEvaluator, QLearningMetrics, LearningStage
    print("✅ Successfully imported Q-learning evaluator core")
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


class SimpleAgent:
    """Minimal agent for testing."""
    
    def __init__(self, agent_id: str):
        self.id = agent_id
        self.learning_rate = 0.1
        self.epsilon = 0.2
        self.action_size = 4
        self.total_reward = 0.0
        self.q_table = SimpleQTable()


class SimpleQTable:
    """Minimal Q-table for testing."""
    
    def get_q_value(self, state, action):
        return 0.1
    
    def get_action_values(self, state):
        return [0.1, 0.2, 0.0, -0.1]
    
    def get_stats(self):
        return {'total_states': 10}


def test_core_functionality():
    """Test the core Q-learning evaluation functionality."""
    print("\n🧪 Testing core Q-learning evaluator...")
    
    # Create evaluator
    evaluator = QLearningEvaluator(evaluation_window=20, update_frequency=5)
    print("✅ Created evaluator")
    
    # Create test agents
    agents = [SimpleAgent(f"agent_{i}") for i in range(3)]
    
    # Register agents
    for agent in agents:
        evaluator.register_agent(agent)
    print(f"✅ Registered {len(agents)} agents")
    
    # Record some learning steps
    for step in range(15):
        for i, agent in enumerate(agents):
            state = (step % 2, (step + i) % 2, 0)
            action = step % 4
            predicted_q = 0.1 + step * 0.01
            actual_reward = predicted_q + (step % 3) * 0.02
            next_state = ((step + 1) % 2, (step + i + 1) % 2, 0)
            
            evaluator.record_q_learning_step(
                agent=agent,
                state=state,
                action=action,
                predicted_q_value=predicted_q,
                actual_reward=actual_reward,
                next_state=next_state
            )
    
    print(f"✅ Recorded {15 * len(agents)} learning steps")
    
    # Test metrics retrieval
    all_metrics = evaluator.get_all_agent_metrics()
    print(f"✅ Retrieved metrics for {len(all_metrics)} agents")
    
    for agent_id, metrics in all_metrics.items():
        print(f"   Agent {agent_id}: MAE={metrics.value_prediction_mae:.4f}")
    
    # Test summary
    summary = evaluator.generate_summary_report()
    print(f"✅ Generated summary report")
    
    if summary.get('status') != 'no_data':
        print(f"   Total agents: {summary.get('total_agents_evaluated', 0)}")
    
    return True


def main():
    """Run the simple test."""
    print("🧠 Simple Q-Learning Evaluator Test")
    print("=" * 40)
    
    try:
        success = test_core_functionality()
        if success:
            print("\n🎉 Test completed successfully!")
            print("The Q-learning evaluation system core functionality is working.")
            return True
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
