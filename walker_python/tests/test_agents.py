import sys
import os
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from agents.basic_agent import BasicAgent

def test_basic_agent_q_learning():
    state_dims = [5, 5]
    action_count = 3
    agent = BasicAgent(state_dims, action_count)
    
    # Set initial state and reward
    agent.set_state((2, 2))
    agent.set_reward(1.0)
    action = agent.select_action(epsilon=0.0)
    agent.take_action(action)
    agent.update(0.1)
    
    # After update, Q-table should have nonzero value for (2,2,action)
    q_value = agent.q_table.get_q_value((2, 2), action)
    assert q_value != 0.0
    
    # Test epsilon-greedy (with epsilon=1.0, should be random)
    actions = set()
    for _ in range(20):
        actions.add(agent.select_action(epsilon=1.0))
    assert len(actions) > 1
    
    # Test reset
    agent.reset()
    assert agent.q_table.get_q_value((2, 2), 0) == 0.0 