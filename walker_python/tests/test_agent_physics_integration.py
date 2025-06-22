import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pymunk
from agents.basic_agent import BasicAgent

def discretize_position(x, x_min, x_max, bins):
    x = max(x_min, min(x, x_max))
    return int((x - x_min) / (x_max - x_min) * (bins - 1))

def test_agent_trains_with_physics_world():
    # Simple 1D world: agent controls a ball left/right/none
    x_min, x_max = -10, 10
    bins = 11
    state_dims = [bins]
    action_count = 3  # 0=left, 1=none, 2=right
    agent = BasicAgent(state_dims, action_count)

    # Set up Pymunk world
    space = pymunk.Space()
    space.gravity = (0, 0)
    mass = 1.0
    radius = 1.0
    moment = pymunk.moment_for_circle(mass, 0, radius)
    body = pymunk.Body(mass, moment)
    body.position = (0, 0)
    shape = pymunk.Circle(body, radius)
    shape.friction = 0.5
    space.add(body, shape)

    # Training loop
    episodes = 20
    steps_per_episode = 30
    learning_curve = []
    for ep in range(episodes):
        # Reset position
        body.position = (0, 0)
        body.velocity = (0, 0)
        total_reward = 0
        for t in range(steps_per_episode):
            # Discretize state
            x = body.position.x
            state = (discretize_position(x, x_min, x_max, bins),)
            agent.set_state(state)

            # Select action
            action = agent.select_action()
            # Apply action as force
            if action == 0:
                body.apply_force_at_local_point((-100, 0))
            elif action == 2:
                body.apply_force_at_local_point((100, 0))
            # else: do nothing

            # Step physics
            space.step(1/30.0)

            # Reward: +1 for moving right, -1 for left, 0 for none
            reward = 0
            if action == 2:
                reward = 1
            elif action == 0:
                reward = -1
            agent.set_reward(reward)
            total_reward += reward

            # Q-learning update
            agent.take_action(action)
            agent.update(1/30.0)
        learning_curve.append(total_reward)

    # Check that Q-table for right action at rightmost state is higher than left
    rightmost_state = (bins - 1,)
    q_right = agent.q_table.get_q_value(rightmost_state, 2)
    q_left = agent.q_table.get_q_value(rightmost_state, 0)
    print(f"Q(rightmost, right)={q_right:.2f}, Q(rightmost, left)={q_left:.2f}")
    print("Learning curve (total reward per episode):", learning_curve)
    assert q_right > q_left
    # Also, Q-table should have nonzero values
    assert np.any(agent.q_table.q_values != 0.0) 