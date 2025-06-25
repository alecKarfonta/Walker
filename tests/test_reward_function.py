import pytest
import numpy as np
import Box2D as b2

from src.agents.crawling_crate_agent import CrawlingCrateAgent

@pytest.fixture
def make_agent():
    world = b2.b2World(gravity=(0, -9.8))
    agent = CrawlingCrateAgent(
        world,
        agent_id=0,
        position=(0, 1),
        category_bits=0x0002,
        mask_bits=0x0001,
    )
    # Allow fixtures to run a few world steps to settle
    for _ in range(5):
        world.Step(1/60, 8, 3)
    return agent


def test_positive_reward_on_forward_progress(make_agent):
    agent = make_agent
    # Simulate arms near ground
    agent.upper_arm.position.y = 0.3
    agent.lower_arm.position.y = 0.15
    prev_x = agent.body.position.x
    # Manually move body forward a small amount
    agent.body.position.x += 0.02
    reward = agent.get_reward(prev_x)
    assert reward > 0, f"Expected positive reward, got {reward}"


def test_negative_reward_when_stuck(make_agent):
    agent = make_agent
    # Arms high in the air, no progress
    agent.upper_arm.position.y = 1.0
    agent.lower_arm.position.y = 1.0
    prev_x = agent.body.position.x
    # No displacement
    reward = agent.get_reward(prev_x)
    assert reward <= 0, f"Expected non-positive reward, got {reward}"


def test_reward_clipped(make_agent):
    agent = make_agent
    # Force huge displacement to test clip upper bound
    prev_x = agent.body.position.x
    agent.body.position.x += 10.0
    reward = agent.get_reward(prev_x)
    assert reward <= 0.2, "Reward should be clipped to 0.2"

    # Force huge negative displacement
    prev_x = agent.body.position.x
    agent.body.position.x -= 10.0
    reward = agent.get_reward(prev_x)
    assert reward >= -0.1, "Reward should be clipped to -0.1" 