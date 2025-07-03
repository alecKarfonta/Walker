#!/usr/bin/env python3
"""
Robot Size Mutation Demo

Demonstrates how robots receive size mutations when respawned while
preserving their Q-network weights and learning progress.

This example shows:
1. Creating a robot with initial physical parameters
2. Simulating some learning (Q-network weights accumulate)
3. Respawning the robot with size mutations
4. Verifying that Q-network weights are preserved while body parts change size
"""

import sys
import os
import time
import random
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import Box2D as b2
from src.agents.physical_parameters import PhysicalParameters
from src.agents.crawling_agent import CrawlingAgent
from src.agents.robot_memory_pool import RobotMemoryPool

try:
    import torch
except ImportError:
    torch = None


def create_simple_world():
    """Create a simple Box2D world for the demo."""
    world = b2.b2World(gravity=(0, -9.81))
    
    # Create ground
    ground_body = world.CreateStaticBody(position=(0, -2))
    ground_body.CreateFixture(
        shape=b2.b2PolygonShape(box=(50, 1)),
        friction=0.3
    )
    
    return world


def print_robot_info(robot, title):
    """Print detailed robot information."""
    print(f"\n{title}")
    print(f"=" * len(title))
    print(f"Robot ID: {robot.id}")
    print(f"Position: ({robot.body.position.x:.2f}, {robot.body.position.y:.2f})")
    
    params = robot.physical_params
    print(f"\nPhysical Parameters:")
    print(f"  Overall Scale: {params.overall_scale:.3f}")
    print(f"  Body Size: {params.body_width:.3f} x {params.body_height:.3f}")
    print(f"  Arm Lengths: upper={params.arm_length:.3f}, lower={params.wrist_length:.3f}")
    print(f"  Arm Widths: upper={params.arm_width:.3f}, lower={params.wrist_width:.3f}")
    print(f"  Wheel Radius: {params.wheel_radius:.3f}")
    print(f"  Leg Spread: {params.leg_spread:.3f}")
    
    # Show Q-network information if available
    if hasattr(robot, '_learning_system') and robot._learning_system:
        learning_system = robot._learning_system
        if hasattr(learning_system, 'q_network') and learning_system.q_network:
            # Count parameters in the Q-network
            param_count = sum(p.numel() for p in learning_system.q_network.parameters())
            print(f"\nQ-Network Information:")
            print(f"  Parameters: {param_count:,}")
            print(f"  Training Steps: {getattr(learning_system, 'training_step', 0)}")
            print(f"  Epsilon: {getattr(learning_system, 'epsilon', 0.0):.3f}")
            
            # Show some sample weights (first layer bias as example)
            try:
                first_layer_weights = list(learning_system.q_network.parameters())[1]  # bias of first layer
                sample_weights = first_layer_weights[:5].detach().cpu().numpy()
                print(f"  Sample weights: {sample_weights}")
            except:
                print(f"  Sample weights: [unavailable]")
    else:
        print(f"\nQ-Network: Not initialized")


def simulate_learning(robot, steps=100):
    """Simulate some learning by taking random actions."""
    print(f"\n🧠 Simulating {steps} learning steps...")
    
    for step in range(steps):
        # Get random action
        if hasattr(robot, 'actions') and robot.actions:
            action_idx = random.randint(0, len(robot.actions) - 1)
            action = robot.actions[action_idx]
        else:
            # Fallback to simple action
            action = (random.uniform(-1, 1), random.uniform(-1, 1))
        
        # Apply action
        try:
            robot.apply_action(action)
        except:
            pass
        
        # Simulate reward
        reward = random.uniform(-0.1, 0.5)
        robot.total_reward += reward
        robot.steps += 1
        
        # Learn from experience
        if hasattr(robot, '_learning_system') and robot._learning_system:
            try:
                state = robot.get_state()
                robot._learning_system.update(state, action_idx, reward, state)
            except:
                pass
    
    print(f"✅ Completed learning simulation")
    print(f"   Total steps: {robot.steps}")
    print(f"   Total reward: {robot.total_reward:.2f}")


def compare_parameters(params1, params2, title):
    """Compare two parameter sets and show differences."""
    print(f"\n{title}")
    print(f"=" * len(title))
    
    changes = []
    
    # Compare key size parameters
    size_params = [
        ('overall_scale', 'Overall Scale'),
        ('body_width', 'Body Width'),
        ('body_height', 'Body Height'),
        ('arm_length', 'Arm Length'),
        ('wrist_length', 'Wrist Length'),
        ('arm_width', 'Arm Width'),
        ('wrist_width', 'Wrist Width'),
        ('wheel_radius', 'Wheel Radius'),
        ('leg_spread', 'Leg Spread')
    ]
    
    for param_name, display_name in size_params:
        old_val = getattr(params1, param_name)
        new_val = getattr(params2, param_name)
        change_pct = ((new_val - old_val) / old_val) * 100 if old_val != 0 else 0
        
        if abs(change_pct) > 0.1:  # Show changes > 0.1%
            changes.append(f"  {display_name}: {old_val:.3f} → {new_val:.3f} ({change_pct:+.1f}%)")
    
    if changes:
        print("Size Changes Detected:")
        for change in changes:
            print(change)
    else:
        print("No significant size changes detected")


def demo_size_mutations():
    """Main demonstration of size mutations during respawning."""
    print("🧬 Robot Size Mutation Demo")
    print("=" * 50)
    
    # Create world and memory pool
    world = create_simple_world()
    memory_pool = RobotMemoryPool(world, min_pool_size=2, max_pool_size=10)
    
    print("✅ Created world and memory pool")
    
    # Create initial robot
    initial_params = PhysicalParameters.random_parameters()
    robot = CrawlingAgent(
        world=world,
        agent_id=None,
        position=(0, 5),
        category_bits=0x0002,
        mask_bits=0x0001,
        physical_params=initial_params
    )
    
    print_robot_info(robot, "🤖 Initial Robot")
    
    # Simulate some learning
    simulate_learning(robot, steps=50)
    print_robot_info(robot, "🧠 Robot After Learning")
    
    # Store original parameters and Q-network state
    original_params = robot.physical_params
    original_total_reward = robot.total_reward
    original_steps = robot.steps
    
    # Get Q-network state before respawning
    original_network_state = None
    if hasattr(robot, '_learning_system') and robot._learning_system:
        try:
            if hasattr(robot._learning_system, 'q_network'):
                # Get a sample of network weights
                original_network_state = {}
                for name, param in robot._learning_system.q_network.named_parameters():
                    original_network_state[name] = param.detach().clone()
                    break  # Just take first parameter as sample
        except:
            pass
    
    # Return robot to memory pool
    memory_pool.return_robot(robot)
    print("\n♻️ Returned robot to memory pool")
    
    # Acquire robot again with size mutations
    print("\n🧬 Respawning robot with size mutations...")
    respawned_robot = memory_pool.acquire_robot(
        position=(10, 5),
        apply_size_mutations=True
    )
    
    print_robot_info(respawned_robot, "🆕 Respawned Robot with Mutations")
    
    # Compare parameters
    compare_parameters(original_params, respawned_robot.physical_params, "📊 Size Mutation Analysis")
    
    # Verify Q-network preservation (if available)
    print(f"\n🧠 Q-Network Preservation Check")
    print(f"=" * 35)
    
    if original_network_state and hasattr(respawned_robot, '_learning_system') and respawned_robot._learning_system:
        try:
            if hasattr(respawned_robot._learning_system, 'q_network'):
                # Compare first parameter
                for name, original_weights in original_network_state.items():
                    current_weights = None
                    for current_name, current_param in respawned_robot._learning_system.q_network.named_parameters():
                        if current_name == name:
                            current_weights = current_param.detach()
                            break
                    
                    if current_weights is not None and torch is not None:
                        weights_match = torch.allclose(original_weights, current_weights, atol=1e-6)
                        print(f"  Network weights preserved: {'✅ YES' if weights_match else '❌ NO'}")
                        print(f"  Sample original: {original_weights.flatten()[:3].numpy()}")
                        print(f"  Sample current:  {current_weights.flatten()[:3].numpy()}")
                    else:
                        print(f"  Network weights: ⚠️ Cannot compare")
                    break
        except Exception as e:
            print(f"  Network comparison failed: {e}")
    else:
        print(f"  Network weights: ⚠️ Not available for comparison")
    
    # Show learning state preservation
    print(f"\nLearning State Preservation:")
    print(f"  Original reward: {original_total_reward:.2f}")
    print(f"  Current reward:  {respawned_robot.total_reward:.2f}")
    print(f"  Original steps:  {original_steps}")
    print(f"  Current steps:   {respawned_robot.steps}")
    
    # Test mutations multiple times
    print(f"\n🔄 Testing Multiple Mutations")
    print(f"=" * 35)
    
    for i in range(3):
        memory_pool.return_robot(respawned_robot)
        respawned_robot = memory_pool.acquire_robot(
            position=(20 + i * 10, 5),
            apply_size_mutations=True
        )
        
        params = respawned_robot.physical_params
        print(f"  Mutation {i+1}: scale={params.overall_scale:.3f}, body={params.body_width:.3f}x{params.body_height:.3f}")
    
    print(f"\n✅ Size mutation demo completed!")
    print(f"🎯 Key Points:")
    print(f"   • Robot body parts mutate in size during respawning")
    print(f"   • Q-network weights are preserved (when available)")
    print(f"   • Learning progress (rewards, steps) reset for new episode")
    print(f"   • Structural features (arm count, segments) remain unchanged")


if __name__ == "__main__":
    demo_size_mutations() 