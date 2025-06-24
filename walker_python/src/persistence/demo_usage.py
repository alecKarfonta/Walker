"""
Demo Usage of Robot Persistence System

This script demonstrates how to use the robot storage and management system
to save, load, and manage robot data including Q-tables and performance metrics.
"""

import time
import Box2D as b2
from pathlib import Path

# Import persistence system
from src.persistence import RobotStorage, StorageManager
from src.agents.evolutionary_crawling_agent import EvolutionaryCrawlingAgent
from src.agents.physical_parameters import PhysicalParameters


def demo_basic_robot_persistence():
    """Demonstrate basic robot save/load functionality."""
    print("🔬 Demo: Basic Robot Persistence")
    print("=" * 50)
    
    # Create a Box2D world
    world = b2.b2World(gravity=(0, -9.8))
    
    # Initialize storage system
    storage = RobotStorage("demo_robot_storage")
    
    # Create a robot with random parameters
    print("1. Creating robot with random parameters...")
    random_params = PhysicalParameters.random_parameters()
    robot = EvolutionaryCrawlingAgent(
        world=world,
        agent_id=None,  # Let it generate UUID
        position=(10, 20),
        physical_params=random_params
    )
    
    # Simulate some training (add some fake data)
    robot.total_reward = 156.7
    robot.max_speed = 12.3
    robot.steps = 1000
    robot.generation = 5
    
    print(f"   Robot ID: {robot.id}")
    print(f"   Total Reward: {robot.total_reward}")
    print(f"   Max Speed: {robot.max_speed}")
    
    # Save the robot
    print("\n2. Saving robot to persistent storage...")
    saved_filename = storage.save_robot(
        robot, 
        notes="Demo robot with random parameters",
        save_method="demo"
    )
    print(f"   Saved as: {saved_filename}")
    
    # List saved robots
    print("\n3. Listing all saved robots...")
    saved_robots = storage.list_saved_robots()
    for robot_info in saved_robots:
        print(f"   Robot {robot_info['robot_id']}: "
              f"Reward={robot_info['total_reward']:.1f}, "
              f"Speed={robot_info['max_speed']:.1f}, "
              f"Generation={robot_info['generation']}")
    
    # Load the robot back
    print("\n4. Loading robot from storage...")
    loaded_robot = storage.load_robot(robot.id, world, position=(50, 25))
    
    print(f"   Loaded Robot ID: {loaded_robot.id}")
    print(f"   Loaded Total Reward: {loaded_robot.total_reward}")
    print(f"   Loaded Max Speed: {loaded_robot.max_speed}")
    print(f"   Loaded Generation: {loaded_robot.generation}")
    
    # Verify data integrity
    assert loaded_robot.total_reward == robot.total_reward
    assert loaded_robot.max_speed == robot.max_speed
    assert loaded_robot.generation == robot.generation
    print("   ✅ Data integrity verified!")
    
    print("\n🎉 Basic persistence demo completed successfully!\n")


def demo_population_management():
    """Demonstrate population-level storage operations."""
    print("🔬 Demo: Population Management")
    print("=" * 50)
    
    # Create world and storage manager
    world = b2.b2World(gravity=(0, -9.8))
    manager = StorageManager("demo_robot_storage")
    
    # Create a population of robots
    print("1. Creating population of robots...")
    population = []
    for i in range(5):
        params = PhysicalParameters.random_parameters()
        robot = EvolutionaryCrawlingAgent(
            world=world,
            physical_params=params,
            position=(10 + i * 5, 20)
        )
        
        # Simulate different performance levels
        robot.total_reward = 50.0 + i * 25.0  # Varying performance
        robot.max_speed = 5.0 + i * 2.0
        robot.generation = i + 1
        population.append(robot)
        
        print(f"   Robot {i+1}: ID={robot.id}, Reward={robot.total_reward}")
    
    # Save population checkpoint
    print("\n2. Saving population checkpoint...")
    checkpoint_name = manager.save_population_checkpoint(population)
    print(f"   Checkpoint saved: {checkpoint_name}")
    
    # Save elite robots only
    print("\n3. Saving elite robots (top 3)...")
    elite_name = manager.save_elite_robots(population, top_n=3)
    print(f"   Elite robots saved: {elite_name}")
    
    # Get storage statistics
    print("\n4. Storage statistics...")
    stats = manager.get_storage_stats()
    print(f"   Total robots stored: {stats.total_robots}")
    print(f"   Total snapshots: {stats.total_snapshots}")
    print(f"   Storage size: {stats.storage_size_mb:.2f} MB")
    
    # Load best robots
    print("\n5. Loading best performing robots...")
    best_robots = manager.load_best_robots(count=3, world=world)
    print(f"   Loaded {len(best_robots)} top performers:")
    for robot in best_robots:
        print(f"     Robot {robot.id}: Reward={robot.total_reward}")
    
    print("\n🎉 Population management demo completed successfully!\n")


def demo_advanced_features():
    """Demonstrate advanced persistence features."""
    print("🔬 Demo: Advanced Features")
    print("=" * 50)
    
    # Initialize systems
    world = b2.b2World(gravity=(0, -9.8))
    storage = RobotStorage("demo_robot_storage")
    manager = StorageManager("demo_robot_storage")
    
    # Create robot and simulate extended training
    print("1. Creating robot and simulating training history...")
    params = PhysicalParameters.random_parameters()
    robot = EvolutionaryCrawlingAgent(world=world, physical_params=params)
    
    # Simulate training progression
    for step in range(10):
        robot.total_reward += 10.0 + step * 2.0
        robot.max_speed = max(robot.max_speed, 5.0 + step * 0.5)
        robot.steps = (step + 1) * 100
        
        # Save progress periodically
        if step % 3 == 0:
            storage.save_robot(robot, notes=f"Training step {step}")
    
    print(f"   Final robot state: Reward={robot.total_reward}, Speed={robot.max_speed}")
    
    # Get performance history
    print("\n2. Analyzing performance history...")
    history = storage.get_performance_history(robot.id)
    if history:
        print(f"   Training sessions recorded: {len(history.reward_history)}")
        print(f"   Performance progression: {history.reward_history}")
        print(f"   Total steps: {history.total_steps}")
    
    # Find similar robots
    print("\n3. Finding similar robots...")
    similar_robots = manager.find_similar_robots(robot.id, similarity_threshold=0.5)
    print(f"   Found {len(similar_robots)} similar robots")
    for similar in similar_robots[:3]:  # Show top 3
        print(f"     Robot {similar['robot_id']}: "
              f"Similarity={similar['similarity']:.2f}, "
              f"Reward={similar['total_reward']:.1f}")
    
    # Export analysis
    print("\n4. Exporting population analysis...")
    analysis_file = manager.export_population_analysis()
    print(f"   Analysis exported to: {analysis_file}")
    
    # Enable auto-save
    print("\n5. Demonstrating auto-save...")
    manager.enable_auto_save(interval_seconds=5)
    
    # Simulate some activity
    robot.total_reward += 50.0
    time.sleep(1)  # Simulate some time passing
    
    # Check if auto-save would trigger
    if manager.check_auto_save([robot]):
        print("   Auto-save triggered!")
    else:
        print("   Auto-save not yet due")
    
    print("\n🎉 Advanced features demo completed successfully!\n")


def main():
    """Run all persistence system demos."""
    print("🚀 Robot Persistence System Demo")
    print("=" * 60)
    print()
    
    try:
        # Run demos
        demo_basic_robot_persistence()
        demo_population_management()
        demo_advanced_features()
        
        print("✨ All demos completed successfully!")
        print("\nThe robot persistence system is ready for use in your training pipeline.")
        print("\nKey features demonstrated:")
        print("  • Complete robot state save/load")
        print("  • Population checkpoint management")
        print("  • Performance history tracking")
        print("  • Elite robot preservation")
        print("  • Similar robot discovery")
        print("  • Auto-save functionality")
        print("  • Storage analytics")
        
        # Cleanup demo files
        import shutil
        demo_storage_path = Path("demo_robot_storage")
        if demo_storage_path.exists():
            shutil.rmtree(demo_storage_path)
            print("\n🧹 Demo storage files cleaned up")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 