#!/usr/bin/env python3
"""
Test script to verify survival Q-learning integration.
Run this to test that the survival patch is working correctly.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import time
import threading
from train_robots_web_visual import TrainingEnvironment

def test_survival_integration():
    """Test the survival Q-learning integration."""
    
    print("🧪 === SURVIVAL Q-LEARNING INTEGRATION TEST ===")
    print("This test will:")
    print("1. Initialize training environment with survival learning")
    print("2. Run training for 2 minutes")
    print("3. Report survival learning progress")
    print("4. Show evidence of food-seeking behavior")
    print()
    
    try:
        # Initialize environment (this will automatically upgrade agents)
        print("🔧 Initializing training environment...")
        env = TrainingEnvironment(num_agents=10, enable_evaluation=False)  # Small test population
        
        # Start training
        print("🚀 Starting training...")
        env.start()
        
        # Monitor for 2 minutes
        print("⏱️  Monitoring for 2 minutes...")
        start_time = time.time()
        
        for i in range(24):  # 24 x 5 seconds = 2 minutes
            time.sleep(5)
            elapsed = time.time() - start_time
            
            print(f"\n📊 === Progress Report ({elapsed:.0f}s) ===")
            
            # Check if survival adapters exist
            if hasattr(env, 'survival_adapters') and env.survival_adapters:
                print(f"✅ Survival adapters active: {len(env.survival_adapters)}")
                
                # Sample first few adapters for detailed stats
                for j, adapter in enumerate(env.survival_adapters[:3]):
                    try:
                        stats = adapter.get_learning_stats()
                        agent_id = adapter.agent.id
                        
                        print(f"🤖 Agent {agent_id}:")
                        print(f"   Learning Stage: {stats.get('learning_stage', 'unknown')}")
                        print(f"   Food Consumed: {stats.get('food_consumed', 0)}")
                        print(f"   Energy Gained: {stats.get('energy_gained', 0.0):.2f}")
                        print(f"   Q-table States: {stats.get('total_states', 0)}")
                        
                        # Check energy levels
                        if agent_id in env.agent_energy_levels:
                            energy = env.agent_energy_levels[agent_id]
                            print(f"   Current Energy: {energy:.2f}")
                            
                            if energy > 0.8:
                                print(f"   🟢 HIGH ENERGY - Learning exploration")
                            elif energy < 0.4:
                                print(f"   🟡 LOW ENERGY - Should seek food!")
                            elif energy < 0.2:
                                print(f"   🔴 CRITICAL ENERGY - Survival mode!")
                        
                    except Exception as e:
                        print(f"   ⚠️ Error getting stats: {e}")
                
                # Check global survival stats
                if hasattr(env, 'survival_learning_stats'):
                    global_stats = env.survival_learning_stats
                    print(f"\n🌍 Global Survival Stats:")
                    print(f"   Total Food Consumed: {global_stats.get('total_food_consumed', 0)}")
                    print(f"   Average Learning Stage: {global_stats.get('average_learning_stage', 'unknown')}")
                    
                    stage_dist = global_stats.get('stage_distribution', {})
                    print(f"   Stage Distribution:")
                    for stage, count in stage_dist.items():
                        print(f"     {stage}: {count} agents")
            
            else:
                print("❌ Survival adapters not found!")
                print("   This means the integration failed to initialize.")
                break
            
            # Check ecosystem
            print(f"\n🌿 Ecosystem Status:")
            food_count = len(env.ecosystem_dynamics.food_sources) if hasattr(env, 'ecosystem_dynamics') else 0
            print(f"   Food Sources: {food_count}")
            
            # Check for evidence of food seeking
            if hasattr(env, 'ecosystem_dynamics') and env.ecosystem_dynamics.food_sources:
                food_seeking_evidence = 0
                for agent in env.agents[:5]:  # Check first 5 agents
                    if getattr(agent, '_destroyed', False) or not agent.body:
                        continue
                        
                    agent_pos = (agent.body.position.x, agent.body.position.y)
                    
                    # Find nearest food
                    nearest_food_dist = float('inf')
                    for food in env.ecosystem_dynamics.food_sources:
                        food_pos = food.position
                        dist = ((agent_pos[0] - food_pos[0])**2 + (agent_pos[1] - food_pos[1])**2)**0.5
                        nearest_food_dist = min(nearest_food_dist, dist)
                    
                    if nearest_food_dist < 5.0:  # Close to food
                        food_seeking_evidence += 1
                
                print(f"   Agents near food: {food_seeking_evidence}/5 (evidence of food-seeking)")
                
                if food_seeking_evidence >= 2:
                    print("   ✅ GOOD: Multiple agents showing food-seeking behavior!")
                elif food_seeking_evidence >= 1:
                    print("   🟡 FAIR: Some food-seeking behavior detected")
                else:
                    print("   🔴 POOR: No clear food-seeking behavior yet")
        
        print(f"\n🏁 === TEST COMPLETE ===")
        print("Results analysis:")
        
        # Final analysis
        if hasattr(env, 'survival_adapters') and env.survival_adapters:
            print("✅ Survival Q-learning integration: SUCCESS")
            
            total_food = env.survival_learning_stats.get('total_food_consumed', 0)
            if total_food > 0:
                print(f"✅ Food consumption: SUCCESS ({total_food} food consumed)")
            else:
                print("🟡 Food consumption: MINIMAL (may need more time)")
            
            avg_stage = env.survival_learning_stats.get('average_learning_stage', 'basic_movement')
            if avg_stage == 'survival_mastery':
                print("✅ Learning progression: EXCELLENT (reached mastery)")
            elif avg_stage == 'food_seeking':
                print("✅ Learning progression: GOOD (reached food seeking)")
            else:
                print("🟡 Learning progression: BASIC (still in movement stage)")
        
        else:
            print("❌ Survival Q-learning integration: FAILED")
            print("   Check console for error messages during initialization")
        
        print("\n📝 Next steps:")
        print("1. If successful, run the full training: python train_robots_web_visual.py")
        print("2. Watch for food-seeking behavior in the web interface")
        print("3. Monitor agent energy levels and survival stats")
        print("4. Look for 'Agent X advanced to FOOD_SEEKING stage' messages")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        try:
            if 'env' in locals():
                env.stop()
                print("🛑 Training stopped")
        except:
            pass

if __name__ == "__main__":
    test_survival_integration() 