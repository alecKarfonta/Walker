"""
Practical example of how to integrate the survival Q-learning system 
with the existing training environment.

This file shows the specific modifications needed to enable enhanced survival learning.
"""

from typing import List, Dict, Any
from .survival_integration_adapter import (
    integrate_survival_learning_with_agent, 
    create_ecosystem_interface
)


def upgrade_training_environment_to_survival_learning(training_env):
    """
    Upgrade an existing training environment to use survival-focused Q-learning.
    
    Args:
        training_env: Existing TrainingEnvironment instance
    """
    
    print("🧬 === UPGRADING TO SURVIVAL Q-LEARNING ===")
    
    # Step 1: Create ecosystem interface
    ecosystem_interface = create_ecosystem_interface(training_env)
    print("✅ Created ecosystem interface")
    
    # Step 2: Integrate survival learning with each agent
    survival_adapters = []
    for agent in training_env.agents:
        if not getattr(agent, '_destroyed', False) and agent.body:
            try:
                survival_adapter = integrate_survival_learning_with_agent(
                    agent, ecosystem_interface
                )
                survival_adapters.append(survival_adapter)
                print(f"✅ Upgraded agent {agent.id} to survival learning")
            except Exception as e:
                print(f"❌ Failed to upgrade agent {agent.id}: {e}")
    
    # Step 3: Store survival adapters in training environment
    training_env.survival_adapters = survival_adapters
    training_env.ecosystem_interface = ecosystem_interface
    
    # Step 4: Add survival-specific monitoring
    _add_survival_monitoring(training_env)
    
    print(f"🧬 === UPGRADE COMPLETE: {len(survival_adapters)} agents upgraded ===")
    
    return survival_adapters


def _add_survival_monitoring(training_env):
    """Add survival-specific monitoring to the training environment."""
    
    # Store original _update_statistics method
    original_update_statistics = training_env._update_statistics
    
    def enhanced_update_statistics():
        """Enhanced statistics with survival metrics."""
        # Run original statistics update
        original_update_statistics()
        
        # Add survival-specific statistics
        if hasattr(training_env, 'survival_adapters'):
            survival_stats = _calculate_survival_statistics(training_env.survival_adapters)
            
            # Add to population stats
            training_env.population_stats.update({
                'survival_metrics': survival_stats,
                'learning_stages': _get_learning_stage_distribution(training_env.survival_adapters),
                'energy_distribution': _get_energy_distribution(training_env),
                'food_seeking_efficiency': _calculate_food_seeking_efficiency(training_env.survival_adapters)
            })
    
    # Replace the method
    training_env._update_statistics = enhanced_update_statistics


def _calculate_survival_statistics(survival_adapters: List) -> Dict[str, Any]:
    """Calculate population-level survival statistics."""
    if not survival_adapters:
        return {}
    
    # Gather individual survival stats
    individual_stats = [adapter.get_survival_learning_stats() for adapter in survival_adapters]
    
    # Calculate population averages
    total_adapters = len(individual_stats)
    avg_consumption_rate = sum(stat['consumption_success_rate'] for stat in individual_stats) / total_adapters
    avg_energy = sum(stat['current_energy'] for stat in individual_stats) / total_adapters
    avg_health = sum(stat['current_health'] for stat in individual_stats) / total_adapters
    
    # Count agents in each learning stage
    stage_counts = {}
    for stat in individual_stats:
        stage = stat['learning_stage']
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    
    # High-value experience totals
    total_high_value_experiences = sum(stat['high_value_experiences'] for stat in individual_stats)
    
    return {
        'total_agents': total_adapters,
        'avg_consumption_success_rate': avg_consumption_rate,
        'avg_energy_level': avg_energy,
        'avg_health_level': avg_health,
        'learning_stage_distribution': stage_counts,
        'total_high_value_experiences': total_high_value_experiences,
        'agents_in_crisis': sum(1 for stat in individual_stats if stat['current_energy'] < 0.2),
        'agents_thriving': sum(1 for stat in individual_stats if stat['current_energy'] > 0.8)
    }


def _get_learning_stage_distribution(survival_adapters: List) -> Dict[str, int]:
    """Get distribution of agents across learning stages."""
    distribution = {}
    for adapter in survival_adapters:
        stage = adapter.survival_q_learning.learning_stage
        distribution[stage] = distribution.get(stage, 0) + 1
    return distribution


def _get_energy_distribution(training_env) -> Dict[str, int]:
    """Get distribution of energy levels across population."""
    if not hasattr(training_env, 'agent_energy_levels'):
        return {}
    
    energy_levels = list(training_env.agent_energy_levels.values())
    if not energy_levels:
        return {}
    
    return {
        'critical': sum(1 for e in energy_levels if e < 0.2),
        'low': sum(1 for e in energy_levels if 0.2 <= e < 0.4),
        'medium': sum(1 for e in energy_levels if 0.4 <= e < 0.7),
        'high': sum(1 for e in energy_levels if 0.7 <= e < 0.9),
        'full': sum(1 for e in energy_levels if e >= 0.9)
    }


def _calculate_food_seeking_efficiency(survival_adapters: List) -> float:
    """Calculate overall food-seeking efficiency."""
    if not survival_adapters:
        return 0.0
    
    total_attempts = sum(adapter.food_seeking_attempts for adapter in survival_adapters)
    total_successes = sum(adapter.successful_consumptions for adapter in survival_adapters)
    
    return total_successes / max(1, total_attempts)


def add_survival_learning_to_new_agents(training_env, new_agents: List):
    """
    Add survival learning to newly created agents.
    
    Args:
        training_env: Training environment with survival learning
        new_agents: List of new agents to upgrade
    """
    
    if not hasattr(training_env, 'ecosystem_interface'):
        print("❌ Training environment not upgraded to survival learning")
        return []
    
    survival_adapters = []
    for agent in new_agents:
        if not getattr(agent, '_destroyed', False) and agent.body:
            try:
                survival_adapter = integrate_survival_learning_with_agent(
                    agent, training_env.ecosystem_interface
                )
                survival_adapters.append(survival_adapter)
                training_env.survival_adapters.append(survival_adapter)
                print(f"✅ Added survival learning to new agent {agent.id}")
            except Exception as e:
                print(f"❌ Failed to add survival learning to agent {agent.id}: {e}")
    
    return survival_adapters


def create_survival_dashboard_data(training_env) -> Dict[str, Any]:
    """
    Create dashboard data specifically for survival learning monitoring.
    
    Args:
        training_env: Training environment with survival learning
        
    Returns:
        Dictionary containing survival dashboard data
    """
    
    if not hasattr(training_env, 'survival_adapters'):
        return {'error': 'Survival learning not enabled'}
    
    # Get survival statistics
    survival_stats = training_env.population_stats.get('survival_metrics', {})
    
    # Get individual agent details for top performers
    top_survivors = []
    if training_env.survival_adapters:
        # Sort by energy level and consumption success rate
        sorted_adapters = sorted(
            training_env.survival_adapters,
            key=lambda a: (a._get_current_energy_level(), a.successful_consumptions),
            reverse=True
        )
        
        for adapter in sorted_adapters[:10]:  # Top 10
            agent_stats = adapter.get_survival_learning_stats()
            top_survivors.append({
                'agent_id': adapter.agent.id,
                'energy': agent_stats['current_energy'],
                'health': agent_stats['current_health'],
                'consumption_rate': agent_stats['consumption_success_rate'],
                'learning_stage': agent_stats['learning_stage'],
                'position': (
                    adapter.agent.body.position.x if adapter.agent.body else 0,
                    adapter.agent.body.position.y if adapter.agent.body else 0
                )
            })
    
    # Food source information
    food_sources = []
    if hasattr(training_env, 'ecosystem_dynamics') and training_env.ecosystem_dynamics:
        for food in training_env.ecosystem_dynamics.food_sources:
            if food.amount > 0:
                food_sources.append({
                    'position': food.position,
                    'type': food.food_type,
                    'amount': food.amount,
                    'capacity': food.max_capacity,
                    'utilization': food.amount / food.max_capacity
                })
    
    # Learning progress metrics
    learning_progress = {
        'stage_transitions': {},
        'convergence_rates': {},
        'exploration_progress': {}
    }
    
    for adapter in training_env.survival_adapters:
        stage = adapter.survival_q_learning.learning_stage
        if stage not in learning_progress['stage_transitions']:
            learning_progress['stage_transitions'][stage] = 0
        learning_progress['stage_transitions'][stage] += 1
    
    return {
        'survival_overview': survival_stats,
        'top_survivors': top_survivors,
        'food_sources': food_sources,
        'learning_progress': learning_progress,
        'energy_distribution': training_env.population_stats.get('energy_distribution', {}),
        'crisis_alerts': {
            'agents_in_crisis': survival_stats.get('agents_in_crisis', 0),
            'low_food_sources': len([f for f in food_sources if f['utilization'] < 0.3]),
            'stagnant_learners': len([
                a for a in training_env.survival_adapters 
                if a.survival_q_learning.learning_stage == 'basic_movement' and 
                a.survival_q_learning.experiences_in_stage > 200
            ])
        }
    }


def print_survival_learning_report(training_env):
    """
    Print a comprehensive survival learning report.
    
    Args:
        training_env: Training environment with survival learning
    """
    
    print("\n🧬 === SURVIVAL LEARNING REPORT ===")
    
    if not hasattr(training_env, 'survival_adapters'):
        print("❌ Survival learning not enabled")
        return
    
    survival_stats = training_env.population_stats.get('survival_metrics', {})
    
    print(f"📊 Population Overview:")
    print(f"   Total Agents: {survival_stats.get('total_agents', 0)}")
    print(f"   Average Energy: {survival_stats.get('avg_energy_level', 0):.2%}")
    print(f"   Average Health: {survival_stats.get('avg_health_level', 0):.2%}")
    print(f"   Food Success Rate: {survival_stats.get('avg_consumption_success_rate', 0):.2%}")
    
    print(f"\n🎓 Learning Stages:")
    stage_dist = survival_stats.get('learning_stage_distribution', {})
    for stage, count in stage_dist.items():
        print(f"   {stage}: {count} agents")
    
    print(f"\n⚡ Energy Status:")
    energy_dist = training_env.population_stats.get('energy_distribution', {})
    for level, count in energy_dist.items():
        print(f"   {level.capitalize()}: {count} agents")
    
    print(f"\n🚨 Crisis Indicators:")
    print(f"   Agents in Crisis: {survival_stats.get('agents_in_crisis', 0)}")
    print(f"   Agents Thriving: {survival_stats.get('agents_thriving', 0)}")
    
    print(f"\n💾 Learning Data:")
    print(f"   High-Value Experiences: {survival_stats.get('total_high_value_experiences', 0)}")
    
    print("🧬 === END REPORT ===\n")


# Example usage function
def example_integration():
    """
    Example of how to integrate survival learning with an existing training setup.
    """
    
    # This would be called after creating your training environment
    # training_env = TrainingEnvironment(num_agents=30)
    # training_env.start()
    
    # Upgrade to survival learning
    # survival_adapters = upgrade_training_environment_to_survival_learning(training_env)
    
    # Monitor survival learning progress
    # def monitoring_loop():
    #     while training_env.is_running:
    #         time.sleep(30)  # Every 30 seconds
    #         print_survival_learning_report(training_env)
    #         
    #         # Get dashboard data for web interface
    #         dashboard_data = create_survival_dashboard_data(training_env)
    #         # Send to web interface or log to file
    
    # Optional: Run monitoring in separate thread
    # import threading
    # monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
    # monitoring_thread.start()
    
    pass 