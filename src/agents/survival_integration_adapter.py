"""
Integration adapter to connect enhanced survival Q-learning with existing agents.
This bridges the gap between the ecosystem and Q-learning systems.
"""

import numpy as np
import math
from typing import Dict, Any, Tuple, Optional
from .enhanced_survival_q_learning import (
    EnhancedSurvivalQLearning, 
    SurvivalState, 
    SurvivalStateProcessor,
    SurvivalRewardCalculator
)


class SurvivalQLearningAdapter:
    """
    Adapter to integrate survival Q-learning with existing crawling agents.
    Handles the transition from basic movement-focused Q-learning to survival-focused learning.
    """
    
    def __init__(self, agent, ecosystem_interface):
        self.agent = agent
        self.ecosystem_interface = ecosystem_interface
        
        # Initialize enhanced survival Q-learning
        self.survival_q_learning = EnhancedSurvivalQLearning(
            action_count=len(agent.actions)
        )
        
        # State tracking
        self.previous_survival_state = None
        self.current_survival_state = None
        self.previous_energy_level = 1.0
        
        # Performance tracking
        self.food_seeking_attempts = 0
        self.successful_consumptions = 0
        self.survival_episodes = 0
        
        # Integration flags
        self.use_survival_learning = True
        self.survival_learning_weight = 0.7  # Balance between old and new systems
        
        print(f"🧬 Initialized Survival Q-Learning Adapter for agent {agent.id}")
    
    def get_enhanced_state(self) -> SurvivalState:
        """Get enhanced survival state representation."""
        # Gather ecosystem data
        ecosystem_data = self._gather_ecosystem_data()
        
        # Process into survival state
        survival_state = self.survival_q_learning.process_state(self.agent, ecosystem_data)
        
        return survival_state
    
    def get_enhanced_reward(self, old_state: SurvivalState, new_state: SurvivalState, 
                          action: Tuple[float, float]) -> float:
        """Calculate enhanced survival-focused reward."""
        # Gather agent data for reward calculation
        agent_data = self._gather_agent_data_for_reward()
        
        # Calculate survival reward
        survival_reward = self.survival_q_learning.calculate_survival_reward(
            old_state, new_state, action, agent_data
        )
        
        # Get original movement-based reward for comparison
        prev_x = getattr(self.agent, 'prev_x', self.agent.body.position.x - 0.01)
        original_reward = self.agent.get_crawling_reward(prev_x)
        
        # Blend rewards (gradually transition to survival focus)
        blended_reward = (
            self.survival_learning_weight * survival_reward + 
            (1.0 - self.survival_learning_weight) * original_reward
        )
        
        return blended_reward
    
    def choose_enhanced_action(self, epsilon: float) -> int:
        """Choose action using enhanced survival Q-learning."""
        current_state = self.get_enhanced_state()
        agent_data = self._gather_agent_data_for_action()
        
        # Use survival-specific action selection
        action_idx = self.survival_q_learning.choose_survival_action(
            current_state, epsilon, agent_data
        )
        
        return action_idx
    
    def update_enhanced_q_learning(self, action_idx: int, reward: float):
        """Update Q-learning with enhanced survival focus."""
        if self.previous_survival_state is None:
            self.previous_survival_state = self.get_enhanced_state()
            return
        
        current_state = self.get_enhanced_state()
        
        # Gather agent data for learning rate adjustment
        agent_data = self._gather_agent_data_for_learning()
        
        # Update with survival-specific prioritization
        self.survival_q_learning.update_with_survival_priority(
            state=self.previous_survival_state,
            action=action_idx,
            reward=reward,
            next_state=current_state,
            learning_rate=self.agent.learning_rate,
            discount_factor=self.agent.discount_factor,
            agent_data=agent_data
        )
        
        # Update state tracking
        self.previous_survival_state = current_state
        
        # Periodic high-value experience replay
        if self.agent.steps % 50 == 0:  # Every 50 steps
            self.survival_q_learning.replay_high_value_experiences(batch_size=16)
    
    def _gather_ecosystem_data(self) -> Dict[str, Any]:
        """Gather ecosystem data for state processing."""
        ecosystem_data = {
            'food_sources': [],
            'agent_energy': {},
            'agent_health': {},
            'nearby_agents': []
        }
        
        # Get ecosystem data from the training environment
        if hasattr(self.ecosystem_interface, 'ecosystem_dynamics'):
            ecosystem = self.ecosystem_interface.ecosystem_dynamics
            
            # Food sources
            ecosystem_data['food_sources'] = [
                {
                    'position': food.position,
                    'type': food.food_type,
                    'amount': food.amount,
                    'max_capacity': food.max_capacity
                }
                for food in ecosystem.food_sources if food.amount > 0
            ]
            
            # Agent energy levels
            if hasattr(self.ecosystem_interface, 'agent_energy_levels'):
                ecosystem_data['agent_energy'] = self.ecosystem_interface.agent_energy_levels
            
            # Agent health data
            if hasattr(self.ecosystem_interface, 'agent_health'):
                ecosystem_data['agent_health'] = self.ecosystem_interface.agent_health
        
        return ecosystem_data
    
    def _gather_agent_data_for_reward(self) -> Dict[str, Any]:
        """Gather agent-specific data for reward calculation."""
        current_energy = self._get_current_energy_level()
        energy_change = current_energy - self.previous_energy_level
        self.previous_energy_level = current_energy
        
        # Calculate displacement
        current_x = self.agent.body.position.x
        prev_x = getattr(self.agent, 'prev_x', current_x - 0.01)
        displacement = current_x - prev_x
        
        # Calculate velocity
        velocity = self.agent.body.linearVelocity
        velocity_magnitude = math.sqrt(velocity.x**2 + velocity.y**2)
        
        return {
            'energy_change': energy_change,
            'energy_level': current_energy,
            'health_level': self._get_current_health_level(),
            'displacement': displacement,
            'velocity_magnitude': velocity_magnitude,
            'body_angle': self.agent.body.angle,
            'exploration_score': self._calculate_exploration_score()
        }
    
    def _gather_agent_data_for_action(self) -> Dict[str, Any]:
        """Gather agent data for action selection."""
        return {
            'energy_level': self._get_current_energy_level(),
            'health_level': self._get_current_health_level(),
            'body_angle': self.agent.body.angle,
            'velocity_magnitude': math.sqrt(
                self.agent.body.linearVelocity.x**2 + 
                self.agent.body.linearVelocity.y**2
            )
        }
    
    def _gather_agent_data_for_learning(self) -> Dict[str, Any]:
        """Gather agent data for learning rate adjustment."""
        return {
            'agent_id': self.agent.id,
            'energy_level': self._get_current_energy_level(),
            'health_level': self._get_current_health_level(),
            'steps_alive': self.agent.steps,
            'recent_performance': self._calculate_recent_performance()
        }
    
    def _get_current_energy_level(self) -> float:
        """Get current energy level from ecosystem."""
        if hasattr(self.ecosystem_interface, 'agent_energy_levels'):
            return self.ecosystem_interface.agent_energy_levels.get(self.agent.id, 1.0)
        return 1.0
    
    def _get_current_health_level(self) -> float:
        """Get current health level from ecosystem."""
        if hasattr(self.ecosystem_interface, 'agent_health'):
            agent_health = self.ecosystem_interface.agent_health.get(self.agent.id, {})
            return agent_health.get('health', 1.0)
        return 1.0
    
    def _calculate_exploration_score(self) -> float:
        """Calculate exploration score based on Q-table coverage."""
        if hasattr(self.agent, 'q_table') and hasattr(self.agent.q_table, 'state_coverage'):
            coverage = len(self.agent.q_table.state_coverage)
            return min(1.0, coverage / 1000.0)  # Normalize
        return 0.0
    
    def _calculate_recent_performance(self) -> float:
        """Calculate recent performance metric."""
        if hasattr(self.agent, 'recent_rewards') and self.agent.recent_rewards:
            recent_rewards = list(self.agent.recent_rewards)[-20:]  # Last 20 rewards
            return sum(recent_rewards) / len(recent_rewards)
        return 0.0
    
    def get_survival_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about survival learning progress."""
        return {
            'learning_stage': self.survival_q_learning.learning_stage,
            'experiences_in_stage': self.survival_q_learning.experiences_in_stage,
            'high_value_experiences': len(self.survival_q_learning.high_value_experiences),
            'food_seeking_attempts': self.food_seeking_attempts,
            'successful_consumptions': self.successful_consumptions,
            'survival_episodes': self.survival_episodes,
            'consumption_success_rate': (
                self.successful_consumptions / max(1, self.food_seeking_attempts)
            ),
            'current_energy': self._get_current_energy_level(),
            'current_health': self._get_current_health_level()
        }
    
    def reset_for_new_episode(self):
        """Reset adapter for new episode."""
        self.previous_survival_state = None
        self.current_survival_state = None
        self.previous_energy_level = 1.0
        self.survival_episodes += 1
    
    def should_use_survival_learning(self) -> bool:
        """Determine if survival learning should be used."""
        # Gradually transition to survival learning based on agent's development
        if self.agent.steps < 100:
            return False  # Use basic learning initially
        elif self.agent.steps < 500:
            self.survival_learning_weight = 0.3  # Gradual introduction
            return True
        else:
            self.survival_learning_weight = 0.8  # Full survival focus
            return True


def integrate_survival_learning_with_agent(agent, ecosystem_interface):
    """
    Factory function to integrate survival learning with an existing agent.
    
    Args:
        agent: Existing crawling agent
        ecosystem_interface: Interface to ecosystem dynamics
        
    Returns:
        SurvivalQLearningAdapter instance
    """
    
    # Create survival adapter
    survival_adapter = SurvivalQLearningAdapter(agent, ecosystem_interface)
    
    # Modify agent's step method to use survival learning
    original_step = agent.step
    original_choose_action = agent.choose_action
    
    def enhanced_step(dt):
        """Enhanced step method with survival learning integration."""
        # Run original step logic
        original_step(dt)
        
        # Add survival learning if appropriate
        if survival_adapter.should_use_survival_learning():
            # Get enhanced state and reward
            current_state = survival_adapter.get_enhanced_state()
            
            # Calculate enhanced reward
            if hasattr(survival_adapter, 'previous_survival_state') and \
               survival_adapter.previous_survival_state is not None:
                
                enhanced_reward = survival_adapter.get_enhanced_reward(
                    survival_adapter.previous_survival_state,
                    current_state,
                    agent.current_action_tuple
                )
                
                # Update with enhanced Q-learning
                survival_adapter.update_enhanced_q_learning(
                    agent.current_action, enhanced_reward
                )
            
            # Store current state for next iteration
            survival_adapter.previous_survival_state = current_state
    
    def enhanced_choose_action():
        """Enhanced action selection with survival focus."""
        if survival_adapter.should_use_survival_learning():
            return survival_adapter.choose_enhanced_action(agent.epsilon)
        else:
            return original_choose_action()
    
    # Replace agent methods
    agent.step = enhanced_step
    agent.choose_action = enhanced_choose_action
    
    # Add survival adapter as attribute
    agent.survival_adapter = survival_adapter
    
    print(f"✅ Successfully integrated survival learning with agent {agent.id}")
    
    return survival_adapter


def create_ecosystem_interface(training_environment):
    """
    Create an interface to connect with the ecosystem dynamics.
    
    Args:
        training_environment: The main training environment instance
        
    Returns:
        Object that provides access to ecosystem data
    """
    
    class EcosystemInterface:
        def __init__(self, env):
            self.env = env
            
        @property
        def ecosystem_dynamics(self):
            return getattr(self.env, 'ecosystem_dynamics', None)
            
        @property
        def agent_energy_levels(self):
            return getattr(self.env, 'agent_energy_levels', {})
            
        @property
        def agent_health(self):
            return getattr(self.env, 'agent_health', {})
            
        @property
        def environmental_system(self):
            return getattr(self.env, 'environmental_system', None)
    
    return EcosystemInterface(training_environment) 