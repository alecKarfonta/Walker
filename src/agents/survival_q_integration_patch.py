"""
Survival Q-Learning Integration Patch
Connects the existing sophisticated Q-learning system with survival features.
NO deep learning required - uses existing EnhancedQTable infrastructure.
"""

import numpy as np
from typing import Dict, Any, Tuple, List
from .q_table import EnhancedQTable
from .enhanced_survival_q_learning import SurvivalState, SurvivalStateProcessor, SurvivalRewardCalculator


class SurvivalAwareQLearning:
    """
    Enhances existing Q-learning agents with survival awareness.
    Uses existing EnhancedQTable but with survival-focused state and rewards.
    """
    
    def __init__(self, existing_agent, ecosystem_interface):
        self.agent = existing_agent
        self.ecosystem_interface = ecosystem_interface
        
        # Enhanced state space: 8 dimensions instead of 3
        # [shoulder_bin(8), elbow_bin(8), energy_bin(5), food_dir_bin(8), 
        #  food_dist_bin(5), velocity_bin(4), contact_bin(2), social_bin(4)]
        self.enhanced_state_dims = [8, 8, 5, 8, 5, 4, 2, 4]
        self.total_states = np.prod(self.enhanced_state_dims)  # ~40,960 states
        
        print(f"🧠 Enhanced state space: {self.enhanced_state_dims} = {self.total_states:,} total states")
        
        # Replace agent's Q-table with survival-aware version
        self.original_q_table = existing_agent.q_table
        existing_agent.q_table = EnhancedQTable(
            action_count=existing_agent.action_size,
            default_value=0.0,
            confidence_threshold=10,  # Reduced for faster confidence
            exploration_bonus=0.2     # Higher bonus for complex state space
        )
        
        # Initialize survival components
        self.state_processor = SurvivalStateProcessor()
        self.reward_calculator = SurvivalRewardCalculator()
        
        # Learning stage progression
        self.learning_stage = 'basic_movement'
        self.stage_experience = 0
        self.stage_thresholds = {
            'basic_movement': 500,    # 500 experiences
            'food_seeking': 1500,     # 1500 experiences  
            'survival_mastery': float('inf')
        }
        
        # Performance tracking
        self.survival_stats = {
            'food_consumed': 0,
            'energy_gained': 0.0,
            'survival_time': 0,
            'death_count': 0
        }
        
        # Transfer knowledge from original Q-table
        self._transfer_basic_knowledge()
    
    def _transfer_basic_knowledge(self):
        """Transfer basic movement knowledge from original Q-table."""
        if not hasattr(self.original_q_table, 'q_values'):
            return
            
        transferred_count = 0
        
        try:
            # Transfer states where we have confidence
            for state_key, action_values in self.original_q_table.q_values.items():
                # Parse old 3D state: "shoulder_bin,elbow_bin,vel_bin"
                old_state_parts = state_key.split(',')
                if len(old_state_parts) == 3:
                    shoulder_bin = int(old_state_parts[0])
                    elbow_bin = int(old_state_parts[1])
                    vel_bin = int(old_state_parts[2])
                    
                    # Map to enhanced state with default survival values
                    # [shoulder, elbow, energy=4(full), food_dir=0, food_dist=2(medium), vel, contact=1, social=0]
                    enhanced_state = (shoulder_bin, elbow_bin, 4, 0, 2, vel_bin, 1, 0)
                    
                    # Transfer Q-values with reduced confidence (50% of original)
                    for action_idx, q_value in enumerate(action_values):
                        if isinstance(q_value, (int, float)) and abs(q_value) > 0.01:
                            reduced_q = q_value * 0.5  # Reduce confidence
                            self.agent.q_table.set_q_value(enhanced_state, action_idx, reduced_q)
                            transferred_count += 1
        except Exception as e:
            print(f"⚠️ Knowledge transfer partially failed: {e}")
        
        print(f"🔄 Transferred {transferred_count} Q-values from basic movement to survival learning")
    
    def get_enhanced_state(self) -> Tuple[int, ...]:
        """Get survival-aware state representation."""
        
        try:
            # Get basic agent data
            agent_pos = (self.agent.body.position.x, self.agent.body.position.y) if self.agent.body else (0, 0)
            
            # Physical state (existing logic)
            shoulder_angle = self.agent.upper_arm.angle if self.agent.upper_arm else 0.0
            elbow_angle = self.agent.lower_arm.angle if self.agent.lower_arm else 0.0
            
            shoulder_bin = max(0, min(7, int((shoulder_angle + np.pi) / (2 * np.pi) * 8)))
            elbow_bin = max(0, min(7, int((elbow_angle + np.pi) / (2 * np.pi) * 8)))
            
            # Survival state from ecosystem
            ecosystem_data = self.ecosystem_interface.get_agent_survival_data(self.agent.id, agent_pos)
            
            energy_level = ecosystem_data.get('energy_level', 1.0)
            energy_bin = max(0, min(4, int(energy_level * 5)))  # 0-4
            
            # Environmental awareness
            food_direction = ecosystem_data.get('nearest_food_direction', 0.0)
            food_direction_bin = max(0, min(7, int(food_direction / (2 * np.pi) * 8)))
            
            food_distance = ecosystem_data.get('nearest_food_distance', 10.0)
            food_distance_bin = max(0, min(4, int(min(food_distance / 4.0, 1.0) * 5)))
            
            # Physical context
            if self.agent.body:
                velocity = self.agent.body.linearVelocity
                vel_magnitude = min(4.0, (velocity.x**2 + velocity.y**2)**0.5)
            else:
                vel_magnitude = 0.0
            velocity_bin = max(0, min(3, int(vel_magnitude)))
            
            # Ground contact (simplified)
            contact_bin = 1  # Assume in contact for now
            
            # Social context
            nearby_agents = ecosystem_data.get('nearby_agents', 0)
            social_bin = max(0, min(3, nearby_agents))
            
            state = (shoulder_bin, elbow_bin, energy_bin, food_direction_bin, 
                    food_distance_bin, velocity_bin, contact_bin, social_bin)
            
            return state
            
        except Exception as e:
            print(f"⚠️ Error getting enhanced state for agent {self.agent.id}: {e}")
            # Fallback to basic state
            return (0, 0, 4, 0, 2, 0, 1, 0)
    
    def calculate_survival_reward(self, old_state: Tuple[int, ...], new_state: Tuple[int, ...], 
                                action: int) -> float:
        """Calculate survival-focused reward."""
        
        try:
            # Get ecosystem data for reward calculation
            agent_pos = (self.agent.body.position.x, self.agent.body.position.y) if self.agent.body else (0, 0)
            ecosystem_data = self.ecosystem_interface.get_agent_survival_data(self.agent.id, agent_pos)
            
            reward = 0.0
            
            # PRIMARY: Energy changes (most important) - SCALED DOWN
            old_energy_bin = old_state[2] if len(old_state) > 2 else 4
            new_energy_bin = new_state[2] if len(new_state) > 2 else 4
            energy_change = new_energy_bin - old_energy_bin
            
            if energy_change > 0:
                reward += 0.2 * energy_change  # REDUCED from 2.0 to 0.2 for Q-learning scale
                # Track energy gain
                self.survival_stats['energy_gained'] += energy_change
                self.survival_stats['food_consumed'] += 1
            elif energy_change < 0:
                reward += 0.05 * energy_change   # REDUCED from 0.5 to 0.05 for Q-learning scale
            
            # SECONDARY: Food-seeking behavior - SCALED DOWN
            old_food_dist_bin = old_state[4] if len(old_state) > 4 else 2
            new_food_dist_bin = new_state[4] if len(new_state) > 4 else 2
            
            if new_food_dist_bin < old_food_dist_bin:
                reward += 0.5  # REDUCED from 3.0
            elif new_food_dist_bin > old_food_dist_bin:
                reward -= 0.2  # REDUCED from 1.0
            
            # TERTIARY: Movement efficiency (reduced importance) - KEPT SAME
            velocity_bin = new_state[5] if len(new_state) > 5 else 0
            if velocity_bin > 0:
                reward += 0.1  # REDUCED from 0.5
            
            # SURVIVAL PENALTIES - SCALED DOWN
            energy_level = ecosystem_data.get('energy_level', 1.0)
            if energy_level < 0.2:
                reward -= 1.0  # REDUCED from 5.0
            elif energy_level < 0.4:
                reward -= 0.2  # REDUCED from 1.0
            
            # Learning stage adjustments - SCALED DOWN
            if self.learning_stage == 'basic_movement':
                reward *= 1.5  # REDUCED from 2.0
            elif self.learning_stage == 'food_seeking':
                # Boost food-related rewards
                if energy_change > 0:
                    reward += 1.0  # REDUCED from 10.0
            
            # Bound the reward to match crawling reward scale
            reward = np.clip(reward, -2.0, 5.0)  # REDUCED from (-10.0, 50.0)
            
            return reward
            
        except Exception as e:
            print(f"⚠️ Error calculating survival reward: {e}")
            return 0.0
    
    def choose_survival_action(self, epsilon: float) -> int:
        """Choose action with survival-aware epsilon adjustment."""
        
        state = self.get_enhanced_state()
        
        # Get ecosystem data for epsilon adjustment
        try:
            agent_pos = (self.agent.body.position.x, self.agent.body.position.y) if self.agent.body else (0, 0)
            ecosystem_data = self.ecosystem_interface.get_agent_survival_data(self.agent.id, agent_pos)
            energy_level = ecosystem_data.get('energy_level', 1.0)
            
            # Adjust epsilon based on survival situation
            if energy_level < 0.2:
                adjusted_epsilon = epsilon * 0.3  # Less exploration when in danger
            elif energy_level > 0.8:
                adjusted_epsilon = min(epsilon * 1.5, 0.8)  # More exploration when safe
            else:
                adjusted_epsilon = epsilon
                
        except Exception as e:
            adjusted_epsilon = epsilon
        
        # Use enhanced epsilon-greedy from existing Q-table
        action = self.agent.q_table.enhanced_epsilon_greedy(
            state, 
            adjusted_epsilon, 
            use_exploration_bonus=True
        )
        
        return action
    
    def update_survival_learning(self, old_state: Tuple[int, ...], action: int, 
                               new_state: Tuple[int, ...]) -> Dict[str, Any]:
        """Update Q-learning with survival awareness."""
        
        # Calculate survival-focused reward
        reward = self.calculate_survival_reward(old_state, new_state, action)
        
        # Update Q-table using existing enhanced method
        self.agent.q_table.update_q_value_enhanced(
            state=old_state,
            action=action,
            reward=reward,
            next_state=new_state,
            base_learning_rate=self.agent.learning_rate,
            discount_factor=self.agent.discount_factor,
            use_adaptive_lr=True
        )
        
        # Track learning progress
        self.stage_experience += 1
        self.survival_stats['survival_time'] += 1
        
        # Check for stage progression
        self._check_stage_progression()
        
        return {
            'reward': reward,
            'learning_stage': self.learning_stage,
            'stage_experience': self.stage_experience,
            'energy_level': new_state[2] if len(new_state) > 2 else 4
        }
    
    def _check_stage_progression(self):
        """Check if agent should advance to next learning stage."""
        if self.learning_stage == 'basic_movement' and self.stage_experience >= self.stage_thresholds['basic_movement']:
            self.learning_stage = 'food_seeking'
            self.stage_experience = 0
            print(f"🎓 Agent {self.agent.id} advanced to FOOD_SEEKING stage")
            
        elif self.learning_stage == 'food_seeking' and self.stage_experience >= self.stage_thresholds['food_seeking']:
            # Additional criteria: must have successfully consumed food
            if self.survival_stats['food_consumed'] > 0:
                self.learning_stage = 'survival_mastery'
                self.stage_experience = 0
                print(f"🏆 Agent {self.agent.id} advanced to SURVIVAL_MASTERY stage")
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics."""
        base_stats = self.agent.q_table.get_stats()
        
        survival_stats = {
            'learning_stage': self.learning_stage,
            'stage_experience': self.stage_experience,
            'food_consumed': self.survival_stats['food_consumed'],
            'energy_gained': self.survival_stats['energy_gained'],
            'survival_time': self.survival_stats['survival_time'],
            'state_space_size': self.total_states,
            'enhanced_state_dims': self.enhanced_state_dims
        }
        
        return {**base_stats, **survival_stats}


# Factory function to upgrade existing agents
def upgrade_agent_to_survival_learning(agent, ecosystem_interface) -> SurvivalAwareQLearning:
    """
    Upgrade an existing agent to use survival-aware Q-learning.
    
    Args:
        agent: Existing CrawlingCrateAgent or EvolutionaryCrawlingAgent
        ecosystem_interface: Interface to ecosystem data
        
    Returns:
        SurvivalAwareQLearning wrapper
    """
    print(f"🧬 Upgrading Agent {agent.id} to survival-aware Q-learning")
    
    survival_q = SurvivalAwareQLearning(agent, ecosystem_interface)
    
    # Modify agent's step method to use survival learning
    original_step = agent.step
    
    def enhanced_step(dt):
        # Get current enhanced state
        current_state = survival_q.get_enhanced_state()
        
        # Store previous state for learning
        if hasattr(agent, '_prev_survival_state'):
            prev_state = agent._prev_survival_state
            prev_action = agent._prev_survival_action
            
            # Update survival learning
            survival_q.update_survival_learning(prev_state, prev_action, current_state)
        
        # Choose action using survival awareness
        if agent.steps % agent.action_interval == 0:
            action_idx = survival_q.choose_survival_action(agent.epsilon)
            agent.current_action = action_idx
            agent.current_action_tuple = agent.actions[action_idx]
            
            # Store for next update
            agent._prev_survival_state = current_state
            agent._prev_survival_action = action_idx
        
        # Continue with original step logic
        original_step(dt)
    
    # Replace step method
    agent.step = enhanced_step
    
    # Add survival stats accessor
    agent.get_survival_stats = survival_q.get_learning_stats
    
    print(f"✅ Agent {agent.id} upgraded! Enhanced state space: {survival_q.total_states:,} states")
    
    return survival_q 