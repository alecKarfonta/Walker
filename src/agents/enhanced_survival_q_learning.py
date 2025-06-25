"""
Enhanced Q-Learning implementation designed specifically for survival tasks.
Integrates food-seeking, energy management, and spatial awareness.
"""

import numpy as np
import math
from typing import Tuple, Dict, Any, List, Optional
from collections import deque, defaultdict
import random
from .q_table import EnhancedQTable


class SurvivalState:
    """Enhanced state representation for survival tasks."""
    
    def __init__(self):
        # Physical state (current arm configuration)
        self.shoulder_angle_bin = 0
        self.elbow_angle_bin = 0
        
        # Survival state
        self.energy_level_bin = 4  # 0-4 (critical, low, medium, high, full)
        self.health_level_bin = 4  # 0-4
        
        # Environmental awareness
        self.nearest_food_direction_bin = 0  # 0-7 (8-directional compass)
        self.nearest_food_distance_bin = 4   # 0-4 (very_close, close, medium, far, very_far)
        self.food_type_bin = 0               # 0-3 (plants, meat, insects, seeds)
        
        # Spatial context
        self.body_velocity_bin = 0           # 0-3 (still, slow, medium, fast)
        self.body_orientation_bin = 0        # 0-7 (8 directions relative to nearest food)
        self.ground_contact_bin = 1          # 0-1 (unstable, stable)
        
        # Social context
        self.nearby_agents_bin = 0           # 0-3 (none, few, some, many)
        self.competition_pressure_bin = 0    # 0-2 (low, medium, high)
    
    def to_tuple(self) -> Tuple:
        """Convert to tuple for Q-table indexing."""
        return (
            self.shoulder_angle_bin,
            self.elbow_angle_bin,
            self.energy_level_bin,
            self.nearest_food_direction_bin,
            self.nearest_food_distance_bin,
            self.body_velocity_bin,
            self.ground_contact_bin
        )
    
    def get_state_size(self) -> Tuple[int, ...]:
        """Get state space dimensions for Q-table initialization."""
        return (8, 8, 5, 8, 5, 4, 2)  # Manageable state space: ~10,240 states


class SurvivalRewardCalculator:
    """Advanced reward calculator for survival-focused learning."""
    
    def __init__(self):
        # Reward weights (tunable hyperparameters)
        self.energy_gain_weight = 100.0      # High priority for energy gain
        self.food_approach_weight = 25.0     # Reward moving toward food
        self.movement_efficiency_weight = 5.0 # Reward efficient movement
        self.survival_penalty_weight = 50.0   # Penalty for low energy/health
        self.thriving_bonus_weight = 10.0     # Bonus for high energy + movement
        
        # Survival thresholds
        self.critical_energy_threshold = 0.2
        self.low_energy_threshold = 0.4
        self.high_energy_threshold = 0.8
        
        # Movement thresholds
        self.min_movement_threshold = 0.003
        self.significant_movement_threshold = 0.01
        
    def calculate_reward(self, old_state: SurvivalState, new_state: SurvivalState, 
                        action: Tuple[float, float], agent_data: Dict[str, Any]) -> float:
        """
        Calculate comprehensive survival reward.
        
        Args:
            old_state: Previous state
            new_state: Current state after action
            action: Action taken
            agent_data: Additional agent information (position, energy, etc.)
            
        Returns:
            Total reward value
        """
        total_reward = 0.0
        
        # 1. ENERGY GAIN REWARD (Highest Priority)
        energy_change = agent_data.get('energy_change', 0.0)
        if energy_change > 0:
            # Major reward for gaining energy (eating) - REDUCED from 20.0 to 5.0
            energy_reward = energy_change * 5.0  # REDUCED SCALE
            total_reward += energy_reward
            
        # 2. FOOD-SEEKING BEHAVIOR REWARD - SCALED DOWN
        food_approach_reward = self._calculate_food_approach_reward(old_state, new_state)
        total_reward += food_approach_reward * 1.0  # REDUCED from 3.0 to 1.0
        
        # 3. MOVEMENT EFFICIENCY REWARD - KEPT SAME
        movement_reward = self._calculate_movement_efficiency(action, agent_data)
        total_reward += movement_reward * self.movement_efficiency_weight
        
        # 4. SURVIVAL PENALTIES - SCALED DOWN
        survival_penalty = self._calculate_survival_penalties(new_state, agent_data)
        total_reward += survival_penalty * 0.5  # REDUCED from 1.0 to 0.5
        
        # 5. THRIVING BONUS - SCALED DOWN
        thriving_bonus = self._calculate_thriving_bonus(new_state, agent_data)
        total_reward += thriving_bonus * 0.5  # REDUCED from 1.0 to 0.5
        
        # 6. BEHAVIORAL BONUSES - KEPT SAME
        behavior_bonus = self._calculate_behavior_bonuses(agent_data)
        total_reward += behavior_bonus
        
        # CRITICAL: Scale down final reward to match crawling reward scale
        total_reward = np.clip(total_reward, -2.0, 10.0)  # Match crawling reward scale
        
        return total_reward
    
    def _calculate_food_approach_reward(self, old_state: SurvivalState, 
                                       new_state: SurvivalState) -> float:
        """Reward for moving toward food sources."""
        # Reward for getting closer to food
        if new_state.nearest_food_distance_bin < old_state.nearest_food_distance_bin:
            return 1.0  # Moved closer to food
        elif new_state.nearest_food_distance_bin > old_state.nearest_food_distance_bin:
            return -0.2  # Moved away from food (small penalty)
        else:
            # Same distance - check if moving in right direction
            if (new_state.nearest_food_direction_bin == new_state.body_orientation_bin or
                abs(new_state.nearest_food_direction_bin - new_state.body_orientation_bin) <= 1):
                return 0.1  # Oriented toward food
        return 0.0
    
    def _calculate_movement_efficiency(self, action: Tuple[float, float], 
                                     agent_data: Dict[str, Any]) -> float:
        """Reward efficient movement."""
        displacement = agent_data.get('displacement', 0.0)
        velocity = agent_data.get('velocity_magnitude', 0.0)
        
        # Reward forward progress
        if displacement > self.min_movement_threshold:
            efficiency_reward = min(displacement * 2.0, 1.0)  # Cap at 1.0
            
            # Efficiency bonus for good speed
            if velocity > 0.5:
                efficiency_reward *= 1.2
            
            return efficiency_reward
        
        # Small penalty for excessive energy use without progress
        energy_cost = abs(action[0]) + abs(action[1])
        if energy_cost > 1.5 and displacement < self.min_movement_threshold:
            return -0.1
        
        return 0.0
    
    def _calculate_survival_penalties(self, state: SurvivalState, 
                                    agent_data: Dict[str, Any]) -> float:
        """Calculate penalties for survival threats."""
        penalty = 0.0
        
        # Energy-based penalties
        energy_level = agent_data.get('energy_level', 1.0)
        if energy_level < self.critical_energy_threshold:
            penalty -= 2.0  # Critical energy penalty
        elif energy_level < self.low_energy_threshold:
            penalty -= 0.5  # Low energy penalty
        
        # Health-based penalties
        health_level = agent_data.get('health_level', 1.0)
        if health_level < 0.5:
            penalty -= 1.0 * (0.5 - health_level)
        
        # Stability penalty
        body_angle = agent_data.get('body_angle', 0.0)
        if abs(body_angle) > math.pi/2:
            penalty -= 1.0  # Flipped over
        elif abs(body_angle) > math.pi/4:
            penalty -= 0.3  # Highly tilted
        
        return penalty
    
    def _calculate_thriving_bonus(self, state: SurvivalState, 
                                 agent_data: Dict[str, Any]) -> float:
        """Bonus for thriving (high energy + good movement)."""
        energy_level = agent_data.get('energy_level', 0.0)
        velocity = agent_data.get('velocity_magnitude', 0.0)
        
        if (energy_level > self.high_energy_threshold and 
            velocity > 0.5 and 
            state.ground_contact_bin == 1):  # Stable contact
            return 1.0
        
        return 0.0
    
    def _calculate_behavior_bonuses(self, agent_data: Dict[str, Any]) -> float:
        """Calculate bonuses for good behaviors."""
        bonus = 0.0
        
        # Bonus for exploring when energy is high
        energy_level = agent_data.get('energy_level', 0.0)
        exploration_score = agent_data.get('exploration_score', 0.0)
        
        if energy_level > 0.6 and exploration_score > 0.1:
            bonus += 0.2
        
        # Bonus for conservative behavior when energy is low
        if energy_level < 0.3:
            velocity = agent_data.get('velocity_magnitude', 0.0)
            if velocity < 0.2:  # Moving slowly to conserve energy
                bonus += 0.1
        
        return bonus


class EnhancedSurvivalQLearning(EnhancedQTable):
    """Enhanced Q-Learning specifically designed for survival tasks."""
    
    def __init__(self, action_count: int, use_signed_x_distance: bool = True):
        # Initialize with survival-appropriate state space
        state_dimensions = SurvivalState().get_state_size()
        super().__init__(action_count, default_value=0.0, 
                        confidence_threshold=20, exploration_bonus=0.2)
        
        self.state_processor = SurvivalStateProcessor(use_signed_x_distance=use_signed_x_distance)
        self.reward_calculator = SurvivalRewardCalculator()
        
        # Survival-specific parameters
        self.energy_awareness_weight = 2.0
        self.food_seeking_bonus = 1.5
        self.survival_learning_rate_multiplier = 1.5
        
        # Experience prioritization (reduced sizes for performance)
        self.experience_buffer = deque(maxlen=2000)  # Reduced from 5000
        self.high_value_experiences = deque(maxlen=400)  # Reduced from 1000
        
        # Curriculum learning stages
        self.learning_stage = 'basic_movement'  # -> 'food_seeking' -> 'survival_mastery'
        self.stage_transition_threshold = {'basic_movement': 100, 'food_seeking': 500}
        self.experiences_in_stage = 0
    
    def process_state(self, agent, ecosystem_data: Dict[str, Any]) -> SurvivalState:
        """Process raw agent data into structured survival state."""
        return self.state_processor.create_survival_state(agent, ecosystem_data)
    
    def calculate_survival_reward(self, old_state: SurvivalState, new_state: SurvivalState,
                                action: Tuple[float, float], agent_data: Dict[str, Any]) -> float:
        """Calculate reward using survival-focused reward function."""
        return self.reward_calculator.calculate_reward(old_state, new_state, action, agent_data)
    
    def update_with_survival_priority(self, state: SurvivalState, action: int, reward: float,
                                    next_state: SurvivalState, learning_rate: float, 
                                    discount_factor: float, agent_data: Dict[str, Any]):
        """Enhanced Q-value update with survival-specific prioritization."""
        
        # Adjust learning rate based on experience importance
        adjusted_lr = self._adjust_learning_rate_for_survival(
            learning_rate, reward, agent_data
        )
        
        # Standard Q-learning update
        state_tuple = state.to_tuple()
        next_state_tuple = next_state.to_tuple()
        
        self.update_q_value_enhanced(
            state_tuple, action, reward, next_state_tuple,
            adjusted_lr, discount_factor, use_adaptive_lr=True
        )
        
        # Store high-value experiences for replay
        if self._is_high_value_experience(reward, agent_data):
            experience = {
                'state': state_tuple,
                'action': action,
                'reward': reward,
                'next_state': next_state_tuple,
                'importance': abs(reward)
            }
            self.high_value_experiences.append(experience)
        
        # Update curriculum learning
        self._update_curriculum_progress()
    
    def choose_survival_action(self, state: SurvivalState, epsilon: float,
                             agent_data: Dict[str, Any]) -> int:
        """Choose action with survival-specific considerations."""
        
        # Adjust epsilon based on survival situation
        adjusted_epsilon = self._adjust_epsilon_for_survival(epsilon, agent_data)
        
        # Use different action selection based on learning stage
        if self.learning_stage == 'basic_movement':
            return self.enhanced_epsilon_greedy(state.to_tuple(), adjusted_epsilon)
        elif self.learning_stage == 'food_seeking':
            return self._food_seeking_action_selection(state, adjusted_epsilon)
        else:  # survival_mastery
            return self._survival_mastery_action_selection(state, adjusted_epsilon, agent_data)
    
    def _adjust_learning_rate_for_survival(self, base_lr: float, reward: float,
                                         agent_data: Dict[str, Any]) -> float:
        """Adjust learning rate based on survival context."""
        
        # Higher learning rate for survival-critical experiences
        energy_level = agent_data.get('energy_level', 1.0)
        
        if energy_level < 0.2:  # Critical energy
            return min(base_lr * 2.0, 0.5)  # Learn faster in crisis
        elif reward > 10.0:  # High reward (likely energy gain)
            return min(base_lr * 1.5, 0.3)  # Learn faster from good experiences
        elif energy_level > 0.8:  # High energy - can explore
            return base_lr * 0.8  # Learn slower when safe
        
        return base_lr
    
    def _adjust_epsilon_for_survival(self, base_epsilon: float, 
                                   agent_data: Dict[str, Any]) -> float:
        """Adjust exploration rate based on survival needs."""
        
        energy_level = agent_data.get('energy_level', 1.0)
        
        if energy_level < 0.2:  # Critical energy
            return base_epsilon * 0.3  # Less exploration, more exploitation
        elif energy_level > 0.8:  # High energy
            return min(base_epsilon * 1.5, 0.8)  # More exploration when safe
        
        return base_epsilon
    
    def _food_seeking_action_selection(self, state: SurvivalState, epsilon: float) -> int:
        """Action selection optimized for food-seeking behavior."""
        
        # If very close to food, bias toward reaching/consumption actions
        if state.nearest_food_distance_bin <= 1:  # Very close
            # Prioritize actions that might help with consumption
            consumption_actions = [0, 1, 2]  # Adjust based on your action space
            if random.random() > epsilon:
                # Choose best consumption action
                state_tuple = state.to_tuple()
                best_action = consumption_actions[0]
                best_value = self.get_q_value(state_tuple, best_action)
                
                for action in consumption_actions:
                    q_value = self.get_q_value(state_tuple, action)
                    if q_value > best_value:
                        best_value = q_value
                        best_action = action
                
                return best_action
        
        # Otherwise, use enhanced epsilon-greedy
        return self.enhanced_epsilon_greedy(state.to_tuple(), epsilon)
    
    def _survival_mastery_action_selection(self, state: SurvivalState, epsilon: float,
                                         agent_data: Dict[str, Any]) -> int:
        """Advanced action selection for survival mastery stage."""
        
        # Multi-objective action selection
        energy_level = agent_data.get('energy_level', 1.0)
        
        if energy_level < 0.3:  # Low energy - focus on conservation and food-seeking
            return self._food_seeking_action_selection(state, epsilon * 0.5)
        elif energy_level > 0.7:  # High energy - can explore and be active
            return self.enhanced_epsilon_greedy(state.to_tuple(), epsilon * 1.2)
        else:  # Medium energy - balanced approach
            return self.enhanced_epsilon_greedy(state.to_tuple(), epsilon)
    
    def _is_high_value_experience(self, reward: float, agent_data: Dict[str, Any]) -> bool:
        """Determine if experience should be prioritized for replay."""
        
        # High positive rewards (likely energy gain)
        if reward > 5.0:
            return True
        
        # Critical survival situations
        energy_level = agent_data.get('energy_level', 1.0)
        if energy_level < 0.2:
            return True
        
        # Significant negative rewards (learning opportunities)
        if reward < -2.0:
            return True
        
        return False
    
    def _update_curriculum_progress(self):
        """Update curriculum learning progress."""
        self.experiences_in_stage += 1
        
        if self.learning_stage in self.stage_transition_threshold:
            threshold = self.stage_transition_threshold[self.learning_stage]
            if self.experiences_in_stage >= threshold:
                if self.learning_stage == 'basic_movement':
                    self.learning_stage = 'food_seeking'
                    print(f"🎓 Advancing to food_seeking stage")
                elif self.learning_stage == 'food_seeking':
                    self.learning_stage = 'survival_mastery'
                    print(f"🎓 Advancing to survival_mastery stage")
                
                self.experiences_in_stage = 0
    
    def replay_high_value_experiences(self, batch_size: int = 32):
        """Replay high-value experiences for accelerated learning."""
        if len(self.high_value_experiences) < batch_size:
            return
        
        # Sample high-value experiences
        batch = random.sample(list(self.high_value_experiences), batch_size)
        
        for exp in batch:
            # Re-learn from high-value experience with boosted learning rate
            current_q = self.get_q_value(exp['state'], exp['action'])
            next_best_q = self.get_best_action(exp['next_state'])[1]
            
            # Boosted learning rate for replay
            boosted_lr = min(0.3, 0.1 * exp['importance'])
            target = exp['reward'] + 0.95 * next_best_q
            new_q = current_q + boosted_lr * (target - current_q)
            
            self.set_q_value(exp['state'], exp['action'], new_q)


class SurvivalStateProcessor:
    """Processes raw agent and environment data into structured survival states."""
    
    def __init__(self, use_signed_x_distance=True):
        # Binning parameters
        self.angle_bins = 8  # 45-degree increments
        self.distance_bins = 5
        self.velocity_bins = 4
        self.energy_bins = 5
        
        # Distance configuration
        self.use_signed_x_distance = use_signed_x_distance  # New: option to use signed x-axis distance
        
        # Distance thresholds for food
        if self.use_signed_x_distance:
            # For signed distance, use symmetric thresholds: [-20, -10, -5, -2, 0, 2, 5, 10, 20]
            self.signed_distance_thresholds = [-20.0, -10.0, -5.0, -2.0, 2.0, 5.0, 10.0, 20.0]
            self.distance_bins = len(self.signed_distance_thresholds) + 1  # 9 bins total
        else:
            # Regular distance thresholds  
            self.food_distance_thresholds = [2.0, 5.0, 10.0, 20.0]  # very_close, close, medium, far, very_far
        
        # Velocity thresholds
        self.velocity_thresholds = [0.1, 0.5, 1.5]  # still, slow, medium, fast
    
    def create_survival_state(self, agent, ecosystem_data: Dict[str, Any]) -> SurvivalState:
        """Create a survival state from agent and ecosystem data."""
        state = SurvivalState()
        
        # Physical state (arm configuration)
        state.shoulder_angle_bin = self._discretize_angle(agent.upper_arm.angle)
        state.elbow_angle_bin = self._discretize_angle(agent.lower_arm.angle)
        
        # Survival state
        energy_level = ecosystem_data.get('agent_energy', {}).get(str(agent.id), 1.0)
        health_level = ecosystem_data.get('agent_health', {}).get(str(agent.id), {}).get('health', 1.0)
        
        state.energy_level_bin = self._discretize_energy(energy_level)
        state.health_level_bin = self._discretize_energy(health_level)
        
        # Environmental awareness
        food_info = self._find_nearest_food(agent, ecosystem_data)
        state.nearest_food_direction_bin = food_info['direction_bin']
        state.nearest_food_distance_bin = food_info['distance_bin']
        state.food_type_bin = food_info['type_bin']
        
        # Spatial context
        velocity_magnitude = math.sqrt(agent.body.linearVelocity.x**2 + agent.body.linearVelocity.y**2)
        state.body_velocity_bin = self._discretize_velocity(velocity_magnitude)
        
        # Body orientation relative to food
        if food_info['distance'] < float('inf'):
            state.body_orientation_bin = self._get_relative_orientation(agent, food_info['position'])
        
        # Ground contact (simplified)
        state.ground_contact_bin = 1 if agent.body.position.y > -1.0 else 0
        
        # Social context
        nearby_agents = self._count_nearby_agents(agent, ecosystem_data)
        state.nearby_agents_bin = min(3, nearby_agents // 2)  # 0-3 bins
        
        return state
    
    def _discretize_angle(self, angle: float) -> int:
        """Discretize angle into bins."""
        # Normalize angle to [0, 2π]
        normalized_angle = (angle + math.pi) % (2 * math.pi)
        return min(self.angle_bins - 1, int(normalized_angle / (2 * math.pi) * self.angle_bins))
    
    def _discretize_energy(self, energy: float) -> int:
        """Discretize energy level into bins."""
        return min(self.energy_bins - 1, int(energy * self.energy_bins))
    
    def _discretize_velocity(self, velocity: float) -> int:
        """Discretize velocity into bins."""
        for i, threshold in enumerate(self.velocity_thresholds):
            if velocity <= threshold:
                return i
        return len(self.velocity_thresholds)
    
    def _find_nearest_food(self, agent, ecosystem_data: Dict[str, Any]) -> Dict[str, Any]:
        """Find nearest food source and calculate direction/distance. For carnivores/scavengers, considers other agents as prey."""
        food_sources = ecosystem_data.get('food_sources', [])
        agent_pos = (agent.body.position.x, agent.body.position.y)
        agent_id = str(agent.id)
        
        # Get agent's ecosystem role
        agent_roles = ecosystem_data.get('agent_roles', {})
        agent_role = agent_roles.get(agent_id, 'omnivore')
        
        # For carnivores and scavengers, other agents are primary food sources
        potential_food_sources = []
        
        # Add environmental food sources for all agents (even carnivores can eat insects)
        for food in food_sources:
            if food.get('amount', 0) > 0.1:  # Only consider food that can actually be consumed (matches consumption threshold)
                potential_food_sources.append({
                    'position': food.get('position', [0, 0]),
                    'type': food.get('type', 'plants'),
                    'source': 'environment'
                })
        
        # For carnivores and scavengers, add other agents as potential prey
        if agent_role in ['carnivore', 'scavenger']:
            nearby_agents_data = ecosystem_data.get('nearby_agents_data', [])
            agent_energy_levels = ecosystem_data.get('agent_energy', {})
            
            for other_agent in nearby_agents_data:
                other_id = other_agent.get('id', '')
                if other_id != agent_id:  # Don't target yourself
                    other_position = other_agent.get('position', [0, 0])
                    other_energy = agent_energy_levels.get(other_id, 1.0)
                    
                    # For scavengers, prefer weak prey (energy < 0.5)
                    # For carnivores, target any prey but bonus for weak prey
                    prey_attractiveness = 1.0
                    if agent_role == 'scavenger':
                        prey_attractiveness = 2.0 if other_energy < 0.5 else 0.5
                    elif agent_role == 'carnivore':
                        prey_attractiveness = 1.5 if other_energy < 0.6 else 1.0
                    
                    potential_food_sources.append({
                        'position': other_position,
                        'type': 'meat',  # Other agents provide meat
                        'source': 'prey',
                        'prey_id': other_id,
                        'prey_energy': other_energy,
                        'attractiveness': prey_attractiveness
                    })
        
        if not potential_food_sources:
            return {
                'distance': float('inf'),
                'direction_bin': 0,
                'distance_bin': 4,  # Very far
                'type_bin': 0,
                'position': (0, 0)
            }
        
        # Find the nearest/most attractive food source
        best_target = None
        best_score = float('inf')  # Lower is better (distance-based with attractiveness modifier)
        
        for target in potential_food_sources:
            target_pos = target['position']
            distance = math.sqrt((agent_pos[0] - target_pos[0])**2 + (agent_pos[1] - target_pos[1])**2)
            
            # For predators, factor in prey attractiveness
            if target.get('source') == 'prey':
                attractiveness = target.get('attractiveness', 1.0)
                # Lower score = better target (distance reduced by attractiveness)
                score = distance / attractiveness
            else:
                score = distance
            
            if score < best_score:
                best_score = score
                best_target = target
        
        if best_target is None:
            return {
                'distance': float('inf'),
                'direction_bin': 0,
                'distance_bin': 4,
                'type_bin': 0,
                'position': (0, 0)
            }
        
        # Calculate direction to target
        target_pos = best_target['position']
        
        # Calculate distance based on configuration
        if self.use_signed_x_distance:
            # Use signed x-axis distance (positive = right, negative = left)
            actual_distance = target_pos[0] - agent_pos[0]
        else:
            # Use regular Euclidean distance
            actual_distance = math.sqrt((agent_pos[0] - target_pos[0])**2 + (agent_pos[1] - target_pos[1])**2)
        
        direction_angle = math.atan2(target_pos[1] - agent_pos[1], target_pos[0] - agent_pos[0])
        direction_bin = self._angle_to_direction_bin(direction_angle)
        
        # Calculate distance bin
        distance_bin = self._distance_to_bin(actual_distance)
        
        # Food type mapping (meat for prey, original type for environmental food)
        food_type_map = {'plants': 0, 'meat': 1, 'insects': 2, 'seeds': 3}
        food_type_bin = food_type_map.get(best_target.get('type', 'plants'), 0)
        
        return {
            'distance': actual_distance,
            'direction_bin': direction_bin,
            'distance_bin': distance_bin,
            'type_bin': food_type_bin,
            'position': target_pos,
            'target_type': best_target.get('source', 'environment'),
            'prey_id': best_target.get('prey_id', None)
        }
    
    def _angle_to_direction_bin(self, angle: float) -> int:
        """Convert angle to 8-directional bin."""
        # Normalize angle to [0, 2π]
        normalized_angle = (angle + math.pi) % (2 * math.pi)
        return int(normalized_angle / (2 * math.pi) * 8) % 8
    
    def _distance_to_bin(self, distance: float) -> int:
        """Convert distance to bin."""
        if self.use_signed_x_distance:
            for i, threshold in enumerate(self.signed_distance_thresholds):
                if distance <= threshold:
                    return i
            return len(self.signed_distance_thresholds)
        else:
            for i, threshold in enumerate(self.food_distance_thresholds):
                if distance <= threshold:
                    return i
            return len(self.food_distance_thresholds)
    
    def _get_relative_orientation(self, agent, food_position: Tuple[float, float]) -> int:
        """Get agent's body orientation relative to food."""
        agent_pos = (agent.body.position.x, agent.body.position.y)
        
        # Calculate angle from agent to food
        food_angle = math.atan2(food_position[1] - agent_pos[1], food_position[0] - agent_pos[0])
        
        # Agent's current orientation
        agent_angle = agent.body.angle
        
        # Relative angle
        relative_angle = (food_angle - agent_angle + math.pi) % (2 * math.pi)
        
        return self._angle_to_direction_bin(relative_angle)
    
    def _count_nearby_agents(self, agent, ecosystem_data: Dict[str, Any]) -> int:
        """Count nearby agents for social context."""
        # This would need to be implemented based on your ecosystem data structure
        # For now, return a placeholder
        return 0 