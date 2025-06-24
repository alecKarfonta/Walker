"""
Q-learning evaluation and analysis.
Tracks learning quality, convergence, and parameter effectiveness.
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field
import math


@dataclass
class QLearningMetrics:
    """Data class to store Q-learning evaluation metrics."""
    robot_id: str
    timestamp: float
    
    # Convergence Analysis
    convergence_rate: float = 0.0
    convergence_stability: float = 0.0
    q_value_variance: List[float] = field(default_factory=list)
    policy_stability: float = 0.0
    
    # Learning Efficiency
    sample_efficiency: float = 0.0  # Learning per sample
    experience_replay_effectiveness: float = 0.0
    temporal_difference_progression: List[float] = field(default_factory=list)
    
    # Parameter Optimization
    learning_rate_effectiveness: List[float] = field(default_factory=list)
    epsilon_decay_optimization: float = 0.0
    discount_factor_impact: float = 0.0
    
    # Q-table Analysis
    q_table_size: int = 0
    state_value_distribution: List[float] = field(default_factory=list)
    action_value_distribution: List[float] = field(default_factory=list)
    value_function_smoothness: float = 0.0
    
    # Learning Progress
    episode_q_improvement: List[float] = field(default_factory=list)
    learning_plateau_detection: bool = False
    exploration_exploitation_balance: float = 0.0


class QLearningEvaluator:
    """
    Evaluates Q-learning effectiveness and convergence.
    Tracks learning quality, parameter effectiveness, and optimization opportunities.
    """
    
    def __init__(self, history_length: int = 1000):
        """
        Initialize the Q-learning evaluator.
        
        Args:
            history_length: Number of historical data points to maintain
        """
        self.history_length = history_length
        self.q_metrics: Dict[str, QLearningMetrics] = {}
        self.q_history: Dict[str, List[QLearningMetrics]] = defaultdict(list)
        
        # Historical tracking for convergence analysis
        self.robot_q_histories: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.robot_td_errors: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self.robot_policy_histories: Dict[str, List[Dict]] = defaultdict(list)
        self.robot_learning_curves: Dict[str, List[float]] = defaultdict(list)
        
        # Parameter tracking
        self.robot_parameter_history: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        
    def evaluate_q_learning(self, agent, step_count: int) -> QLearningMetrics:
        """
        Evaluate Q-learning performance and return comprehensive metrics.
        
        Args:
            agent: The robot agent to evaluate
            step_count: Current training step
            
        Returns:
            QLearningMetrics object with current evaluation
        """
        try:
            robot_id = str(agent.id)
            timestamp = time.time()
            
            # Update historical tracking
            self._update_q_learning_tracking(agent, robot_id, step_count)
            
            # Create new metrics object
            metrics = QLearningMetrics(robot_id=robot_id, timestamp=timestamp)
            
            # Convergence Analysis
            metrics.convergence_rate = self._calculate_convergence_rate(agent, robot_id)
            metrics.convergence_stability = self._calculate_convergence_stability(robot_id)
            metrics.q_value_variance = self._calculate_q_value_variance(agent)
            metrics.policy_stability = self._calculate_policy_stability(robot_id)
            
            # Learning Efficiency
            metrics.sample_efficiency = self._calculate_sample_efficiency(agent, robot_id)
            metrics.experience_replay_effectiveness = self._calculate_replay_effectiveness(agent)
            metrics.temporal_difference_progression = self._get_td_progression(robot_id)
            
            # Parameter Optimization
            metrics.learning_rate_effectiveness = self._analyze_learning_rate_effectiveness(robot_id)
            metrics.epsilon_decay_optimization = self._analyze_epsilon_decay(robot_id)
            metrics.discount_factor_impact = self._analyze_discount_factor_impact(agent)
            
            # Q-table Analysis
            metrics.q_table_size = self._get_q_table_size(agent)
            metrics.state_value_distribution = self._get_state_value_distribution(agent)
            metrics.action_value_distribution = self._get_action_value_distribution(agent)
            metrics.value_function_smoothness = self._calculate_value_function_smoothness(agent)
            
            # Learning Progress
            metrics.episode_q_improvement = self._calculate_episode_improvement(robot_id)
            metrics.learning_plateau_detection = self._detect_learning_plateau(robot_id)
            metrics.exploration_exploitation_balance = self._calculate_exploration_balance(agent)
            
            # Store metrics
            self.q_metrics[robot_id] = metrics
            self.q_history[robot_id].append(metrics)
            
            # Trim history
            if len(self.q_history[robot_id]) > self.history_length:
                self.q_history[robot_id] = self.q_history[robot_id][-self.history_length:]
            
            return metrics
            
        except Exception as e:
            print(f"⚠️  Error evaluating Q-learning for robot {getattr(agent, 'id', 'unknown')}: {e}")
            return QLearningMetrics(robot_id=str(getattr(agent, 'id', 'unknown')), timestamp=time.time())
    
    def _update_q_learning_tracking(self, agent, robot_id: str, step_count: int):
        """Update Q-learning historical tracking."""
        try:
            # Store current Q-learning state
            q_state = {
                'step': step_count,
                'convergence': getattr(agent, 'q_table', None) and agent.q_table.get_convergence_estimate() or 0.0,
                'q_table_size': len(getattr(agent, 'q_table', {}).q_values or {}),
                'total_reward': getattr(agent, 'total_reward', 0.0),
                'epsilon': getattr(agent, 'epsilon', 0.0),
                'learning_rate': getattr(agent, 'learning_rate', 0.0)
            }
            self.robot_q_histories[robot_id].append(q_state)
            
            # Trim history
            if len(self.robot_q_histories[robot_id]) > 1000:
                self.robot_q_histories[robot_id] = self.robot_q_histories[robot_id][-1000:]
            
            # Track TD errors
            if hasattr(agent, 'q_table') and hasattr(agent.q_table, 'q_value_history'):
                recent_history = agent.q_table.q_value_history[-10:]
                for entry in recent_history:
                    if 'td_error' in entry:
                        self.robot_td_errors[robot_id].append(entry['td_error'])
            
            # Track parameters
            self.robot_parameter_history[robot_id]['epsilon'].append(getattr(agent, 'epsilon', 0.0))
            self.robot_parameter_history[robot_id]['learning_rate'].append(getattr(agent, 'learning_rate', 0.0))
            self.robot_parameter_history[robot_id]['total_reward'].append(getattr(agent, 'total_reward', 0.0))
            
            # Trim parameter history
            for param_list in self.robot_parameter_history[robot_id].values():
                if len(param_list) > 500:
                    param_list[:] = param_list[-500:]
            
            # Store current policy
            if hasattr(agent, 'q_table'):
                policy_sample = self._sample_current_policy(agent)
                self.robot_policy_histories[robot_id].append(policy_sample)
                if len(self.robot_policy_histories[robot_id]) > 100:
                    self.robot_policy_histories[robot_id] = self.robot_policy_histories[robot_id][-100:]
            
        except Exception as e:
            print(f"⚠️  Error updating Q-learning tracking for robot {robot_id}: {e}")
    
    def _sample_current_policy(self, agent) -> Dict:
        """Sample current policy for stability analysis."""
        try:
            policy_sample = {}
            if hasattr(agent, 'q_table') and hasattr(agent.q_table, 'q_values'):
                # Sample a few states and their best actions
                q_values = agent.q_table.q_values
                sample_states = list(q_values.keys())[:10]  # Sample first 10 states
                
                for state in sample_states:
                    state_actions = q_values[state]
                    if state_actions:
                        best_action = max(state_actions.items(), key=lambda x: x[1])[0]
                        policy_sample[state] = best_action
            
            return policy_sample
        except:
            return {}
    
    def _calculate_convergence_rate(self, agent, robot_id: str) -> float:
        """Calculate convergence rate based on recent Q-value changes."""
        try:
            if hasattr(agent, 'q_table'):
                return float(agent.q_table.get_convergence_estimate())
            return 0.0
        except:
            return 0.0
    
    def _calculate_convergence_stability(self, robot_id: str) -> float:
        """Calculate stability of convergence over time."""
        try:
            history = self.robot_q_histories[robot_id]
            if len(history) < 10:
                return 0.0
            
            # Get recent convergence values
            recent_convergence = [entry['convergence'] for entry in history[-10:]]
            
            # Calculate stability as inverse of variance
            if len(recent_convergence) > 1:
                variance = np.var(recent_convergence)
                stability = 1.0 / (1.0 + variance)
                return float(stability)
            
            return 0.0
        except:
            return 0.0
    
    def _calculate_q_value_variance(self, agent) -> List[float]:
        """Calculate variance in Q-values."""
        try:
            if hasattr(agent, 'q_table') and hasattr(agent.q_table, 'q_values'):
                all_q_values = []
                for state_actions in agent.q_table.q_values.values():
                    all_q_values.extend(state_actions.values())
                
                if all_q_values:
                    return [float(np.var(all_q_values))]
            return [0.0]
        except:
            return [0.0]
    
    def _calculate_policy_stability(self, robot_id: str) -> float:
        """Calculate stability of the policy over time."""
        try:
            policy_history = self.robot_policy_histories[robot_id]
            if len(policy_history) < 2:
                return 0.0
            
            # Compare recent policies
            recent_policies = policy_history[-5:]
            if len(recent_policies) < 2:
                return 0.0
            
            # Calculate similarity between consecutive policies
            similarities = []
            for i in range(len(recent_policies) - 1):
                policy1 = recent_policies[i]
                policy2 = recent_policies[i + 1]
                
                # Calculate Jaccard similarity
                if policy1 and policy2:
                    common_states = set(policy1.keys()) & set(policy2.keys())
                    if common_states:
                        same_actions = sum(1 for state in common_states if policy1[state] == policy2[state])
                        similarity = same_actions / len(common_states)
                        similarities.append(similarity)
            
            return float(np.mean(similarities)) if similarities else 0.0
        except:
            return 0.0
    
    def _calculate_sample_efficiency(self, agent, robot_id: str) -> float:
        """Calculate sample efficiency (learning per sample)."""
        try:
            history = self.robot_q_histories[robot_id]
            if len(history) < 2:
                return 0.0
            
            # Calculate improvement per step
            recent_history = history[-10:]
            if len(recent_history) < 2:
                return 0.0
            
            improvement = recent_history[-1]['total_reward'] - recent_history[0]['total_reward']
            steps = recent_history[-1]['step'] - recent_history[0]['step']
            
            if steps > 0:
                return improvement / steps
            return 0.0
        except:
            return 0.0
    
    def _calculate_replay_effectiveness(self, agent) -> float:
        """Calculate effectiveness of experience replay."""
        try:
            if hasattr(agent, 'replay_buffer'):
                buffer_size = len(agent.replay_buffer.buffer) if hasattr(agent.replay_buffer, 'buffer') else 0
                buffer_capacity = getattr(agent.replay_buffer, 'capacity', 1)
                
                # Simple measure: buffer utilization
                utilization = buffer_size / buffer_capacity
                
                # Could be enhanced with more sophisticated metrics
                return float(utilization)
            return 0.0
        except:
            return 0.0
    
    def _get_td_progression(self, robot_id: str) -> List[float]:
        """Get temporal difference error progression."""
        try:
            td_errors = list(self.robot_td_errors[robot_id])
            return [float(error) for error in td_errors[-20:]]  # Last 20 TD errors
        except:
            return []
    
    def _analyze_learning_rate_effectiveness(self, robot_id: str) -> List[float]:
        """Analyze learning rate effectiveness over time."""
        try:
            learning_rates = self.robot_parameter_history[robot_id]['learning_rate']
            rewards = self.robot_parameter_history[robot_id]['total_reward']
            
            if len(learning_rates) < 2 or len(rewards) < 2:
                return [0.0]
            
            # Calculate correlation between learning rate and reward improvement
            min_len = min(len(learning_rates), len(rewards))
            if min_len > 10:
                recent_lrs = learning_rates[-min_len:]
                recent_rewards = rewards[-min_len:]
                
                # Simple effectiveness: reward improvement when learning rate changes
                lr_changes = np.diff(recent_lrs)
                reward_changes = np.diff(recent_rewards)
                
                if len(lr_changes) > 0 and len(reward_changes) > 0:
                    # Correlation between learning rate changes and reward improvements
                    correlation = np.corrcoef(lr_changes, reward_changes)[0, 1]
                    return [float(correlation) if not np.isnan(correlation) else 0.0]
            
            return [0.0]
        except:
            return [0.0]
    
    def _analyze_epsilon_decay(self, robot_id: str) -> float:
        """Analyze epsilon decay optimization."""
        try:
            epsilons = self.robot_parameter_history[robot_id]['epsilon']
            rewards = self.robot_parameter_history[robot_id]['total_reward']
            
            if len(epsilons) < 10 or len(rewards) < 10:
                return 0.0
            
            # Analyze if epsilon decay is well-timed with performance
            min_len = min(len(epsilons), len(rewards))
            recent_epsilons = epsilons[-min_len:]
            recent_rewards = rewards[-min_len:]
            
            # Check if epsilon decreases as performance improves
            epsilon_trend = np.polyfit(range(len(recent_epsilons)), recent_epsilons, 1)[0]
            reward_trend = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
            
            # Good epsilon decay: epsilon decreases while rewards increase
            if epsilon_trend < 0 and reward_trend > 0:
                return abs(epsilon_trend) * reward_trend
            
            return 0.0
        except:
            return 0.0
    
    def _analyze_discount_factor_impact(self, agent) -> float:
        """Analyze impact of discount factor on learning."""
        try:
            if hasattr(agent, 'discount_factor'):
                discount = agent.discount_factor
                
                # Simple heuristic: optimal discount factor is usually around 0.9-0.99
                optimal_range = (0.85, 0.99)
                if optimal_range[0] <= discount <= optimal_range[1]:
                    # Distance from optimal center (0.92)
                    return 1.0 - abs(discount - 0.92) / 0.07
                else:
                    return 0.0
            return 0.0
        except:
            return 0.0
    
    def _get_q_table_size(self, agent) -> int:
        """Get Q-table size."""
        try:
            if hasattr(agent, 'q_table') and hasattr(agent.q_table, 'q_values'):
                return len(agent.q_table.q_values)
            return 0
        except:
            return 0
    
    def _get_state_value_distribution(self, agent) -> List[float]:
        """Get distribution of state values."""
        try:
            if hasattr(agent, 'q_table') and hasattr(agent.q_table, 'q_values'):
                state_values = []
                for state_actions in agent.q_table.q_values.values():
                    if state_actions:
                        max_value = max(state_actions.values())
                        state_values.append(max_value)
                
                return state_values[:50]  # Limit to first 50 values
            return []
        except:
            return []
    
    def _get_action_value_distribution(self, agent) -> List[float]:
        """Get distribution of action values."""
        try:
            if hasattr(agent, 'q_table') and hasattr(agent.q_table, 'q_values'):
                action_values = []
                for state_actions in agent.q_table.q_values.values():
                    action_values.extend(state_actions.values())
                
                return action_values[:100]  # Limit to first 100 values
            return []
        except:
            return []
    
    def _calculate_value_function_smoothness(self, agent) -> float:
        """Calculate smoothness of value function."""
        try:
            state_values = self._get_state_value_distribution(agent)
            if len(state_values) > 2:
                # Calculate variance of differences (smoothness measure)
                differences = np.diff(state_values)
                smoothness = 1.0 / (1.0 + np.var(differences))
                return float(smoothness)
            return 0.0
        except:
            return 0.0
    
    def _calculate_episode_improvement(self, robot_id: str) -> List[float]:
        """Calculate Q-learning improvement per episode."""
        try:
            history = self.robot_q_histories[robot_id]
            if len(history) < 2:
                return [0.0]
            
            # Calculate improvements over recent history
            improvements = []
            recent_history = history[-10:]
            
            for i in range(1, len(recent_history)):
                improvement = recent_history[i]['total_reward'] - recent_history[i-1]['total_reward']
                improvements.append(improvement)
            
            return improvements
        except:
            return [0.0]
    
    def _detect_learning_plateau(self, robot_id: str) -> bool:
        """Detect if learning has plateaued."""
        try:
            history = self.robot_q_histories[robot_id]
            if len(history) < 20:
                return False
            
            # Check if recent improvements are consistently small
            recent_rewards = [entry['total_reward'] for entry in history[-20:]]
            
            # Calculate trend
            if len(recent_rewards) > 10:
                trend = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
                
                # Plateau if trend is very small (less than 0.001 per step)
                return abs(trend) < 0.001
            
            return False
        except:
            return False
    
    def _calculate_exploration_balance(self, agent) -> float:
        """Calculate exploration-exploitation balance."""
        try:
            if hasattr(agent, 'epsilon') and hasattr(agent, 'total_reward'):
                epsilon = agent.epsilon
                performance = getattr(agent, 'total_reward', 0.0)
                
                # Good balance: moderate epsilon with good performance
                # Or low epsilon with very good performance
                if performance > 0:
                    # Balance score based on epsilon and performance
                    if epsilon > 0.3:  # High exploration
                        return min(1.0, performance / 10.0)  # Performance should be building
                    elif epsilon < 0.1:  # Low exploration
                        return min(1.0, performance / 5.0)   # Performance should be high
                    else:  # Medium exploration
                        return min(1.0, performance / 7.0)   # Balanced performance expected
            
            return 0.0
        except:
            return 0.0
    
    def get_q_learning_summary(self, robot_id: str) -> Dict[str, Any]:
        """Get Q-learning summary for a robot."""
        if robot_id not in self.q_metrics:
            return {}
        
        metrics = self.q_metrics[robot_id]
        return {
            'robot_id': robot_id,
            'convergence_rate': metrics.convergence_rate,
            'convergence_stability': metrics.convergence_stability,
            'sample_efficiency': metrics.sample_efficiency,
            'policy_stability': metrics.policy_stability,
            'q_table_size': metrics.q_table_size,
            'learning_plateau': metrics.learning_plateau_detection,
            'exploration_balance': metrics.exploration_exploitation_balance,
            'replay_effectiveness': metrics.experience_replay_effectiveness
        }
    
    def get_learning_diagnostics(self, robot_id: str) -> Dict[str, Any]:
        """Get detailed learning diagnostics for troubleshooting."""
        if robot_id not in self.q_metrics:
            return {}
        
        metrics = self.q_metrics[robot_id]
        
        # Identify potential issues
        issues = []
        recommendations = []
        
        if metrics.convergence_rate < 0.3:
            issues.append("Low convergence rate")
            recommendations.append("Consider increasing learning rate or adjusting reward function")
        
        if metrics.policy_stability < 0.5:
            issues.append("Unstable policy")
            recommendations.append("Consider reducing learning rate or epsilon")
        
        if metrics.learning_plateau_detection:
            issues.append("Learning plateau detected")
            recommendations.append("Consider curriculum learning or exploration bonus")
        
        if metrics.exploration_exploitation_balance < 0.3:
            issues.append("Poor exploration-exploitation balance")
            recommendations.append("Adjust epsilon decay schedule")
        
        return {
            'robot_id': robot_id,
            'issues_detected': issues,
            'recommendations': recommendations,
            'overall_health': 'good' if len(issues) == 0 else 'needs_attention',
            'convergence_trend': self._get_convergence_trend(robot_id),
            'performance_trend': self._get_performance_trend(robot_id)
        }
    
    def _get_convergence_trend(self, robot_id: str) -> str:
        """Get convergence trend description."""
        try:
            history = self.robot_q_histories[robot_id]
            if len(history) < 5:
                return "insufficient_data"
            
            recent_convergence = [entry['convergence'] for entry in history[-5:]]
            trend = np.polyfit(range(len(recent_convergence)), recent_convergence, 1)[0]
            
            if trend > 0.01:
                return "improving"
            elif trend < -0.01:
                return "deteriorating"
            else:
                return "stable"
        except:
            return "unknown"
    
    def _get_performance_trend(self, robot_id: str) -> str:
        """Get performance trend description."""
        try:
            history = self.robot_q_histories[robot_id]
            if len(history) < 5:
                return "insufficient_data"
            
            recent_rewards = [entry['total_reward'] for entry in history[-5:]]
            trend = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
            
            if trend > 0.1:
                return "improving"
            elif trend < -0.1:
                return "deteriorating"
            else:
                return "stable"
        except:
            return "unknown" 