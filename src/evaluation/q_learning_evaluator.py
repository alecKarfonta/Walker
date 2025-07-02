"""
Q-Learning Performance Evaluator
Comprehensive evaluation system for Q-learning effectiveness across different agent types.
"""

import time
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import math

# Handle numpy import gracefully
try:
    import numpy as np
except ImportError:
    # Create a minimal numpy substitute for basic operations
    class NumpySubstitute:
        @staticmethod
        def mean(data):
            return sum(data) / len(data) if data else 0.0
        
        @staticmethod
        def std(data):
            if not data:
                return 0.0
            mean_val = sum(data) / len(data)
            variance = sum((x - mean_val) ** 2 for x in data) / len(data)
            return variance ** 0.5
        
        @staticmethod
        def min(data):
            return min(data) if data else 0.0
        
        @staticmethod
        def max(data):
            return max(data) if data else 0.0
        
        @staticmethod
        def sqrt(x):
            return x ** 0.5
        
        @staticmethod
        def clip(x, min_val, max_val):
            return max(min_val, min(max_val, x))
        
        @staticmethod
        def array(data):
            return list(data)
        
        @staticmethod
        def abs(data):
            if isinstance(data, (list, tuple)):
                return [abs(x) for x in data]
            return abs(data)
        
        @staticmethod
        def diff(data):
            if len(data) < 2:
                return []
            return [data[i+1] - data[i] for i in range(len(data)-1)]
        
        @staticmethod
        def var(data):
            if not data:
                return 0.0
            mean_val = sum(data) / len(data)
            return sum((x - mean_val) ** 2 for x in data) / len(data)
        
        @staticmethod
        def log2(x):
            if x <= 0:
                return 0.0
            return math.log(x) / math.log(2)
        
        @staticmethod
        def histogram(data, bins=10):
            if not data:
                return [0] * bins, list(range(bins + 1))
            
            min_val, max_val = min(data), max(data)
            if min_val == max_val:
                hist = [len(data)] + [0] * (bins - 1)
                edges = [min_val - 0.5 + i for i in range(bins + 1)]
                return hist, edges
            
            range_val = max_val - min_val
            bin_size = range_val / bins
            hist = [0] * bins
            edges = [min_val + i * bin_size for i in range(bins + 1)]
            
            for value in data:
                bin_idx = int((value - min_val) / bin_size)
                if bin_idx >= bins:
                    bin_idx = bins - 1
                hist[bin_idx] += 1
            
            return hist, edges
    
    np = NumpySubstitute()
    print("⚠️ NumPy not available, using basic substitute")


class LearningStage(Enum):
    """Learning stages for curriculum evaluation."""
    EXPLORATION = "exploration"
    LEARNING = "learning"
    CONVERGENCE = "convergence"
    MASTERY = "mastery"
    PLATEAU = "plateau"


@dataclass
class QLearningMetrics:
    """Comprehensive Q-learning metrics for a single agent."""
    agent_id: str
    agent_type: str
    
    # Value Prediction Accuracy (core metric requested)
    value_prediction_error: float  # |Q(s,a) - actual_reward_received|
    value_prediction_mae: float    # Mean Absolute Error over time
    value_prediction_rmse: float   # Root Mean Square Error over time
    
    # Q-Value Analysis
    q_value_mean: float
    q_value_std: float
    q_value_range: Tuple[float, float]
    q_value_distribution: Dict[str, int]  # Distribution of Q-values by bins
    
    # Learning Progress
    learning_rate_current: float
    epsilon_current: float
    exploration_ratio: float  # % of actions that were exploratory
    exploitation_ratio: float # % of actions that were exploitative
    
    # Convergence Metrics
    convergence_score: float      # How stable are Q-values (0-1)
    value_change_rate: float      # Rate of Q-value change per update
    policy_stability: float       # How often does best action change (0-1)
    
    # Experience Quality
    experience_diversity: float   # Diversity of state-action pairs visited
    state_coverage: float         # % of relevant state space explored
    action_preference_entropy: float  # Entropy of action selection distribution
    
    # Learning Efficiency
    steps_to_first_reward: int
    steps_to_stable_policy: int
    reward_improvement_rate: float
    learning_efficiency_score: float  # Overall efficiency rating
    
    # Performance Trends
    recent_performance_trend: str  # "improving", "stable", "declining"
    learning_velocity: float       # Rate of performance improvement
    plateau_duration: int          # Steps since last significant improvement
    
    # Problem Indicators
    learning_issues: List[str]     # Detected learning problems
    recommendations: List[str]     # Suggested improvements
    
    # Temporal metrics
    timestamp: float
    steps_evaluated: int
    total_updates: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'value_prediction': {
                'current_error': self.value_prediction_error,
                'mean_absolute_error': self.value_prediction_mae,
                'root_mean_square_error': self.value_prediction_rmse
            },
            'q_values': {
                'mean': self.q_value_mean,
                'std': self.q_value_std,
                'min': self.q_value_range[0],
                'max': self.q_value_range[1],
                'distribution': self.q_value_distribution
            },
            'learning_parameters': {
                'learning_rate': self.learning_rate_current,
                'epsilon': self.epsilon_current,
                'exploration_ratio': self.exploration_ratio,
                'exploitation_ratio': self.exploitation_ratio
            },
            'convergence': {
                'score': self.convergence_score,
                'value_change_rate': self.value_change_rate,
                'policy_stability': self.policy_stability
            },
            'experience': {
                'diversity': self.experience_diversity,
                'state_coverage': self.state_coverage,
                'action_entropy': self.action_preference_entropy
            },
            'efficiency': {
                'steps_to_first_reward': self.steps_to_first_reward,
                'steps_to_stable_policy': self.steps_to_stable_policy,
                'improvement_rate': self.reward_improvement_rate,
                'efficiency_score': self.learning_efficiency_score
            },
            'trends': {
                'performance_trend': self.recent_performance_trend,
                'learning_velocity': self.learning_velocity,
                'plateau_duration': self.plateau_duration
            },
            'diagnostics': {
                'issues': self.learning_issues,
                'recommendations': self.recommendations
            },
            'metadata': {
                'timestamp': self.timestamp,
                'steps_evaluated': self.steps_evaluated,
                'total_updates': self.total_updates
            }
        }


class QLearningEvaluator:
    """Evaluates Q-learning performance across different agent types."""
    
    def __init__(self, evaluation_window: int = 1000, update_frequency: int = 125):  # Increased by 25%
        self.evaluation_window = evaluation_window
        self.update_frequency = update_frequency
        
        # Storage for metrics
        self.agent_metrics_history: Dict[str, List[QLearningMetrics]] = defaultdict(list)
        self.agent_value_predictions: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self.agent_actual_rewards: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self.agent_q_value_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.agent_action_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self.agent_state_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        
        # Learning milestones tracking
        self.learning_milestones: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.performance_baselines: Dict[str, float] = defaultdict(float)
        self.last_update_time: Dict[str, float] = defaultdict(float)
        
        # Agent type classification
        self.agent_type_mapping: Dict[str, str] = {}
        
        # Comparative analysis
        self.type_performance_comparison: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        print("🧠 Q-Learning Evaluator initialized")
    
    def register_agent(self, agent) -> None:
        """Register an agent for evaluation."""
        agent_id = str(agent.id)
        
        # Determine agent type
        if hasattr(agent, 'learning_approach'):
            agent_type = agent.learning_approach
        elif hasattr(agent, 'q_table') and hasattr(agent.q_table, '__class__'):
            if 'Enhanced' in agent.q_table.__class__.__name__:
                agent_type = 'enhanced_q_learning'
            elif 'Survival' in agent.q_table.__class__.__name__:
                agent_type = 'survival_q_learning'
            else:
                agent_type = 'basic_q_learning'
        else:
            agent_type = 'unknown'
        
        self.agent_type_mapping[agent_id] = agent_type
        
        # Initialize baseline performance
        if agent_id not in self.performance_baselines:
            self.performance_baselines[agent_id] = getattr(agent, 'total_reward', 0.0)
        
        print(f"📊 Registered agent {agent_id} as type: {agent_type}")
    
    def record_q_learning_step(self, agent, state, action: int, 
                              predicted_q_value: float, actual_reward: float,
                              next_state, learning_occurred: bool = True) -> None:
        """
        Record a Q-learning step for evaluation.
        
        Args:
            agent: The agent that took the action
            state: The state the agent was in
            action: The action taken
            predicted_q_value: The Q-value the agent predicted for this state-action
            actual_reward: The actual reward received
            next_state: The resulting state
            learning_occurred: Whether a Q-learning update happened
        """
        agent_id = str(agent.id)
        
        # Ensure agent is registered
        if agent_id not in self.agent_type_mapping:
            self.register_agent(agent)
        
        # Record the prediction vs reality
        self.agent_value_predictions[agent_id].append(predicted_q_value)
        self.agent_actual_rewards[agent_id].append(actual_reward)
        
        # Record state-action history
        self.agent_action_history[agent_id].append(action)
        if hasattr(state, '__iter__'):
            state_hash = hash(tuple(state) if isinstance(state, (list, tuple)) else state)
        else:
            state_hash = hash(state)
        self.agent_state_history[agent_id].append(state_hash)
        
        # Record Q-value if available
        if hasattr(agent, 'q_table'):
            try:
                # Get current Q-values for the state
                if hasattr(agent.q_table, 'get_action_values'):
                    action_values = agent.q_table.get_action_values(state)
                    if action_values:
                        mean_q = np.mean(action_values)
                        self.agent_q_value_history[agent_id].append(mean_q)
                elif hasattr(agent.q_table, 'get_q_value'):
                    q_val = agent.q_table.get_q_value(state, action)
                    self.agent_q_value_history[agent_id].append(q_val)
            except Exception as e:
                # Silently handle Q-table access errors
                pass
        
        # Track learning milestones
        self._update_learning_milestones(agent_id, actual_reward)
        
        # Periodic evaluation
        current_time = time.time()
        if (current_time - self.last_update_time[agent_id]) > (self.update_frequency / 10.0):
            if len(self.agent_value_predictions[agent_id]) >= 10:  # Minimum data for evaluation
                self._evaluate_agent_performance(agent)
            self.last_update_time[agent_id] = current_time

    def _update_learning_milestones(self, agent_id: str, reward: float) -> None:
        """Track important learning milestones."""
        if agent_id not in self.learning_milestones:
            self.learning_milestones[agent_id] = {
                'first_positive_reward_step': -1,
                'first_significant_reward_step': -1,
                'stable_policy_step': -1,
                'total_steps': 0
            }
        
        milestones = self.learning_milestones[agent_id]
        milestones['total_steps'] += 1
        
        # First positive reward
        if reward > 0 and milestones['first_positive_reward_step'] == -1:
            milestones['first_positive_reward_step'] = milestones['total_steps']
        
        # First significant reward (> 0.1)
        if reward > 0.1 and milestones['first_significant_reward_step'] == -1:
            milestones['first_significant_reward_step'] = milestones['total_steps']

    def _calculate_exploration_ratios(self, agent, action_history: List[int]) -> Tuple[float, float]:
        """Calculate exploration vs exploitation ratios."""
        if not action_history or len(action_history) < 10:
            return 0.5, 0.5
        
        # Simple heuristic: if agent has epsilon, use that
        epsilon = getattr(agent, 'epsilon', 0.1)
        
        # Also look at action diversity
        unique_actions = len(set(action_history[-50:]))  # Last 50 actions
        total_actions = getattr(agent, 'action_size', len(set(action_history)))
        
        if total_actions > 0:
            action_diversity = unique_actions / total_actions
            exploration_estimate = max(epsilon, action_diversity)
        else:
            exploration_estimate = epsilon
        
        exploration_ratio = min(1.0, exploration_estimate)
        exploitation_ratio = 1.0 - exploration_ratio
        
        return exploration_ratio, exploitation_ratio

    def _calculate_convergence_score(self, q_history: List[float]) -> float:
        """Calculate how converged the Q-values are (0 = changing rapidly, 1 = stable)."""
        if len(q_history) < 20:
            return 0.0
        
        # Look at recent changes in Q-values
        recent_values = np.array(q_history[-20:])
        if len(recent_values) < 2:
            return 0.0
        
        # Calculate variance in recent changes
        changes = np.diff(recent_values)
        change_variance = np.var(changes) if len(changes) > 0 else 0.0
        
        # Convert to convergence score (lower variance = higher convergence)
        convergence_score = 1.0 / (1.0 + change_variance * 10)
        return float(np.clip(convergence_score, 0.0, 1.0))

    def _calculate_value_change_rate(self, q_history: List[float]) -> float:
        """Calculate the rate of Q-value change."""
        if len(q_history) < 2:
            return 0.0
        
        changes = np.diff(q_history[-20:]) if len(q_history) >= 20 else np.diff(q_history)
        return float(np.mean(np.abs(changes))) if len(changes) > 0 else 0.0

    def _calculate_policy_stability(self, agent, state_history: List[int], 
                                   action_history: List[int]) -> float:
        """Calculate how stable the policy is."""
        if len(action_history) < 20 or not hasattr(agent, 'q_table'):
            return 0.0
        
        # Look at action consistency for recently visited states
        state_action_pairs = list(zip(state_history[-20:], action_history[-20:]))
        state_counts = defaultdict(list)
        
        for state, action in state_action_pairs:
            state_counts[state].append(action)
        
        # Calculate consistency for each state
        consistencies = []
        for state, actions in state_counts.items():
            if len(actions) > 1:
                most_common_action = max(set(actions), key=actions.count)
                consistency = actions.count(most_common_action) / len(actions)
                consistencies.append(consistency)
        
        return float(np.mean(consistencies)) if consistencies else 0.0

    def _calculate_experience_diversity(self, state_history: List[int], 
                                       action_history: List[int]) -> float:
        """Calculate diversity of experienced state-action pairs."""
        if not state_history or not action_history:
            return 0.0
        
        # Count unique state-action pairs
        state_action_pairs = list(zip(state_history, action_history))
        unique_pairs = len(set(state_action_pairs))
        total_pairs = len(state_action_pairs)
        
        return unique_pairs / total_pairs if total_pairs > 0 else 0.0

    def _calculate_state_coverage(self, agent, state_history: List[int]) -> float:
        """Calculate what percentage of the state space has been explored."""
        if not state_history:
            return 0.0
        
        unique_states = len(set(state_history))
        
        # Estimate total state space size
        if hasattr(agent, 'q_table') and hasattr(agent.q_table, 'get_stats'):
            stats = agent.q_table.get_stats()
            total_states = stats.get('total_states', unique_states * 2)
        else:
            # Conservative estimate
            total_states = max(unique_states * 2, 100)
        
        return min(1.0, unique_states / total_states)

    def _calculate_action_entropy(self, action_history: List[int]) -> float:
        """Calculate entropy of action selection distribution."""
        if not action_history:
            return 0.0
        
        # Count action frequencies
        action_counts = defaultdict(int)
        for action in action_history[-100:]:  # Recent actions
            action_counts[action] += 1
        
        total_actions = sum(action_counts.values())
        if total_actions == 0:
            return 0.0
        
        # Calculate entropy
        entropy = 0.0
        for count in action_counts.values():
            if count > 0:
                prob = count / total_actions
                entropy -= prob * np.log2(prob)
        
        return float(entropy)

    def _calculate_reward_improvement_rate(self, reward_history: List[float]) -> float:
        """Calculate the rate of reward improvement."""
        if len(reward_history) < 20:
            return 0.0
        
        # Compare recent rewards to earlier rewards
        recent_rewards = np.array(reward_history[-10:])
        earlier_rewards = np.array(reward_history[-20:-10])
        
        recent_mean = np.mean(recent_rewards)
        earlier_mean = np.mean(earlier_rewards)
        
        improvement = recent_mean - earlier_mean
        return float(improvement)

    def _calculate_efficiency_score(self, steps_to_first_reward: int, 
                                   prediction_mae: float, convergence_score: float) -> float:
        """Calculate overall learning efficiency score."""
        efficiency_components = []
        
        # Speed to first reward (lower is better)
        if steps_to_first_reward > 0:
            reward_speed_score = max(0.0, 1.0 - (steps_to_first_reward / 1000.0))
            efficiency_components.append(reward_speed_score)
        
        # Prediction accuracy (lower MAE is better)
        accuracy_score = max(0.0, 1.0 - min(prediction_mae, 1.0))
        efficiency_components.append(accuracy_score)
        
        # Convergence (higher is better)
        efficiency_components.append(convergence_score)
        
        return float(np.mean(efficiency_components)) if efficiency_components else 0.0

    def _analyze_performance_trend(self, reward_history: List[float]) -> str:
        """Analyze recent performance trend."""
        if len(reward_history) < 30:
            return "insufficient_data"
        
        # Compare recent thirds
        recent_third = np.mean(reward_history[-10:])
        middle_third = np.mean(reward_history[-20:-10])
        early_third = np.mean(reward_history[-30:-20])
        
        # Determine trend
        if recent_third > middle_third > early_third:
            return "improving"
        elif recent_third < middle_third < early_third:
            return "declining"
        elif abs(recent_third - middle_third) < 0.01:
            return "stable"
        else:
            return "fluctuating"

    def _calculate_learning_velocity(self, reward_history: List[float], 
                                    q_history: List[float]) -> float:
        """Calculate the velocity of learning (rate of change)."""
        if len(reward_history) < 10:
            return 0.0
        
        # Combine reward improvement and Q-value stabilization
        reward_velocity = 0.0
        if len(reward_history) >= 20:
            recent_rewards = np.array(reward_history[-10:])
            earlier_rewards = np.array(reward_history[-20:-10])
            reward_velocity = (np.mean(recent_rewards) - np.mean(earlier_rewards)) / 10
        
        q_stability_velocity = 0.0
        if len(q_history) >= 20:
            recent_q_std = np.std(q_history[-10:])
            earlier_q_std = np.std(q_history[-20:-10])
            q_stability_velocity = max(0, earlier_q_std - recent_q_std) / 10  # Decreasing std is good
        
        return float(reward_velocity + q_stability_velocity)

    def _calculate_plateau_duration(self, reward_history: List[float]) -> int:
        """Calculate how long the agent has been on a performance plateau."""
        if len(reward_history) < 50:
            return 0
        
        # Look for the last significant improvement
        improvement_threshold = 0.02  # Minimum improvement to count
        steps_since_improvement = 0
        
        recent_mean = np.mean(reward_history[-10:])
        
        for i in range(10, min(len(reward_history), 200), 10):
            historical_mean = np.mean(reward_history[-(i+10):-i])
            improvement = recent_mean - historical_mean
            
            if improvement > improvement_threshold:
                break
            steps_since_improvement = i
        
        return steps_since_improvement

    def _diagnose_learning_issues(self, prediction_mae: float, convergence_score: float,
                                 exploration_ratio: float, trend: str, plateau_duration: int) -> List[str]:
        """Diagnose potential learning issues."""
        issues = []
        
        # Poor value prediction
        if prediction_mae > 0.5:
            issues.append("high_value_prediction_error")
        
        # Poor convergence
        if convergence_score < 0.3:
            issues.append("poor_convergence")
        
        # Exploration issues
        if exploration_ratio < 0.05:
            issues.append("insufficient_exploration")
        elif exploration_ratio > 0.8:
            issues.append("excessive_exploration")
        
        # Performance issues
        if trend == "declining":
            issues.append("declining_performance")
        
        if plateau_duration > 100:
            issues.append("learning_plateau")
        
        return issues

    def _generate_recommendations(self, issues: List[str], agent_type: str) -> List[str]:
        """Generate recommendations based on detected issues."""
        recommendations = []
        
        if "high_value_prediction_error" in issues:
            recommendations.append("Consider adjusting learning rate or improving state representation")
        
        if "poor_convergence" in issues:
            recommendations.append("Reduce learning rate or increase experience replay")
        
        if "insufficient_exploration" in issues:
            recommendations.append("Increase epsilon or add exploration bonus")
        
        if "excessive_exploration" in issues:
            recommendations.append("Decrease epsilon or improve exploitation strategy")
        
        if "declining_performance" in issues:
            recommendations.append("Check for learning rate decay or catastrophic forgetting")
        
        if "learning_plateau" in issues:
            if agent_type == "basic_q_learning":
                recommendations.append("Consider upgrading to enhanced Q-learning")
            else:
                recommendations.append("Try curriculum learning or reward shaping")
        
        return recommendations

    def _evaluate_agent_performance(self, agent) -> QLearningMetrics:
        """Evaluate comprehensive Q-learning performance for an agent."""
        agent_id = str(agent.id)
        agent_type = self.agent_type_mapping.get(agent_id, 'unknown')
        
        # Get recent data
        predictions = list(self.agent_value_predictions[agent_id])
        actual_rewards = list(self.agent_actual_rewards[agent_id])
        q_history = list(self.agent_q_value_history[agent_id])
        action_history = list(self.agent_action_history[agent_id])
        state_history = list(self.agent_state_history[agent_id])
        
        # Calculate value prediction accuracy (CORE METRIC)
        value_prediction_error = 0.0
        value_prediction_mae = 0.0
        value_prediction_rmse = 0.0
        
        if len(predictions) > 0 and len(actual_rewards) > 0:
            min_len = min(len(predictions), len(actual_rewards))
            pred_list = predictions[-min_len:]
            actual_list = actual_rewards[-min_len:]
            
            # Calculate errors manually to work with NumpySubstitute
            errors = [abs(p - a) for p, a in zip(pred_list, actual_list)]
            value_prediction_error = float(errors[-1]) if len(errors) > 0 else 0.0
            value_prediction_mae = float(np.mean(errors))
            # Calculate RMSE manually
            squared_errors = [e * e for e in errors]
            value_prediction_rmse = float(np.sqrt(np.mean(squared_errors)))
        
        # Q-value analysis
        q_value_mean = float(np.mean(q_history)) if q_history else 0.0
        q_value_std = float(np.std(q_history)) if q_history else 0.0
        q_value_range = (float(np.min(q_history)), float(np.max(q_history))) if q_history else (0.0, 0.0)
        
        # Q-value distribution
        q_value_distribution = {}
        if q_history:
            hist, bin_edges = np.histogram(q_history, bins=10)
            for i, count in enumerate(hist):
                bin_center = (bin_edges[i] + bin_edges[i+1]) / 2
                q_value_distribution[f"bin_{bin_center:.3f}"] = int(count)
        
        # Learning parameters
        learning_rate_current = getattr(agent, 'learning_rate', 0.0)
        epsilon_current = getattr(agent, 'epsilon', 0.0)
        
        # Exploration vs exploitation analysis
        exploration_ratio, exploitation_ratio = self._calculate_exploration_ratios(agent, action_history)
        
        # Convergence metrics
        convergence_score = self._calculate_convergence_score(q_history)
        value_change_rate = self._calculate_value_change_rate(q_history)
        policy_stability = self._calculate_policy_stability(agent, state_history, action_history)
        
        # Experience diversity
        experience_diversity = self._calculate_experience_diversity(state_history, action_history)
        state_coverage = self._calculate_state_coverage(agent, state_history)
        action_preference_entropy = self._calculate_action_entropy(action_history)
        
        # Learning efficiency
        milestones = self.learning_milestones[agent_id]
        steps_to_first_reward = milestones.get('first_positive_reward_step', -1)
        steps_to_stable_policy = milestones.get('stable_policy_step', -1)
        reward_improvement_rate = self._calculate_reward_improvement_rate(actual_rewards)
        learning_efficiency_score = self._calculate_efficiency_score(
            steps_to_first_reward, value_prediction_mae, convergence_score
        )
        
        # Performance trends
        recent_performance_trend = self._analyze_performance_trend(actual_rewards)
        learning_velocity = self._calculate_learning_velocity(actual_rewards, q_history)
        plateau_duration = self._calculate_plateau_duration(actual_rewards)
        
        # Problem diagnosis
        learning_issues = self._diagnose_learning_issues(
            value_prediction_mae, convergence_score, exploration_ratio, 
            recent_performance_trend, plateau_duration
        )
        recommendations = self._generate_recommendations(learning_issues, agent_type)
        
        # Create metrics object
        metrics = QLearningMetrics(
            agent_id=agent_id,
            agent_type=agent_type,
            value_prediction_error=value_prediction_error,
            value_prediction_mae=value_prediction_mae,
            value_prediction_rmse=value_prediction_rmse,
            q_value_mean=q_value_mean,
            q_value_std=q_value_std,
            q_value_range=q_value_range,
            q_value_distribution=q_value_distribution,
            learning_rate_current=learning_rate_current,
            epsilon_current=epsilon_current,
            exploration_ratio=exploration_ratio,
            exploitation_ratio=exploitation_ratio,
            convergence_score=convergence_score,
            value_change_rate=value_change_rate,
            policy_stability=policy_stability,
            experience_diversity=experience_diversity,
            state_coverage=state_coverage,
            action_preference_entropy=action_preference_entropy,
            steps_to_first_reward=steps_to_first_reward,
            steps_to_stable_policy=steps_to_stable_policy,
            reward_improvement_rate=reward_improvement_rate,
            learning_efficiency_score=learning_efficiency_score,
            recent_performance_trend=recent_performance_trend,
            learning_velocity=learning_velocity,
            plateau_duration=plateau_duration,
            learning_issues=learning_issues,
            recommendations=recommendations,
            timestamp=time.time(),
            steps_evaluated=len(predictions),
            total_updates=milestones['total_steps']
        )
        
        # Store metrics
        self.agent_metrics_history[agent_id].append(metrics)
        if len(self.agent_metrics_history[agent_id]) > 100:  # Keep last 100 evaluations
            self.agent_metrics_history[agent_id].pop(0)
        
        return metrics

    def get_agent_metrics(self, agent_id: str) -> Optional[QLearningMetrics]:
        """Get the latest metrics for a specific agent."""
        if agent_id in self.agent_metrics_history and self.agent_metrics_history[agent_id]:
            return self.agent_metrics_history[agent_id][-1]
        return None

    def get_all_agent_metrics(self) -> Dict[str, QLearningMetrics]:
        """Get latest metrics for all agents."""
        metrics = {}
        for agent_id, history in self.agent_metrics_history.items():
            if history:
                metrics[agent_id] = history[-1]
        return metrics

    def get_type_comparison(self) -> Dict[str, Dict[str, float]]:
        """Get comparative analysis across agent types."""
        type_metrics = defaultdict(list)
        
        # Group metrics by agent type
        for agent_id, history in self.agent_metrics_history.items():
            if history:
                latest_metrics = history[-1]
                agent_type = latest_metrics.agent_type
                type_metrics[agent_type].append(latest_metrics)
        
        # Calculate averages per type
        comparison = {}
        for agent_type, metrics_list in type_metrics.items():
            if metrics_list:
                comparison[agent_type] = {
                    'avg_prediction_mae': np.mean([m.value_prediction_mae for m in metrics_list]),
                    'avg_convergence_score': np.mean([m.convergence_score for m in metrics_list]),
                    'avg_efficiency_score': np.mean([m.learning_efficiency_score for m in metrics_list]),
                    'avg_learning_velocity': np.mean([m.learning_velocity for m in metrics_list]),
                    'agent_count': len(metrics_list),
                    'avg_steps_to_first_reward': np.mean([m.steps_to_first_reward for m in metrics_list if m.steps_to_first_reward > 0]),
                    'common_issues': self._get_common_issues([m.learning_issues for m in metrics_list])
                }
        
        return dict(comparison)

    def _get_common_issues(self, issues_lists: List[List[str]]) -> List[str]:
        """Find the most common issues across agents."""
        all_issues = [issue for issues in issues_lists for issue in issues]
        issue_counts = defaultdict(int)
        for issue in all_issues:
            issue_counts[issue] += 1
        
        # Return issues that affect more than 25% of agents
        total_agents = len(issues_lists)
        threshold = max(1, total_agents * 0.25)
        
        return [issue for issue, count in issue_counts.items() if count >= threshold]

    def get_learning_diagnostics(self, agent_id: str) -> Dict[str, Any]:
        """Get comprehensive learning diagnostics for an agent."""
        if agent_id not in self.agent_metrics_history or not self.agent_metrics_history[agent_id]:
            return {'status': 'no_data'}
        
        latest_metrics = self.agent_metrics_history[agent_id][-1]
        
        # Determine overall health
        health_score = 0
        max_score = 5
        
        if latest_metrics.value_prediction_mae < 0.3:
            health_score += 1
        if latest_metrics.convergence_score > 0.5:
            health_score += 1
        if latest_metrics.learning_efficiency_score > 0.5:
            health_score += 1
        if latest_metrics.recent_performance_trend in ['improving', 'stable']:
            health_score += 1
        if latest_metrics.plateau_duration < 50:
            health_score += 1
        
        if health_score >= 4:
            overall_health = 'excellent'
        elif health_score >= 3:
            overall_health = 'good'
        elif health_score >= 2:
            overall_health = 'fair'
        else:
            overall_health = 'needs_attention'
        
        return {
            'agent_id': agent_id,
            'agent_type': latest_metrics.agent_type,
            'overall_health': overall_health,
            'health_score': f"{health_score}/{max_score}",
            'key_metrics': {
                'value_prediction_accuracy': 1.0 - min(1.0, latest_metrics.value_prediction_mae),
                'convergence_score': latest_metrics.convergence_score,
                'learning_efficiency': latest_metrics.learning_efficiency_score,
                'performance_trend': latest_metrics.recent_performance_trend
            },
            'issues_detected': latest_metrics.learning_issues,
            'recommendations': latest_metrics.recommendations,
            'learning_stage': self._determine_learning_stage(latest_metrics),
            'time_since_evaluation': time.time() - latest_metrics.timestamp
        }

    def _determine_learning_stage(self, metrics: QLearningMetrics) -> str:
        """Determine what stage of learning the agent is in."""
        if metrics.steps_to_first_reward == -1:
            return LearningStage.EXPLORATION.value
        elif metrics.convergence_score < 0.3:
            return LearningStage.LEARNING.value
        elif metrics.convergence_score > 0.7 and metrics.plateau_duration < 20:
            return LearningStage.CONVERGENCE.value
        elif metrics.learning_efficiency_score > 0.8:
            return LearningStage.MASTERY.value
        else:
            return LearningStage.PLATEAU.value

    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a comprehensive summary report."""
        all_metrics = self.get_all_agent_metrics()
        type_comparison = self.get_type_comparison()
        
        if not all_metrics:
            return {'status': 'no_data', 'message': 'No Q-learning data available'}
        
        # Overall statistics
        all_mae_values = [m.value_prediction_mae for m in all_metrics.values()]
        all_efficiency_scores = [m.learning_efficiency_score for m in all_metrics.values()]
        all_convergence_scores = [m.convergence_score for m in all_metrics.values()]
        
        # Best performing agents
        best_accuracy_agent = min(all_metrics.items(), key=lambda x: x[1].value_prediction_mae)
        best_efficiency_agent = max(all_metrics.items(), key=lambda x: x[1].learning_efficiency_score)
        best_convergence_agent = max(all_metrics.items(), key=lambda x: x[1].convergence_score)
        
        return {
            'timestamp': time.time(),
            'total_agents_evaluated': len(all_metrics),
            'overall_statistics': {
                'avg_prediction_mae': float(np.mean(all_mae_values)),
                'avg_efficiency_score': float(np.mean(all_efficiency_scores)),
                'avg_convergence_score': float(np.mean(all_convergence_scores)),
                'prediction_accuracy_range': [float(np.min(all_mae_values)), float(np.max(all_mae_values))]
            },
            'best_performers': {
                'most_accurate': {
                    'agent_id': best_accuracy_agent[0],
                    'agent_type': best_accuracy_agent[1].agent_type,
                    'mae': best_accuracy_agent[1].value_prediction_mae
                },
                'most_efficient': {
                    'agent_id': best_efficiency_agent[0],
                    'agent_type': best_efficiency_agent[1].agent_type,
                    'efficiency_score': best_efficiency_agent[1].learning_efficiency_score
                },
                'best_convergence': {
                    'agent_id': best_convergence_agent[0],
                    'agent_type': best_convergence_agent[1].agent_type,
                    'convergence_score': best_convergence_agent[1].convergence_score
                }
            },
            'agent_type_comparison': type_comparison,
            'system_health': {
                'agents_with_issues': len([m for m in all_metrics.values() if m.learning_issues]),
                'agents_learning_well': len([m for m in all_metrics.values() if m.learning_efficiency_score > 0.6]),
                'agents_converged': len([m for m in all_metrics.values() if m.convergence_score > 0.7])
            }
        }
