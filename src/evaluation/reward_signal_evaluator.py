"""
Reward Signal Quality Evaluator
Comprehensive evaluation system for analyzing the quality and effectiveness of reward signals in RL.
"""

import time
import math
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum

# Handle numpy import gracefully and add missing entropy function
try:
    import numpy as np
    # Add entropy function to numpy if it doesn't exist
    if not hasattr(np, 'entropy'):
        def entropy(data):
            if not data: return 0.0
            # Calculate entropy of reward distribution
            counts = {}
            for x in data:
                rounded = round(x, 3)  # Round for binning
                counts[rounded] = counts.get(rounded, 0) + 1
            total = len(data)
            entropy_val = 0.0
            for count in counts.values():
                p = count / total
                if p > 0:
                    entropy_val -= p * math.log2(p)
            return entropy_val
        np.entropy = entropy
except ImportError:
    # Basic numpy substitute for statistical operations
    class NumpySubstitute:
        @staticmethod
        def mean(data): return sum(data) / len(data) if data else 0.0
        @staticmethod
        def std(data):
            if not data: return 0.0
            mean_val = sum(data) / len(data)
            return (sum((x - mean_val) ** 2 for x in data) / len(data)) ** 0.5
        @staticmethod
        def var(data):
            if not data: return 0.0
            mean_val = sum(data) / len(data)
            return sum((x - mean_val) ** 2 for x in data) / len(data)
        @staticmethod
        def percentile(data, p):
            if not data: return 0.0
            sorted_data = sorted(data)
            k = (len(sorted_data) - 1) * p / 100
            f = math.floor(k)
            c = math.ceil(k)
            if f == c: return sorted_data[int(k)]
            return sorted_data[int(f)] * (c - k) + sorted_data[int(c)] * (k - f)
        @staticmethod
        def entropy(data):
            if not data: return 0.0
            # Calculate entropy of reward distribution
            counts = {}
            for x in data:
                rounded = round(x, 3)  # Round for binning
                counts[rounded] = counts.get(rounded, 0) + 1
            total = len(data)
            entropy = 0.0
            for count in counts.values():
                p = count / total
                if p > 0:
                    entropy -= p * math.log2(p)
            return entropy
    np = NumpySubstitute()


class RewardQualityIssue(Enum):
    """Types of reward signal quality issues."""
    SPARSE_REWARDS = "sparse_rewards"
    NOISY_REWARDS = "noisy_rewards"
    BIASED_REWARDS = "biased_rewards"
    INCONSISTENT_REWARDS = "inconsistent_rewards"
    POOR_EXPLORATION_INCENTIVE = "poor_exploration_incentive"
    REWARD_HACKING = "reward_hacking"
    DELAYED_REWARDS = "delayed_rewards"
    SATURATED_REWARDS = "saturated_rewards"


@dataclass
class RewardSignalMetrics:
    """Comprehensive metrics for reward signal quality."""
    agent_id: str
    
    # Basic reward statistics
    reward_mean: float
    reward_std: float
    reward_range: Tuple[float, float]
    reward_sparsity: float  # Percentage of zero rewards
    reward_density: float   # Percentage of non-zero rewards
    
    # Signal quality metrics
    signal_to_noise_ratio: float  # Mean absolute reward / reward std
    reward_consistency: float     # Consistency across similar states
    temporal_consistency: float   # Consistency over time
    
    # Distribution analysis
    reward_entropy: float         # Information content of reward distribution
    reward_skewness: float       # Asymmetry of reward distribution
    positive_reward_ratio: float # Percentage of positive rewards
    
    # Learning effectiveness
    exploration_incentive: float  # How well rewards encourage exploration
    convergence_support: float   # How well rewards support convergence
    behavioral_alignment: float  # How well rewards align with desired behavior
    
    # Temporal patterns
    reward_autocorrelation: float # Correlation between consecutive rewards
    reward_smoothness: float      # How smooth the reward signal is
    reward_lag: float            # Average delay between action and reward
    
    # Problem indicators
    quality_issues: List[RewardQualityIssue]
    quality_score: float         # Overall quality score (0-1)
    recommendations: List[str]   # Improvement suggestions
    
    # Metadata
    timestamp: float
    steps_analyzed: int
    total_rewards_received: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'agent_id': self.agent_id,
            'basic_statistics': {
                'mean': self.reward_mean,
                'std': self.reward_std,
                'min': self.reward_range[0],
                'max': self.reward_range[1],
                'sparsity': self.reward_sparsity,
                'density': self.reward_density
            },
            'signal_quality': {
                'signal_to_noise_ratio': self.signal_to_noise_ratio,
                'consistency': self.reward_consistency,
                'temporal_consistency': self.temporal_consistency
            },
            'distribution': {
                'entropy': self.reward_entropy,
                'skewness': self.reward_skewness,
                'positive_ratio': self.positive_reward_ratio
            },
            'learning_support': {
                'exploration_incentive': self.exploration_incentive,
                'convergence_support': self.convergence_support,
                'behavioral_alignment': self.behavioral_alignment
            },
            'temporal_patterns': {
                'autocorrelation': self.reward_autocorrelation,
                'smoothness': self.reward_smoothness,
                'lag': self.reward_lag
            },
            'quality_assessment': {
                'overall_score': self.quality_score,
                'issues': [issue.value for issue in self.quality_issues],
                'recommendations': self.recommendations
            },
            'metadata': {
                'timestamp': self.timestamp,
                'steps_analyzed': self.steps_analyzed,
                'total_rewards': self.total_rewards_received
            }
        }


class RewardSignalEvaluator:
    """Evaluates the quality and effectiveness of reward signals."""
    
    def __init__(self, window_size: int = 1000, min_samples: int = 50):
        self.window_size = window_size
        self.min_samples = min_samples
        
        # Storage for reward data
        self.agent_rewards: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.agent_states: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.agent_actions: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.agent_timestamps: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        
        # Metrics history
        self.metrics_history: Dict[str, List[RewardSignalMetrics]] = defaultdict(list)
        
        # State-action reward tracking for consistency analysis
        self.state_action_rewards: Dict[str, Dict[Tuple, List[float]]] = defaultdict(lambda: defaultdict(list))
        
        print("📊 Reward Signal Evaluator initialized")
    
    def record_reward(self, agent_id: str, state: Any, action: int, reward: float, timestamp: Optional[float] = None):
        """Record a reward signal for analysis."""
        if timestamp is None:
            timestamp = time.time()
        
        try:
            # Store the data
            self.agent_rewards[agent_id].append(reward)
            self.agent_states[agent_id].append(state)
            self.agent_actions[agent_id].append(action)
            self.agent_timestamps[agent_id].append(timestamp)
            
            # Store for state-action consistency analysis with improved state handling
            try:
                # Handle different state types more robustly
                if hasattr(state, '__iter__') and not isinstance(state, (str, bytes)):
                    # For iterables (lists, tuples, numpy arrays), convert to tuple
                    if hasattr(state, 'tolist'):  # numpy array
                        state_key = tuple(state.tolist())
                    else:
                        state_key = tuple(state)
                elif hasattr(state, '__dict__'):
                    # For objects with attributes, use a string representation
                    state_key = str(state)
                else:
                    # For simple types (int, float, string)
                    state_key = state
                    
                action_key = (state_key, action)
                self.state_action_rewards[agent_id][action_key].append(reward)
                
            except Exception as state_error:
                # If state conversion fails, use a fallback but still record the reward
                print(f"⚠️ State conversion failed for agent {agent_id}: {state_error}")
                fallback_key = (f"state_hash_{hash(str(state))}", action)
                self.state_action_rewards[agent_id][fallback_key].append(reward)
            
            # Periodic evaluation
            if len(self.agent_rewards[agent_id]) >= self.min_samples:
                if len(self.agent_rewards[agent_id]) % 100 == 0:  # Evaluate every 100 samples
                    self._evaluate_agent_reward_quality(agent_id)
                    
        except Exception as e:
            print(f"❌ Failed to record reward for agent {agent_id}: {e}")
            print(f"   State type: {type(state)}, Action: {action}, Reward: {reward}")
            # Don't let recording failures break the system
            pass
    
    def _evaluate_agent_reward_quality(self, agent_id: str) -> RewardSignalMetrics:
        """Evaluate reward signal quality for an agent."""
        rewards = list(self.agent_rewards[agent_id])
        states = list(self.agent_states[agent_id])
        actions = list(self.agent_actions[agent_id])
        timestamps = list(self.agent_timestamps[agent_id])
        
        if len(rewards) < self.min_samples:
            return None
        
        # Basic statistics
        reward_mean = np.mean(rewards)
        reward_std = np.std(rewards)
        reward_range = (min(rewards), max(rewards))
        
        # Sparsity analysis
        zero_rewards = sum(1 for r in rewards if abs(r) < 1e-6)
        reward_sparsity = zero_rewards / len(rewards)
        reward_density = 1.0 - reward_sparsity
        
        # Signal quality
        signal_to_noise_ratio = abs(reward_mean) / (reward_std + 1e-8)
        reward_consistency = self._calculate_reward_consistency(agent_id)
        temporal_consistency = self._calculate_temporal_consistency(rewards, timestamps)
        
        # Distribution analysis
        reward_entropy = np.entropy(rewards)
        reward_skewness = self._calculate_skewness(rewards)
        positive_rewards = sum(1 for r in rewards if r > 0)
        positive_reward_ratio = positive_rewards / len(rewards)
        
        # Learning effectiveness
        exploration_incentive = self._calculate_exploration_incentive(agent_id)
        convergence_support = self._calculate_convergence_support(rewards)
        behavioral_alignment = self._calculate_behavioral_alignment(agent_id)
        
        # Temporal patterns
        reward_autocorrelation = self._calculate_autocorrelation(rewards)
        reward_smoothness = self._calculate_smoothness(rewards)
        reward_lag = self._calculate_reward_lag(timestamps)
        
        # Quality assessment
        quality_issues = self._identify_quality_issues(rewards, reward_sparsity, signal_to_noise_ratio, 
                                                      reward_consistency, exploration_incentive)
        quality_score = self._calculate_overall_quality_score(signal_to_noise_ratio, reward_consistency,
                                                             exploration_incentive, convergence_support)
        recommendations = self._generate_recommendations(quality_issues, quality_score)
        
        # Create metrics object
        metrics = RewardSignalMetrics(
            agent_id=agent_id,
            reward_mean=reward_mean,
            reward_std=reward_std,
            reward_range=reward_range,
            reward_sparsity=reward_sparsity,
            reward_density=reward_density,
            signal_to_noise_ratio=signal_to_noise_ratio,
            reward_consistency=reward_consistency,
            temporal_consistency=temporal_consistency,
            reward_entropy=reward_entropy,
            reward_skewness=reward_skewness,
            positive_reward_ratio=positive_reward_ratio,
            exploration_incentive=exploration_incentive,
            convergence_support=convergence_support,
            behavioral_alignment=behavioral_alignment,
            reward_autocorrelation=reward_autocorrelation,
            reward_smoothness=reward_smoothness,
            reward_lag=reward_lag,
            quality_issues=quality_issues,
            quality_score=quality_score,
            recommendations=recommendations,
            timestamp=time.time(),
            steps_analyzed=len(rewards),
            total_rewards_received=len([r for r in rewards if abs(r) > 1e-6])
        )
        
        # Store metrics
        self.metrics_history[agent_id].append(metrics)
        if len(self.metrics_history[agent_id]) > 50:  # Keep last 50 evaluations
            self.metrics_history[agent_id].pop(0)
        
        return metrics
    
    def _calculate_reward_consistency(self, agent_id: str) -> float:
        """Calculate how consistent rewards are for similar state-action pairs."""
        state_action_rewards = self.state_action_rewards[agent_id]
        
        if not state_action_rewards:
            return 0.0
        
        consistencies = []
        for state_action, reward_list in state_action_rewards.items():
            if len(reward_list) > 1:
                # Calculate coefficient of variation (std/mean) for this state-action
                reward_std = np.std(reward_list)
                reward_mean = abs(np.mean(reward_list))
                if reward_mean > 1e-6:
                    cv = reward_std / reward_mean
                    consistency = max(0.0, 1.0 - cv)  # Lower CV = higher consistency
                    consistencies.append(consistency)
        
        return np.mean(consistencies) if consistencies else 0.0
    
    def _calculate_temporal_consistency(self, rewards: List[float], timestamps: List[float]) -> float:
        """Calculate temporal consistency of reward signal."""
        if len(rewards) < 10:
            return 0.0
        
        # Split into time windows and compare variance
        window_size = len(rewards) // 5  # 5 windows
        if window_size < 2:
            return 1.0
        
        window_variances = []
        for i in range(0, len(rewards) - window_size + 1, window_size):
            window_rewards = rewards[i:i + window_size]
            if len(window_rewards) > 1:
                window_variances.append(np.var(window_rewards))
        
        if not window_variances:
            return 1.0
        
        # Lower variance in variances = higher temporal consistency
        variance_of_variances = np.var(window_variances)
        mean_variance = np.mean(window_variances)
        
        if mean_variance > 1e-6:
            consistency = max(0.0, 1.0 - (variance_of_variances / mean_variance))
        else:
            consistency = 1.0
        
        return consistency
    
    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness of reward distribution."""
        if len(data) < 3:
            return 0.0
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        if std_val < 1e-6:
            return 0.0
        
        skewness = sum(((x - mean_val) / std_val) ** 3 for x in data) / len(data)
        return skewness
    
    def _calculate_exploration_incentive(self, agent_id: str) -> float:
        """Calculate how well the reward signal incentivizes exploration."""
        state_action_rewards = self.state_action_rewards[agent_id]
        
        if not state_action_rewards:
            return 0.0
        
        # Measure diversity of rewards across different state-actions
        unique_state_actions = len(state_action_rewards)
        total_samples = sum(len(rewards) for rewards in state_action_rewards.values())
        
        if total_samples == 0:
            return 0.0
        
        # Calculate exploration incentive based on:
        # 1. Diversity of state-actions explored
        # 2. Variance in rewards across state-actions
        
        diversity_score = min(1.0, unique_state_actions / 50.0)  # Normalize to reasonable range
        
        # Calculate reward variance across different state-actions
        all_means = [np.mean(rewards) for rewards in state_action_rewards.values()]
        if len(all_means) > 1:
            variance_score = min(1.0, np.std(all_means) / (abs(np.mean(all_means)) + 1e-6))
        else:
            variance_score = 0.0
        
        return (diversity_score + variance_score) / 2.0
    
    def _calculate_convergence_support(self, rewards: List[float]) -> float:
        """Calculate how well the reward signal supports convergence."""
        if len(rewards) < 20:
            return 0.0
        
        # Look at trend in reward variance over time
        window_size = len(rewards) // 4
        if window_size < 2:
            return 1.0
        
        early_rewards = rewards[:window_size]
        late_rewards = rewards[-window_size:]
        
        early_variance = np.var(early_rewards)
        late_variance = np.var(late_rewards)
        
        # Good convergence support: variance decreases over time
        if early_variance > 1e-6:
            convergence_support = max(0.0, (early_variance - late_variance) / early_variance)
        else:
            convergence_support = 1.0 if late_variance < 1e-6 else 0.0
        
        return min(1.0, convergence_support)
    
    def _calculate_behavioral_alignment(self, agent_id: str) -> float:
        """Calculate how well rewards align with desired behavior."""
        rewards = list(self.agent_rewards[agent_id])
        
        if not rewards:
            return 0.0
        
        # Simple heuristic: positive rewards should be more common than negative
        # for tasks where we want to encourage behavior
        positive_rewards = sum(1 for r in rewards if r > 0)
        negative_rewards = sum(1 for r in rewards if r < 0)
        
        if positive_rewards + negative_rewards == 0:
            return 0.0
        
        positive_ratio = positive_rewards / (positive_rewards + negative_rewards)
        
        # Also consider reward magnitude - larger positive rewards are better
        if positive_rewards > 0:
            avg_positive = np.mean([r for r in rewards if r > 0])
            avg_negative = abs(np.mean([r for r in rewards if r < 0])) if negative_rewards > 0 else 0
            
            magnitude_ratio = avg_positive / (avg_positive + avg_negative + 1e-6)
        else:
            magnitude_ratio = 0.0
        
        return (positive_ratio + magnitude_ratio) / 2.0
    
    def _calculate_autocorrelation(self, rewards: List[float]) -> float:
        """Calculate autocorrelation in reward signal."""
        if len(rewards) < 3:
            return 0.0
        
        # Calculate lag-1 autocorrelation
        mean_reward = np.mean(rewards)
        numerator = sum((rewards[i] - mean_reward) * (rewards[i+1] - mean_reward) 
                       for i in range(len(rewards) - 1))
        denominator = sum((r - mean_reward) ** 2 for r in rewards)
        
        if denominator < 1e-6:
            return 0.0
        
        return numerator / denominator
    
    def _calculate_smoothness(self, rewards: List[float]) -> float:
        """Calculate smoothness of reward signal."""
        if len(rewards) < 2:
            return 1.0
        
        # Calculate total variation (sum of absolute differences)
        total_variation = sum(abs(rewards[i+1] - rewards[i]) for i in range(len(rewards) - 1))
        reward_range = max(rewards) - min(rewards)
        
        if reward_range < 1e-6:
            return 1.0
        
        # Normalize by range - lower variation = higher smoothness
        smoothness = max(0.0, 1.0 - (total_variation / (reward_range * len(rewards))))
        return smoothness
    
    def _calculate_reward_lag(self, timestamps: List[float]) -> float:
        """Calculate average delay in reward signal."""
        if len(timestamps) < 2:
            return 0.0
        
        # Simple heuristic: average time between consecutive rewards
        time_diffs = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps) - 1)]
        return np.mean(time_diffs)
    
    def _identify_quality_issues(self, rewards: List[float], sparsity: float, 
                               snr: float, consistency: float, exploration: float) -> List[RewardQualityIssue]:
        """Identify potential quality issues with the reward signal."""
        issues = []
        
        # Sparse rewards
        if sparsity > 0.9:
            issues.append(RewardQualityIssue.SPARSE_REWARDS)
        
        # Noisy rewards (low SNR)
        if snr < 0.5:
            issues.append(RewardQualityIssue.NOISY_REWARDS)
        
        # Inconsistent rewards
        if consistency < 0.3:
            issues.append(RewardQualityIssue.INCONSISTENT_REWARDS)
        
        # Poor exploration incentive
        if exploration < 0.2:
            issues.append(RewardQualityIssue.POOR_EXPLORATION_INCENTIVE)
        
        # Biased rewards (all positive or all negative)
        positive_count = sum(1 for r in rewards if r > 0)
        negative_count = sum(1 for r in rewards if r < 0)
        if positive_count == 0 or negative_count == 0:
            if len(rewards) > 100:  # Only flag if we have enough samples
                issues.append(RewardQualityIssue.BIASED_REWARDS)
        
        # Saturated rewards (limited range)
        reward_range = max(rewards) - min(rewards)
        if reward_range < 0.01 and len(rewards) > 50:
            issues.append(RewardQualityIssue.SATURATED_REWARDS)
        
        return issues
    
    def _calculate_overall_quality_score(self, snr: float, consistency: float,
                                       exploration: float, convergence: float) -> float:
        """Calculate overall reward signal quality score."""
        # Weighted combination of key metrics
        weights = {
            'snr': 0.3,
            'consistency': 0.25,
            'exploration': 0.25,
            'convergence': 0.2
        }
        
        # Normalize metrics to 0-1 range
        normalized_snr = min(1.0, snr / 2.0)  # SNR > 2 is excellent
        
        score = (weights['snr'] * normalized_snr +
                weights['consistency'] * consistency +
                weights['exploration'] * exploration +
                weights['convergence'] * convergence)
        
        return max(0.0, min(1.0, score))
    
    def _generate_recommendations(self, issues: List[RewardQualityIssue], quality_score: float) -> List[str]:
        """Generate recommendations for improving reward signal quality."""
        recommendations = []
        
        if RewardQualityIssue.SPARSE_REWARDS in issues:
            recommendations.append("Add intermediate rewards or reward shaping to provide more frequent feedback")
        
        if RewardQualityIssue.NOISY_REWARDS in issues:
            recommendations.append("Reduce reward noise by smoothing or improving reward calculation")
        
        if RewardQualityIssue.INCONSISTENT_REWARDS in issues:
            recommendations.append("Ensure consistent rewards for similar state-action pairs")
        
        if RewardQualityIssue.POOR_EXPLORATION_INCENTIVE in issues:
            recommendations.append("Add exploration bonuses or curiosity-driven rewards")
        
        if RewardQualityIssue.BIASED_REWARDS in issues:
            recommendations.append("Balance positive and negative rewards to provide clear learning signals")
        
        if RewardQualityIssue.SATURATED_REWARDS in issues:
            recommendations.append("Increase reward range or improve reward sensitivity")
        
        if quality_score < 0.3:
            recommendations.append("Consider redesigning reward function with domain expertise")
        elif quality_score < 0.6:
            recommendations.append("Fine-tune reward parameters and test different reward formulations")
        
        return recommendations
    
    def get_agent_metrics(self, agent_id: str) -> Optional[RewardSignalMetrics]:
        """Get latest reward signal metrics for an agent."""
        if agent_id in self.metrics_history and self.metrics_history[agent_id]:
            return self.metrics_history[agent_id][-1]
        return None
    
    def get_all_metrics(self) -> Dict[str, RewardSignalMetrics]:
        """Get latest metrics for all agents."""
        metrics = {}
        for agent_id, history in self.metrics_history.items():
            if history:
                metrics[agent_id] = history[-1]
        return metrics
    
    def generate_comparative_report(self) -> Dict[str, Any]:
        """Generate a comparative report across all agents."""
        all_metrics = self.get_all_metrics()
        
        if not all_metrics:
            return {'status': 'no_data', 'message': 'No reward signal data available'}
        
        # Aggregate statistics
        quality_scores = [m.quality_score for m in all_metrics.values()]
        snr_values = [m.signal_to_noise_ratio for m in all_metrics.values()]
        consistency_values = [m.reward_consistency for m in all_metrics.values()]
        
        # Identify best and worst performers
        best_agent = max(all_metrics.items(), key=lambda x: x[1].quality_score)
        worst_agent = min(all_metrics.items(), key=lambda x: x[1].quality_score)
        
        # Common issues
        all_issues = [issue for metrics in all_metrics.values() for issue in metrics.quality_issues]
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue.value] = issue_counts.get(issue.value, 0) + 1
        
        return {
            'timestamp': time.time(),
            'total_agents_analyzed': len(all_metrics),
            'overall_statistics': {
                'avg_quality_score': np.mean(quality_scores),
                'avg_snr': np.mean(snr_values),
                'avg_consistency': np.mean(consistency_values),
                'quality_range': [min(quality_scores), max(quality_scores)]
            },
            'best_performer': {
                'agent_id': best_agent[0],
                'quality_score': best_agent[1].quality_score,
                'snr': best_agent[1].signal_to_noise_ratio
            },
            'worst_performer': {
                'agent_id': worst_agent[0],
                'quality_score': worst_agent[1].quality_score,
                'issues': [issue.value for issue in worst_agent[1].quality_issues]
            },
            'common_issues': issue_counts,
            'system_recommendations': self._generate_system_recommendations(all_metrics)
        }
    
    def _generate_system_recommendations(self, all_metrics: Dict[str, RewardSignalMetrics]) -> List[str]:
        """Generate system-level recommendations."""
        recommendations = []
        
        avg_quality = np.mean([m.quality_score for m in all_metrics.values()])
        avg_sparsity = np.mean([m.reward_sparsity for m in all_metrics.values()])
        avg_snr = np.mean([m.signal_to_noise_ratio for m in all_metrics.values()])
        
        if avg_quality < 0.4:
            recommendations.append("Overall reward signal quality is poor - consider fundamental reward redesign")
        
        if avg_sparsity > 0.8:
            recommendations.append("Rewards are too sparse across all agents - add more frequent intermediate rewards")
        
        if avg_snr < 0.5:
            recommendations.append("Signal-to-noise ratio is low - reduce reward variance or increase signal strength")
        
        # Check for consistency across agents
        quality_scores = [m.quality_score for m in all_metrics.values()]
        if np.std(quality_scores) > 0.3:
            recommendations.append("Inconsistent reward quality across agents - check for agent-specific reward bugs")
        
        return recommendations 