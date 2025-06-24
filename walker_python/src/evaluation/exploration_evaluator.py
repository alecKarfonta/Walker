"""
Exploration and action space evaluation.
Analyzes how effectively robots explore their state and action spaces.
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import math


@dataclass
class ExplorationMetrics:
    """Data class to store exploration metrics."""
    robot_id: str
    timestamp: float
    
    # Coverage Metrics
    state_space_coverage: float = 0.0
    coverage_efficiency: float = 0.0  # Coverage per step
    exploration_breadth: float = 0.0  # How wide the exploration
    exploration_depth: float = 0.0   # How deep in promising areas
    
    # Discovery Metrics
    novel_states_per_episode: List[int] = field(default_factory=list)
    exploration_reward_ratio: float = 0.0
    curiosity_driven_discoveries: int = 0
    
    # Efficiency Metrics
    exploration_redundancy: float = 0.0  # Revisiting known areas
    targeted_exploration_success: float = 0.0  # Successful exploration of promising areas


@dataclass
class ActionSpaceMetrics:
    """Data class to store action space analysis metrics."""
    robot_id: str
    timestamp: float
    
    # Action Diversity
    action_entropy: float = 0.0
    action_sequence_complexity: float = 0.0
    unique_action_combinations: int = 0
    
    # Action Effectiveness
    action_reward_correlation: Dict[str, float] = field(default_factory=dict)
    action_success_rates: Dict[str, float] = field(default_factory=dict)
    context_dependent_actions: Dict[str, List[str]] = field(default_factory=dict)


class ExplorationEvaluator:
    """
    Evaluates how effectively robots explore their state space.
    Tracks coverage, efficiency, and discovery patterns.
    """
    
    def __init__(self, history_length: int = 500):
        """
        Initialize the exploration evaluator.
        
        Args:
            history_length: Number of historical data points to maintain
        """
        self.history_length = history_length
        self.exploration_metrics: Dict[str, ExplorationMetrics] = {}
        self.exploration_history: Dict[str, List[ExplorationMetrics]] = defaultdict(list)
        
        # State tracking for each robot
        self.robot_state_visits: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.robot_state_rewards: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self.robot_exploration_timeline: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        
    def evaluate_exploration(self, agent, step_count: int) -> ExplorationMetrics:
        """
        Evaluate robot's exploration patterns and return metrics.
        
        Args:
            agent: The robot agent to evaluate
            step_count: Current training step
            
        Returns:
            ExplorationMetrics object with current evaluation
        """
        try:
            robot_id = str(agent.id)
            timestamp = time.time()
            
            # Update state visit tracking
            self._update_state_tracking(agent, robot_id)
            
            # Create new metrics object
            metrics = ExplorationMetrics(robot_id=robot_id, timestamp=timestamp)
            
            # Coverage Metrics
            metrics.state_space_coverage = self._calculate_state_coverage(robot_id)
            metrics.coverage_efficiency = self._calculate_coverage_efficiency(agent, robot_id)
            metrics.exploration_breadth = self._calculate_exploration_breadth(robot_id)
            metrics.exploration_depth = self._calculate_exploration_depth(robot_id)
            
            # Discovery Metrics
            metrics.novel_states_per_episode = self._calculate_novel_states_per_episode(robot_id)
            metrics.exploration_reward_ratio = self._calculate_exploration_reward_ratio(robot_id)
            metrics.curiosity_driven_discoveries = self._calculate_curiosity_discoveries(robot_id)
            
            # Efficiency Metrics
            metrics.exploration_redundancy = self._calculate_exploration_redundancy(robot_id)
            metrics.targeted_exploration_success = self._calculate_targeted_exploration_success(robot_id)
            
            # Store metrics
            self.exploration_metrics[robot_id] = metrics
            self.exploration_history[robot_id].append(metrics)
            
            # Trim history
            if len(self.exploration_history[robot_id]) > self.history_length:
                self.exploration_history[robot_id] = self.exploration_history[robot_id][-self.history_length:]
            
            return metrics
            
        except Exception as e:
            print(f"⚠️  Error evaluating exploration for robot {getattr(agent, 'id', 'unknown')}: {e}")
            return ExplorationMetrics(robot_id=str(getattr(agent, 'id', 'unknown')), timestamp=time.time())
    
    def _update_state_tracking(self, agent, robot_id: str):
        """Update state visit tracking for the robot."""
        try:
            if hasattr(agent, 'current_state') and agent.current_state is not None:
                state_key = str(agent.current_state)
                self.robot_state_visits[robot_id][state_key] += 1
                
                # Track reward for this state
                if hasattr(agent, 'immediate_reward'):
                    self.robot_state_rewards[robot_id][state_key].append(agent.immediate_reward)
                
                # Track exploration timeline
                timestamp = time.time()
                self.robot_exploration_timeline[robot_id].append((state_key, timestamp))
                
                # Trim timeline if too long
                if len(self.robot_exploration_timeline[robot_id]) > 1000:
                    self.robot_exploration_timeline[robot_id] = self.robot_exploration_timeline[robot_id][-1000:]
                    
        except Exception as e:
            print(f"⚠️  Error updating state tracking for robot {robot_id}: {e}")
    
    def _calculate_state_coverage(self, robot_id: str) -> float:
        """Calculate total state space coverage."""
        try:
            return float(len(self.robot_state_visits[robot_id]))
        except:
            return 0.0
    
    def _calculate_coverage_efficiency(self, agent, robot_id: str) -> float:
        """Calculate coverage efficiency (coverage per step)."""
        try:
            coverage = len(self.robot_state_visits[robot_id])
            steps = getattr(agent, 'steps', 1)
            return coverage / max(steps, 1)
        except:
            return 0.0
    
    def _calculate_exploration_breadth(self, robot_id: str) -> float:
        """Calculate exploration breadth (how widely the robot explores)."""
        try:
            state_visits = self.robot_state_visits[robot_id]
            if not state_visits:
                return 0.0
            
            # Calculate entropy of state visits (higher entropy = more breadth)
            total_visits = sum(state_visits.values())
            if total_visits == 0:
                return 0.0
            
            entropy = 0.0
            for visits in state_visits.values():
                prob = visits / total_visits
                if prob > 0:
                    entropy -= prob * math.log2(prob)
            
            # Normalize by maximum possible entropy
            max_entropy = math.log2(len(state_visits)) if len(state_visits) > 1 else 1
            return entropy / max_entropy
            
        except:
            return 0.0
    
    def _calculate_exploration_depth(self, robot_id: str) -> float:
        """Calculate exploration depth (how deeply the robot explores promising areas)."""
        try:
            state_rewards = self.robot_state_rewards[robot_id]
            if not state_rewards:
                return 0.0
            
            # Find states with high average rewards
            promising_states = []
            for state, rewards in state_rewards.items():
                if rewards:
                    avg_reward = np.mean(rewards)
                    if avg_reward > 0:  # Positive reward threshold
                        promising_states.append(state)
            
            if not promising_states:
                return 0.0
            
            # Calculate how much time is spent in promising states
            state_visits = self.robot_state_visits[robot_id]
            total_visits = sum(state_visits.values())
            promising_visits = sum(state_visits.get(state, 0) for state in promising_states)
            
            return promising_visits / max(total_visits, 1)
            
        except:
            return 0.0
    
    def _calculate_novel_states_per_episode(self, robot_id: str) -> List[int]:
        """Calculate novel states discovered per episode."""
        try:
            # This would require episode tracking, for now return recent discovery rate
            timeline = self.robot_exploration_timeline[robot_id]
            if len(timeline) < 2:
                return [0]
            
            # Look at recent discoveries
            recent_timeline = timeline[-100:]  # Last 100 state visits
            unique_states = len(set(state for state, _ in recent_timeline))
            return [unique_states]
            
        except:
            return [0]
    
    def _calculate_exploration_reward_ratio(self, robot_id: str) -> float:
        """Calculate ratio of reward from exploration vs exploitation."""
        try:
            state_visits = self.robot_state_visits[robot_id]
            state_rewards = self.robot_state_rewards[robot_id]
            
            if not state_visits or not state_rewards:
                return 0.0
            
            exploration_reward = 0.0
            exploitation_reward = 0.0
            
            for state, visits in state_visits.items():
                rewards = state_rewards.get(state, [])
                if rewards:
                    avg_reward = np.mean(rewards)
                    if visits <= 3:  # Exploration (first few visits)
                        exploration_reward += avg_reward * min(visits, 3)
                    else:  # Exploitation
                        exploitation_reward += avg_reward * (visits - 3)
            
            total_reward = exploration_reward + exploitation_reward
            if total_reward > 0:
                return exploration_reward / total_reward
            return 0.0
            
        except:
            return 0.0
    
    def _calculate_curiosity_discoveries(self, robot_id: str) -> int:
        """Calculate curiosity-driven discoveries."""
        try:
            # States visited with low visit counts that led to positive rewards
            state_visits = self.robot_state_visits[robot_id]
            state_rewards = self.robot_state_rewards[robot_id]
            
            curiosity_discoveries = 0
            for state, visits in state_visits.items():
                if visits <= 2:  # Low visit count (curiosity-driven)
                    rewards = state_rewards.get(state, [])
                    if rewards and np.mean(rewards) > 0:
                        curiosity_discoveries += 1
            
            return curiosity_discoveries
            
        except:
            return 0
    
    def _calculate_exploration_redundancy(self, robot_id: str) -> float:
        """Calculate exploration redundancy (revisiting known areas)."""
        try:
            state_visits = self.robot_state_visits[robot_id]
            if not state_visits:
                return 0.0
            
            total_visits = sum(state_visits.values())
            unique_states = len(state_visits)
            
            if unique_states == 0:
                return 1.0
            
            # Perfect exploration would have each state visited once
            # Redundancy = (total_visits - unique_states) / total_visits
            redundancy = max(0, (total_visits - unique_states) / total_visits)
            return redundancy
            
        except:
            return 0.0
    
    def _calculate_targeted_exploration_success(self, robot_id: str) -> float:
        """Calculate success of targeted exploration in promising areas."""
        try:
            state_rewards = self.robot_state_rewards[robot_id]
            if not state_rewards:
                return 0.0
            
            # Find promising areas (states with positive average rewards)
            promising_areas = []
            for state, rewards in state_rewards.items():
                if rewards and np.mean(rewards) > 0:
                    promising_areas.append(state)
            
            if not promising_areas:
                return 0.0
            
            # Calculate success rate in promising areas
            total_promising_visits = 0
            successful_promising_visits = 0
            
            for state in promising_areas:
                rewards = state_rewards[state]
                total_promising_visits += len(rewards)
                successful_promising_visits += sum(1 for r in rewards if r > 0)
            
            if total_promising_visits > 0:
                return successful_promising_visits / total_promising_visits
            return 0.0
            
        except:
            return 0.0
    
    def get_exploration_summary(self, robot_id: str) -> Dict[str, Any]:
        """Get exploration summary for a robot."""
        if robot_id not in self.exploration_metrics:
            return {}
        
        metrics = self.exploration_metrics[robot_id]
        return {
            'robot_id': robot_id,
            'state_coverage': metrics.state_space_coverage,
            'coverage_efficiency': metrics.coverage_efficiency,
            'exploration_breadth': metrics.exploration_breadth,
            'exploration_depth': metrics.exploration_depth,
            'exploration_redundancy': metrics.exploration_redundancy,
            'targeted_success': metrics.targeted_exploration_success,
            'novel_discoveries': metrics.curiosity_driven_discoveries
        }


class ActionSpaceAnalyzer:
    """
    Analyzes how robots use their action space.
    Tracks action diversity, effectiveness, and patterns.
    """
    
    def __init__(self, history_length: int = 500):
        """
        Initialize the action space analyzer.
        
        Args:
            history_length: Number of historical data points to maintain
        """
        self.history_length = history_length
        self.action_metrics: Dict[str, ActionSpaceMetrics] = {}
        self.action_history: Dict[str, List[ActionSpaceMetrics]] = defaultdict(list)
        
        # Action tracking for each robot
        self.robot_action_counts: Dict[str, Counter] = defaultdict(Counter)
        self.robot_action_rewards: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self.robot_action_sequences: Dict[str, List[str]] = defaultdict(list)
        self.robot_context_actions: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
        
    def analyze_action_space(self, agent, step_count: int) -> ActionSpaceMetrics:
        """
        Analyze robot's action space usage and return metrics.
        
        Args:
            agent: The robot agent to evaluate
            step_count: Current training step
            
        Returns:
            ActionSpaceMetrics object with current analysis
        """
        try:
            robot_id = str(agent.id)
            timestamp = time.time()
            
            # Update action tracking
            self._update_action_tracking(agent, robot_id)
            
            # Create new metrics object
            metrics = ActionSpaceMetrics(robot_id=robot_id, timestamp=timestamp)
            
            # Action Diversity
            metrics.action_entropy = self._calculate_action_entropy(robot_id)
            metrics.action_sequence_complexity = self._calculate_sequence_complexity(robot_id)
            metrics.unique_action_combinations = self._calculate_unique_combinations(robot_id)
            
            # Action Effectiveness
            metrics.action_reward_correlation = self._calculate_action_reward_correlation(robot_id)
            metrics.action_success_rates = self._calculate_action_success_rates(robot_id)
            metrics.context_dependent_actions = self._analyze_context_dependent_actions(robot_id)
            
            # Store metrics
            self.action_metrics[robot_id] = metrics
            self.action_history[robot_id].append(metrics)
            
            # Trim history
            if len(self.action_history[robot_id]) > self.history_length:
                self.action_history[robot_id] = self.action_history[robot_id][-self.history_length:]
            
            return metrics
            
        except Exception as e:
            print(f"⚠️  Error analyzing action space for robot {getattr(agent, 'id', 'unknown')}: {e}")
            return ActionSpaceMetrics(robot_id=str(getattr(agent, 'id', 'unknown')), timestamp=time.time())
    
    def _update_action_tracking(self, agent, robot_id: str):
        """Update action tracking for the robot."""
        try:
            if hasattr(agent, 'current_action_tuple') and agent.current_action_tuple is not None:
                action_key = str(agent.current_action_tuple)
                self.robot_action_counts[robot_id][action_key] += 1
                
                # Track reward for this action
                if hasattr(agent, 'immediate_reward'):
                    self.robot_action_rewards[robot_id][action_key].append(agent.immediate_reward)
                
                # Track action sequences
                self.robot_action_sequences[robot_id].append(action_key)
                if len(self.robot_action_sequences[robot_id]) > 200:
                    self.robot_action_sequences[robot_id] = self.robot_action_sequences[robot_id][-200:]
                
                # Track context-dependent actions
                if hasattr(agent, 'current_state') and agent.current_state is not None:
                    context_key = str(agent.current_state)
                    self.robot_context_actions[robot_id][context_key][action_key] += 1
                    
        except Exception as e:
            print(f"⚠️  Error updating action tracking for robot {robot_id}: {e}")
    
    def _calculate_action_entropy(self, robot_id: str) -> float:
        """Calculate entropy of action distribution."""
        try:
            action_counts = self.robot_action_counts[robot_id]
            if not action_counts:
                return 0.0
            
            total_actions = sum(action_counts.values())
            if total_actions == 0:
                return 0.0
            
            entropy = 0.0
            for count in action_counts.values():
                prob = count / total_actions
                if prob > 0:
                    entropy -= prob * math.log2(prob)
            
            return entropy
            
        except:
            return 0.0
    
    def _calculate_sequence_complexity(self, robot_id: str) -> float:
        """Calculate complexity of action sequences."""
        try:
            sequences = self.robot_action_sequences[robot_id]
            if len(sequences) < 3:
                return 0.0
            
            # Count unique 3-action sequences
            trigrams = set()
            for i in range(len(sequences) - 2):
                trigram = (sequences[i], sequences[i+1], sequences[i+2])
                trigrams.add(trigram)
            
            # Complexity = unique trigrams / total possible trigrams
            total_trigrams = len(sequences) - 2
            if total_trigrams > 0:
                return len(trigrams) / total_trigrams
            return 0.0
            
        except:
            return 0.0
    
    def _calculate_unique_combinations(self, robot_id: str) -> int:
        """Calculate number of unique action combinations."""
        try:
            return len(self.robot_action_counts[robot_id])
        except:
            return 0
    
    def _calculate_action_reward_correlation(self, robot_id: str) -> Dict[str, float]:
        """Calculate correlation between actions and rewards."""
        try:
            action_rewards = self.robot_action_rewards[robot_id]
            correlations = {}
            
            for action, rewards in action_rewards.items():
                if rewards:
                    correlations[action] = float(np.mean(rewards))
            
            return correlations
            
        except:
            return {}
    
    def _calculate_action_success_rates(self, robot_id: str) -> Dict[str, float]:
        """Calculate success rates for each action."""
        try:
            action_rewards = self.robot_action_rewards[robot_id]
            success_rates = {}
            
            for action, rewards in action_rewards.items():
                if rewards:
                    successful = sum(1 for r in rewards if r > 0)
                    success_rates[action] = successful / len(rewards)
            
            return success_rates
            
        except:
            return {}
    
    def _analyze_context_dependent_actions(self, robot_id: str) -> Dict[str, List[str]]:
        """Analyze context-dependent action patterns."""
        try:
            context_actions = self.robot_context_actions[robot_id]
            context_patterns = {}
            
            for context, action_counts in context_actions.items():
                if action_counts:
                    # Get most common actions for this context
                    most_common = action_counts.most_common(3)
                    context_patterns[context] = [action for action, _ in most_common]
            
            return context_patterns
            
        except:
            return {}
    
    def get_action_summary(self, robot_id: str) -> Dict[str, Any]:
        """Get action space summary for a robot."""
        if robot_id not in self.action_metrics:
            return {}
        
        metrics = self.action_metrics[robot_id]
        return {
            'robot_id': robot_id,
            'action_entropy': metrics.action_entropy,
            'sequence_complexity': metrics.action_sequence_complexity,
            'unique_combinations': metrics.unique_action_combinations,
            'best_actions': self._get_best_actions(robot_id),
            'action_diversity_score': len(self.robot_action_counts[robot_id])
        }
    
    def _get_best_actions(self, robot_id: str) -> List[Tuple[str, float]]:
        """Get best performing actions for a robot."""
        try:
            action_rewards = self.robot_action_rewards[robot_id]
            best_actions = []
            
            for action, rewards in action_rewards.items():
                if rewards:
                    avg_reward = np.mean(rewards)
                    best_actions.append((action, avg_reward))
            
            # Sort by average reward
            best_actions.sort(key=lambda x: x[1], reverse=True)
            return best_actions[:5]  # Top 5 actions
            
        except:
            return [] 