"""
Individual robot evaluation and behavior analysis.
Tracks learning performance, behavioral patterns, and physical parameter effectiveness.
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass, field
import json


@dataclass
class IndividualMetrics:
    """Data class to store individual robot metrics."""
    robot_id: str
    timestamp: float
    
    # Q-Learning Effectiveness
    q_learning_convergence: float = 0.0
    q_value_distribution: List[float] = field(default_factory=list)
    td_error_history: List[float] = field(default_factory=list)
    state_coverage_progress: float = 0.0
    action_diversity_score: float = 0.0
    
    # Exploration vs Exploitation
    exploration_efficiency: float = 0.0  # Reward gained per exploration
    epsilon_adaptation_rate: float = 0.0
    novel_state_discovery_rate: float = 0.0
    action_sequence_patterns: List[str] = field(default_factory=list)
    
    # Performance Progression
    cumulative_reward_curve: List[float] = field(default_factory=list)
    episode_length_trend: List[int] = field(default_factory=list)
    success_rate_over_time: List[float] = field(default_factory=list)
    stability_score_progression: List[float] = field(default_factory=list)
    
    # Physical Parameter Effectiveness
    parameter_fitness_correlation: Dict[str, float] = field(default_factory=dict)
    motor_efficiency_score: float = 0.0
    energy_consumption_rate: float = 0.0


@dataclass
class BehaviorMetrics:
    """Data class to store behavioral analysis metrics."""
    robot_id: str
    timestamp: float
    
    # Movement Patterns
    gait_analysis: Dict[str, Any] = field(default_factory=dict)
    movement_efficiency: float = 0.0
    trajectory_optimization: float = 0.0
    speed_consistency: float = 0.0
    
    # Learning Patterns
    learning_velocity: float = 0.0  # How fast the robot learns
    knowledge_retention: float = 0.0  # How well it retains learned behaviors
    adaptation_speed: float = 0.0  # How quickly it adapts to changes
    generalization_ability: float = 0.0  # How well it applies learning to new situations


class IndividualRobotEvaluator:
    """
    Evaluates individual robot performance across multiple dimensions.
    Tracks learning progress, behavioral patterns, and parameter effectiveness.
    """
    
    def __init__(self, history_length: int = 1000):
        """
        Initialize the individual robot evaluator.
        
        Args:
            history_length: Number of historical data points to maintain
        """
        self.history_length = history_length
        self.robot_metrics: Dict[str, IndividualMetrics] = {}
        self.metrics_history: Dict[str, List[IndividualMetrics]] = defaultdict(list)
        
        # Performance tracking windows
        self.reward_window = 100
        self.convergence_window = 200
        self.stability_window = 50
        
    def evaluate_robot(self, agent, step_count: int) -> IndividualMetrics:
        """
        Evaluate a single robot and return comprehensive metrics.
        
        Args:
            agent: The robot agent to evaluate
            step_count: Current training step
            
        Returns:
            IndividualMetrics object with current evaluation
        """
        try:
            robot_id = str(agent.id)
            timestamp = time.time()
            
            # Create new metrics object
            metrics = IndividualMetrics(robot_id=robot_id, timestamp=timestamp)
            
            # Q-Learning Effectiveness
            metrics.q_learning_convergence = self._calculate_convergence(agent)
            metrics.q_value_distribution = self._get_q_value_distribution(agent)
            metrics.td_error_history = self._get_td_error_history(agent)
            metrics.state_coverage_progress = self._calculate_state_coverage(agent)
            metrics.action_diversity_score = self._calculate_action_diversity(agent)
            
            # Exploration vs Exploitation
            metrics.exploration_efficiency = self._calculate_exploration_efficiency(agent)
            metrics.epsilon_adaptation_rate = self._calculate_epsilon_adaptation(agent)
            metrics.novel_state_discovery_rate = self._calculate_novel_state_rate(agent)
            metrics.action_sequence_patterns = self._analyze_action_sequences(agent)
            
            # Performance Progression
            metrics.cumulative_reward_curve = self._get_reward_progression(agent)
            metrics.episode_length_trend = self._get_episode_length_trend(agent)
            metrics.success_rate_over_time = self._calculate_success_rate_trend(agent)
            metrics.stability_score_progression = self._get_stability_progression(agent)
            
            # Physical Parameter Effectiveness
            metrics.parameter_fitness_correlation = self._calculate_parameter_correlations(agent)
            metrics.motor_efficiency_score = self._calculate_motor_efficiency(agent)
            metrics.energy_consumption_rate = self._calculate_energy_consumption(agent)
            
            # Store metrics
            self.robot_metrics[robot_id] = metrics
            self.metrics_history[robot_id].append(metrics)
            
            # Trim history if too long
            if len(self.metrics_history[robot_id]) > self.history_length:
                self.metrics_history[robot_id] = self.metrics_history[robot_id][-self.history_length:]
            
            return metrics
            
        except Exception as e:
            print(f"⚠️  Error evaluating robot {getattr(agent, 'id', 'unknown')}: {e}")
            # Return empty metrics on error
            return IndividualMetrics(robot_id=str(getattr(agent, 'id', 'unknown')), timestamp=time.time())
    
    def _calculate_convergence(self, agent) -> float:
        """Calculate Q-learning convergence score."""
        try:
            if hasattr(agent, 'q_table') and hasattr(agent.q_table, 'get_convergence_estimate'):
                return float(agent.q_table.get_convergence_estimate())
            return 0.0
        except:
            return 0.0
    
    def _get_q_value_distribution(self, agent) -> List[float]:
        """Get distribution of Q-values."""
        try:
            if hasattr(agent, 'q_table') and hasattr(agent.q_table, 'q_values'):
                q_values = []
                for state_actions in agent.q_table.q_values.values():
                    q_values.extend(state_actions.values())
                return q_values[:100]  # Limit to first 100 values
            return []
        except:
            return []
    
    def _get_td_error_history(self, agent) -> List[float]:
        """Get temporal difference error history."""
        try:
            if hasattr(agent, 'q_table') and hasattr(agent.q_table, 'q_value_history'):
                return [abs(entry.get('td_error', 0)) for entry in agent.q_table.q_value_history[-50:]]
            return []
        except:
            return []
    
    def _calculate_state_coverage(self, agent) -> float:
        """Calculate state space coverage."""
        try:
            if hasattr(agent, 'q_table') and hasattr(agent.q_table, 'state_coverage'):
                return float(len(agent.q_table.state_coverage))
            return 0.0
        except:
            return 0.0
    
    def _calculate_action_diversity(self, agent) -> float:
        """Calculate action diversity score."""
        try:
            if hasattr(agent, 'action_history') and agent.action_history:
                unique_actions = len(set(map(tuple, agent.action_history)))
                total_actions = len(agent.action_history)
                return unique_actions / max(total_actions, 1)
            return 0.0
        except:
            return 0.0
    
    def _calculate_exploration_efficiency(self, agent) -> float:
        """Calculate exploration efficiency (reward per exploration)."""
        try:
            if hasattr(agent, 'total_reward') and hasattr(agent, 'epsilon'):
                exploration_actions = getattr(agent, 'steps', 1) * agent.epsilon
                if exploration_actions > 0:
                    return agent.total_reward / exploration_actions
            return 0.0
        except:
            return 0.0
    
    def _calculate_epsilon_adaptation(self, agent) -> float:
        """Calculate epsilon adaptation rate."""
        try:
            robot_id = str(agent.id)
            if robot_id in self.robot_metrics:
                previous_epsilon = getattr(self.robot_metrics[robot_id], 'epsilon_adaptation_rate', 0.0)
                current_epsilon = getattr(agent, 'epsilon', 0.0)
                return abs(current_epsilon - previous_epsilon)
            return 0.0
        except:
            return 0.0
    
    def _calculate_novel_state_rate(self, agent) -> float:
        """Calculate rate of novel state discovery."""
        try:
            robot_id = str(agent.id)
            if robot_id in self.robot_metrics and hasattr(agent, 'q_table'):
                previous_coverage = self.robot_metrics[robot_id].state_coverage_progress
                current_coverage = len(getattr(agent.q_table, 'state_coverage', set()))
                steps_since_last = getattr(agent, 'steps', 1) - getattr(self.robot_metrics[robot_id], 'last_step', 0)
                if steps_since_last > 0:
                    return (current_coverage - previous_coverage) / steps_since_last
            return 0.0
        except:
            return 0.0
    
    def _analyze_action_sequences(self, agent) -> List[str]:
        """Analyze action sequence patterns."""
        try:
            if hasattr(agent, 'action_history') and agent.action_history:
                # Convert recent actions to string patterns
                recent_actions = agent.action_history[-10:]
                patterns = []
                for i in range(len(recent_actions) - 2):
                    pattern = f"{recent_actions[i]}->{recent_actions[i+1]}->{recent_actions[i+2]}"
                    patterns.append(pattern)
                return patterns
            return []
        except:
            return []
    
    def _get_reward_progression(self, agent) -> List[float]:
        """Get cumulative reward progression."""
        try:
            if hasattr(agent, 'recent_rewards') and agent.recent_rewards:
                return list(agent.recent_rewards)[-50:]  # Last 50 rewards
            elif hasattr(agent, 'total_reward'):
                return [float(agent.total_reward)]
            return []
        except:
            return []
    
    def _get_episode_length_trend(self, agent) -> List[int]:
        """Get episode length trend."""
        try:
            # This would need to be tracked separately in the agent
            # For now, return current steps
            if hasattr(agent, 'steps'):
                return [int(agent.steps)]
            return []
        except:
            return []
    
    def _calculate_success_rate_trend(self, agent) -> List[float]:
        """Calculate success rate over time."""
        try:
            if hasattr(agent, 'recent_rewards') and agent.recent_rewards:
                rewards = list(agent.recent_rewards)[-20:]  # Last 20 rewards
                success_threshold = 0.01  # Define success threshold
                success_rate = sum(1 for r in rewards if r > success_threshold) / len(rewards)
                return [success_rate]
            return [0.0]
        except:
            return [0.0]
    
    def _get_stability_progression(self, agent) -> List[float]:
        """Get stability score progression."""
        try:
            if hasattr(agent, 'body') and agent.body:
                stability = max(0, 1.0 - abs(agent.body.angle))
                return [stability]
            return [0.0]
        except:
            return [0.0]
    
    def _calculate_parameter_correlations(self, agent) -> Dict[str, float]:
        """Calculate correlations between physical parameters and fitness."""
        try:
            correlations = {}
            if hasattr(agent, 'physical_params') and hasattr(agent, 'total_reward'):
                params = agent.physical_params
                fitness = agent.total_reward
                
                # Simple correlation approximation
                correlations['motor_torque'] = fitness * 0.001  # Placeholder
                correlations['learning_rate'] = fitness * 0.01
                correlations['epsilon'] = fitness * 0.01
                
            return correlations
        except:
            return {}
    
    def _calculate_motor_efficiency(self, agent) -> float:
        """Calculate motor efficiency score."""
        try:
            if hasattr(agent, 'physical_params') and hasattr(agent, 'total_reward'):
                motor_power = getattr(agent.physical_params, 'motor_torque', 100)
                performance = agent.total_reward
                if motor_power > 0:
                    return performance / motor_power
            return 0.0
        except:
            return 0.0
    
    def _calculate_energy_consumption(self, agent) -> float:
        """Calculate energy consumption rate."""
        try:
            if hasattr(agent, 'physical_params') and hasattr(agent, 'steps'):
                motor_power = getattr(agent.physical_params, 'motor_torque', 100)
                steps = agent.steps
                if steps > 0:
                    return motor_power * steps / 1000.0  # Normalized energy consumption
            return 0.0
        except:
            return 0.0
    
    def get_robot_summary(self, robot_id: str) -> Dict[str, Any]:
        """Get a summary of robot performance."""
        if robot_id not in self.robot_metrics:
            return {}
        
        metrics = self.robot_metrics[robot_id]
        return {
            'robot_id': robot_id,
            'convergence_score': metrics.q_learning_convergence,
            'exploration_efficiency': metrics.exploration_efficiency,
            'action_diversity': metrics.action_diversity_score,
            'motor_efficiency': metrics.motor_efficiency_score,
            'stability_trend': metrics.stability_score_progression[-1] if metrics.stability_score_progression else 0.0,
            'recent_performance': metrics.cumulative_reward_curve[-1] if metrics.cumulative_reward_curve else 0.0
        }
    
    def get_all_robots_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary for all evaluated robots."""
        return {robot_id: self.get_robot_summary(robot_id) for robot_id in self.robot_metrics.keys()}


class BehaviorAnalyzer:
    """
    Analyzes behavioral patterns and learning dynamics of individual robots.
    """
    
    def __init__(self, analysis_window: int = 200):
        """
        Initialize the behavior analyzer.
        
        Args:
            analysis_window: Number of steps to analyze for behavioral patterns
        """
        self.analysis_window = analysis_window
        self.behavior_metrics: Dict[str, BehaviorMetrics] = {}
        self.behavior_history: Dict[str, List[BehaviorMetrics]] = defaultdict(list)
        
    def analyze_behavior(self, agent, step_count: int) -> BehaviorMetrics:
        """
        Analyze robot behavior and return behavioral metrics.
        
        Args:
            agent: The robot agent to analyze
            step_count: Current training step
            
        Returns:
            BehaviorMetrics object with current analysis
        """
        try:
            robot_id = str(agent.id)
            timestamp = time.time()
            
            # Create new behavior metrics object
            metrics = BehaviorMetrics(robot_id=robot_id, timestamp=timestamp)
            
            # Movement Patterns
            metrics.gait_analysis = self._analyze_gait(agent)
            metrics.movement_efficiency = self._calculate_movement_efficiency(agent)
            metrics.trajectory_optimization = self._calculate_trajectory_optimization(agent)
            metrics.speed_consistency = self._calculate_speed_consistency(agent)
            
            # Learning Patterns
            metrics.learning_velocity = self._calculate_learning_velocity(agent)
            metrics.knowledge_retention = self._calculate_knowledge_retention(agent)
            metrics.adaptation_speed = self._calculate_adaptation_speed(agent)
            metrics.generalization_ability = self._calculate_generalization_ability(agent)
            
            # Store metrics
            self.behavior_metrics[robot_id] = metrics
            self.behavior_history[robot_id].append(metrics)
            
            # Trim history
            if len(self.behavior_history[robot_id]) > 100:
                self.behavior_history[robot_id] = self.behavior_history[robot_id][-100:]
            
            return metrics
            
        except Exception as e:
            print(f"⚠️  Error analyzing behavior for robot {getattr(agent, 'id', 'unknown')}: {e}")
            return BehaviorMetrics(robot_id=str(getattr(agent, 'id', 'unknown')), timestamp=time.time())
    
    def _analyze_gait(self, agent) -> Dict[str, Any]:
        """Analyze gait patterns."""
        try:
            gait_data = {}
            if hasattr(agent, 'action_history') and agent.action_history:
                actions = agent.action_history[-20:]  # Analyze last 20 actions
                gait_data['action_frequency'] = len(set(map(tuple, actions))) / len(actions) if actions else 0
                gait_data['action_consistency'] = self._calculate_action_consistency(actions)
            return gait_data
        except:
            return {}
    
    def _calculate_movement_efficiency(self, agent) -> float:
        """Calculate movement efficiency."""
        try:
            if hasattr(agent, 'body') and hasattr(agent, 'total_reward'):
                distance = getattr(agent, 'body').position.x - getattr(agent, 'initial_position', [0, 0])[0]
                energy_used = getattr(agent, 'steps', 1) * 0.1  # Approximation
                if energy_used > 0:
                    return distance / energy_used
            return 0.0
        except:
            return 0.0
    
    def _calculate_trajectory_optimization(self, agent) -> float:
        """Calculate trajectory optimization score."""
        try:
            # Simplified: straighter path = higher score
            if hasattr(agent, 'body'):
                forward_progress = agent.body.position.x - getattr(agent, 'initial_position', [0, 0])[0]
                lateral_deviation = abs(agent.body.position.y - getattr(agent, 'initial_position', [0, 0])[1])
                if forward_progress > 0:
                    return forward_progress / (forward_progress + lateral_deviation)
            return 0.0
        except:
            return 0.0
    
    def _calculate_speed_consistency(self, agent) -> float:
        """Calculate speed consistency."""
        try:
            if hasattr(agent, 'recent_rewards') and agent.recent_rewards:
                rewards = list(agent.recent_rewards)[-10:]
                if len(rewards) > 2:
                    return 1.0 - (np.std(rewards) / (np.mean(rewards) + 1e-6))
            return 0.0
        except:
            return 0.0
    
    def _calculate_learning_velocity(self, agent) -> float:
        """Calculate how fast the robot learns."""
        try:
            robot_id = str(agent.id)
            if robot_id in self.behavior_history and len(self.behavior_history[robot_id]) >= 2:
                recent_performance = getattr(agent, 'total_reward', 0)
                past_performance = self.behavior_history[robot_id][-2].learning_velocity
                return recent_performance - past_performance
            return 0.0
        except:
            return 0.0
    
    def _calculate_knowledge_retention(self, agent) -> float:
        """Calculate knowledge retention."""
        try:
            if hasattr(agent, 'q_table') and hasattr(agent.q_table, 'q_value_history'):
                history = agent.q_table.q_value_history
                if len(history) > 10:
                    # Check if Q-values are stable over time
                    recent_changes = [abs(entry.get('td_error', 0)) for entry in history[-10:]]
                    return 1.0 - np.mean(recent_changes)
            return 0.0
        except:
            return 0.0
    
    def _calculate_adaptation_speed(self, agent) -> float:
        """Calculate adaptation speed."""
        try:
            if hasattr(agent, 'time_since_good_value'):
                # Lower time since good value = faster adaptation
                return 1.0 / (1.0 + agent.time_since_good_value * 0.01)
            return 0.0
        except:
            return 0.0
    
    def _calculate_generalization_ability(self, agent) -> float:
        """Calculate generalization ability."""
        try:
            if hasattr(agent, 'q_table') and hasattr(agent.q_table, 'state_coverage'):
                coverage = len(agent.q_table.state_coverage)
                performance = getattr(agent, 'total_reward', 0)
                if coverage > 0:
                    return performance / coverage  # Performance per state explored
            return 0.0
        except:
            return 0.0
    
    def _calculate_action_consistency(self, actions) -> float:
        """Calculate consistency of actions."""
        try:
            if not actions:
                return 0.0
            
            # Calculate how consistent the action patterns are
            action_strings = [str(action) for action in actions]
            unique_actions = set(action_strings)
            return len(unique_actions) / len(action_strings)
        except:
            return 0.0
    
    def get_behavior_summary(self, robot_id: str) -> Dict[str, Any]:
        """Get behavior summary for a robot."""
        if robot_id not in self.behavior_metrics:
            return {}
        
        metrics = self.behavior_metrics[robot_id]
        return {
            'robot_id': robot_id,
            'movement_efficiency': metrics.movement_efficiency,
            'learning_velocity': metrics.learning_velocity,
            'adaptation_speed': metrics.adaptation_speed,
            'trajectory_optimization': metrics.trajectory_optimization,
            'speed_consistency': metrics.speed_consistency,
            'knowledge_retention': metrics.knowledge_retention,
            'generalization_ability': metrics.generalization_ability
        } 