import numpy as np
import random
import time
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Any, Optional
import copy
import logging

logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """Track and evaluate agent performance for elite identification."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.timestamps = defaultdict(lambda: deque(maxlen=window_size))
    
    def update(self, agent_id: str, metrics: Dict[str, float]):
        """Update metrics for an agent."""
        current_time = time.time()
        
        for metric_name, value in metrics.items():
            self.metrics[f"{agent_id}_{metric_name}"].append(value)
            self.timestamps[f"{agent_id}_{metric_name}"].append(current_time)
    
    def get_performance_score(self, agent_id: str, weights: Dict[str, float] = None) -> float:
        """Calculate weighted performance score for an agent."""
        if weights is None:
            weights = {
                'total_reward': 0.4,
                'survival_time': 0.2,
                'food_consumption': 0.2,
                'distance_traveled': 0.15,
                'learning_efficiency': 0.05
            }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            metric_key = f"{agent_id}_{metric}"
            if metric_key in self.metrics and self.metrics[metric_key]:
                # Use recent average performance
                recent_values = list(self.metrics[metric_key])[-50:]  # Last 50 updates
                avg_value = np.mean(recent_values)
                
                # Normalize based on metric type
                if metric == 'total_reward':
                    normalized = max(0, min(1, (avg_value + 5) / 15))  # Assume reward range -5 to 10
                elif metric == 'survival_time':
                    normalized = max(0, min(1, avg_value / 1000))  # Assume max 1000 steps
                elif metric == 'food_consumption':
                    normalized = max(0, min(1, avg_value / 5))  # Assume max 5 food units
                elif metric == 'distance_traveled':
                    normalized = max(0, min(1, avg_value / 100))  # Assume max 100 units
                elif metric == 'learning_efficiency':
                    normalized = max(0, min(1, avg_value))  # Already 0-1
                else:
                    normalized = max(0, min(1, avg_value))
                
                score += weight * normalized
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def get_elite_agents(self, agent_ids: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Identify elite agents based on performance."""
        agent_scores = []
        
        for agent_id in agent_ids:
            score = self.get_performance_score(agent_id)
            agent_scores.append((agent_id, score))
        
        # Sort by score (descending) and return top k
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        return agent_scores[:top_k]

class BehavioralPattern:
    """Represents a behavioral pattern extracted from an elite agent."""
    
    def __init__(self, state_action_pairs: List[Tuple[Any, int]], 
                 context: Dict[str, Any], effectiveness: float):
        self.state_action_pairs = state_action_pairs
        self.context = context  # Situation where this pattern was effective
        self.effectiveness = effectiveness
        self.usage_count = 0
        self.success_rate = 0.0
    
    def matches_context(self, current_context: Dict[str, Any], threshold: float = 0.7) -> bool:
        """Check if current context matches this pattern's context."""
        if not self.context or not current_context:
            return False
        
        matches = 0
        total = 0
        
        for key, value in self.context.items():
            if key in current_context:
                total += 1
                if isinstance(value, (int, float)) and isinstance(current_context[key], (int, float)):
                    # Numerical similarity
                    if abs(value - current_context[key]) / (abs(value) + 1e-6) < 0.3:  # 30% tolerance
                        matches += 1
                elif value == current_context[key]:
                    # Exact match
                    matches += 1
        
        return (matches / total) >= threshold if total > 0 else False
    
    def get_action_for_state(self, state) -> Optional[int]:
        """Get the action that elite agent took in a similar state."""
        # Simple state matching - could be enhanced with more sophisticated similarity metrics
        for stored_state, action in self.state_action_pairs:
            if self._states_similar(state, stored_state):
                return action
        return None
    
    def _states_similar(self, state1, state2, threshold: float = 0.8) -> bool:
        """Check if two states are similar enough."""
        try:
            if isinstance(state1, (list, tuple, np.ndarray)) and isinstance(state2, (list, tuple, np.ndarray)):
                state1_arr = np.array(state1)
                state2_arr = np.array(state2)
                
                if state1_arr.shape != state2_arr.shape:
                    return False
                
                # Cosine similarity
                dot_product = np.dot(state1_arr, state2_arr)
                norm1 = np.linalg.norm(state1_arr)
                norm2 = np.linalg.norm(state2_arr)
                
                if norm1 == 0 or norm2 == 0:
                    return np.array_equal(state1_arr, state2_arr)
                
                similarity = dot_product / (norm1 * norm2)
                return similarity >= threshold
            else:
                return state1 == state2
        except:
            return False

class EliteImitationLearning:
    """Main system for elite agent imitation learning."""
    
    def __init__(self, imitation_probability: float = 0.3, 
                 elite_update_interval: int = 500,
                 pattern_extraction_window: int = 100):
        self.imitation_probability = imitation_probability
        self.elite_update_interval = elite_update_interval
        self.pattern_extraction_window = pattern_extraction_window
        
        # Core components
        self.performance_metrics = PerformanceMetrics()
        self.behavioral_patterns = []
        
        # Elite tracking
        self.current_elites = []
        self.elite_trajectories = defaultdict(list)  # Store recent state-action pairs
        self.last_elite_update = 0
        
        # Learning statistics
        self.imitation_attempts = 0
        self.successful_imitations = 0
        self.knowledge_transfers = 0
        
        print("👥 Elite Imitation Learning System initialized")
        print(f"   🎯 Imitation probability: {imitation_probability}")
        print(f"   🔄 Elite update interval: {elite_update_interval} steps")
        print(f"   📊 Pattern extraction window: {pattern_extraction_window} steps")
    
    def update_agent_performance(self, agent_id: str, performance_data: Dict[str, float]):
        """Update performance metrics for an agent."""
        self.performance_metrics.update(agent_id, performance_data)
    
    def record_agent_trajectory(self, agent_id: str, state: Any, action: int, 
                              reward: float, context: Dict[str, Any] = None):
        """Record state-action pairs for potential pattern extraction."""
        trajectory_data = {
            'state': state,
            'action': action,
            'reward': reward,
            'context': context or {},
            'timestamp': time.time()
        }
        
        self.elite_trajectories[agent_id].append(trajectory_data)
        
        # Keep only recent trajectory data
        if len(self.elite_trajectories[agent_id]) > self.pattern_extraction_window:
            self.elite_trajectories[agent_id].pop(0)
    
    def update_elites(self, agent_ids: List[str]):
        """Update the list of elite agents based on current performance."""
        self.last_elite_update = time.time()
        
        # Get top performing agents
        new_elites = self.performance_metrics.get_elite_agents(agent_ids, top_k=5)
        
        # Extract patterns from newly identified elites
        for agent_id, score in new_elites:
            if agent_id not in [elite[0] for elite in self.current_elites]:
                self._extract_patterns_from_agent(agent_id, score)
        
        self.current_elites = new_elites
        
        if self.current_elites:
            elite_names = [f"{agent_id[:8]}({score:.3f})" for agent_id, score in self.current_elites]
            print(f"👑 Elite agents updated: {', '.join(elite_names)}")
    
    def _extract_patterns_from_agent(self, agent_id: str, performance_score: float):
        """Extract behavioral patterns from an elite agent's trajectory."""
        if agent_id not in self.elite_trajectories:
            return
        
        trajectory = self.elite_trajectories[agent_id]
        if len(trajectory) < 10:  # Need minimum trajectory length
            return
        
        # Extract successful subsequences (those leading to positive rewards)
        patterns_extracted = 0
        
        for i in range(len(trajectory) - 5):  # Look at sequences of 5 steps
            sequence = trajectory[i:i+5]
            
            # Check if this sequence led to good outcomes
            total_reward = sum(step['reward'] for step in sequence)
            if total_reward > 0.1:  # Positive outcome threshold
                
                # Extract state-action pairs
                state_action_pairs = [(step['state'], step['action']) for step in sequence]
                
                # Extract context (average of contexts in sequence)
                context = self._merge_contexts([step['context'] for step in sequence])
                
                # Create pattern
                effectiveness = min(1.0, total_reward * performance_score)
                pattern = BehavioralPattern(state_action_pairs, context, effectiveness)
                
                self.behavioral_patterns.append(pattern)
                patterns_extracted += 1
        
        # Keep only best patterns
        if len(self.behavioral_patterns) > 1000:
            self.behavioral_patterns.sort(key=lambda p: p.effectiveness, reverse=True)
            self.behavioral_patterns = self.behavioral_patterns[:800]
        
        if patterns_extracted > 0:
            print(f"   📚 Extracted {patterns_extracted} patterns from elite agent {agent_id[:8]}")
            self.knowledge_transfers += patterns_extracted
    
    def _merge_contexts(self, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple contexts into a representative context."""
        if not contexts:
            return {}
        
        merged = {}
        for key in contexts[0].keys():
            values = [ctx.get(key) for ctx in contexts if key in ctx]
            if values:
                if isinstance(values[0], (int, float)):
                    merged[key] = np.mean(values)
                else:
                    # For non-numeric values, take the most common
                    merged[key] = max(set(values), key=values.count)
        
        return merged
    
    def get_imitation_action(self, agent_id: str, current_state: Any, 
                           current_context: Dict[str, Any] = None) -> Optional[int]:
        """Get an action suggestion based on elite agent patterns."""
        if random.random() > self.imitation_probability:
            return None  # No imitation this time
        
        self.imitation_attempts += 1
        
        # Find matching patterns
        matching_patterns = []
        for pattern in self.behavioral_patterns:
            if pattern.matches_context(current_context or {}):
                matching_patterns.append(pattern)
        
        if not matching_patterns:
            return None
        
        # Sort by effectiveness and try best patterns first
        matching_patterns.sort(key=lambda p: p.effectiveness, reverse=True)
        
        for pattern in matching_patterns[:3]:  # Try top 3 patterns
            action = pattern.get_action_for_state(current_state)
            if action is not None:
                pattern.usage_count += 1
                print(f"   🎭 Agent {agent_id[:8]} imitating elite pattern (effectiveness: {pattern.effectiveness:.3f})")
                return action
        
        return None
    
    def should_update_elites(self, current_step: int) -> bool:
        """Check if it's time to update elite agents."""
        return current_step % self.elite_update_interval == 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the imitation learning system."""
        imitation_success_rate = (self.successful_imitations / self.imitation_attempts 
                                if self.imitation_attempts > 0 else 0.0)
        
        return {
            'current_elites': len(self.current_elites),
            'elite_agents': [(agent_id[:8], score) for agent_id, score in self.current_elites],
            'behavioral_patterns': len(self.behavioral_patterns),
            'imitation_attempts': self.imitation_attempts,
            'successful_imitations': self.successful_imitations,
            'imitation_success_rate': imitation_success_rate,
            'knowledge_transfers': self.knowledge_transfers,
            'total_trajectories': sum(len(traj) for traj in self.elite_trajectories.values())
        } 