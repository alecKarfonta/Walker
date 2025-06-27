"""
Reward Signal Integration Adapter
Connects the reward signal evaluator with the existing agent system.
"""

import time
from typing import Dict, Any, Optional, List

# Handle both relative and absolute imports
try:
    from .reward_signal_evaluator import RewardSignalEvaluator, RewardSignalMetrics
except ImportError:
    from reward_signal_evaluator import RewardSignalEvaluator, RewardSignalMetrics

class RewardSignalIntegrationAdapter:
    """
    Integration adapter for connecting reward signal evaluation with existing agents.
    Non-invasive integration that hooks into reward calculations.
    """
    
    def __init__(self):
        self.evaluator = RewardSignalEvaluator()
        self.registered_agents: Dict[str, Dict[str, Any]] = {}
        self.active = True
        
        print("🔗 Reward Signal Integration Adapter initialized")
    
    def register_agent(self, agent_id: str, agent_type: str = "unknown", 
                      metadata: Optional[Dict[str, Any]] = None):
        """Register an agent for reward signal evaluation."""
        if not self.active:
            return
        
        self.registered_agents[agent_id] = {
            'type': agent_type,
            'metadata': metadata or {},
            'registered_at': time.time(),
            'reward_count': 0
        }
        
        print(f"📝 Registered agent {agent_id} ({agent_type}) for reward signal evaluation")
    
    def record_reward_signal(self, agent_id: str, state: Any, action: int, 
                           reward: float, timestamp: Optional[float] = None):
        """Record a reward signal from an agent."""
        if not self.active:
            return
        
        # Auto-register if not already registered
        if agent_id not in self.registered_agents:
            self.register_agent(agent_id, "auto_detected")
        
        # Debug logging for first few calls
        if self.registered_agents[agent_id]['reward_count'] < 5:
            print(f"🔧 DEBUG: Recording reward {self.registered_agents[agent_id]['reward_count']+1} for agent {agent_id}: {reward}")
        
        # Record in evaluator
        try:
            self.evaluator.record_reward(agent_id, state, action, reward, timestamp)
            # Debug: confirm evaluator call succeeded
            if self.registered_agents[agent_id]['reward_count'] < 5:
                print(f"  ✅ Successfully called evaluator.record_reward")
        except Exception as e:
            print(f"❌ Failed to record in evaluator for agent {agent_id}: {e}")
        
        # Update tracking
        self.registered_agents[agent_id]['reward_count'] += 1
        self.registered_agents[agent_id]['last_reward_time'] = timestamp or time.time()
        self.registered_agents[agent_id]['last_reward'] = reward
    
    def get_agent_reward_metrics(self, agent_id: str) -> Optional[RewardSignalMetrics]:
        """Get reward signal metrics for a specific agent."""
        return self.evaluator.get_agent_metrics(agent_id)
    
    def get_all_reward_metrics(self) -> Dict[str, RewardSignalMetrics]:
        """Get reward signal metrics for all agents."""
        return self.evaluator.get_all_metrics()
    
    def get_reward_comparative_report(self) -> Dict[str, Any]:
        """Get comparative report across all agents."""
        return self.evaluator.generate_comparative_report()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status for reward signal evaluation."""
        all_metrics = self.evaluator.get_all_metrics()
        
        return {
            'active': self.active,
            'total_agents': len(self.registered_agents),
            'agents_with_metrics': len(all_metrics),
            'total_rewards_recorded': sum(
                agent_info['reward_count'] for agent_info in self.registered_agents.values()
            ),
            'registered_agents': {
                agent_id: {
                    'type': info['type'],
                    'reward_count': info['reward_count'],
                    'last_reward': info.get('last_reward', 0),
                    'last_reward_time': info.get('last_reward_time', 0)
                }
                for agent_id, info in self.registered_agents.items()
            },
            'timestamp': time.time()
        }
    
    def get_agent_diagnostics(self, agent_id: str) -> Dict[str, Any]:
        """Get detailed diagnostics for a specific agent's reward signals."""
        if agent_id not in self.registered_agents:
            return {'error': f'Agent {agent_id} not registered'}
        
        metrics = self.get_agent_reward_metrics(agent_id)
        agent_info = self.registered_agents[agent_id]
        
        if not metrics:
            return {
                'agent_id': agent_id,
                'status': 'insufficient_data',
                'agent_info': agent_info,
                'message': 'Not enough reward data for analysis'
            }
        
        # Detailed diagnostics
        diagnostics = {
            'agent_id': agent_id,
            'agent_type': agent_info['type'],
            'reward_signal_analysis': {
                'overall_quality': {
                    'score': metrics.quality_score,
                    'rating': self._get_quality_rating(metrics.quality_score),
                    'issues': [issue.value for issue in metrics.quality_issues],
                    'recommendations': metrics.recommendations
                },
                'signal_characteristics': {
                    'sparsity': {
                        'value': metrics.reward_sparsity,
                        'interpretation': self._interpret_sparsity(metrics.reward_sparsity)
                    },
                    'noise_level': {
                        'snr': metrics.signal_to_noise_ratio,
                        'interpretation': self._interpret_snr(metrics.signal_to_noise_ratio)
                    },
                    'consistency': {
                        'value': metrics.reward_consistency,
                        'interpretation': self._interpret_consistency(metrics.reward_consistency)
                    },
                    'exploration_support': {
                        'value': metrics.exploration_incentive,
                        'interpretation': self._interpret_exploration(metrics.exploration_incentive)
                    }
                },
                'learning_implications': {
                    'convergence_support': metrics.convergence_support,
                    'behavioral_alignment': metrics.behavioral_alignment,
                    'temporal_consistency': metrics.temporal_consistency
                },
                'distribution_analysis': {
                    'mean': metrics.reward_mean,
                    'std': metrics.reward_std,
                    'range': metrics.reward_range,
                    'skewness': metrics.reward_skewness,
                    'entropy': metrics.reward_entropy,
                    'positive_ratio': metrics.positive_reward_ratio
                }
            },
            'data_summary': {
                'total_rewards_recorded': agent_info['reward_count'],
                'steps_analyzed': metrics.steps_analyzed,
                'non_zero_rewards': metrics.total_rewards_received,
                'analysis_timestamp': metrics.timestamp
            }
        }
        
        return diagnostics
    
    def _get_quality_rating(self, score: float) -> str:
        """Convert quality score to rating."""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        elif score >= 0.2:
            return "Poor"
        else:
            return "Very Poor"
    
    def _interpret_sparsity(self, sparsity: float) -> str:
        """Interpret reward sparsity value."""
        if sparsity < 0.1:
            return "Dense rewards - good for learning"
        elif sparsity < 0.5:
            return "Moderate reward frequency"
        elif sparsity < 0.8:
            return "Sparse rewards - may slow learning"
        else:
            return "Very sparse rewards - consider adding intermediate rewards"
    
    def _interpret_snr(self, snr: float) -> str:
        """Interpret signal-to-noise ratio."""
        if snr > 2.0:
            return "Excellent signal clarity"
        elif snr > 1.0:
            return "Good signal clarity"
        elif snr > 0.5:
            return "Moderate signal clarity"
        else:
            return "Poor signal clarity - high noise"
    
    def _interpret_consistency(self, consistency: float) -> str:
        """Interpret reward consistency."""
        if consistency > 0.8:
            return "Highly consistent rewards"
        elif consistency > 0.6:
            return "Good consistency"
        elif consistency > 0.4:
            return "Moderate consistency"
        else:
            return "Inconsistent rewards - may confuse learning"
    
    def _interpret_exploration(self, exploration: float) -> str:
        """Interpret exploration incentive."""
        if exploration > 0.8:
            return "Strong exploration incentive"
        elif exploration > 0.6:
            return "Good exploration incentive"
        elif exploration > 0.4:
            return "Moderate exploration incentive"
        else:
            return "Weak exploration incentive - may lead to local optima"
    
    def disable(self):
        """Disable reward signal evaluation."""
        self.active = False
        print("⏸️ Reward Signal Integration Adapter disabled")
    
    def enable(self):
        """Enable reward signal evaluation."""
        self.active = True
        print("▶️ Reward Signal Integration Adapter enabled")


# Singleton pattern to ensure only one instance exists
_reward_signal_adapter_instance = None

def get_reward_signal_adapter():
    """Get the singleton reward signal adapter instance."""
    global _reward_signal_adapter_instance
    if _reward_signal_adapter_instance is None:
        _reward_signal_adapter_instance = RewardSignalIntegrationAdapter()
    return _reward_signal_adapter_instance

# Global instance for easy integration (uses singleton)
reward_signal_adapter = get_reward_signal_adapter()


# Helper functions for easy integration
def record_reward(agent_id: str, state: Any, action: int, reward: float, 
                 timestamp: Optional[float] = None):
    """Convenience function to record a reward signal."""
    reward_signal_adapter.record_reward_signal(agent_id, state, action, reward, timestamp)


def get_reward_metrics(agent_id: str) -> Optional[RewardSignalMetrics]:
    """Convenience function to get reward metrics for an agent."""
    return reward_signal_adapter.get_agent_reward_metrics(agent_id)


def get_all_reward_metrics() -> Dict[str, RewardSignalMetrics]:
    """Convenience function to get all reward metrics."""
    return reward_signal_adapter.get_all_reward_metrics()


def register_agent(agent_id: str, agent_type: str = "unknown", 
                  metadata: Optional[Dict[str, Any]] = None):
    """Convenience function to register an agent."""
    reward_signal_adapter.register_agent(agent_id, agent_type, metadata) 