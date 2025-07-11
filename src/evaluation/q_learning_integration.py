"""
Q-Learning Evaluation Integration
Adapters to integrate the Q-learning evaluator with existing agent systems.
"""

from typing import Any, Optional
from .q_learning_evaluator import QLearningEvaluator
import time


class QLearningIntegrationAdapter:
    """Adapter to integrate Q-learning evaluation with existing agents."""
    
    def __init__(self, evaluator: QLearningEvaluator):
        self.evaluator = evaluator
        self.original_methods = {}
        print("🔗 Q-Learning Integration Adapter initialized")
    
    def integrate_agent(self, agent) -> None:
        """Integrate an agent with Q-learning evaluation."""
        # Register the agent
        self.evaluator.register_agent(agent)
        
        # Store original methods
        agent_id = str(agent.id)
        self.original_methods[agent_id] = {}
        
        # Hook into the Q-learning update process
        if hasattr(agent, 'update_q_value'):
            self.original_methods[agent_id]['update_q_value'] = agent.update_q_value
            agent.update_q_value = self._create_q_update_wrapper(agent, agent.update_q_value)
        
        # Hook into action selection
        if hasattr(agent, 'choose_action'):
            self.original_methods[agent_id]['choose_action'] = agent.choose_action
            agent.choose_action = self._create_action_wrapper(agent, agent.choose_action)
        
        # Hook into step method
        if hasattr(agent, 'step'):
            self.original_methods[agent_id]['step'] = agent.step
            agent.step = self._create_step_wrapper(agent, agent.step)
        
        print(f"🔗 Integrated agent {agent_id} with Q-learning evaluation")
    
    def _create_q_update_wrapper(self, agent, original_method):
        """Create a wrapper for Q-value update methods."""
        def wrapped_update_q_value(*args, **kwargs):
            # Call original method
            result = original_method(*args, **kwargs)
            
            # Try to extract the information for evaluation
            try:
                # Different agents may have different signatures
                if len(args) >= 2:  # next_state, reward, etc.
                    next_state = args[0]
                    reward = args[1]
                    
                    # Get current state and action
                    current_state = getattr(agent, 'current_state', None)
                    current_action = getattr(agent, 'current_action', 0)
                    
                    # Get predicted Q-value before update
                    if hasattr(agent, 'q_table') and current_state is not None:
                        try:
                            predicted_q = agent.q_table.get_q_value(current_state, current_action)
                        except:
                            predicted_q = 0.0
                    else:
                        predicted_q = 0.0
                    
                    # Record the Q-learning step
                    self.evaluator.record_q_learning_step(
                        agent=agent,
                        state=current_state,
                        action=current_action,
                        predicted_q_value=predicted_q,
                        actual_reward=reward,
                        next_state=next_state,
                        learning_occurred=True
                    )
            except Exception as e:
                # Silently handle any integration errors
                pass
            
            return result
        
        return wrapped_update_q_value
    
    def _create_action_wrapper(self, agent, original_method):
        """Create a wrapper for action selection methods."""
        def wrapped_choose_action(*args, **kwargs):
            # Call original method
            action = original_method(*args, **kwargs)
            
            # Store current state and action for later use
            try:
                # STRICT: Only use get_state_representation() for 19D states - NO FALLBACKS
                if hasattr(agent, 'get_state_representation'):
                    state = agent.get_state_representation()
                    # Verify it's actually 19D - FAIL FAST if not
                    if hasattr(state, '__len__') and len(state) != 19:
                        raise ValueError(f"Agent {getattr(agent, 'id', 'unknown')} get_state_representation() returned {len(state)}D state, expected 19D!")
                    agent.current_state = state
                    # Minimal logging: only log once per agent
                    if not hasattr(agent, '_qlearn_state_confirmed'):
                        agent_id = str(getattr(agent, 'id', 'unknown'))[:8]
                        print(f"✅ Q-Learning: Agent {agent_id} using 19D states")
                        agent._qlearn_state_confirmed = True
                else:
                    # NO FALLBACKS - If agent doesn't have get_state_representation(), that's an ERROR
                    raise AttributeError(f"Agent {getattr(agent, 'id', 'unknown')} missing get_state_representation() method - cannot integrate with Q-learning!")
                
                agent.current_action = action
                agent.last_action_time = time.time()
            except Exception as e:
                # NO FALLBACK PADDING - Let the error bubble up so we can fix the root cause
                print(f"🚨 Q-Learning Integration Error for agent {getattr(agent, 'id', 'unknown')}: {e}")
                raise  # Re-raise the exception instead of hiding it
            
            return action
        
        return wrapped_choose_action
    
    def _create_step_wrapper(self, agent, original_method):
        """Create a wrapper for the step method to capture rewards."""
        def wrapped_step(*args, **kwargs):
            # Store previous reward for comparison
            prev_reward = getattr(agent, 'total_reward', 0.0)
            
            # Call original method
            result = original_method(*args, **kwargs)
            
            # Calculate reward received in this step
            try:
                current_reward = getattr(agent, 'total_reward', 0.0)
                step_reward = current_reward - prev_reward
                
                # Store for Q-learning evaluation
                agent.last_step_reward = step_reward
                
                # If we have state/action info and enough time has passed, record evaluation
                if (hasattr(agent, 'current_state') and hasattr(agent, 'current_action') and
                    hasattr(agent, 'last_action_time')):
                    
                    time_since_action = time.time() - agent.last_action_time
                    if time_since_action > 0.1:  # Give some time for the action to take effect
                        
                        # Get predicted Q-value
                        predicted_q = 0.0
                        if hasattr(agent, 'q_table') and agent.current_state is not None:
                            try:
                                predicted_q = agent.q_table.get_q_value(agent.current_state, agent.current_action)
                            except:
                                predicted_q = 0.0
                        
                        # Record the step for evaluation
                        self.evaluator.record_q_learning_step(
                            agent=agent,
                            state=agent.current_state,
                            action=agent.current_action,
                            predicted_q_value=predicted_q,
                            actual_reward=step_reward,
                            next_state=agent.current_state,  # Next state is current after step
                            learning_occurred=True
                        )
            except Exception as e:
                # Silently handle any integration errors
                pass
            
            return result
        
        return wrapped_step
    
    def restore_agent(self, agent) -> None:
        """Restore an agent to its original state (remove integration)."""
        agent_id = str(agent.id)
        
        if agent_id in self.original_methods:
            # Restore original methods
            for method_name, original_method in self.original_methods[agent_id].items():
                setattr(agent, method_name, original_method)
            
            # Clean up stored state
            if hasattr(agent, 'current_state'):
                delattr(agent, 'current_state')
            if hasattr(agent, 'current_action'):
                delattr(agent, 'current_action')
            if hasattr(agent, 'last_action_time'):
                delattr(agent, 'last_action_time')
            if hasattr(agent, 'last_step_reward'):
                delattr(agent, 'last_step_reward')
            
            del self.original_methods[agent_id]
            print(f"🔗 Restored agent {agent_id} to original state")


def create_evaluator_for_training_environment(training_env) -> QLearningEvaluator:
    """Create and integrate a Q-learning evaluator with a training environment."""
    # Create the evaluator
    evaluator = QLearningEvaluator(evaluation_window=1000, update_frequency=50)
    
    # Create the integration adapter
    adapter = QLearningIntegrationAdapter(evaluator)
    
    # Store the adapter on the training environment for later access
    training_env.q_learning_evaluator = evaluator
    training_env.q_learning_adapter = adapter
    
    # Integrate existing agents
    if hasattr(training_env, 'agents'):
        for agent in training_env.agents:
            try:
                adapter.integrate_agent(agent)
            except Exception as e:
                print(f"⚠️ Failed to integrate agent {agent.id}: {e}")
    
    print(f"🧠 Q-Learning evaluator created and integrated with {len(getattr(training_env, 'agents', []))} agents")
    return evaluator


def get_q_learning_status_for_api(training_env) -> dict:
    """Get Q-learning evaluation status for API endpoints."""
    if not hasattr(training_env, 'q_learning_evaluator'):
        return {'status': 'not_initialized', 'message': 'Q-learning evaluator not initialized'}
    
    evaluator = training_env.q_learning_evaluator
    
    try:
        # Get summary report
        summary = evaluator.generate_summary_report()
        
        # Add integration status
        summary['integration_status'] = {
            'evaluator_active': True,
            'agents_monitored': len(evaluator.agent_type_mapping),
            'agent_types': list(set(evaluator.agent_type_mapping.values())),
            'last_evaluation': max([metrics.timestamp for metrics in evaluator.get_all_agent_metrics().values()] 
                                 if evaluator.get_all_agent_metrics() else [0])
        }
        
        return summary
    except Exception as e:
        return {'status': 'error', 'message': f'Error generating Q-learning status: {e}'}
