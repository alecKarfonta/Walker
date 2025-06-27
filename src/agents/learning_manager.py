"""
Flexible Learning Manager
Allows dynamic switching between different Q-learning approaches for individual robots.
"""

import time
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from .q_table import EnhancedQTable, SparseQTable
from .survival_q_integration_patch import SurvivalAwareQLearning

# Create logger for this module
logger = logging.getLogger(__name__)


class LearningApproach(Enum):
    """Available learning approaches."""
    BASIC_Q_LEARNING = "basic_q_learning"
    ENHANCED_Q_LEARNING = "enhanced_q_learning"
    SURVIVAL_Q_LEARNING = "survival_q_learning"
    DEEP_Q_LEARNING = "deep_q_learning"  # For future use
    ATTENTION_DEEP_Q_LEARNING = "attention_deep_q_learning"
    ELITE_IMITATION_LEARNING = "elite_imitation_learning"


class LearningManager:
    """
    Manages different learning approaches for individual robots.
    Allows dynamic switching between approaches during training.
    """
    
    def __init__(self, ecosystem_interface=None, training_environment=None):
        self.ecosystem_interface = ecosystem_interface
        self.training_environment = training_environment
        
        # Track which approach each agent is using
        self.agent_approaches: Dict[str, LearningApproach] = {}
        
        # Store learning adapters for each agent
        self.agent_adapters: Dict[str, Any] = {}
        
        # Store original Q-tables for fallback
        self.agent_original_qtables: Dict[str, Any] = {}
        
        # Deep learning training coordination
        self._agents_currently_training: set = set()
        
        # Individual network creation counter for logging
        self._network_creation_count = 0
        
        # ATTENTION NETWORK MEMORY POOL - Reuse networks to prevent resource explosion
        self._attention_network_pool: List[Any] = []  # Available networks
        self._attention_network_pool_max_size = 10    # Max pooled networks
        self._attention_networks_in_use: Dict[str, Any] = {}  # agent_id -> network
        
        # NETWORK CREATION TRACKING
        self._network_stats = {
            'total_created': 0,
            'total_reused': 0,
            'current_in_use': 0,
            'current_in_pool': 0,
            'peak_networks_in_use': 0,
            'networks_destroyed': 0
        }
        
        # GPU USAGE TRACKING
        self._gpu_stats = {
            'initial_memory_mb': 0,
            'current_memory_mb': 0,
            'peak_memory_mb': 0,
            'last_update_time': 0
        }
        
        # Initialize GPU baseline
        self._update_gpu_stats()
        
        # Performance tracking for comparison
        self.approach_performance: Dict[LearningApproach, Dict[str, float]] = {
            approach: {
                'total_reward': 0.0,
                'learning_speed': 0.0,
                'food_consumed': 0,
                'survival_time': 0.0,
                'agent_count': 0
            }
            for approach in LearningApproach
        }
        
        # Learning approach metadata
        self.approach_info = {
            LearningApproach.BASIC_Q_LEARNING: {
                'name': 'Basic Q-Learning',
                'description': 'Simple tabular Q-learning with fixed exploration',
                'color': '#808080',  # Gray
                'icon': '🔤',
                'state_space': '~144 states',
                'advantages': ['Fast', 'Simple', 'Interpretable'],
                'disadvantages': ['Limited state representation', 'Slow learning']
            },
            LearningApproach.ENHANCED_Q_LEARNING: {
                'name': 'Enhanced Q-Learning',
                'description': 'Advanced tabular Q-learning with adaptive rates and exploration bonuses',
                'color': '#3498db',  # Blue
                'icon': '⚡',
                'state_space': '~144 states',
                'advantages': ['Adaptive learning', 'Confidence-based actions', 'Experience replay'],
                'disadvantages': ['Still limited state space', 'Movement-focused rewards']
            },
            LearningApproach.SURVIVAL_Q_LEARNING: {
                'name': 'Survival Q-Learning',
                'description': 'Enhanced Q-learning focused on survival with ecosystem awareness',
                'color': '#27ae60',  # Green
                'icon': '🍃',
                'state_space': '~40,960 states',
                'advantages': ['Survival-focused', 'Food awareness', 'Progressive learning stages'],
                'disadvantages': ['Larger state space', 'More complex']
            },
            LearningApproach.DEEP_Q_LEARNING: {
                'name': 'Deep Q-Learning',
                'description': 'Neural network-based Q-learning with continuous state representation',
                'color': '#9b59b6',  # Purple
                'icon': '🧠',
                'state_space': 'Continuous',
                'advantages': ['Scalable', 'Continuous states', 'High performance ceiling'],
                'disadvantages': ['Requires GPU', 'Slower startup', 'Less interpretable']
            },
            LearningApproach.ATTENTION_DEEP_Q_LEARNING: {
                'name': 'Attention Deep Q-Learning',
                'icon': '🔍',
                'description': 'Neural network with multi-head attention mechanisms'
            },
            LearningApproach.ELITE_IMITATION_LEARNING: {
                'name': 'Elite Imitation Learning',
                'icon': '🎭',
                'description': 'Learn from elite agent behavioral patterns'
            }
        }
        
        # Import enhanced learning systems
        try:
            from .attention_deep_q_learning import AttentionDeepQLearning
            from .elite_imitation_learning import EliteImitationLearning
            self.attention_deep_q_available = True
            self.elite_imitation_available = True
        except ImportError as e:
            logger.warning(f"Enhanced learning systems not available: {e}")
            self.attention_deep_q_available = False
            self.elite_imitation_available = False
        
        # Elite imitation learning system (shared across all agents)
        if self.elite_imitation_available:
            self.elite_imitation = EliteImitationLearning(
                imitation_probability=0.3,
                elite_update_interval=500,
                pattern_extraction_window=100
            )
        else:
            self.elite_imitation = None
        
        print("🎛️ LearningManager initialized - Ready for flexible approach switching")
    
    def get_agent_approach(self, agent_id: str) -> LearningApproach:
        """Get the current learning approach for an agent."""
        return self.agent_approaches.get(agent_id, LearningApproach.ENHANCED_Q_LEARNING)
    
    def set_agent_approach(self, agent, approach: LearningApproach) -> bool:
        """Set learning approach for an agent with enhanced systems."""
        try:
            agent_id = agent.id
            current_approach = self.get_agent_approach(agent_id)
            
            if current_approach == approach:
                print(f"🔄 Agent {agent_id} already using {approach.value}")
                return True
            
            print(f"🔄 Switching Agent {agent_id}: {current_approach.value} → {approach.value}")
            
            # Store current Q-table for knowledge transfer (always update to get latest learned data)
            self.agent_original_qtables[agent_id] = agent.q_table
            
            # Perform the switch
            success = self._perform_approach_switch(agent, current_approach, approach)
            
            if success:
                self.agent_approaches[agent_id] = approach
                self._update_performance_tracking(agent_id, current_approach, approach)
                print(f"✅ Agent {agent_id} successfully switched to {approach.value}")
                return True
            else:
                print(f"❌ Failed to switch Agent {agent_id} to {approach.value}")
                return False
                
        except Exception as e:
            print(f"❌ Error switching Agent {agent_id} to {approach.value}: {e}")
            return False
    
    def _perform_approach_switch(self, agent, from_approach: LearningApproach, 
                                to_approach: LearningApproach) -> bool:
        """Perform the actual switching between learning approaches."""
        
        agent_id = agent.id
        
        try:
            # Step 1: Clean up current approach
            self._cleanup_current_approach(agent, from_approach)
            
            # Step 2: Create new learning approach instance
            new_adapter = self._create_learning_approach(agent, to_approach)
            if not new_adapter:
                print(f"❌ Failed to create {to_approach.value} for agent {agent_id}")
                return False
            
            # Step 3: Store new adapter if needed
            if new_adapter != agent:  # Only store if it's a separate adapter
                self.agent_adapters[agent_id] = new_adapter
            
            # Step 4: Transfer knowledge from old approach
            if agent_id in self.agent_original_qtables:
                self._transfer_knowledge(agent, self.agent_original_qtables[agent_id], to_approach)
            
            print(f"✅ Agent {agent_id} successfully switched to {to_approach.value}")
            return True
                
        except Exception as e:
            print(f"❌ Error performing approach switch: {e}")
            return False
    
    def _create_learning_approach(self, agent, approach: LearningApproach):
        """Create a new learning approach instance for the agent."""
        
        try:
            if approach == LearningApproach.BASIC_Q_LEARNING:
                # Simple sparse Q-table
                agent.q_table = SparseQTable(
                    action_count=agent.action_size,
                    default_value=0.0
                )
                agent.learning_rate = 0.1
                agent.epsilon = 0.3
                agent.epsilon_decay = 0.999
                agent.learning_approach = approach.value
                return agent
            
            elif approach == LearningApproach.ENHANCED_Q_LEARNING:
                # Enhanced Q-table with sophisticated features
                agent.q_table = EnhancedQTable(
                    action_count=agent.action_size,
                    default_value=0.0,
                    confidence_threshold=15,
                    exploration_bonus=0.15
                )
                agent.learning_rate = 0.05
                agent.epsilon = 0.3
                agent.epsilon_decay = 0.9999
                agent.learning_approach = approach.value
                return agent
            
            elif approach == LearningApproach.SURVIVAL_Q_LEARNING:
                if not self.ecosystem_interface:
                    print("❌ Cannot create survival Q-learning: No ecosystem interface")
                    return None
                
                from .survival_q_integration_patch import upgrade_agent_to_survival_learning
                survival_adapter = upgrade_agent_to_survival_learning(agent, self.ecosystem_interface)
                return survival_adapter
            
            elif approach == LearningApproach.DEEP_Q_LEARNING:
                from .deep_survival_q_learning import DeepSurvivalQLearning, TORCH_AVAILABLE
                
                if not TORCH_AVAILABLE:
                    print("❌ PyTorch not available. Cannot use Deep Q-Learning.")
                    return None
                
                # Create Deep Q-Learning instance
                state_dim = 15
                action_dim = agent.action_size
                
                deep_q_learner = DeepSurvivalQLearning(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    learning_rate=3e-4,
                    gamma=0.99,
                    epsilon_start=1.0,
                    epsilon_end=0.01,
                    epsilon_decay=150000,
                    buffer_size=50000,  # Increased from 25k to 50k for better learning
                    batch_size=32,
                    target_update_freq=2000,
                    device="auto",
                    use_dueling=True,
                    use_prioritized_replay=True
                )
                
                # Setup deep learning wrapper
                self._setup_deep_learning_wrapper(agent, deep_q_learner)
                agent.learning_approach = approach.value
                return deep_q_learner
            
            elif approach == LearningApproach.ATTENTION_DEEP_Q_LEARNING:
                if not self.attention_deep_q_available:
                    print(f"❌ Attention Deep Q-Learning not available for agent {agent.id}")
                    return None
                
                # Acquire attention network from pool (reuse existing or create new if needed)
                attention_dqn = self._acquire_attention_network(agent.id)
                if not attention_dqn:
                    print(f"❌ Failed to acquire attention network for agent {agent.id}")
                    return None
                
                # Setup attention learning wrapper
                self._setup_attention_learning_wrapper(agent, attention_dqn)
                agent.learning_approach = approach.value
                return attention_dqn
            
            elif approach == LearningApproach.ELITE_IMITATION_LEARNING:
                if not self.elite_imitation_available:
                    print(f"❌ Elite Imitation Learning not available for agent {agent.id}")
                    return None
                
                # Setup elite imitation wrapper
                self._setup_elite_imitation_wrapper(agent)
                agent.learning_approach = approach.value
                return agent
            
            else:
                print(f"⚠️ Unknown learning approach: {approach}")
                return None
                
        except Exception as e:
            print(f"❌ Error creating learning approach {approach.value}: {e}")
            return None
    
    def _setup_deep_learning_wrapper(self, agent, deep_q_learner):
        """Setup the deep learning step wrapper for an agent."""
        # Initialize deep learning attributes with OPTIMIZED frequencies
        agent._deep_step_count = 0
        agent._deep_experience_collection_freq = 20  # INCREASED from 10 to 20 (collect less frequently)
        agent._deep_training_freq = 6000  # INCREASED from 3000 to 6000 (train less frequently)
        agent._deep_min_buffer_size = 3000
        agent._deep_ready_to_train = False
        agent._deep_prev_state = None
        agent._deep_prev_action = None
        agent._deep_prev_reward = 0.0
        
        # Performance tracking
        agent._deep_last_cleanup = time.time()
        agent._deep_cleanup_interval = 60.0  # Clean up every minute
        
        # Store original step method
        if not hasattr(agent, '_original_step_method'):
            agent._original_step_method = agent.step
        
        # Create wrapper step method
        def deep_learning_step(dt: float):
            try:
                result = agent._original_step_method(dt)
                agent._deep_step_count += 1
                
                # PERFORMANCE: Periodic cleanup
                current_time = time.time()
                if current_time - agent._deep_last_cleanup > agent._deep_cleanup_interval:
                    self._cleanup_agent_deep_learning_data(agent, deep_q_learner)
                    agent._deep_last_cleanup = current_time
                
                # Experience collection (less frequent)
                if agent._deep_step_count % agent._deep_experience_collection_freq == 0:
                    self._collect_deep_experience(agent, deep_q_learner)
                
                # Training (much less frequent)
                if (agent._deep_ready_to_train and 
                    agent._deep_step_count % agent._deep_training_freq == 0):
                    self._train_deep_model(agent, deep_q_learner)
                
                return result
            except Exception as e:
                return agent._original_step_method(dt)
        
        agent.step = deep_learning_step
    
    def _setup_attention_learning_wrapper(self, agent, attention_dqn):
        """Setup the attention learning wrapper for an agent."""
        agent._attention_dqn = attention_dqn
        agent._original_choose_action = getattr(agent, 'choose_action', None)
        
        def attention_choose_action():
            if not hasattr(agent, '_attention_dqn') or agent._attention_dqn is None:
                if agent._original_choose_action:
                    return agent._original_choose_action()
                return 0
            
            training_env = getattr(agent, 'training_environment', None)
            state_data = self._get_agent_state_data(agent, training_env)
            state_vector = agent._attention_dqn.get_arm_control_state_representation(state_data)
            action = agent._attention_dqn.choose_action(state_vector, state_data)
            return action
        
        agent.choose_action = attention_choose_action
    
    def _setup_elite_imitation_wrapper(self, agent):
        """Setup the elite imitation learning wrapper for an agent."""
        agent._original_choose_action = getattr(agent, 'choose_action', None)
        agent._use_elite_imitation = True
        
        def imitation_choose_action():
            context = self._get_agent_context(agent)
            current_state = self._get_agent_state(agent)
            
            if self.elite_imitation:
                imitation_action = self.elite_imitation.get_imitation_action(
                    agent.id, current_state, context
                )
                if imitation_action is not None:
                    return imitation_action
            
            if agent._original_choose_action:
                return agent._original_choose_action()
            else:
                return agent.choose_action() if hasattr(agent, 'choose_action') else 0
        
        agent.choose_action = imitation_choose_action
    
    def _collect_deep_experience(self, agent, deep_q_learner):
        """Collect experience for deep learning."""
        try:
            current_state = [
                getattr(agent.body, 'angle', 0.0),
                agent.body.linearVelocity.x if hasattr(agent.body, 'linearVelocity') else 0.0,
                agent.body.linearVelocity.y if hasattr(agent.body, 'linearVelocity') else 0.0,
                agent.body.position.x if hasattr(agent.body, 'position') else 0.0,
                agent.body.position.y if hasattr(agent.body, 'position') else 0.0,
            ]
            
            if len(current_state) == 5:
                extended_state_list = current_state + [0.0] * 10
                extended_state_array = np.array(extended_state_list, dtype=np.float32)
                action_idx = deep_q_learner.choose_action(extended_state_array)
                
                if hasattr(agent, 'actions') and action_idx is not None and action_idx < len(agent.actions):
                    agent.current_action = action_idx
                    agent.current_action_tuple = agent.actions[action_idx]
                
                if (agent._deep_prev_state is not None and 
                    agent._deep_prev_action is not None and
                    action_idx is not None):
                    reward = getattr(agent, 'total_reward', 0.0) - getattr(agent, '_deep_prev_reward', 0.0)
                    deep_q_learner.store_experience(
                        agent._deep_prev_state, 
                        agent._deep_prev_action, 
                        reward, 
                        extended_state_array, 
                        False
                    )
                    
                    agent._deep_prev_state = extended_state_array
                    agent._deep_prev_action = action_idx
                    agent._deep_prev_reward = getattr(agent, 'total_reward', 0.0)
                    
        except Exception as e:
            print(f"❌ Experience collection error Agent {agent.id}: {e}")
    
    def _train_deep_model(self, agent, deep_q_learner):
        """Train the deep learning model."""
        if len(self._agents_currently_training) >= 1:
            import random
            agent._deep_step_count -= random.randint(1000, 3000)
            return
        
        self._agents_currently_training.add(agent.id)
        
        try:
            import time
            import threading
            
            print(f"🔥 GPU TRAINING START: Agent {agent.id}")
            training_start_time = time.time()
            
            def background_training():
                try:
                    learning_stats = deep_q_learner.learn()
                    training_duration = time.time() - training_start_time
                    print(f"✅ GPU TRAINING COMPLETE: Agent {agent.id} ({training_duration:.3f}s)")
                except Exception as e:
                    print(f"❌ GPU TRAINING FAILED: Agent {agent.id} - {e}")
                finally:
                    self._agents_currently_training.discard(agent.id)
            
            training_thread = threading.Thread(target=background_training, daemon=True)
            training_thread.start()
            
        except Exception as e:
            print(f"❌ GPU TRAINING FAILED: Agent {agent.id} - {e}")
            self._agents_currently_training.discard(agent.id)
    
    def _transfer_knowledge(self, agent, source_qtable, approach: LearningApproach):
        """Transfer knowledge from source Q-table to new approach."""
        try:
            if approach == LearningApproach.BASIC_Q_LEARNING:
                self._transfer_basic_knowledge(agent, source_qtable)
            elif approach == LearningApproach.ENHANCED_Q_LEARNING:
                self._transfer_enhanced_knowledge(agent, source_qtable)
            elif approach == LearningApproach.SURVIVAL_Q_LEARNING:
                self._transfer_survival_knowledge(agent, source_qtable)
            elif approach == LearningApproach.DEEP_Q_LEARNING:
                self._transfer_deep_knowledge(agent, source_qtable)
            # Note: Attention and Elite approaches don't need specific knowledge transfer
            
        except Exception as e:
            print(f"⚠️ Error transferring knowledge for {approach.value}: {e}")
    
    def _cleanup_current_approach(self, agent, approach: LearningApproach):
        """Clean up the current learning approach."""
        agent_id = agent.id
        
        try:
            # Remove survival adapter if it exists
            if agent_id in self.agent_adapters:
                del self.agent_adapters[agent_id]
            
            # Restore original step method if it was modified
            if hasattr(agent, '_original_step_method'):
                agent.step = agent._original_step_method
                delattr(agent, '_original_step_method')
            
            # CRITICAL FIX: Return attention networks to pool instead of destroying them
            if hasattr(agent, '_attention_dqn'):
                try:
                    # Return network to pool for reuse
                    self._return_attention_network(agent_id)
                    
                    # Remove reference from agent
                    delattr(agent, '_attention_dqn')
                    print(f"♻️ Returned attention network to pool from agent {agent_id}")
                except Exception as e:
                    print(f"⚠️ Error returning attention network for agent {agent_id}: {e}")
            
            # Restore original choose_action method
            if hasattr(agent, '_original_choose_action'):
                agent.choose_action = agent._original_choose_action
                delattr(agent, '_original_choose_action')
            
            # Clean up survival-specific attributes
            if hasattr(agent, '_prev_survival_state'):
                delattr(agent, '_prev_survival_state')
            if hasattr(agent, '_prev_survival_action'):
                delattr(agent, '_prev_survival_action')
            if hasattr(agent, 'get_survival_stats'):
                delattr(agent, 'get_survival_stats')
                
        except Exception as e:
            print(f"⚠️ Error cleaning up approach for agent {agent_id}: {e}")
    
    def _transfer_basic_knowledge(self, agent, source_qtable):
        """Transfer basic knowledge from source Q-table."""
        try:
            if hasattr(source_qtable, 'q_values') and hasattr(agent.q_table, 'q_values'):
                # Transfer a subset of Q-values
                transfer_count = 0
                for state_key, action_values in list(source_qtable.q_values.items())[:100]:  # Limit transfer
                    agent.q_table.q_values[state_key] = [v * 0.5 for v in action_values]  # Reduced confidence
                    transfer_count += 1
                
                print(f"📚 Transferred {transfer_count} Q-values to basic learning")
                
        except Exception as e:
            print(f"⚠️ Error transferring basic knowledge: {e}")
    
    def _transfer_enhanced_knowledge(self, agent, source_qtable):
        """Transfer enhanced knowledge from source Q-table."""
        try:
            if hasattr(source_qtable, 'learn_from_other_table') and hasattr(agent.q_table, 'learn_from_other_table'):
                # Use the sophisticated knowledge transfer method with high learning rate to preserve more data
                agent.q_table.learn_from_other_table(source_qtable, learning_rate=0.8)
                print(f"📚 Enhanced knowledge transfer completed (80% of learned data preserved)")
            elif hasattr(source_qtable, 'q_values') and hasattr(agent.q_table, 'q_values'):
                # Fallback: Direct Q-value transfer for maximum preservation
                transferred_states = 0
                for state_key, action_values in source_qtable.q_values.items():
                    agent.q_table.q_values[state_key] = action_values.copy()
                    if hasattr(source_qtable, 'visit_counts') and hasattr(agent.q_table, 'visit_counts'):
                        agent.q_table.visit_counts[state_key] = source_qtable.visit_counts[state_key].copy()
                    transferred_states += 1
                
                # Copy other important data
                if hasattr(source_qtable, 'update_count'):
                    agent.q_table.update_count = source_qtable.update_count
                
                print(f"📚 Direct knowledge transfer: {transferred_states} states preserved")
                
        except Exception as e:
            print(f"⚠️ Error transferring enhanced knowledge: {e}")
    
    def _transfer_survival_knowledge(self, agent, source_qtable):
        """Transfer survival Q-learning knowledge including ecosystem awareness."""
        try:
            if hasattr(source_qtable, 'q_values') and hasattr(agent.q_table, 'q_values'):
                # Direct Q-value transfer for survival states
                transferred_states = 0
                for state_key, action_values in source_qtable.q_values.items():
                    if isinstance(action_values, list) and len(action_values) > 0:
                        agent.q_table.q_values[state_key] = action_values.copy()
                        transferred_states += 1
                        
                        # Transfer visit counts if available
                        if hasattr(source_qtable, 'visit_counts') and hasattr(agent.q_table, 'visit_counts'):
                            if state_key in source_qtable.visit_counts:
                                agent.q_table.visit_counts[state_key] = source_qtable.visit_counts[state_key].copy()
                
                print(f"📚 Survival knowledge transfer: {transferred_states} survival states preserved")
                
                # Transfer survival-specific metadata
                if hasattr(source_qtable, 'stage_progression'):
                    agent.q_table.stage_progression = getattr(source_qtable, 'stage_progression', 'basic_movement')
                
                if hasattr(source_qtable, 'survival_stats'):
                    agent.q_table.survival_stats = getattr(source_qtable, 'survival_stats', {}).copy()
                    
        except Exception as e:
            print(f"⚠️ Error transferring survival knowledge: {e}")

    def _transfer_deep_knowledge(self, agent, source_qtable):
        """Transfer deep Q-learning knowledge including neural network weights."""
        try:
            # Get the deep learning adapter
            deep_learner = self.agent_adapters.get(agent.id)
            if not deep_learner:
                print(f"⚠️ No deep learning adapter found for agent {agent.id}")
                return
                
            # Transfer neural network weights if source has them
            if hasattr(source_qtable, '_deep_learner_weights'):
                try:
                    deep_learner.load_state_dict(source_qtable._deep_learner_weights)
                    print(f"📚 Deep learning neural network weights transferred")
                except Exception as e:
                    print(f"⚠️ Could not load neural network weights: {e}")
            
            # Transfer experience replay buffer
            if hasattr(source_qtable, '_deep_experience_buffer') and hasattr(deep_learner, 'memory'):
                try:
                    source_buffer = source_qtable._deep_experience_buffer
                    if source_buffer and len(source_buffer) > 0:
                        # Transfer up to 10,000 most recent experiences
                        transfer_count = min(10000, len(source_buffer))
                        for i in range(-transfer_count, 0):  # Get last N experiences
                            deep_learner.memory.push(*source_buffer[i])
                        print(f"📚 Transferred {transfer_count} experiences to replay buffer")
                except Exception as e:
                    print(f"⚠️ Could not transfer experience buffer: {e}")
            
            # Transfer learning statistics
            if hasattr(source_qtable, '_deep_training_stats'):
                deep_learner._training_stats = getattr(source_qtable, '_deep_training_stats', {}).copy()
                
        except Exception as e:
            print(f"⚠️ Error transferring deep learning knowledge: {e}")

    def transfer_complete_knowledge(self, source_agent, target_agent, approach: LearningApproach):
        """
        Comprehensive knowledge transfer between agents for any learning approach.
        
        Args:
            source_agent: Agent to copy knowledge from
            target_agent: Agent to copy knowledge to  
            approach: Learning approach to use for transfer
        """
        try:
            print(f"🧠 Transferring {approach.value} knowledge: {source_agent.id} → {target_agent.id}")
            
            # Store source Q-table for transfer
            source_qtable = source_agent.q_table
            
            # Store approach-specific data from source
            if approach == LearningApproach.DEEP_Q_LEARNING:
                # Store neural network weights and experience buffer
                deep_adapter = self.agent_adapters.get(source_agent.id)
                if deep_adapter:
                    try:
                        source_qtable._deep_learner_weights = deep_adapter.state_dict()
                        source_qtable._deep_experience_buffer = list(deep_adapter.memory.buffer) if hasattr(deep_adapter.memory, 'buffer') else []
                        source_qtable._deep_training_stats = getattr(deep_adapter, '_training_stats', {})
                    except Exception as e:
                        print(f"⚠️ Error storing deep learning data: {e}")
                        
            elif approach == LearningApproach.SURVIVAL_Q_LEARNING:
                # Store survival-specific data
                survival_adapter = self.agent_adapters.get(source_agent.id)
                if survival_adapter:
                    try:
                        source_qtable.stage_progression = getattr(survival_adapter, 'learning_stage', 'basic_movement')
                        source_qtable.survival_stats = getattr(survival_adapter, 'survival_stats', {}).copy()
                    except Exception as e:
                        print(f"⚠️ Error storing survival data: {e}")
            
            # Set up target agent with the same approach
            success = self.set_agent_approach(target_agent, approach)
            if not success:
                print(f"❌ Failed to set up {approach.value} on target agent")
                return False
            
            # Perform knowledge transfer based on approach
            if approach == LearningApproach.BASIC_Q_LEARNING:
                self._transfer_basic_knowledge(target_agent, source_qtable)
            elif approach == LearningApproach.ENHANCED_Q_LEARNING:
                self._transfer_enhanced_knowledge(target_agent, source_qtable)
            elif approach == LearningApproach.SURVIVAL_Q_LEARNING:
                self._transfer_survival_knowledge(target_agent, source_qtable)
            elif approach == LearningApproach.DEEP_Q_LEARNING:
                self._transfer_deep_knowledge(target_agent, source_qtable)
            
            # Copy learning parameters
            target_agent.learning_rate = getattr(source_agent, 'learning_rate', 0.1)
            target_agent.epsilon = getattr(source_agent, 'epsilon', 0.3)
            if hasattr(source_agent, 'epsilon_decay'):
                target_agent.epsilon_decay = source_agent.epsilon_decay
                
            print(f"✅ Complete knowledge transfer successful for {approach.value}")
            return True
            
        except Exception as e:
            print(f"❌ Error in complete knowledge transfer: {e}")
            return False
    
    def _update_performance_tracking(self, agent_id: str, from_approach: LearningApproach, 
                                   to_approach: LearningApproach):
        """Update performance tracking when switching approaches."""
        try:
            # Decrement count for old approach
            if self.approach_performance[from_approach]['agent_count'] > 0:
                self.approach_performance[from_approach]['agent_count'] -= 1
            
            # Increment count for new approach
            self.approach_performance[to_approach]['agent_count'] += 1
            
        except Exception as e:
            print(f"⚠️ Error updating performance tracking: {e}")
    
    def get_approach_statistics(self) -> Dict[str, Any]:
        """Get statistics about learning approach usage and performance."""
        try:
            stats = {
                'approach_distribution': {},
                'approach_performance': {},
                'total_agents': sum(perf['agent_count'] for perf in self.approach_performance.values()),
                'approach_info': {}
            }
            
            # Calculate distribution
            for approach in LearningApproach:
                count = self.approach_performance[approach]['agent_count']
                stats['approach_distribution'][approach.value] = count
                
                # Add approach info
                stats['approach_info'][approach.value] = self.approach_info[approach]
                
                # Calculate performance metrics
                if count > 0:
                    perf = self.approach_performance[approach]
                    stats['approach_performance'][approach.value] = {
                        'avg_reward': perf['total_reward'] / count,
                        'avg_food_consumed': perf['food_consumed'] / count,
                        'avg_survival_time': perf['survival_time'] / count,
                        'agent_count': count
                    }
                else:
                    stats['approach_performance'][approach.value] = {
                        'avg_reward': 0.0,
                        'avg_food_consumed': 0.0,
                        'avg_survival_time': 0.0,
                        'agent_count': 0
                    }
            
            return stats
            
        except Exception as e:
            print(f"⚠️ Error getting approach statistics: {e}")
            return {}
    
    def get_agent_learning_info(self, agent_id: str) -> Dict[str, Any]:
        """Get detailed learning information for a specific agent."""
        try:
            approach = self.get_agent_approach(agent_id)
            info = self.approach_info[approach].copy()
            
            # Add current stats if we have an adapter
            if agent_id in self.agent_adapters:
                try:
                    adapter_stats = self.agent_adapters[agent_id].get_learning_stats()
                    info['current_stats'] = adapter_stats
                except:
                    pass
            
            info['current_approach'] = approach.value
            
            return info
            
        except Exception as e:
            print(f"⚠️ Error getting agent learning info: {e}")
            return {}
    
    def bulk_switch_approach(self, agent_ids: List[str], agents: List[Any], 
                           new_approach: LearningApproach) -> Dict[str, bool]:
        """Switch multiple agents to a new learning approach."""
        results = {}
        
        for agent_id, agent in zip(agent_ids, agents):
            results[agent_id] = self.set_agent_approach(agent, new_approach)
        
        success_count = sum(1 for success in results.values() if success)
        print(f"🔄 Bulk switch complete: {success_count}/{len(agent_ids)} agents switched to {new_approach.value}")
        
        return results
    
    def set_training_environment(self, training_environment):
        """Set the training environment reference for accessing real food data."""
        self.training_environment = training_environment
    
    def inject_training_environment_into_agents(self, agents):
        """Inject training environment reference into all agents for state data access."""
        for agent in agents:
            if hasattr(agent, 'id'):
                agent.training_environment = self.training_environment
        print(f"🌍 Training environment injected into {len(agents)} agents")
    
    def recommend_approach(self, agent, performance_history: Dict[str, float]) -> LearningApproach:
        """Recommend the best learning approach for an agent based on performance."""
        try:
            # Simple recommendation logic based on performance
            avg_reward = performance_history.get('avg_reward', 0.0)
            learning_speed = performance_history.get('learning_speed', 0.0)
            survival_time = performance_history.get('survival_time', 0.0)
            
            # If agent is performing poorly, suggest survival learning
            if avg_reward < -0.5 or survival_time < 30:
                return LearningApproach.SURVIVAL_Q_LEARNING
            
            # If agent is learning slowly, suggest enhanced learning
            elif learning_speed < 0.1:
                return LearningApproach.ENHANCED_Q_LEARNING
            
            # If agent is doing well, keep current approach
            else:
                return self.get_agent_approach(agent.id)
                
        except Exception as e:
            print(f"⚠️ Error recommending approach: {e}")
            return LearningApproach.ENHANCED_Q_LEARNING 

    def update_agent_performance(self, agent, step_count: int):
        """Update agent performance for elite identification."""
        if self.elite_imitation and hasattr(agent, 'id'):
            try:
                # Calculate performance metrics
                # Calculate distance traveled safely
                distance_traveled = 0.0
                if hasattr(agent, 'body') and agent.body and hasattr(agent.body, 'position'):
                    distance_traveled = abs(agent.body.position.x)
                
                performance_data = {
                    'total_reward': getattr(agent, 'total_reward', 0.0),
                    'survival_time': getattr(agent, 'steps', 0),
                    'food_consumption': getattr(agent, 'total_reward', 0.0) * 0.1,  # Estimate
                    'distance_traveled': distance_traveled,
                    'learning_efficiency': 0.5  # Default value
                }
                
                self.elite_imitation.update_agent_performance(agent.id, performance_data)
                
                # Record trajectory for pattern extraction
                if hasattr(agent, 'current_state') and hasattr(agent, 'current_action'):
                    context = self._get_agent_context(agent)
                    reward = getattr(agent, 'immediate_reward', 0.0)
                    
                    self.elite_imitation.record_agent_trajectory(
                        agent.id, agent.current_state, agent.current_action, reward, context
                    )
                
            except Exception as e:
                logger.warning(f"Failed to update performance for agent {agent.id}: {e}")
    
    def update_elites(self, all_agents, current_step: int):
        """Update elite agents periodically."""
        if (self.elite_imitation and 
            self.elite_imitation.should_update_elites(current_step)):
            
            agent_ids = [agent.id for agent in all_agents if hasattr(agent, 'id')]
            self.elite_imitation.update_elites(agent_ids)
    
    def get_enhanced_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics including enhanced learning systems."""
        stats: Dict[str, Any] = {
            'attention_deep_q_available': self.attention_deep_q_available,
            'elite_imitation_available': self.elite_imitation_available,
        }
        
        if self.elite_imitation:
            stats['elite_imitation'] = self.elite_imitation.get_statistics()
        
        return stats
    
    def _get_agent_state_data(self, agent, training_environment=None) -> Dict[str, Any]:
        """Extract state data from agent for enhanced learning systems."""
        try:
            state_data = {}
            
            # Physical state - REQUIRE real body data
            if not (hasattr(agent, 'body') and agent.body):
                raise ValueError(f"Agent {agent.id} missing body physics data")
                
            state_data['position'] = (agent.body.position.x, agent.body.position.y)
            state_data['velocity'] = (agent.body.linearVelocity.x, agent.body.linearVelocity.y)
            state_data['body_angle'] = agent.body.angle
            
            # Arm angles - REQUIRE real arm data
            if not (hasattr(agent, 'upper_arm') and agent.upper_arm and 
                   hasattr(agent, 'lower_arm') and agent.lower_arm):
                raise ValueError(f"Agent {agent.id} missing arm physics data")
                
            state_data['arm_angles'] = {
                'shoulder': agent.upper_arm.angle,
                'elbow': agent.lower_arm.angle
            }
            
            # Energy and health
            state_data['energy'] = getattr(agent, 'energy_level', 1.0)
            state_data['health'] = getattr(agent, 'health_level', 1.0)
            
            # Real food information from training environment - REQUIRE training environment
            if not training_environment:
                raise ValueError(f"Agent {agent.id} missing training environment reference")
                
            food_info = training_environment._get_closest_food_distance_for_agent(agent)
            
            # Use signed_x_distance directly as direction (matches survival Q-learning approach)
            # This eliminates the data format mismatch that caused KeyError fallbacks
            state_data['nearest_food'] = {
                'distance': food_info['distance'],
                'direction': food_info['signed_x_distance'],  # Use signed x-distance as direction
                'type': food_info['food_type']
            }
            
            # Ground contact detection using Box2D physics
            ground_contact = self._detect_ground_contact(agent)
            state_data['ground_contact'] = ground_contact
            
            # Physics body for advanced ground detection
            state_data['physics_body'] = agent.body
            
            return state_data
            
        except Exception as e:
            # NO FALLBACK VALUES - Let it fail so we can see the real problems
            raise RuntimeError(f"Critical state extraction failure for agent {getattr(agent, 'id', 'unknown')}: {e}")
            
    def _convert_to_continuous_state(self, state_data: Dict[str, Any]) -> np.ndarray:
        """Convert state data to continuous vector for neural networks."""
        try:
            # FIXED: Use standalone state representation function instead of creating new instances
            return self._get_enhanced_state_representation_static(state_data)
        except Exception as e:
            print(f"⚠️ Error converting to continuous state: {e}")
            return np.zeros(15, dtype=np.float32)
    
    def _get_enhanced_state_representation_static(self, agent_data: Dict[str, Any]) -> np.ndarray:
        """Standalone state representation function that doesn't require instantiating classes."""
        try:
            # Physical features (8 dimensions)
            position_x = agent_data.get('position', (0, 0))[0] / 100.0  # Normalize
            position_y = agent_data.get('position', (0, 0))[1] / 100.0
            velocity_x = agent_data.get('velocity', (0, 0))[0] / 10.0
            velocity_y = agent_data.get('velocity', (0, 0))[1] / 10.0
            
            arm_angles = agent_data.get('arm_angles', {'shoulder': 0, 'elbow': 0})
            shoulder_angle = arm_angles['shoulder'] / np.pi  # Normalize to [-1, 1]
            elbow_angle = arm_angles['elbow'] / np.pi
            
            body_angle = agent_data.get('body_angle', 0) / np.pi
            stability = 1.0 - abs(body_angle)  # Higher is more stable
            
            physical = [position_x, position_y, velocity_x, velocity_y, 
                       shoulder_angle, elbow_angle, body_angle, stability]
            
            # Energy features (2 dimensions)
            energy = agent_data.get('energy', 1.0)
            health = agent_data.get('health', 1.0)
            energy_features = [energy, health]
            
            # Food features (3 dimensions)
            food_info = agent_data.get('nearest_food', {'distance': float('inf'), 'direction': 0, 'type': 0})
            food_distance = min(1.0, food_info['distance'] / 50.0)  # Normalize and cap
            food_direction = food_info['direction'] / np.pi  # Normalize to [-1, 1]
            food_type = food_info.get('type', 0) / 3.0  # Normalize food type
            food_features = [food_distance, food_direction, food_type]
            
            # Social features (2 dimensions)
            nearby_agents = min(1.0, agent_data.get('nearby_agents', 0) / 10.0)  # Normalize
            competition = agent_data.get('competition_pressure', 0.5)
            social_features = [nearby_agents, competition]
            
            # Combine all features (15 dimensions total)
            state = np.array(physical + energy_features + food_features + social_features, dtype=np.float32)
            
            # Ensure no NaN or infinite values
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return state
            
        except Exception as e:
            print(f"⚠️ Error creating state representation: {e}")
            # Return default state
            return np.zeros(15, dtype=np.float32)
    
    def _detect_ground_contact(self, agent) -> bool:
        """Detect if the agent is in contact with the ground using Box2D physics."""
        try:
            if not (hasattr(agent, 'body') and agent.body):
                return False
            
            # Check if lower arm is in contact with ground
            if hasattr(agent, 'lower_arm') and agent.lower_arm:
                for contact_edge in agent.lower_arm.contacts:
                    contact = contact_edge.contact
                    if contact.touching:
                        fixture_a = contact.fixtureA
                        fixture_b = contact.fixtureB
                        # Check if contact is with ground (category bit 0x0001)
                        if ((fixture_a.filterData.categoryBits & 0x0001) or 
                            (fixture_b.filterData.categoryBits & 0x0001)):
                            return True
            
            return False
            
        except Exception as e:
            # Simple fallback for ground contact only
            try:
                if hasattr(agent, 'body') and agent.body:
                    return agent.body.position.y <= 1.0  # Ground level approximation
            except:
                pass
            return False
    
    def _get_agent_context(self, agent) -> Dict[str, Any]:
        """Get contextual information for elite imitation learning."""
        try:
            context = {}
            
            # Energy level context
            energy = getattr(agent, 'energy_level', 1.0)
            context['energy_level'] = energy
            
            # Food distance context (simplified)
            context['food_distance'] = 10.0  # Default
            
            # Performance context
            context['total_reward'] = getattr(agent, 'total_reward', 0.0)
            context['steps_alive'] = getattr(agent, 'steps', 0)
            
            # Body state context
            if hasattr(agent, 'body') and agent.body:
                context['position_x'] = agent.body.position.x
                context['position_y'] = agent.body.position.y
                context['velocity'] = (agent.body.linearVelocity.x**2 + agent.body.linearVelocity.y**2)**0.5
            
            return context
            
        except Exception as e:
            logger.warning(f"Error getting agent context: {e}")
            return {}
    
    def _get_agent_state(self, agent):
        """Get current state representation for imitation learning."""
        try:
            if hasattr(agent, 'current_state'):
                return agent.current_state
            elif hasattr(agent, 'get_discretized_state'):
                return agent.get_discretized_state()
            else:
                # Fallback to simple state
                return (0, 0)  # Default state
        except Exception as e:
            logger.warning(f"Error getting agent state: {e}")
            return (0, 0)
    
    def _acquire_attention_network(self, agent_id: str):
        """Acquire an attention network from the pool or create a new one if pool is empty."""
        try:
            # CRITICAL: Check if agent already has a network (from transfer)
            if agent_id in self._attention_networks_in_use:
                existing_network = self._attention_networks_in_use[agent_id]
                print(f"♻️ Agent {agent_id[:8]} already has attention network (transferred)")
                return existing_network
            
            # Try to reuse an existing network from the pool
            if self._attention_network_pool:
                network = self._attention_network_pool.pop()
                self._attention_networks_in_use[agent_id] = network
                
                # Update tracking stats
                self._network_stats['total_reused'] += 1
                self._network_stats['current_in_use'] = len(self._attention_networks_in_use)
                self._network_stats['current_in_pool'] = len(self._attention_network_pool)
                
                # Update peak usage if needed
                if self._network_stats['current_in_use'] > self._network_stats['peak_networks_in_use']:
                    self._network_stats['peak_networks_in_use'] = self._network_stats['current_in_use']
                
                print(f"♻️ Reused Attention Network for agent {agent_id[:8]} (pool: {len(self._attention_network_pool)}, reused: {self._network_stats['total_reused']})")
                return network
            
            # Pool is empty, create a new network
            from .attention_deep_q_learning import AttentionDeepQLearning
            
            self._network_creation_count += 1
            self._network_stats['total_created'] += 1
            
            network = AttentionDeepQLearning(
                state_dim=5,  # Fixed: [arm_angle, elbow_angle, food_distance, food_direction, ground_contact]
                action_dim=6,  # Fixed: Must match agent action count (not 9)
                learning_rate=0.001
            )
            
            self._attention_networks_in_use[agent_id] = network
            
            # Update tracking stats
            self._network_stats['current_in_use'] = len(self._attention_networks_in_use)
            self._network_stats['current_in_pool'] = len(self._attention_network_pool)
            
            # Update peak usage if needed
            if self._network_stats['current_in_use'] > self._network_stats['peak_networks_in_use']:
                self._network_stats['peak_networks_in_use'] = self._network_stats['current_in_use']
            
            # Update GPU stats after creating network
            self._update_gpu_stats()
            
            print(f"🧠 NEW Attention Network #{self._network_creation_count} for agent {agent_id[:8]} (created: {self._network_stats['total_created']}, GPU: {self._gpu_stats['current_memory_mb']}MB)")
            
            return network
            
        except Exception as e:
            print(f"❌ Error acquiring attention network for agent {agent_id}: {e}")
            return None
    
    def _return_attention_network(self, agent_id: str):
        """Return an attention network to the pool when agent no longer needs it."""
        try:
            if agent_id not in self._attention_networks_in_use:
                return  # Agent doesn't have a network
            
            network = self._attention_networks_in_use.pop(agent_id)
            
            # Update stats - network no longer in use
            self._network_stats['current_in_use'] = len(self._attention_networks_in_use)
            
            # Only return to pool if we have space and network is valid
            if (len(self._attention_network_pool) < self._attention_network_pool_max_size and 
                network is not None):
                
                # Reset network state for reuse (optional - could preserve some learning)
                try:
                    # Clear attention history but keep learned weights
                    if hasattr(network, 'attention_history'):
                        network.attention_history.clear()
                except Exception as e:
                    print(f"⚠️ Error resetting network state: {e}")
                
                self._attention_network_pool.append(network)
                
                # Update pool stats
                self._network_stats['current_in_pool'] = len(self._attention_network_pool)
                
                print(f"♻️ Returned Attention Network to pool from agent {agent_id[:8]} (pool: {len(self._attention_network_pool)}, in_use: {self._network_stats['current_in_use']})")
            else:
                # Pool is full or network is invalid, destroy it
                try:
                    if hasattr(network, 'device') and 'cuda' in str(network.device):
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    del network
                    
                    # Update destruction stats
                    self._network_stats['networks_destroyed'] += 1
                    
                    # Update GPU stats after destroying network
                    self._update_gpu_stats()
                    
                    print(f"🧹 Destroyed excess Attention Network from agent {agent_id[:8]} (destroyed: {self._network_stats['networks_destroyed']}, GPU: {self._gpu_stats['current_memory_mb']}MB)")
                except Exception as e:
                    print(f"⚠️ Error destroying attention network: {e}")
                    
        except Exception as e:
            print(f"⚠️ Error returning attention network for agent {agent_id}: {e}")
    
    def get_attention_pool_stats(self) -> Dict[str, int]:
        """Get statistics about the attention network pool."""
        return {
            'networks_in_pool': len(self._attention_network_pool),
            'networks_in_use': len(self._attention_networks_in_use), 
            'total_networks_created': self._network_creation_count,
            'pool_max_size': self._attention_network_pool_max_size
        }
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics including network creation and GPU usage."""
        # Update current stats
        self._network_stats['current_in_use'] = len(self._attention_networks_in_use)
        self._network_stats['current_in_pool'] = len(self._attention_network_pool)
        self._update_gpu_stats()
        
        return {
            'network_stats': self._network_stats.copy(),
            'gpu_stats': self._gpu_stats.copy(),
            'efficiency_metrics': {
                'reuse_rate': (self._network_stats['total_reused'] / 
                              max(1, self._network_stats['total_created'] + self._network_stats['total_reused']) * 100),
                'memory_efficiency': (self._gpu_stats['current_memory_mb'] - self._gpu_stats['initial_memory_mb']),
                'pool_utilization': (len(self._attention_network_pool) / self._attention_network_pool_max_size * 100)
            }
        }
    
    def log_resource_usage(self, force: bool = False):
        """Log network creation and GPU usage statistics periodically."""
        import time
        
        # Log every 60 seconds or when forced
        current_time = time.time()
        if not force and current_time - getattr(self, '_last_log_time', 0) < 60:
            return
        
        self._last_log_time = current_time
        stats = self.get_comprehensive_stats()
        
        print(f"""
📊 ATTENTION NETWORK POOL STATS:
   🧠 Networks: Created={stats['network_stats']['total_created']}, Reused={stats['network_stats']['total_reused']}, Destroyed={stats['network_stats']['networks_destroyed']}
   🔄 Current: In Use={stats['network_stats']['current_in_use']}, In Pool={stats['network_stats']['current_in_pool']}, Peak={stats['network_stats']['peak_networks_in_use']}
   📈 Efficiency: Reuse Rate={stats['efficiency_metrics']['reuse_rate']:.1f}%, Pool Utilization={stats['efficiency_metrics']['pool_utilization']:.1f}%
   
🖥️ GPU MEMORY USAGE:
   💾 Current: {stats['gpu_stats']['current_memory_mb']}MB, Peak: {stats['gpu_stats']['peak_memory_mb']}MB
   📊 Memory Growth: +{stats['efficiency_metrics']['memory_efficiency']:.1f}MB since startup
""")
    
    def reset_tracking_stats(self):
        """Reset network creation and GPU tracking statistics."""
        self._network_stats = {
            'total_created': 0,
            'total_reused': 0,
            'current_in_use': len(self._attention_networks_in_use),
            'current_in_pool': len(self._attention_network_pool),
            'peak_networks_in_use': 0,
            'networks_destroyed': 0
        }
        
        # Reset GPU baseline
        self._update_gpu_stats()
        self._gpu_stats['initial_memory_mb'] = self._gpu_stats['current_memory_mb']
        self._gpu_stats['peak_memory_mb'] = self._gpu_stats['current_memory_mb']
        
        print("🔄 Network and GPU tracking statistics reset")

    def _update_gpu_stats(self):
        """Update GPU usage statistics."""
        try:
            import time
            
            # Try to get CUDA GPU memory usage first
            current_memory_mb = 0
            try:
                import torch
                if torch.cuda.is_available():
                    # Get GPU memory usage in MB
                    current_memory_mb = int(torch.cuda.memory_allocated() / (1024**2))
                    
                    # Set initial baseline on first call
                    if self._gpu_stats['initial_memory_mb'] == 0:
                        self._gpu_stats['initial_memory_mb'] = current_memory_mb
            except ImportError:
                # Fallback to nvidia-ml-py if available
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    current_memory_mb = int(info.used / (1024**2))
                    
                    if self._gpu_stats['initial_memory_mb'] == 0:
                        self._gpu_stats['initial_memory_mb'] = current_memory_mb
                except:
                    pass  # No GPU monitoring available
            
            # Update peak memory usage
            if current_memory_mb > self._gpu_stats['peak_memory_mb']:
                self._gpu_stats['peak_memory_mb'] = current_memory_mb
            
            # Update current memory usage
            self._gpu_stats['current_memory_mb'] = current_memory_mb
            
            # Update last update time
            self._gpu_stats['last_update_time'] = int(time.time())
            
        except Exception as e:
            print(f"⚠️ Error updating GPU stats: {e}")
        
        return self._gpu_stats
    
    def _cleanup_agent_deep_learning_data(self, agent, deep_q_learner):
        """Clean up accumulated deep learning data for an agent."""
        try:
            # Clean up attention network data if present
            if hasattr(agent, '_attention_dqn') and agent._attention_dqn:
                if hasattr(agent._attention_dqn, '_cleanup_attention_data'):
                    agent._attention_dqn._cleanup_attention_data()
                
                # Limit experience replay buffer size
                if hasattr(agent._attention_dqn, 'memory'):
                    buffer = agent._attention_dqn.memory
                    if hasattr(buffer, 'buffer') and len(buffer.buffer) > 15000:  # Limit to 15k experiences
                        # Remove oldest 25% of experiences
                        remove_count = len(buffer.buffer) // 4
                        for _ in range(remove_count):
                            if buffer.buffer:
                                buffer.buffer.popleft()
                        print(f"🧹 Trimmed attention network buffer for agent {agent.id[:8]}: removed {remove_count} old experiences")
            
            # Clean up standard deep Q-learning data
            if hasattr(deep_q_learner, 'memory'):
                buffer = deep_q_learner.memory
                if hasattr(buffer, 'buffer') and len(buffer.buffer) > 15000:  # Limit to 15k experiences
                    remove_count = len(buffer.buffer) // 4
                    for _ in range(remove_count):
                        if buffer.buffer:
                            buffer.buffer.popleft()
                    print(f"🧹 Trimmed deep Q-learning buffer for agent {agent.id[:8]}: removed {remove_count} old experiences")
            
        except Exception as e:
            print(f"⚠️ Error cleaning up deep learning data for agent {agent.id}: {e}") 