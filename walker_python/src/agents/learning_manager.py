"""
Flexible Learning Manager
Allows dynamic switching between different Q-learning approaches for individual robots.
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from .q_table import EnhancedQTable, SparseQTable
from .survival_q_integration_patch import SurvivalAwareQLearning


class LearningApproach(Enum):
    """Available learning approaches."""
    BASIC_Q_LEARNING = "basic_q_learning"
    ENHANCED_Q_LEARNING = "enhanced_q_learning"
    SURVIVAL_Q_LEARNING = "survival_q_learning"
    DEEP_Q_LEARNING = "deep_q_learning"  # For future use


class LearningManager:
    """
    Manages different learning approaches for individual robots.
    Allows dynamic switching between approaches during training.
    """
    
    def __init__(self, ecosystem_interface=None):
        self.ecosystem_interface = ecosystem_interface
        
        # Track which approach each agent is using
        self.agent_approaches: Dict[str, LearningApproach] = {}
        
        # Store learning adapters for each agent
        self.agent_adapters: Dict[str, Any] = {}
        
        # Store original Q-tables for fallback
        self.agent_original_qtables: Dict[str, Any] = {}
        
        # Deep learning training coordination
        self._agents_currently_training: set = set()
        
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
            }
        }
        
        print("🎛️ LearningManager initialized - Ready for flexible approach switching")
    
    def get_agent_approach(self, agent_id: str) -> LearningApproach:
        """Get the current learning approach for an agent."""
        return self.agent_approaches.get(agent_id, LearningApproach.ENHANCED_Q_LEARNING)
    
    def set_agent_approach(self, agent, new_approach: LearningApproach) -> bool:
        """
        Switch an agent to a new learning approach.
        
        Args:
            agent: The agent to switch
            new_approach: The new learning approach to use
            
        Returns:
            bool: True if switch was successful, False otherwise
        """
        try:
            agent_id = agent.id
            current_approach = self.get_agent_approach(agent_id)
            
            if current_approach == new_approach:
                print(f"🔄 Agent {agent_id} already using {new_approach.value}")
                return True
            
            print(f"🔄 Switching Agent {agent_id}: {current_approach.value} → {new_approach.value}")
            
            # Store current Q-table for knowledge transfer (always update to get latest learned data)
            self.agent_original_qtables[agent_id] = agent.q_table
            
            # Perform the switch
            success = self._perform_approach_switch(agent, current_approach, new_approach)
            
            if success:
                self.agent_approaches[agent_id] = new_approach
                self._update_performance_tracking(agent_id, current_approach, new_approach)
                print(f"✅ Agent {agent_id} successfully switched to {new_approach.value}")
                return True
            else:
                print(f"❌ Failed to switch Agent {agent_id} to {new_approach.value}")
                return False
                
        except Exception as e:
            print(f"❌ Error switching Agent {agent_id} to {new_approach.value}: {e}")
            return False
    
    def _perform_approach_switch(self, agent, from_approach: LearningApproach, 
                                to_approach: LearningApproach) -> bool:
        """Perform the actual switching between learning approaches."""
        
        agent_id = agent.id
        
        try:
            # Clean up current approach
            self._cleanup_current_approach(agent, from_approach)
            
            # Set up new approach
            if to_approach == LearningApproach.BASIC_Q_LEARNING:
                return self._setup_basic_q_learning(agent)
            
            elif to_approach == LearningApproach.ENHANCED_Q_LEARNING:
                return self._setup_enhanced_q_learning(agent)
            
            elif to_approach == LearningApproach.SURVIVAL_Q_LEARNING:
                return self._setup_survival_q_learning(agent)
            
            elif to_approach == LearningApproach.DEEP_Q_LEARNING:
                return self._setup_deep_q_learning(agent)
            
            else:
                print(f"⚠️ Unknown learning approach: {to_approach}")
                return False
                
        except Exception as e:
            print(f"❌ Error performing approach switch: {e}")
            return False
    
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
            
            # Clean up survival-specific attributes
            if hasattr(agent, '_prev_survival_state'):
                delattr(agent, '_prev_survival_state')
            if hasattr(agent, '_prev_survival_action'):
                delattr(agent, '_prev_survival_action')
            if hasattr(agent, 'get_survival_stats'):
                delattr(agent, 'get_survival_stats')
                
        except Exception as e:
            print(f"⚠️ Error cleaning up approach for agent {agent_id}: {e}")
    
    def _setup_basic_q_learning(self, agent) -> bool:
        """Set up basic Q-learning approach."""
        try:
            # Use simple sparse Q-table
            agent.q_table = SparseQTable(
                action_count=agent.action_size,
                default_value=0.0
            )
            
            # Reset learning parameters to basic values
            agent.learning_rate = 0.1
            agent.epsilon = 0.3
            agent.epsilon_decay = 0.999
            
            # Transfer some knowledge from original Q-table if available
            if agent.id in self.agent_original_qtables:
                self._transfer_basic_knowledge(agent, self.agent_original_qtables[agent.id])
            
            return True
            
        except Exception as e:
            print(f"❌ Error setting up basic Q-learning: {e}")
            return False
    
    def _setup_enhanced_q_learning(self, agent) -> bool:
        """Set up enhanced Q-learning approach."""
        try:
            # Use enhanced Q-table with all the sophisticated features
            agent.q_table = EnhancedQTable(
                action_count=agent.action_size,
                default_value=0.0,
                confidence_threshold=15,
                exploration_bonus=0.15
            )
            
            # Enhanced learning parameters
            agent.learning_rate = 0.05
            agent.epsilon = 0.3
            agent.epsilon_decay = 0.9999
            
            # Transfer knowledge from original Q-table
            if agent.id in self.agent_original_qtables:
                self._transfer_enhanced_knowledge(agent, self.agent_original_qtables[agent.id])
            
            return True
            
        except Exception as e:
            print(f"❌ Error setting up enhanced Q-learning: {e}")
            return False
    
    def _setup_survival_q_learning(self, agent) -> bool:
        """Set up survival Q-learning approach."""
        try:
            if not self.ecosystem_interface:
                print("❌ Cannot setup survival Q-learning: No ecosystem interface")
                return False
            
            from .survival_q_integration_patch import upgrade_agent_to_survival_learning
            
            # Upgrade agent to survival learning
            survival_adapter = upgrade_agent_to_survival_learning(agent, self.ecosystem_interface)
            self.agent_adapters[agent.id] = survival_adapter
            
            # Transfer knowledge from original Q-table if available
            if agent.id in self.agent_original_qtables:
                self._transfer_survival_knowledge(agent, self.agent_original_qtables[agent.id])
            
            return True
            
        except Exception as e:
            print(f"❌ Error setting up survival Q-learning: {e}")
            return False
    
    def _setup_deep_q_learning(self, agent) -> bool:
        """Set up deep Q-learning approach with GPU acceleration."""
        try:
            from .deep_survival_q_learning import DeepSurvivalQLearning, TORCH_AVAILABLE
            
            if not TORCH_AVAILABLE:
                print("❌ PyTorch not available. Cannot use Deep Q-Learning. Falling back to Enhanced Q-Learning.")
                return self._setup_enhanced_q_learning(agent)
            
            # Create Deep Q-Learning instance with GPU support
            state_dim = 15  # Continuous state vector dimensions (matches get_continuous_state_vector output)
            action_dim = agent.action_size
            
            deep_q_learner = DeepSurvivalQLearning(
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate=3e-4,
                gamma=0.99,
                epsilon_start=1.0,
                epsilon_end=0.01,
                epsilon_decay=150000,  # Slower decay
                buffer_size=25000,     # Reduced from 100k to 25k
                batch_size=32,         # Reduced from 64 to 32  
                target_update_freq=2000,  # Less frequent updates
                device="auto",  # Auto-detect GPU
                use_dueling=True,
                use_prioritized_replay=True
            )
            
                            # Store the deep learner in agent adapters
            self.agent_adapters[agent.id] = deep_q_learner
            
            # Initialize experience collection (lightweight) vs training (heavy) separation
            agent._deep_step_count = 0
            agent._deep_experience_collection_freq = 10    # Collect experience every 10 steps (reduced frequency)
            agent._deep_training_freq = 15000             # Train every 15,000 steps (much less frequent) 
            agent._deep_min_buffer_size = 5000            # Minimum 5k experiences (reduced from 10k)
            agent._deep_ready_to_train = False            # No training until buffer is full
            agent._deep_prev_state = None
            agent._deep_prev_action = None                # Initialize previous action
            agent._deep_prev_reward = 0.0                 # Initialize previous reward
            
            # Store original step method and replace with deep learning step
            if not hasattr(agent, '_original_step_method'):
                agent._original_step_method = agent.step
            
            # Create wrapper for efficient Deep Q-Learning with separated experience collection and training
            def deep_learning_step(dt: float):
                try:
                    # CRITICAL: Always run original simulation step first 
                    result = agent._original_step_method(dt)
                    
                    # Initialize deep learning attributes if needed
                    if not hasattr(agent, '_deep_step_count'):
                        agent._deep_step_count = 0
                        agent._deep_experience_collection_freq = 10
                        agent._deep_training_freq = 15000
                        agent._deep_min_buffer_size = 5000
                        agent._deep_ready_to_train = False
                        agent._deep_prev_state = None
                        agent._deep_prev_action = None
                        agent._deep_prev_reward = 0.0
                        print(f"🔧 Deep Q-Learning attributes initialized for Agent {agent.id}")
                    
                    agent._deep_step_count += 1
                    
                    # PHASE 1: Lightweight experience collection (every few steps)
                    if agent._deep_step_count % agent._deep_experience_collection_freq == 0:
                        try:
                            # Quick state collection (minimal computation)
                            current_state = [
                                getattr(agent.body, 'angle', 0.0),                              # body angle
                                agent.body.linearVelocity.x if hasattr(agent.body, 'linearVelocity') else 0.0,  # velocity x
                                agent.body.linearVelocity.y if hasattr(agent.body, 'linearVelocity') else 0.0,  # velocity y
                                agent.body.position.x if hasattr(agent.body, 'position') else 0.0,              # position x
                                agent.body.position.y if hasattr(agent.body, 'position') else 0.0,              # position y
                            ]
                            
                            # Choose action (neural network inference - relatively fast)
                            if len(current_state) == 5:  # Basic state, extend to 15 dimensions
                                extended_state_list = current_state + [0.0] * 10  # Pad to 15 dimensions
                                extended_state_array = np.array(extended_state_list, dtype=np.float32)
                                action_idx = deep_q_learner.choose_action(extended_state_array)
                                
                                # Set action on agent (lightweight)
                                if hasattr(agent, 'actions') and action_idx is not None and action_idx < len(agent.actions):
                                    agent.current_action = action_idx
                                    agent.current_action_tuple = agent.actions[action_idx]
                                
                                # Store experience if we have previous state (lightweight)
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
                                    
                                    # Debug: Log experience storage (every 100 experiences)
                                    if hasattr(deep_q_learner, 'memory') and len(deep_q_learner.memory) % 100 == 0:
                                        print(f"💾 Agent {agent.id}: Stored experience #{len(deep_q_learner.memory)} (reward: {reward:.3f})")
                                
                                # Store current state for next step
                                agent._deep_prev_state = extended_state_array
                                agent._deep_prev_action = action_idx
                                agent._deep_prev_reward = getattr(agent, 'total_reward', 0.0)
                                
                        except Exception as e:
                            # Log experience collection errors instead of silent fail
                            print(f"❌ Experience collection error Agent {agent.id}: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    # PHASE 2: Heavy training (very infrequent, only when buffer is full)
                    if not agent._deep_ready_to_train:
                        # Check if buffer is ready for training
                        if hasattr(deep_q_learner, 'memory') and len(deep_q_learner.memory) >= agent._deep_min_buffer_size:
                            agent._deep_ready_to_train = True
                            print(f"🎯 TRAINING READY: Agent {agent.id} buffer filled ({len(deep_q_learner.memory):,} experiences)")
                            print(f"   🔥 GPU training will begin every {agent._deep_training_freq:,} steps")
                        else:
                            # Periodic buffer status update (every 1000 steps)
                            if hasattr(deep_q_learner, 'memory') and agent._deep_step_count % 1000 == 0:
                                current_buffer_size = len(deep_q_learner.memory)
                                progress = (current_buffer_size / agent._deep_min_buffer_size) * 100
                                print(f"📊 Buffer Progress: Agent {agent.id} - {current_buffer_size:,}/{agent._deep_min_buffer_size:,} ({progress:.1f}%)")
                    
                    # Heavy training phase (very infrequent) with training coordination
                    if (agent._deep_ready_to_train and 
                        agent._deep_step_count % agent._deep_training_freq == 0 and
                        hasattr(deep_q_learner, 'memory') and 
                        len(deep_q_learner.memory) >= deep_q_learner.batch_size):
                        
                        # Training coordination - prevent multiple agents training simultaneously
                        # (initialized in __init__ method)
                        
                        # Skip training if too many agents are already training
                        if len(self._agents_currently_training) >= 2:  # Max 2 agents training simultaneously
                            # Defer training by small random amount
                            import random
                            agent._deep_step_count -= random.randint(100, 500)
                            return result
                        
                        # Add this agent to training set
                        self._agents_currently_training.add(agent.id)
                        
                        try:
                            import time
                            import threading
                            buffer_size = len(deep_q_learner.memory)
                            batch_size = deep_q_learner.batch_size
                            
                            print(f"🔥 GPU TRAINING START: Agent {agent.id} (step {agent._deep_step_count})")
                            print(f"   📊 Buffer: {buffer_size:,} experiences | Batch: {batch_size}")
                            
                            training_start_time = time.time()
                            
                            # BACKGROUND THREADING: Move heavy training to background thread
                            def background_training():
                                try:
                                    # Heavy neural network training (GPU intensive)
                                    learning_stats = deep_q_learner.learn()
                                    
                                    training_duration = time.time() - training_start_time
                                    print(f"✅ GPU TRAINING COMPLETE: Agent {agent.id}")
                                    print(f"   ⏱️  Duration: {training_duration:.3f}s | Loss: {learning_stats.get('loss', 'N/A')}")
                                    
                                    # Memory cleanup handled automatically by buffer maxlen
                                except Exception as e:
                                    print(f"❌ GPU TRAINING FAILED: Agent {agent.id} - {e}")
                                finally:
                                    # Remove from training set
                                    self._agents_currently_training.discard(agent.id)
                            
                            # Start background training (non-blocking)
                            training_thread = threading.Thread(target=background_training, daemon=True)
                            training_thread.start()
                                
                        except Exception as e:
                            print(f"❌ GPU TRAINING FAILED: Agent {agent.id} - {e}")
                            self._agents_currently_training.discard(agent.id)
                    
                    return result
                    
                except Exception as e:
                    # Always fall back to original simulation
                    return agent._original_step_method(dt)
            
            # Replace the step method
            agent.step = deep_learning_step
            
            # Transfer knowledge from original Q-table if available
            if agent.id in self.agent_original_qtables:
                self._transfer_deep_knowledge(agent, self.agent_original_qtables[agent.id])
            
            print(f"🧠 Agent {agent.id} now using Deep Q-Learning with GPU acceleration")
            return True
            
        except Exception as e:
            print(f"❌ Error setting up deep Q-learning: {e}")
            # Fall back to enhanced Q-learning if deep learning fails
            return self._setup_enhanced_q_learning(agent)
    
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