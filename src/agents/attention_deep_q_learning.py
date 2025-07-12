import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import time
import logging
from typing import Tuple, List, Dict, Any, Union, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleAttention(nn.Module):
    """Simplified attention mechanism for arm-based food-seeking behavior."""
    
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = np.sqrt(self.head_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Handle single state input - add sequence dimension if needed
        if len(x.shape) == 2:  # [batch_size, embed_dim]
            x = x.unsqueeze(1)  # [batch_size, 1, embed_dim]
        
        seq_len = x.size(1)
        
        # Self-attention: query, key, value are all the same input
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads and apply output linear layer
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )
        
        output = self.out_linear(attended)
        
        # If input was single state, squeeze back to [batch_size, embed_dim]
        if seq_len == 1:
            output = output.squeeze(1)
        
        return output, attention_weights

class EnhancedRobotEncoder(nn.Module):
    """Enhanced encoder for expanded robot state representation (29 dimensions with ray sensing)."""
    
    def __init__(self, state_size: int = 29, embed_dim: int = 128):
        super().__init__()
        
        # Strict validation - no silent fixes
        if state_size != 29:
            raise ValueError(f"EnhancedRobotEncoder requires exactly 29 state dimensions, got {state_size}. "
                           f"This encoder is specifically designed for robot state with ray sensing. "
                           f"Use a different encoder or fix the state representation.")
        
        if embed_dim != 128:
            raise ValueError(f"EnhancedRobotEncoder requires exactly 128 embed dimensions, got {embed_dim}. "
                           f"The feature encoders are hard-coded to sum to 128 dimensions.")
        
        self.embed_dim = embed_dim
        self.state_size = state_size
        
        # Specialized encoders for different feature groups (dimensions add up to exactly embed_dim)
        self.joint_encoder = nn.Linear(4, 21)      # Joint angles + velocities
        self.body_encoder = nn.Linear(6, 21)       # Body position, velocity, orientation
        self.food_encoder = nn.Linear(4, 21)       # Food targeting information
        self.physics_encoder = nn.Linear(3, 16)    # Physics feedback
        self.action_encoder = nn.Linear(2, 16)     # Action history
        self.ray_encoder = nn.Linear(10, 33)       # Ray sensing data (5 rays × 2 values) - extra 1 dim to reach 128 total
        
        # Verify encoder dimensions sum to embed_dim
        total_dims = 21 + 21 + 21 + 16 + 16 + 33
        if total_dims != embed_dim:
            raise ValueError(f"Encoder dimensions sum to {total_dims}, expected {embed_dim}")
        
        # Feature combination layers
        self.feature_combiner = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim)
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, state):
        """
        Encode expanded robot state: 29 dimensions total
        - [0:4]: Joint angles and velocities
        - [4:10]: Body state (position, velocity, orientation)
        - [10:14]: Food targeting information
        - [14:17]: Physics feedback
        - [17:19]: Action history
        - [19:29]: Ray sensing data (5 rays × 2 values each)
        """
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        # STRICT: Verify state dimensions - NO AUTO-FIXING
        if state.shape[1] != self.state_size:
            raise ValueError(f"EnhancedRobotEncoder expected {self.state_size}D state, got {state.shape[1]}D state! "
                           f"Shape: {state.shape}. This indicates a bug in state generation - fix the source instead of padding!")
        
        # Validate state contains valid numbers
        if torch.isnan(state).any() or torch.isinf(state).any():
            raise ValueError(f"State contains NaN or Inf values: {state}")
        
        # Split state into feature groups
        joint_state = state[:, :4]       # Joint angles and velocities
        body_state = state[:, 4:10]      # Body state
        food_state = state[:, 10:14]     # Food targeting
        physics_state = state[:, 14:17]  # Physics feedback
        action_state = state[:, 17:19]   # Action history
        ray_state = state[:, 19:29]      # Ray sensing data
        
        # Encode each feature group
        joint_embed = F.relu(self.joint_encoder(joint_state))
        body_embed = F.relu(self.body_encoder(body_state))
        food_embed = F.relu(self.food_encoder(food_state))
        physics_embed = F.relu(self.physics_encoder(physics_state))
        action_embed = F.relu(self.action_encoder(action_state))
        ray_embed = F.relu(self.ray_encoder(ray_state))
        
        # Combine all features
        combined = torch.cat([joint_embed, body_embed, food_embed, physics_embed, action_embed, ray_embed], dim=-1)
        
        # Verify combined dimensions
        if combined.shape[1] != self.embed_dim:
            raise RuntimeError(f"Combined features have {combined.shape[1]} dimensions, expected {self.embed_dim}")
        
        combined = self.feature_combiner(combined)
        combined = self.layer_norm(combined)
        
        return combined

class EnhancedRobotAttentionDQN(nn.Module):
    """Enhanced Dueling DQN with attention for full robot state representation."""
    
    def __init__(self, state_size: int = 29, action_size: int = 15, embed_dim: int = 128, num_heads: int = 4):
        super().__init__()
        
        # Strict validation - no silent fixes
        if state_size != 29:
            raise ValueError(f"EnhancedRobotAttentionDQN requires exactly 29 state dimensions, got {state_size}")
        
        # FIXED: Support variable action dimensions for different agent morphologies
        if action_size < 5 or action_size > 100:
            raise ValueError(f"EnhancedRobotAttentionDQN action dimensions must be between 5 and 100, got {action_size}")
        
        if embed_dim != 128:
            raise ValueError(f"EnhancedRobotAttentionDQN requires exactly 128 embed dimensions, got {embed_dim}")
        
        self.state_size = state_size
        self.action_size = action_size
        self.embed_dim = embed_dim
        
        # Enhanced robot state encoder
        self.feature_encoder = EnhancedRobotEncoder(state_size, embed_dim)
        
        # Attention layer
        self.attention = SimpleAttention(embed_dim, num_heads)
        
        # Feed-forward processing
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Dueling heads
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )
        
        self.advantage_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, action_size)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, state):
        # Encode arm control features
        features = self.feature_encoder(state)
        
        # Apply attention to focus on relevant arm/target features
        attended_features, attention_weights = self.attention(features)
        features = self.norm1(features + attended_features)
        
        # Feed-forward processing
        ff_output = self.ff(features)
        features = self.norm2(features + ff_output)
        
        # Ensure features are properly shaped [batch_size, embed_dim]
        if len(features.shape) > 2:
            features = features.squeeze(1)  # Remove sequence dimension if present
        
        # Dueling Q-value computation
        value = self.value_head(features)
        advantages = self.advantage_head(features)
        
        # Combine value and advantages
        q_values = value + (advantages - advantages.mean(dim=-1, keepdim=True))
        
        return q_values, {
            'attention_weights': attention_weights,
            'features': features
        }

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer for enhanced learning."""
    
    def __init__(self, capacity: int = 2000, alpha: float = 0.6):
        if capacity <= 0:
            raise ValueError(f"Buffer capacity must be positive, got {capacity}")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")
            
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
    
    def add(self, state, action, reward, next_state, done):
        """Add experience with maximum priority."""
        # Validate inputs
        if state is None or next_state is None:
            raise ValueError("State and next_state cannot be None")
        
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            # Buffer not full yet, just append
            self.buffer.append(experience)
            self.priorities[len(self.buffer) - 1] = self.max_priority
        else:
            # Buffer is full, overwrite at current position
            self.buffer[self.position] = experience
            self.priorities[self.position] = self.max_priority
            self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4):
        """Sample batch with prioritized sampling."""
        if len(self.buffer) == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {batch_size}")
        
        # Ensure we don't sample more than available
        actual_batch_size = min(batch_size, len(self.buffer))
        
        # Get priorities for existing buffer items
        buffer_size = len(self.buffer)
        prios = self.priorities[:buffer_size]
        
        # Ensure priorities are valid (no zeros)
        prios = np.maximum(prios, 1e-6)
        probs = prios ** self.alpha
        
        if np.sum(probs) == 0:
            raise ValueError("All priorities are zero")
        
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(buffer_size, actual_batch_size, p=probs, replace=True)
        experiences = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        weights = (buffer_size * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return experiences, indices, weights
    
    def update_priorities(self, indices, priorities):
        """Update priorities for given indices."""
        if len(indices) != len(priorities):
            raise ValueError(f"Indices and priorities must have same length: {len(indices)} vs {len(priorities)}")
        
        for idx, priority in zip(indices, priorities):
            if not (0 <= idx < len(self.buffer)):
                raise IndexError(f"Index {idx} out of range for buffer size {len(self.buffer)}")
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)
    
    def __iter__(self):
        """Make buffer iterable for compatibility."""
        return iter(self.buffer)

class AttentionDeepQLearning:
    """Enhanced Deep Q-Learning with Double DQN, Prioritized Replay, and attention."""
    
    def __init__(self, state_dim: int = 29, action_dim: int = 15, learning_rate: float = 0.001):
        """
        Initialize Enhanced Deep Q-Learning with Attention for Robot Crawling.
        
        CRITICAL PARAMETERS:
        ===================
        state_dim: Must be 29 (matches CrawlingAgent.get_state_representation() with ray sensing)
        action_dim: Must be 15 (matches CrawlingAgent locomotion action space)
        
        This ensures dimension consistency between robot state generation 
        and neural network expectations. Any mismatch will cause training errors.
        """
        # STRICT VALIDATION - no silent fixes
        if state_dim != 29:
            raise ValueError(f"AttentionDeepQLearning requires exactly 29 state dimensions, got {state_dim}. "
                           f"This must match agent.get_state_representation() with ray sensing.")
        
        # FIXED: Support variable action dimensions for different agent morphologies
        if action_dim < 5 or action_dim > 100:
            raise ValueError(f"AttentionDeepQLearning action dimensions must be between 5 and 100, got {action_dim}. "
                           f"This allows for different robot morphologies with varying joint counts.")
        
        if learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive, got {learning_rate}")
            
        # Initialize base Deep Q-Learning attributes
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Enhanced Q-Learning hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9995
        self.batch_size = 32
        self.target_update_freq = 500
        self.steps_done = 0
        
        # NEW: Training run counter
        self.training_runs = 0
        self.last_training_time = 0.0
        
        # Epsilon cycling to prevent permanent local minima
        self.epsilon_cycle_steps = 10000
        self.epsilon_reset_value = 0.3
        self.last_epsilon_reset = 0
        
        # Prioritized Experience Replay
        self.use_prioritized_replay = True
        self.memory: Union[PrioritizedReplayBuffer, deque] = PrioritizedReplayBuffer(capacity=2000, alpha=0.6)
        self.beta = 0.4
        self.beta_increment = 0.001
        
        # Enhanced robot attention-based network
        self.q_network = EnhancedRobotAttentionDQN(state_dim, action_dim).to(self.device)
        self.target_network = EnhancedRobotAttentionDQN(state_dim, action_dim).to(self.device)
        
        # Update optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.attention_history = deque(maxlen=200)
        
        logger.info(f"Attention-based Deep Q-Learning initialized with {self.device}")
        logger.info(f"Network parameters: {sum(p.numel() for p in self.q_network.parameters()):,}")
        logger.info(f"Attention history: {self.attention_history.maxlen} records")
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        # Validate inputs
        if state is None or next_state is None:
            raise ValueError("State and next_state cannot be None")
        
        if not isinstance(action, int) or action < 0 or action >= self.action_dim:
            raise ValueError(f"Action must be integer between 0 and {self.action_dim-1}, got {action}")
        
        # Store based on memory type
        if isinstance(self.memory, PrioritizedReplayBuffer):
            self.memory.add(state, action, reward, next_state, done)
        elif isinstance(self.memory, deque):
            self.memory.append((state, action, reward, next_state, done))
        else:
            raise TypeError(f"Unsupported memory type: {type(self.memory)}")
    
    def choose_action(self, state_vector: np.ndarray, agent_data: Optional[Dict[str, Any]] = None) -> int:
        """Choose action with epsilon-greedy policy."""
        if state_vector is None:
            raise ValueError("State vector cannot be None")
        
        if len(state_vector) != self.state_dim:
            raise ValueError(f"State vector must have {self.state_dim} dimensions, got {len(state_vector)}")
        
        current_time = time.time()
        
        # Simple epsilon-greedy exploration
        if np.random.random() > self.epsilon:
            self.q_network.eval()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
                q_values, attention_info = self.q_network(state_tensor)
                action = q_values.argmax().item()
                
                # Store attention information for analysis
                self.attention_history.append({
                    'attention_weights': attention_info['attention_weights'].cpu().numpy(),
                    'selected_action': action,
                    'timestamp': current_time
                })
            self.q_network.train()
            return action
        else:
            return np.random.randint(0, self.action_dim)
    
    def learn(self) -> Dict[str, float]:
        """Enhanced learning with Double DQN and Prioritized Replay."""
        if len(self.memory) < self.batch_size:
            logger.warning(f"Insufficient experience for learning: {len(self.memory)} < {self.batch_size}")
            return {}
        
        # Sample experiences
        try:
            if isinstance(self.memory, PrioritizedReplayBuffer):
                experiences, indices, weights = self.memory.sample(self.batch_size, self.beta)
                weights = torch.FloatTensor(weights).to(self.device)
                # Increase beta over time for importance sampling
                self.beta = min(1.0, self.beta + self.beta_increment)
            elif isinstance(self.memory, deque):
                experiences = random.sample(list(self.memory), self.batch_size)
                indices = None
                weights = torch.ones(self.batch_size).to(self.device)
            else:
                raise TypeError(f"Unsupported memory type: {type(self.memory)}")
        except Exception as e:
            logger.error(f"Failed to sample experiences: {e}")
            raise
        
        # Log training initialization (first time only)
        if not hasattr(self, '_first_training_logged'):
            logger.info(f"First training session - Memory: {len(self.memory)}, Batch: {self.batch_size}")
            logger.info(f"Using Double DQN: True")
            logger.info(f"Using Prioritized Replay: {isinstance(self.memory, PrioritizedReplayBuffer)}")
            self._first_training_logged = True
        
        # Prepare batch
        from collections import namedtuple
        Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])
        batch = Experience(*zip(*experiences))
        
        # Convert to tensors with proper error handling
        try:
            states = torch.FloatTensor(batch.state).to(self.device)
            actions = torch.LongTensor(batch.action).to(self.device)
            rewards = torch.FloatTensor(batch.reward).to(self.device)
            next_states = torch.FloatTensor(batch.next_state).to(self.device)
            dones = torch.BoolTensor(batch.done).to(self.device)
        except Exception as e:
            logger.error(f"Failed to convert batch to tensors: {e}")
            logger.error(f"Batch shapes - states: {np.array(batch.state).shape}, actions: {np.array(batch.action).shape}")
            raise ValueError(f"Tensor conversion failed: {e}. This indicates inconsistent state dimensions in experience buffer.")
        
        # Current Q-values with attention
        current_q_values, current_attention = self.q_network(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
        
        # DOUBLE DQN: Use main network to select actions, target network to evaluate
        with torch.no_grad():
            # Use main network to select best actions
            next_q_values_main, _ = self.q_network(next_states)
            next_actions = next_q_values_main.argmax(1)
            
            # Use target network to evaluate selected actions
            next_q_values_target, _ = self.target_network(next_states)
            next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute TD errors and loss
        td_errors = target_q_values.unsqueeze(1) - current_q_values
        loss = (weights.unsqueeze(1) * td_errors.pow(2)).mean()
        
        # Update priorities for prioritized replay
        if isinstance(self.memory, PrioritizedReplayBuffer) and indices is not None:
            try:
                priorities = torch.abs(td_errors).detach().cpu().numpy().flatten() + 1e-6
                self.memory.update_priorities(indices, priorities)
            except Exception as e:
                logger.error(f"Failed to update priorities: {e}")
                raise
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update epsilon for exploration decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Epsilon cycling to prevent permanent local minima
        if self.steps_done - self.last_epsilon_reset >= self.epsilon_cycle_steps:
            old_epsilon = self.epsilon
            self.epsilon = self.epsilon_reset_value
            self.last_epsilon_reset = self.steps_done
            logger.info(f"Epsilon cycling: {old_epsilon:.3f} → {self.epsilon:.3f} (step {self.steps_done})")
        
        # Update target network
        if self.steps_done % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.steps_done += 1
        
        # NEW: Increment training run counter and update timestamp
        self.training_runs += 1
        self.last_training_time = time.time()
        
        # Debug Q-values if they're consistently 0
        mean_q_val = current_q_values.mean().item()
        if abs(mean_q_val) < 1e-6 and self.training_runs % 50 == 0:
            print(f"⚠️ DEBUG: Q-values near zero! mean_q={mean_q_val:.6f}, network_weights_sum={sum(p.data.sum().item() for p in self.q_network.parameters()):.6f}")
            print(f"   Sample Q-values: {current_q_values[:3].flatten().detach().cpu().numpy()}")
            print(f"   Sample rewards: {rewards[:3].detach().cpu().numpy()}")
            print(f"   Sample states: {states[0][:5].detach().cpu().numpy()}")
        
        return {
            'loss': loss.item(),
            'mean_q_value': mean_q_val,
            'attention_entropy': self._calculate_attention_entropy(current_attention['attention_weights']),
            'epsilon': self.epsilon,
            'beta': self.beta if isinstance(self.memory, PrioritizedReplayBuffer) else 0.0,
            'mean_td_error': torch.abs(td_errors).mean().item(),
            'training_runs': self.training_runs  # NEW: Include training run count in return stats
        }
    
    def _calculate_attention_entropy(self, attention_weights):
        """Calculate entropy of attention weights."""
        if attention_weights is None:
            raise ValueError("Attention weights cannot be None")
        
        try:
            avg_attention = attention_weights.mean(dim=(0, 1))
            
            # Ensure no zeros for log calculation
            avg_attention = torch.clamp(avg_attention, min=1e-10)
            entropy = -(avg_attention * torch.log(avg_attention)).sum()
            
            if torch.isnan(entropy) or torch.isinf(entropy):
                raise ValueError(f"Invalid entropy calculation: {entropy}")
            
            return entropy.item()
        except Exception as e:
            logger.error(f"Failed to calculate attention entropy: {e}")
            raise
    
    def get_arm_control_state_representation(self, agent_data: Dict[str, Any]) -> np.ndarray:
        """
        DEPRECATED: This method returned 5-dimensional states which caused neural network errors.
        
        All agents now use standardized 29-dimensional state representation via 
        agent.get_state_representation() method for consistency.
        """
        raise DeprecationWarning(
            "get_arm_control_state_representation is deprecated. "
            "Use agent.get_state_representation() for 29-dimensional state consistency."
        )
    
    def get_attention_analysis(self) -> Dict[str, Any]:
        """Analyze attention patterns for arm control features."""
        if not self.attention_history:
            return {}
        
        recent_attention = list(self.attention_history)[-50:]  # Last 50 decisions
        
        try:
            # Average attention weights
            avg_attention = np.mean([a['attention_weights'] for a in recent_attention], axis=0)
            
            # Feature focus analysis for arm control
            feature_names = ['arm_angle', 'elbow_angle', 'food_distance', 'food_direction', 'ground_contact']
            
            # Calculate entropy for each attention pattern
            attention_entropies = []
            for a in recent_attention:
                weights = a['attention_weights']
                weights = np.clip(weights, 1e-10, 1.0)  # Prevent log(0)
                entropy = -np.sum(weights * np.log(weights))
                attention_entropies.append(entropy)
            
            return {
                'average_attention_weights': avg_attention.tolist(),
                'feature_names': feature_names,
                'attention_entropy': float(np.mean(attention_entropies)),
                'decisions_analyzed': len(recent_attention),
                'state_dimensions': self.state_dim,
                'focus_areas': {
                    'arm_control': np.mean(avg_attention[:, :, :2]) if len(avg_attention.shape) > 2 else 0.0,
                    'food_targeting': np.mean(avg_attention[:, :, 2:4]) if len(avg_attention.shape) > 2 else 0.0,
                    'physics_contact': np.mean(avg_attention[:, :, 4:5]) if len(avg_attention.shape) > 2 else 0.0
                }
            }
        except Exception as e:
            logger.error(f"Failed to analyze attention patterns: {e}")
            raise
    
    def _cleanup_attention_data(self):
        """Clean up accumulated attention data to prevent memory growth."""
        try:
            current_time = time.time()
            if len(self.attention_history) > 150:
                cutoff_time = current_time - 3600.0  # 1 hour old
                
                # Filter old entries
                old_size = len(self.attention_history)
                filtered_entries = [
                    entry for entry in list(self.attention_history)
                    if entry.get('timestamp', current_time) > cutoff_time
                ]
                
                # Clear and refill
                self.attention_history.clear()
                for entry in filtered_entries[-150:]:
                    self.attention_history.append(entry)
                
                if old_size - len(self.attention_history) > 50:
                    logger.info(f"Cleaned attention data: {len(self.attention_history)} records remaining "
                              f"(removed {old_size - len(self.attention_history)} old entries)")
            
        except Exception as e:
            logger.error(f"Failed to clean attention data: {e}")
            raise
    
    def reset_target_network_sync(self):
        """Reset target network synchronization for transferred networks."""
        try:
            # Force immediate target network sync with main network
            self.target_network.load_state_dict(self.q_network.state_dict())
            
            # Reset step counter to ensure proper target update timing
            self.steps_done = self.steps_done % self.target_update_freq
            
            logger.info(f"Target network resynced after transfer "
                       f"(next update in {self.target_update_freq - self.steps_done} steps)")
            
        except Exception as e:
            logger.error(f"Failed to reset target network sync: {e}")
            raise
    
    def get_network_sync_status(self) -> Dict[str, Any]:
        """Get information about main and target network synchronization."""
        try:
            steps_until_sync = self.target_update_freq - (self.steps_done % self.target_update_freq)
            
            # Compare a few parameters to check if networks are in sync
            main_params = list(self.q_network.parameters())
            target_params = list(self.target_network.parameters())
            
            param_differences = []
            for i, (main_param, target_param) in enumerate(zip(main_params[:3], target_params[:3])):
                diff = torch.abs(main_param - target_param).mean().item()
                param_differences.append(diff)
            
            avg_param_diff = np.mean(param_differences) if param_differences else 0.0
            
            return {
                'steps_done': self.steps_done,
                'target_update_freq': self.target_update_freq,
                'steps_until_next_sync': steps_until_sync,
                'last_sync_step': (self.steps_done // self.target_update_freq) * self.target_update_freq,
                'avg_parameter_difference': avg_param_diff,
                'networks_likely_synced': avg_param_diff < 1e-6
            }
            
        except Exception as e:
            logger.error(f"Failed to get network sync status: {e}")
            raise 