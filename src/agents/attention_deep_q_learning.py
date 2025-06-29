import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import time
from typing import Tuple, List, Dict, Any
from .deep_survival_q_learning import DeepSurvivalQLearning

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

class ArmControlEncoder(nn.Module):
    """Encoder for arm control and physics-based state representation."""
    
    def __init__(self, state_size: int = 5, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Specialized encoders for different aspects
        self.arm_encoder = nn.Linear(2, embed_dim // 2)      # arm_angle, elbow_angle
        self.target_encoder = nn.Linear(2, embed_dim // 4)   # food_distance, food_direction
        self.physics_encoder = nn.Linear(1, embed_dim // 4)  # is_ground_contact
        
        # Combine features
        self.feature_combiner = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, state):
        """
        Encode arm control state: [arm_angle, elbow_angle, food_distance, food_direction, is_ground_contact] = 5 dimensions
        """
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        # Split state into different feature types
        arm_state = state[:, :2]        # arm_angle, elbow_angle
        target_state = state[:, 2:4]    # food_distance, food_direction
        physics_state = state[:, 4:5]   # is_ground_contact
        
        # Encode each feature type
        arm_embed = F.relu(self.arm_encoder(arm_state))
        target_embed = F.relu(self.target_encoder(target_state))
        physics_embed = F.relu(self.physics_encoder(physics_state))
        
        # Combine and normalize
        combined = torch.cat([arm_embed, target_embed, physics_embed], dim=-1)
        combined = self.feature_combiner(combined)
        combined = self.layer_norm(combined)
        
        return combined

class ArmControlAttentionDQN(nn.Module):
    """Dueling DQN with attention for arm-based food-seeking behavior."""
    
    def __init__(self, state_size: int = 5, action_size: int = 9, embed_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.embed_dim = embed_dim
        
        # Arm control feature encoder
        self.feature_encoder = ArmControlEncoder(state_size, embed_dim)
        
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

class AttentionDeepQLearning(DeepSurvivalQLearning):
    """Deep Q-Learning with attention for arm-based food-seeking behavior."""
    
    def __init__(self, state_dim: int = 5, action_dim: int = 9, learning_rate: float = 0.001):
        # Initialize parent class with arm control dimensions
        super().__init__(state_dim, action_dim, learning_rate)
        
        # Replace with arm control attention-based network
        self.q_network = ArmControlAttentionDQN(state_dim, action_dim).to(self.device)
        self.target_network = ArmControlAttentionDQN(state_dim, action_dim).to(self.device)
        
        # Update optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.attention_history = deque(maxlen=5000)  # Increased from 1000 to 5000
        
        # Performance optimization tracking - MUCH less aggressive
        self._last_cleanup_time = time.time()
        self._cleanup_interval = 300.0  # Clean up every 5 minutes instead of 30 seconds
        
        print(f"🦾 Arm Control Attention-based Deep Q-Learning initialized with {self.device}")
        print(f"   📊 Network parameters: {sum(p.numel() for p in self.q_network.parameters()):,}")
        print(f"   🧹 Attention history: {self.attention_history.maxlen} records (optimized)")
    
    def choose_action(self, state_vector: np.ndarray, agent_data: Dict[str, Any] = None) -> int:
        """Choose action with arm control attention-based analysis."""
        # PERFORMANCE: Periodic cleanup of attention history
        current_time = time.time()
        if current_time - self._last_cleanup_time > self._cleanup_interval:
            self._cleanup_attention_data()
            self._last_cleanup_time = current_time
        
        # Calculate epsilon
        epsilon = self._calculate_survival_epsilon(agent_data)
        
        if np.random.random() > epsilon:
            self.q_network.eval()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
                q_values, attention_info = self.q_network(state_tensor)
                action = q_values.argmax().item()
                
                # Store attention information for analysis - ONLY store essential data
                if len(self.attention_history) < self.attention_history.maxlen:
                    self.attention_history.append({
                        'attention_weights': attention_info['attention_weights'].cpu().numpy(),
                        'selected_action': action,
                        'timestamp': current_time  # Add timestamp for cleanup
                    })
            self.q_network.train()
            return action
        else:
            return self._survival_biased_exploration(agent_data)
    
    def learn(self) -> Dict[str, float]:
        """Enhanced learning with arm control attention."""
        if self.use_prioritized_replay:
            if len(self.memory) < self.batch_size:
                return {}
            
            experiences, indices, weights = self.memory.sample(self.batch_size)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            if len(self.memory) < self.batch_size:
                return {}
            
            experiences = random.sample(self.memory, self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)
        
        # Log training initialization (first time only)
        if not hasattr(self, '_first_training_logged'):
            print(f"🧠 Neural Network: First training session - Memory: {len(self.memory)}, Batch: {self.batch_size}")
            self._first_training_logged = True
        
        # Prepare batch
        from collections import namedtuple
        Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])
        batch = Experience(*zip(*experiences))
        
        states = torch.FloatTensor(batch.state).to(self.device)
        actions = torch.LongTensor(batch.action).to(self.device)
        rewards = torch.FloatTensor(batch.reward).to(self.device)
        next_states = torch.FloatTensor(batch.next_state).to(self.device)
        dones = torch.BoolTensor(batch.done).to(self.device)
        
        # Current Q-values with attention
        current_q_values, current_attention = self.q_network(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values, _ = self.target_network(next_states)
            next_q_values = next_q_values.max(1)[0].detach()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        td_errors = target_q_values.unsqueeze(1) - current_q_values
        loss = (weights.unsqueeze(1) * td_errors.pow(2)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update priorities if using prioritized replay
        if self.use_prioritized_replay:
            priorities = abs(td_errors.detach().cpu().numpy()) + 1e-6
            self.memory.update_priorities(indices, priorities.flatten())
            self.memory.frame += 1
        
        # Update target network
        if self.steps_done % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.steps_done += 1
        
        return {
            'loss': loss.item(),
            'mean_q_value': current_q_values.mean().item(),
            'attention_entropy': self._calculate_attention_entropy(current_attention['attention_weights'])
        }
    
    def _calculate_attention_entropy(self, attention_weights):
        """Calculate entropy of attention weights."""
        try:
            avg_attention = attention_weights.mean(dim=(0, 1))
            entropy = -(avg_attention * torch.log(avg_attention + 1e-10)).sum()
            return entropy.item()
        except:
            return 0.0
    
    def get_arm_control_state_representation(self, agent_data: Dict[str, Any]) -> np.ndarray:
        """Convert agent data to arm control state representation."""
        # NO try/catch fallback - let failures be explicit
        
        # Arm joint angles (2 dimensions) - normalized to [-1, 1]
        arm_angles = agent_data['arm_angles']
        arm_angle = arm_angles['shoulder'] / np.pi  # Normalize shoulder angle
        elbow_angle = arm_angles['elbow'] / np.pi   # Normalize elbow angle
        
        # Food targeting information (2 dimensions)
        food_info = agent_data['nearest_food']
        
        # Food distance (normalized to [0, 1])
        food_distance = min(1.0, food_info['distance'] / 100.0)
        
        # Food direction (signed x-distance normalized to [-1, 1])
        # Positive = food is to the right, Negative = food is to the left
        signed_distance = food_info['direction']
        food_direction = np.clip(signed_distance / 50.0, -1.0, 1.0)  # Clamp to [-1, 1]
        
        # Ground contact detection using Box2D physics (1 dimension)
        is_ground_contact = 0.0
        if agent_data.get('ground_contact', False):
            is_ground_contact = 1.0
        elif 'physics_body' in agent_data and agent_data['physics_body']:
            # Check Box2D contacts if available
            body = agent_data['physics_body']
            for contact_edge in body.contacts:
                contact = contact_edge.contact
                if contact.touching:
                    fixture_a = contact.fixtureA
                    fixture_b = contact.fixtureB
                    # Check if contact is with ground (category bit 0x0001)
                    if ((fixture_a.filterData.categoryBits & 0x0001) or 
                        (fixture_b.filterData.categoryBits & 0x0001)):
                        is_ground_contact = 1.0
                        break
        
        # Create arm control state vector: [arm_angle, elbow_angle, food_distance, food_direction, is_ground_contact]
        state = np.array([arm_angle, elbow_angle, food_distance, food_direction, is_ground_contact], dtype=np.float32)
        
        # Ensure no NaN or infinite values (but don't fallback to zeros)
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            raise ValueError(f"Invalid state values detected: {state}")
        
        return state
    
    def get_attention_analysis(self) -> Dict[str, Any]:
        """Analyze attention patterns for arm control features."""
        if not self.attention_history:
            return {}
        
        recent_attention = list(self.attention_history)[-50:]  # Last 50 decisions
        
        # Average attention weights
        avg_attention = np.mean([a['attention_weights'] for a in recent_attention], axis=0)
        
        # Feature focus analysis for arm control
        feature_names = ['arm_angle', 'elbow_angle', 'food_distance', 'food_direction', 'ground_contact']
        
        return {
            'average_attention_weights': avg_attention.tolist(),
            'feature_names': feature_names,
            'attention_entropy': float(np.mean([
                -np.sum(a['attention_weights'] * np.log(a['attention_weights'] + 1e-10))
                for a in recent_attention
            ])),
            'decisions_analyzed': len(recent_attention),
            'state_dimensions': 5,
            'focus_areas': {
                'arm_control': np.mean(avg_attention[:, :, :2]) if len(avg_attention.shape) > 2 else 0.0,
                'food_targeting': np.mean(avg_attention[:, :, 2:4]) if len(avg_attention.shape) > 2 else 0.0,
                'physics_contact': np.mean(avg_attention[:, :, 4:5]) if len(avg_attention.shape) > 2 else 0.0
            }
        }
    
    def _cleanup_attention_data(self):
        """Clean up accumulated attention data to prevent memory growth - LESS AGGRESSIVE."""
        try:
            # Much more conservative cleanup to preserve learning data
            current_time = time.time()
            if hasattr(self, 'attention_history'):
                # Remove entries older than 30 minutes (was 5 minutes)
                cutoff_time = current_time - 1800.0  # 30 minutes instead of 5 minutes
                
                # Keep more records - minimum 500, maximum 5000
                old_size = len(self.attention_history)
                self.attention_history = deque([
                    entry for entry in self.attention_history 
                    if entry.get('timestamp', current_time) > cutoff_time
                ], maxlen=5000)  # Increased from 50 to 5000
                
                # Only log if significant cleanup happened
                if old_size - len(self.attention_history) > 100:
                    print(f"🧹 Cleaned attention data: {len(self.attention_history)} records remaining (removed {old_size - len(self.attention_history)} old entries)")
            
            # REMOVED: GPU cache clearing - this was too aggressive and frequent
            
        except Exception as e:
            print(f"⚠️ Error cleaning attention data: {e}")
    
    def reset_target_network_sync(self):
        """Reset target network synchronization for transferred networks."""
        try:
            # Force immediate target network sync with main network
            self.target_network.load_state_dict(self.q_network.state_dict())
            
            # Reset step counter to ensure proper target update timing
            # Keep training progress but reset target sync timing
            self.steps_done = self.steps_done % self.target_update_freq
            
            print(f"🎯 Target network resynced after transfer (next update in {self.target_update_freq - self.steps_done} steps)")
            
        except Exception as e:
            print(f"⚠️ Error resetting target network sync: {e}")
    
    def get_network_sync_status(self) -> Dict[str, Any]:
        """Get information about main and target network synchronization."""
        try:
            steps_until_sync = self.target_update_freq - (self.steps_done % self.target_update_freq)
            
            # Compare a few parameters to check if networks are in sync
            main_params = list(self.q_network.parameters())
            target_params = list(self.target_network.parameters())
            
            param_differences = []
            for i, (main_param, target_param) in enumerate(zip(main_params[:3], target_params[:3])):  # Check first 3 layers
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
            print(f"⚠️ Error getting network sync status: {e}")
            return {'error': str(e)} 