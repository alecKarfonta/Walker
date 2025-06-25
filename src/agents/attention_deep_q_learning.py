import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import time
from typing import Tuple, List, Dict, Any
from .deep_survival_q_learning import DeepSurvivalQLearning

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for focusing on relevant state features."""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
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
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and split into heads
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads and apply output linear layer
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, -1, self.embed_dim
        )
        
        output = self.out_linear(attended)
        
        return output, attention_weights

class EnvironmentalFeatureEncoder(nn.Module):
    """Encodes different types of environmental features with specialized attention."""
    
    def __init__(self, state_size: int, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Feature extraction layers for different aspects
        self.physical_encoder = nn.Linear(8, embed_dim // 4)  # Position, velocity, angles
        self.energy_encoder = nn.Linear(2, embed_dim // 4)    # Energy, health
        self.food_encoder = nn.Linear(3, embed_dim // 4)      # Food distance, direction, type
        self.social_encoder = nn.Linear(2, embed_dim // 4)    # Nearby agents, competition
        
        # Combine all features
        self.feature_combiner = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, state):
        """
        Encode state features with specialized attention.
        Expected state format: [physical(8), energy(2), food(3), social(2)] = 15 dimensions
        """
        # Split state into different feature types
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        physical_features = state[:, :8]    # Position, velocity, arm angles, etc.
        energy_features = state[:, 8:10]    # Energy, health levels
        food_features = state[:, 10:13]     # Food distance, direction, type
        social_features = state[:, 13:15]   # Nearby agents, competition
        
        # Encode each feature type
        physical_embed = F.relu(self.physical_encoder(physical_features))
        energy_embed = F.relu(self.energy_encoder(energy_features))
        food_embed = F.relu(self.food_encoder(food_features))
        social_embed = F.relu(self.social_encoder(social_features))
        
        # Combine features
        combined = torch.cat([physical_embed, energy_embed, food_embed, social_embed], dim=-1)
        combined = self.feature_combiner(combined)
        combined = self.layer_norm(combined)
        
        return combined

class AttentionDuelingDQN(nn.Module):
    """Dueling DQN with multi-head attention for focusing on relevant features."""
    
    def __init__(self, state_size: int, action_size: int, embed_dim: int = 128, num_heads: int = 8):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.embed_dim = embed_dim
        
        # Environmental feature encoder
        self.feature_encoder = EnvironmentalFeatureEncoder(state_size, embed_dim)
        
        # Multi-head attention layers
        self.self_attention = MultiHeadAttention(embed_dim, num_heads)
        self.cross_attention = MultiHeadAttention(embed_dim, num_heads)
        
        # Feed-forward layers
        self.ff1 = nn.Linear(embed_dim, embed_dim * 2)
        self.ff2 = nn.Linear(embed_dim * 2, embed_dim)
        self.dropout = nn.Dropout(0.1)
        
        # Dueling architecture
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
        self.norm3 = nn.LayerNorm(embed_dim)
        
    def forward(self, state):
        # Encode environmental features
        features = self.feature_encoder(state)
        
        # Self-attention to focus on important features
        attended_features, attention_weights = self.self_attention(features, features, features)
        features = self.norm1(features + attended_features)
        
        # Feed-forward processing
        ff_output = self.ff2(F.relu(self.ff1(features)))
        ff_output = self.dropout(ff_output)
        features = self.norm2(features + ff_output)
        
        # Cross-attention for context integration
        context_features, context_weights = self.cross_attention(features, features, features)
        features = self.norm3(features + context_features)
        
        # Dueling Q-value computation
        value = self.value_head(features)
        advantages = self.advantage_head(features)
        
        # Combine value and advantages
        q_values = value + (advantages - advantages.mean(dim=-1, keepdim=True))
        
        return q_values, {
            'attention_weights': attention_weights,
            'context_weights': context_weights,
            'features': features
        }

class AttentionDeepQLearning(DeepSurvivalQLearning):
    """Enhanced Deep Q-Learning with attention mechanisms."""
    
    def __init__(self, state_dim: int = 15, action_dim: int = 9, learning_rate: float = 0.001):
        # Initialize parent class
        super().__init__(state_dim, action_dim, learning_rate)
        
        # Replace the DQN with attention-based version
        self.q_network = AttentionDuelingDQN(state_dim, action_dim).to(self.device)
        self.target_network = AttentionDuelingDQN(state_dim, action_dim).to(self.device)
        
        # Update optimizer with new network
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Attention analysis
        self.attention_history = deque(maxlen=1000)
        
        print(f"🔍 Attention-based Deep Q-Learning initialized with {self.device}")
        print(f"   📊 Network parameters: {sum(p.numel() for p in self.q_network.parameters()):,}")
    
    def choose_action(self, state_vector: np.ndarray, agent_data: Dict[str, Any] = None) -> int:
        """Choose action with attention-based feature analysis."""
        # Calculate epsilon
        epsilon = self._calculate_survival_epsilon(agent_data)
        
        if np.random.random() > epsilon:
            self.q_network.eval()  # Set to eval mode for single-sample inference
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
                q_values, attention_info = self.q_network(state_tensor)
                action = q_values.argmax().item()
                
                # Store attention information for analysis
                self.attention_history.append({
                    'attention_weights': attention_info['attention_weights'].cpu().numpy(),
                    'context_weights': attention_info['context_weights'].cpu().numpy(),
                    'q_values': q_values.cpu().numpy(),
                    'selected_action': action
                })
            self.q_network.train()  # Set back to train mode
            return action
        else:
            return self._survival_biased_exploration(agent_data)
    
    def store_experience(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool, agent_data: Dict[str, Any] = None):
        """Enhanced store experience with attention-based learning."""
        # Use parent class method which handles both PrioritizedReplayBuffer and deque
        super().store_experience(state, action, reward, next_state, done, agent_data)
    
    def learn(self) -> Dict[str, float]:
        """Enhanced learning with attention-focused training."""
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
            next_q_values, next_attention = self.target_network(next_states)
            next_q_values = next_q_values.max(1)[0].detach()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss with importance sampling weights
        td_errors = target_q_values.unsqueeze(1) - current_q_values
        loss = (weights.unsqueeze(1) * td_errors.pow(2)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update priorities if using prioritized replay
        if self.use_prioritized_replay:
            priorities = abs(td_errors.detach().cpu().numpy()) + 1e-6
            self.memory.update_priorities(indices, priorities.flatten())
            self.memory.frame += 1
        
        # Update target network periodically
        if self.steps_done % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.steps_done += 1
        
        return {
            'loss': loss.item(),
            'mean_q_value': current_q_values.mean().item(),
            'attention_entropy': self._calculate_attention_entropy(current_attention)
        }
    
    def _calculate_attention_entropy(self, attention_weights):
        """Calculate entropy of attention weights for monitoring."""
        try:
            # Average across batch and heads
            avg_attention = attention_weights.mean(dim=(0, 1))
            entropy = -(avg_attention * torch.log(avg_attention + 1e-10)).sum()
            return entropy.item()
        except:
            return 0.0
    
    def get_attention_analysis(self) -> Dict[str, Any]:
        """Analyze attention patterns for debugging and insights."""
        if not self.attention_history:
            return {}
        
        recent_attention = list(self.attention_history)[-100:]  # Last 100 decisions
        
        # Average attention weights across heads and time
        avg_attention = np.mean([a['attention_weights'] for a in recent_attention], axis=0)
        avg_context = np.mean([a['context_weights'] for a in recent_attention], axis=0)
        
        # Most attended features
        feature_importance = np.mean(avg_attention, axis=(0, 1))  # Average across heads and positions
        
        return {
            'average_attention_weights': avg_attention.tolist(),
            'average_context_weights': avg_context.tolist(),
            'feature_importance': feature_importance.tolist(),
            'most_important_features': np.argsort(feature_importance)[-5:].tolist(),
            'attention_entropy': float(np.mean([
                -np.sum(a['attention_weights'] * np.log(a['attention_weights'] + 1e-10))
                for a in recent_attention
            ])),
            'decisions_analyzed': len(recent_attention)
        }
    
    def get_enhanced_state_representation(self, agent_data: Dict[str, Any]) -> np.ndarray:
        """Convert agent data to attention-friendly state representation."""
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
            print(f"⚠️ Error creating attention state representation: {e}")
            # Return default state
            return np.zeros(15, dtype=np.float32) 