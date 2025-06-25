"""
Deep Q-Network (DQN) implementation for survival learning.
Provides neural network-based value function approximation for complex state spaces.

Requirements:
    pip install torch torchvision numpy

Optional GPU acceleration:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
"""

import logging
import sys

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️ PyTorch not available. Install with: pip install torch torchvision numpy")
    TORCH_AVAILABLE = False
    # Mock torch for type checking
    class torch:
        class nn:
            class Module: pass
        class device: pass
        class FloatTensor: pass
        class LongTensor: pass
        class BoolTensor: pass

import numpy as np
import random
from collections import deque, namedtuple
from typing import Tuple, List, Dict, Any, Optional, Union
import math

# Configure logging
logger = logging.getLogger(__name__)

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class SurvivalDQN(nn.Module):
    """
    Deep Q-Network for survival learning.
    Processes continuous state inputs and outputs Q-values for discrete actions.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256, 128]):
        super(SurvivalDQN, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build network layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
        
        # Output layer for Q-values
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize network weights."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    def forward(self, state):
        """Forward pass through the network."""
        return self.network(state)


class DuelingDQN(nn.Module):
    """
    Dueling DQN that separates value and advantage estimation.
    Better for survival learning where action values vary significantly.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
        super(DuelingDQN, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared feature layers
        self.feature_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[1])
        )
        
        feature_dim = hidden_dims[1]
        
        # Value stream (estimates V(s))
        self.value_stream = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage stream (estimates A(s,a))
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize network weights."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    def forward(self, state):
        """Forward pass with dueling architecture."""
        features = self.feature_layers(state)
        
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantage: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = values + advantages - advantages.mean(dim=1, keepdim=True)
        
        return q_values


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer for more efficient learning.
    Prioritizes experiences based on TD error magnitude.
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4, beta_frames: int = 100000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
        # Storage
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    
    def beta(self):
        """Calculate current beta value for importance sampling."""
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)
    
    def push(self, experience, priority):
        """Add experience with priority."""
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        
        self.priorities[self.pos] = max(priority, 1e-6)
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size: int):
        """Sample batch with priorities."""
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]
        
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-self.beta())
        weights /= weights.max()
        
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = max(priority, 1e-6)
    
    def __len__(self):
        return len(self.buffer)


class DeepSurvivalQLearning:
    """
    Deep Q-Learning implementation specifically designed for survival tasks.
    Uses neural networks for continuous state representation and complex value functions.
    """
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: int = 100000,
                 buffer_size: int = 100000,
                 batch_size: int = 64,
                 target_update_freq: int = 1000,
                 device: str = None,
                 use_dueling: bool = True,
                 use_prioritized_replay: bool = True):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Device setup
        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🧠 Deep Q-Learning using device: {self.device}")
        
        # Debug logging for GPU training initialization
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(self.device)
            gpu_memory = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            logger.debug(f"🚀 GPU TRAINING INITIALIZED")
            logger.debug(f"   • GPU Device: {gpu_name}")
            logger.debug(f"   • GPU Memory: {gpu_memory:.1f} GB")
            logger.debug(f"   • CUDA Version: {torch.version.cuda}")
            logger.debug(f"   • PyTorch Version: {torch.__version__}")
            logger.debug(f"   • State Dimensions: {state_dim}")
            logger.debug(f"   • Action Dimensions: {action_dim}")
            logger.debug(f"   • Network Architecture: {'Dueling DQN' if use_dueling else 'Standard DQN'}")
            logger.debug(f"   • Replay Buffer: {'Prioritized' if use_prioritized_replay else 'Standard'}")
            print(f"🚀 GPU TRAINING ACTIVE: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            logger.debug(f"🖥️ CPU TRAINING INITIALIZED")
            logger.debug(f"   • Device: {self.device}")
            logger.debug(f"   • State Dimensions: {state_dim}")
            logger.debug(f"   • Action Dimensions: {action_dim}")
            print(f"🖥️ CPU Training mode active")
        
        # Networks
        if use_dueling:
            self.q_network = DuelingDQN(state_dim, action_dim).to(self.device)
            self.target_network = DuelingDQN(state_dim, action_dim).to(self.device)
        else:
            self.q_network = SurvivalDQN(state_dim, action_dim).to(self.device)
            self.target_network = SurvivalDQN(state_dim, action_dim).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay
        if use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(buffer_size)
        else:
            self.memory = deque(maxlen=buffer_size)
        
        self.use_prioritized_replay = use_prioritized_replay
        
        # Training state
        self.steps_done = 0
        self.learning_started = False
        
        # Survival-specific features
        self.survival_priorities = {
            'energy_gain': 3.0,      # Prioritize energy gain experiences
            'critical_energy': 2.0,  # Prioritize low energy situations
            'death_avoidance': 2.5   # Prioritize near-death experiences
        }
        
    def get_continuous_state_vector(self, survival_state_data: Dict[str, Any]) -> np.ndarray:
        """
        Convert survival state data to continuous vector for neural network.
        
        Args:
            survival_state_data: Dictionary containing survival state information
            
        Returns:
            Numpy array representing continuous state
        """
        state_vector = []
        
        # Physical state (normalized angles)
        shoulder_angle = survival_state_data.get('shoulder_angle', 0.0)
        elbow_angle = survival_state_data.get('elbow_angle', 0.0)
        state_vector.extend([
            np.sin(shoulder_angle), np.cos(shoulder_angle),  # Angle encoding
            np.sin(elbow_angle), np.cos(elbow_angle)
        ])
        
        # Survival state (already normalized 0-1)
        state_vector.extend([
            survival_state_data.get('energy_level', 1.0),
            survival_state_data.get('health_level', 1.0)
        ])
        
        # Environmental awareness
        food_distance = survival_state_data.get('food_distance', float('inf'))
        food_distance_norm = min(1.0, food_distance / 20.0)  # Normalize to 0-1
        
        food_direction = survival_state_data.get('food_direction', 0.0)
        state_vector.extend([
            food_distance_norm,
            np.sin(food_direction), np.cos(food_direction)  # Direction encoding
        ])
        
        # Spatial context
        velocity_magnitude = survival_state_data.get('velocity_magnitude', 0.0)
        velocity_norm = min(1.0, velocity_magnitude / 5.0)  # Normalize
        
        body_angle = survival_state_data.get('body_angle', 0.0)
        ground_contact = survival_state_data.get('ground_contact', 1.0)
        
        state_vector.extend([
            velocity_norm,
            np.sin(body_angle), np.cos(body_angle),
            ground_contact
        ])
        
        # Social context
        nearby_agents = survival_state_data.get('nearby_agents', 0)
        nearby_agents_norm = min(1.0, nearby_agents / 10.0)  # Normalize
        competition = survival_state_data.get('competition_pressure', 0.0)
        
        state_vector.extend([
            nearby_agents_norm,
            competition
        ])
        
        return np.array(state_vector, dtype=np.float32)
    
    def choose_action(self, state_vector: np.ndarray, agent_data: Dict[str, Any] = None) -> int:
        """Choose action using epsilon-greedy with survival-aware exploration."""
        
        # Calculate epsilon with survival-aware decay
        epsilon = self._calculate_survival_epsilon(agent_data)
        
        if random.random() > epsilon:
            # Exploitation: use Q-network in eval mode to fix BatchNorm issues
            self.q_network.eval()  # Set to eval mode for single-sample inference
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                action = q_values.max(1)[1].item()
            self.q_network.train()  # Set back to train mode for learning
        else:
            # Exploration: survival-biased random action
            action = self._survival_biased_exploration(agent_data)
        
        return action
    
    def _calculate_survival_epsilon(self, agent_data: Dict[str, Any] = None) -> float:
        """Calculate epsilon with survival-aware adjustments."""
        # Base epsilon decay
        base_epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                      math.exp(-1. * self.steps_done / self.epsilon_decay)
        
        if agent_data is None:
            return base_epsilon
        
        # Adjust based on survival situation
        energy_level = agent_data.get('energy_level', 1.0)
        
        if energy_level < 0.2:  # Critical energy
            return base_epsilon * 0.3  # Less exploration, more exploitation
        elif energy_level > 0.8:  # High energy
            return min(base_epsilon * 1.5, 0.8)  # More exploration when safe
        
        return base_epsilon
    
    def _survival_biased_exploration(self, agent_data: Dict[str, Any] = None) -> int:
        """Survival-biased random action selection."""
        # For now, return random action
        # Could be enhanced to bias toward survival-positive actions
        return random.randint(0, self.action_dim - 1)
    
    def store_experience(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool, agent_data: Dict[str, Any] = None):
        """Store experience with survival-based prioritization."""
        
        experience = Experience(state, action, reward, next_state, done)
        
        if self.use_prioritized_replay:
            # Calculate priority based on survival importance
            priority = abs(reward) + 1e-6  # Base priority from reward magnitude
            
            # Boost priority for survival-critical experiences
            if agent_data:
                energy_level = agent_data.get('energy_level', 1.0)
                energy_change = agent_data.get('energy_change', 0.0)
                
                if energy_change > 0:  # Energy gain
                    priority *= self.survival_priorities['energy_gain']
                elif energy_level < 0.2:  # Critical energy
                    priority *= self.survival_priorities['critical_energy']
                elif done and energy_level <= 0:  # Death
                    priority *= self.survival_priorities['death_avoidance']
            
            self.memory.push(experience, priority)
        else:
            self.memory.append(experience)
    
    def learn(self) -> Dict[str, float]:
        """Perform one learning step."""
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
        batch = Experience(*zip(*experiences))
        
        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
        done_batch = torch.BoolTensor(batch.done).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (self.gamma * next_q_values * ~done_batch)
        
        # Compute loss with importance sampling weights
        td_errors = target_q_values.unsqueeze(1) - current_q_values
        loss = (weights.unsqueeze(1) * td_errors.pow(2)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
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
        
        # Debug logging for first learning step (batch training start)
        if not self.learning_started:
            self.learning_started = True
            if self.device.type == 'cuda':
                logger.debug(f"🎯 GPU BATCH TRAINING STARTED")
                logger.debug(f"   • First learning step completed")
                logger.debug(f"   • Batch size: {self.batch_size}")
                logger.debug(f"   • Memory buffer size: {len(self.memory)}")
                logger.debug(f"   • Training loss: {loss.item():.4f}")
                logger.debug(f"   • Mean Q-value: {current_q_values.mean().item():.4f}")
                print(f"🎯 GPU BATCH TRAINING: Processing {self.batch_size} experiences")
            else:
                logger.debug(f"🎯 CPU BATCH TRAINING STARTED")
                logger.debug(f"   • First learning step completed")
                logger.debug(f"   • Batch size: {self.batch_size}")
                logger.debug(f"   • Memory buffer size: {len(self.memory)}")
        
        return {
            'loss': loss.item(),
            'mean_q_value': current_q_values.mean().item(),
            'mean_target_q': target_q_values.mean().item(),
            'epsilon': self._calculate_survival_epsilon()
        }
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }, filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        
        print(f"📂 Model loaded from {filepath}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            'steps_done': self.steps_done,
            'epsilon': self._calculate_survival_epsilon(),
            'memory_size': len(self.memory),
            'learning_started': self.learning_started,
            'device': str(self.device),
            'use_prioritized_replay': self.use_prioritized_replay
        } 