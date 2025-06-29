"""
Simplified Learning Manager for Attention-based Deep Q-Learning
Manages attention network pooling and resource allocation.
"""

import time
import numpy as np
import logging
from typing import Dict, Any, List, Optional

# Create logger for this module
logger = logging.getLogger(__name__)


class LearningManager:
    """
    Manages attention-based deep Q-learning networks with pooling and resource management.
    """
    
    def __init__(self, ecosystem_interface=None, training_environment=None):
        """Initialize Learning Manager with attention network pooling."""
        try:
            # Core systems
            self.ecosystem_interface = ecosystem_interface
            self.training_environment = training_environment
            
            # Check for attention deep Q-learning availability
            self.attention_deep_q_available = self._check_attention_deep_q_availability()
            if not self.attention_deep_q_available:
                raise ImportError("Attention Deep Q-Learning not available")
            
            # DIMENSION-AWARE ATTENTION NETWORK POOLING
            self._attention_network_pools = {}  # Dict[(state_size, action_size)] = [networks]
            self._attention_networks_in_use = {}  # agent_id -> network
            self._attention_network_pool_max_size = 8  # Max networks per dimension combination
            
            # Network creation and performance tracking
            self._agents_currently_training = set()
            
            # Statistics tracking
            self._network_stats = {
                'total_created': 0,
                'total_reused': 0,
                'current_in_use': 0,
                'current_in_pool': 0,
                'peak_networks_in_use': 0,
                'networks_destroyed': 0,
                'pool_architectures': 0
            }
            
            # GPU monitoring
            self._gpu_stats = {
                'current_memory_mb': 0,
                'peak_memory_mb': 0,
                'initial_memory_mb': 0,
                'last_update_time': 0
            }
            
            print("🔧 Learning Manager initialized with attention network pooling")
            
        except Exception as e:
            print(f"❌ Error initializing Learning Manager: {e}")
            raise
    
    def set_training_environment(self, training_environment):
        """Set the training environment reference for accessing real food data."""
        self.training_environment = training_environment
    
    def inject_training_environment_into_agents(self, agents):
        """Inject training environment reference and assign attention networks to all agents."""
        for agent in agents:
            if hasattr(agent, 'id'):
                agent.training_environment = self.training_environment
        print(f"🌍 Training environment injected into {len(agents)} agents")
        
        # Assign attention networks to all agents
        self.assign_attention_networks_to_agents(agents)
    
    def assign_attention_networks_to_agents(self, agents):
        """Assign attention networks from the pool to all agents."""
        assigned_count = 0
        for agent in agents:
            if hasattr(agent, 'id'):
                # Get the correct action space size for this agent's morphology
                action_size = 6  # Default for basic 2-joint robots
                if hasattr(agent, 'action_size'):
                    action_size = agent.action_size
                elif hasattr(agent, 'actions') and agent.actions:
                    action_size = len(agent.actions)
                
                # Get attention network with correct action space for this agent
                network = self._acquire_attention_network(agent.id, action_size, agent.state_size)
                if network:
                    self._setup_attention_learning_wrapper(agent, network)
                    assigned_count += 1
                    print(f"🧠 Agent {agent.id[:8]}: {action_size} actions → attention network assigned")
                else:
                    print(f"❌ Failed to assign attention network to agent {agent.id[:8]}")
        
        print(f"🔗 Assigned attention networks to {assigned_count}/{len(agents)} agents")
    
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
    
    def cleanup_agent(self, agent):
        """Clean up an agent's attention network and return it to the pool."""
        try:
            agent_id = agent.id
            
            # Return attention network to pool
            if hasattr(agent, '_attention_dqn'):
                self._return_attention_network(agent_id)
                delattr(agent, '_attention_dqn')
                print(f"♻️ Returned attention network to pool from agent {agent_id}")
            
            # Restore original choose_action method
            if hasattr(agent, '_original_choose_action'):
                agent.choose_action = agent._original_choose_action
                delattr(agent, '_original_choose_action')
                
        except Exception as e:
            print(f"⚠️ Error cleaning up agent {getattr(agent, 'id', 'unknown')}: {e}")
    
    def _get_agent_state_data(self, agent, training_environment=None) -> Dict[str, Any]:
        """Extract state data from agent for attention deep Q-learning."""
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
            
            # Real food information from training environment
            if not training_environment:
                raise ValueError(f"Agent {agent.id} missing training environment reference")
                
            food_info = training_environment._get_closest_food_distance_for_agent(agent)
            
            state_data['nearest_food'] = {
                'distance': food_info['distance'],
                'direction': food_info['signed_x_distance'],
                'type': food_info['food_type']
            }
            
            # Ground contact detection using Box2D physics
            ground_contact = self._detect_ground_contact(agent)
            state_data['ground_contact'] = ground_contact
            
            # Physics body for advanced ground detection
            state_data['physics_body'] = agent.body
            
            return state_data
            
        except Exception as e:
            raise RuntimeError(f"Critical state extraction failure for agent {getattr(agent, 'id', 'unknown')}: {e}")
    
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
            
        except Exception:
            # Simple fallback for ground contact
            try:
                if hasattr(agent, 'body') and agent.body:
                    return agent.body.position.y <= 1.0  # Ground level approximation
            except:
                pass
            return False
    
    def _acquire_attention_network(self, agent_id: str, action_size: int = 6, state_size: int = 5):
        """Acquire an attention network from the pool or create a new one if pool is empty."""
        try:
            # Check if agent already has a network
            if agent_id in self._attention_networks_in_use:
                existing_network = self._attention_networks_in_use[agent_id]
                print(f"♻️ Agent {agent_id[:8]} already has attention network")
                return existing_network
            
            # Try to reuse from pool first
            pool_key = (state_size, action_size)
            if self._attention_network_pools.get(pool_key):
                network = self._attention_network_pools[pool_key].pop()
                self._attention_networks_in_use[agent_id] = network
                self._network_stats['total_reused'] += 1
                print(f"♻️ REUSED Attention Network from pool for agent {agent_id[:8]} (pool: {len(self._attention_network_pools[pool_key])} left)")
                return network
            
            # Create new network only if pool is empty
            from .attention_deep_q_learning import AttentionDeepQLearning
            
            attention_dqn = AttentionDeepQLearning(
                state_dim=state_size,
                action_dim=action_size,
                learning_rate=0.001
            )
            
            self._attention_networks_in_use[agent_id] = attention_dqn
            self._network_stats['total_created'] += 1
            
            # Log with GPU memory info
            gpu_mem = ""
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_mem = f" (GPU: {torch.cuda.memory_allocated() // 1024 // 1024}MB)"
            except:
                pass
                
            print(f"🧠 NEW Attention Network #{self._network_stats['total_created']} for agent {agent_id[:8]} (created: {self._network_stats['total_created']}{gpu_mem})")
            
            return attention_dqn
            
        except Exception as e:
            print(f"❌ Error acquiring attention network for agent {agent_id}: {e}")
            return None
    
    def _return_attention_network(self, agent_id: str):
        """Return an attention network to the pool when agent no longer needs it."""
        try:
            if agent_id not in self._attention_networks_in_use:
                return
            
            network = self._attention_networks_in_use.pop(agent_id)
            
            # Update stats
            self._network_stats['current_in_use'] = len(self._attention_networks_in_use)
            
            # Get network dimensions
            network_state_dim = getattr(network, 'state_dim', None)
            network_action_dim = getattr(network, 'action_dim', None)
            
            if network_state_dim is None or network_action_dim is None:
                # Can't determine dimensions, destroy the network
                print(f"🧹 Destroying network with unknown dimensions from agent {agent_id[:8]}")
                try:
                    del network
                    self._network_stats['networks_destroyed'] += 1
                except:
                    pass
                return
            
            # Initialize pool for this architecture if needed
            pool_key = (network_state_dim, network_action_dim)
            if pool_key not in self._attention_network_pools:
                self._attention_network_pools[pool_key] = []
            
            # Return to pool if we have space
            if len(self._attention_network_pools[pool_key]) < self._attention_network_pool_max_size:
                # Reset network state for reuse
                try:
                    if hasattr(network, 'attention_history'):
                        network.attention_history.clear()
                except Exception as e:
                    print(f"⚠️ Error resetting network state: {e}")
                
                self._attention_network_pools[pool_key].append(network)
                
                # Update pool stats
                self._network_stats['current_in_pool'] = sum(len(networks) for networks in self._attention_network_pools.values())
                self._network_stats['pool_architectures'] = len(self._attention_network_pools)
                
                print(f"♻️ Returned Attention Network to pool from agent {agent_id[:8]} (pool[{network_state_dim},{network_action_dim}]: {len(self._attention_network_pools[pool_key])}, in_use: {self._network_stats['current_in_use']})")
            else:
                # Pool is full, destroy the network
                try:
                    if hasattr(network, 'device') and 'cuda' in str(network.device):
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    del network
                    
                    self._network_stats['networks_destroyed'] += 1
                    self._update_gpu_stats()
                    
                    print(f"🧹 Destroyed excess Attention Network from agent {agent_id[:8]} (destroyed: {self._network_stats['networks_destroyed']}, GPU: {self._gpu_stats['current_memory_mb']}MB)")
                except Exception as e:
                    print(f"⚠️ Error destroying attention network: {e}")
                    
        except Exception as e:
            print(f"⚠️ Error returning attention network for agent {agent_id}: {e}")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics including network creation and GPU usage."""
        # Update current stats
        self._network_stats['current_in_use'] = len(self._attention_networks_in_use)
        self._network_stats['current_in_pool'] = sum(len(networks) for networks in self._attention_network_pools.values())
        self._update_gpu_stats()
        
        return {
            'network_stats': self._network_stats.copy(),
            'gpu_stats': self._gpu_stats.copy(),
            'efficiency_metrics': {
                'reuse_rate': (self._network_stats['total_reused'] / 
                              max(1, self._network_stats['total_created'] + self._network_stats['total_reused']) * 100),
                'memory_efficiency': (self._gpu_stats['current_memory_mb'] - self._gpu_stats['initial_memory_mb']),
                'pool_utilization': (sum(len(networks) for networks in self._attention_network_pools.values()) / 
                                    (self._attention_network_pool_max_size * max(1, len(self._attention_network_pools))) * 100)
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
    
    def _update_gpu_stats(self):
        """Update GPU usage statistics."""
        try:
            import time
            
            current_memory_mb = 0
            try:
                import torch
                if torch.cuda.is_available():
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
                    pass
            
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
    
    def _check_attention_deep_q_availability(self) -> bool:
        """Check if attention deep Q-learning is available."""
        try:
            from .attention_deep_q_learning import AttentionDeepQLearning
            return True
        except ImportError:
            return False 