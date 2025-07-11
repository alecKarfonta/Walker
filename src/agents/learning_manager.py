"""
Learning Manager: Handles neural network pooling and knowledge transfer for agents.
Ensures that when agents respawn, they inherit learned knowledge instead of starting fresh.
"""

import torch
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from collections import deque, defaultdict
import uuid
import logging

from .attention_deep_q_learning import AttentionDeepQLearning

logger = logging.getLogger(__name__)


class NetworkPool:
    """Pool of neural networks for efficient reuse and knowledge transfer."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.available_networks: deque = deque()
        self.network_history: Dict[str, Dict] = {}  # network_id -> metadata
        self.creation_count = 0
        self.reuse_count = 0
        self.lock = threading.Lock()
        
    def get_network(self, state_dim: int, action_dim: int) -> Tuple[AttentionDeepQLearning, str]:
        """Get a neural network from the pool or create a new one."""
        with self.lock:
            # Try to reuse an existing network
            if self.available_networks:
                network_data = self.available_networks.popleft()
                network = network_data['network']
                network_id = network_data['id']
                is_prebuffered = network_data.get('prebuffered', False)
                
                # Reset target network sync for transferred networks
                network.reset_target_network_sync()
                
                self.reuse_count += 1
                
                # Enhanced logging for pre-buffered networks
                if is_prebuffered:
                    logger.info(f"⚡ Used pre-buffered network {network_id} (pool: {len(self.available_networks)} remaining)")
                else:
                    logger.info(f"♻️ Reused returned network {network_id} (pool: {len(self.available_networks)} remaining)")
                
                return network, network_id
            
            # Create new network if pool is empty
            network = AttentionDeepQLearning(
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate=0.001
            )
            
            network_id = str(uuid.uuid4())[:8]
            self.network_history[network_id] = {
                'created_at': time.time(),
                'reuse_count': 0,
                'state_dim': state_dim,
                'action_dim': action_dim
            }
            
            self.creation_count += 1
            logger.warning(f"🆘 Buffer depleted! Created emergency network {network_id} for action_size={action_dim}")
            return network, network_id
    
    def return_network(self, network: AttentionDeepQLearning, network_id: str):
        """Return a neural network to the pool for reuse."""
        with self.lock:
            if len(self.available_networks) >= self.max_size:
                logger.warning(f"Network pool full, discarding network {network_id}")
                return
            
            # Update metadata
            if network_id in self.network_history:
                self.network_history[network_id]['reuse_count'] += 1
                self.network_history[network_id]['last_used'] = time.time()
            
            # Add to pool
            self.available_networks.append({
                'network': network,
                'id': network_id,
                'returned_at': time.time()
            })
            
            logger.info(f"♻️ Returned neural network {network_id} to pool")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self.lock:
            return {
                'available_networks': len(self.available_networks),
                'creation_count': self.creation_count,
                'reuse_count': self.reuse_count,
                'max_size': self.max_size,
                'total_networks_created': len(self.network_history)
            }


class LearningManager:
    """Manages neural network allocation and knowledge transfer across agents."""
    
    def __init__(self, max_networks_per_pool: int = 50):
        self.network_pools: Dict[str, NetworkPool] = {}
        self.agent_networks: Dict[str, Tuple[AttentionDeepQLearning, str]] = {}  # agent_id -> (network, network_id)
        self.knowledge_transfer_log: List[Dict] = []
        self.max_networks_per_pool = max_networks_per_pool
        self.lock = threading.Lock()
        
        # Elite network preservation
        self.elite_networks: Dict[str, AttentionDeepQLearning] = {}
        self.elite_threshold = 0.8  # Top 20% performers
        
        logger.info("🧠 Learning Manager initialized")
        
        # PRE-INITIALIZE NETWORK BUFFERS to avoid "pool empty" messages
        self._pre_initialize_network_buffers()
    
    def _pre_initialize_network_buffers(self):
        """Pre-create a buffer of neural networks for common action sizes to eliminate 'pool empty' situations."""
        try:
            state_size = 29  # Fixed state size for all agents
            
            # Pre-create networks for common evolutionary agent action sizes
            # Based on morphology: 2×joints + 5, where joints range from 2-18
            common_action_sizes = [
                9,   # 2 joints (1 limb × 2 segments)
                11,  # 3 joints  
                13,  # 4 joints
                15,  # 5 joints (2 limbs × 2-3 segments) - most common
                17,  # 6 joints
                19,  # 7 joints
                21,  # 8 joints (2 limbs × 3 segments)
                23,  # 9 joints
                25,  # 10 joints (3-4 limbs × 2-3 segments)
                27,  # 11 joints
                29,  # 12 joints
                31,  # 13 joints
                33,  # 14 joints
                35,  # 15 joints
                37,  # 16 joints
                39,  # 17 joints
                41   # 18 joints (6 limbs × 3 segments) - complex robots
            ]
            
            # Buffer size: Create 3-5 networks per common action size
            buffer_size_per_action = min(5, max(3, self.max_networks_per_pool // len(common_action_sizes)))
            
            total_networks_created = 0
            
            for action_size in common_action_sizes:
                pool = self._get_or_create_pool(state_size, action_size)
                
                # Pre-create buffer networks for this action size
                for i in range(buffer_size_per_action):
                    network = AttentionDeepQLearning(
                        state_dim=state_size,
                        action_dim=action_size,
                        learning_rate=0.001
                    )
                    
                    network_id = f"prebuf_{action_size}_{i}_{str(uuid.uuid4())[:4]}"
                    
                    # Add to pool's available networks
                    with pool.lock:
                        pool.available_networks.append({
                            'network': network,
                            'id': network_id,
                            'returned_at': time.time(),
                            'prebuffered': True  # Mark as pre-buffered
                        })
                        pool.creation_count += 1
                        pool.network_history[network_id] = {
                            'created_at': time.time(),
                            'reuse_count': 0,
                            'state_dim': state_size,
                            'action_dim': action_size,
                            'prebuffered': True
                        }
                    
                    total_networks_created += 1
            
            logger.info(f"🚀 Pre-initialized {total_networks_created} neural networks across {len(common_action_sizes)} action sizes")
            logger.info(f"💾 Buffer: {buffer_size_per_action} networks per action size, ready for instant assignment")
            
            # Log buffer status
            for action_size in common_action_sizes[:5]:  # Show first 5 for brevity
                pool_key = self._get_pool_key(state_size, action_size)
                if pool_key in self.network_pools:
                    available = len(self.network_pools[pool_key].available_networks)
                    logger.info(f"   📦 Pool {pool_key}: {available} networks ready")
                    
        except Exception as e:
            logger.error(f"❌ Failed to pre-initialize network buffers: {e}")
            # Don't fail initialization if buffer pre-creation fails
    
    def _get_pool_key(self, state_dim: int, action_dim: int) -> str:
        """Get the pool key for networks with specific dimensions."""
        return f"{state_dim}x{action_dim}"
    
    def _get_or_create_pool(self, state_dim: int, action_dim: int) -> NetworkPool:
        """Get or create a network pool for specific dimensions."""
        pool_key = self._get_pool_key(state_dim, action_dim)
        
        if pool_key not in self.network_pools:
            self.network_pools[pool_key] = NetworkPool(max_size=self.max_networks_per_pool)
            logger.info(f"Created network pool for {pool_key}")
        
        return self.network_pools[pool_key]
    
    def _acquire_attention_network(self, agent_id: str, action_size: int, state_size: int) -> Optional[AttentionDeepQLearning]:
        """Acquire a neural network for an agent."""
        try:
            # Validate state dimensions (fixed)
            if state_size != 29:
                logger.error(f"Invalid state_size {state_size}, expected 29")
                return None
            
            # FIXED: Support variable action sizes for different agent morphologies
            # Action sizes can range from 5 (minimum viable) to 50+ (complex robots)
            if action_size < 5 or action_size > 100:
                logger.error(f"Invalid action_size {action_size}, must be between 5 and 100")
                return None
            
            # Get appropriate pool
            pool = self._get_or_create_pool(state_size, action_size)
            
            # Get network from pool
            network, network_id = pool.get_network(state_size, action_size)
            
            # Track assignment
            with self.lock:
                self.agent_networks[agent_id] = (network, network_id)
            
            logger.info(f"🧠 Assigned network {network_id} to agent {agent_id}")
            return network
            
        except Exception as e:
            logger.error(f"Failed to acquire network for agent {agent_id}: {e}")
            return None
    
    def release_agent_network(self, agent_id: str, performance_score: float = 0.0):
        """Release an agent's network back to the pool."""
        with self.lock:
            if agent_id not in self.agent_networks:
                logger.warning(f"Agent {agent_id} has no network to release")
                return
            
            network, network_id = self.agent_networks[agent_id]
            
            # Check if this should be an elite network
            if performance_score >= self.elite_threshold:
                self.elite_networks[network_id] = network
                logger.info(f"🏆 Preserved elite network {network_id} (score: {performance_score:.3f})")
            
            # Return to appropriate pool
            pool_key = self._get_pool_key(network.state_dim, network.action_dim)
            if pool_key in self.network_pools:
                self.network_pools[pool_key].return_network(network, network_id)
            
            # Remove from agent tracking
            del self.agent_networks[agent_id]
            
            logger.info(f"Released network {network_id} from agent {agent_id}")
    
    def transfer_knowledge(self, source_agent_id: str, target_agent_id: str) -> bool:
        """Transfer knowledge from source agent to target agent."""
        with self.lock:
            if source_agent_id not in self.agent_networks:
                logger.warning(f"Source agent {source_agent_id} has no network")
                return False
            
            if target_agent_id not in self.agent_networks:
                logger.warning(f"Target agent {target_agent_id} has no network")
                return False
            
            source_network, source_id = self.agent_networks[source_agent_id]
            target_network, target_id = self.agent_networks[target_agent_id]
            
            try:
                # Transfer main network weights
                target_network.q_network.load_state_dict(source_network.q_network.state_dict())
                target_network.target_network.load_state_dict(source_network.target_network.state_dict())
                
                # Transfer learning parameters
                target_network.epsilon = source_network.epsilon
                target_network.steps_done = source_network.steps_done
                
                # Reset target network sync
                target_network.reset_target_network_sync()
                
                # Log transfer
                transfer_record = {
                    'timestamp': time.time(),
                    'source_agent': source_agent_id,
                    'target_agent': target_agent_id,
                    'source_network': source_id,
                    'target_network': target_id
                }
                self.knowledge_transfer_log.append(transfer_record)
                
                logger.info(f"🧠 Transferred knowledge: {source_agent_id} → {target_agent_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to transfer knowledge: {e}")
                return False
    
    def get_best_network_for_agent(self, agent_id: str, performance_threshold: float = 0.7) -> Optional[AttentionDeepQLearning]:
        """Get the best available network for an agent based on performance."""
        with self.lock:
            if not self.elite_networks:
                return None
            
            # For now, return a random elite network
            # Could be improved with better selection criteria
            import random
            elite_id = random.choice(list(self.elite_networks.keys()))
            elite_network = self.elite_networks[elite_id]
            
            logger.info(f"🏆 Assigned elite network {elite_id} to agent {agent_id}")
            return elite_network
    
    def _refill_buffers_if_needed(self):
        """Refill network buffers that have fallen below minimum threshold."""
        try:
            state_size = 29
            min_buffer_threshold = 2  # Refill when below 2 networks
            refill_count = 3  # Add 3 networks when refilling
            
            # Common action sizes that need buffer maintenance
            common_action_sizes = [9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41]
            
            total_refilled = 0
            
            for action_size in common_action_sizes:
                pool_key = self._get_pool_key(state_size, action_size)
                
                if pool_key in self.network_pools:
                    pool = self.network_pools[pool_key]
                    
                    with pool.lock:
                        available_count = len(pool.available_networks)
                        
                        if available_count <= min_buffer_threshold:
                            # Refill the buffer
                            for i in range(refill_count):
                                network = AttentionDeepQLearning(
                                    state_dim=state_size,
                                    action_dim=action_size,
                                    learning_rate=0.001
                                )
                                
                                network_id = f"refill_{action_size}_{i}_{str(uuid.uuid4())[:4]}"
                                
                                pool.available_networks.append({
                                    'network': network,
                                    'id': network_id,
                                    'returned_at': time.time(),
                                    'prebuffered': True,
                                    'refilled': True
                                })
                                pool.creation_count += 1
                                pool.network_history[network_id] = {
                                    'created_at': time.time(),
                                    'reuse_count': 0,
                                    'state_dim': state_size,
                                    'action_dim': action_size,
                                    'prebuffered': True,
                                    'refilled': True
                                }
                                
                                total_refilled += 1
                            
                            logger.info(f"🔄 Refilled buffer for action_size={action_size}: {available_count} → {available_count + refill_count} networks")
            
            if total_refilled > 0:
                logger.info(f"📈 Buffer maintenance: Refilled {total_refilled} networks across {len([a for a in common_action_sizes if self._get_pool_key(state_size, a) in self.network_pools and len(self.network_pools[self._get_pool_key(state_size, a)].available_networks) > min_buffer_threshold])} pools")
                
        except Exception as e:
            logger.error(f"❌ Failed to refill network buffers: {e}")

    def cleanup_inactive_agents(self, active_agent_ids: List[str]):
        """Clean up networks from inactive agents and refill buffers."""
        with self.lock:
            inactive_agents = set(self.agent_networks.keys()) - set(active_agent_ids)
            
            for agent_id in inactive_agents:
                logger.info(f"Cleaning up inactive agent {agent_id}")
                self.release_agent_network(agent_id)
        
        # Refill buffers after cleanup
        self._refill_buffers_if_needed()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive learning manager statistics."""
        with self.lock:
            pool_stats = {}
            total_buffered_networks = 0
            
            for pool_key, pool in self.network_pools.items():
                stats = pool.get_stats()
                pool_stats[pool_key] = stats
                total_buffered_networks += stats['available_networks']
            
            return {
                'active_agents': len(self.agent_networks),
                'elite_networks': len(self.elite_networks),
                'knowledge_transfers': len(self.knowledge_transfer_log),
                'total_buffered_networks': total_buffered_networks,
                'pools': pool_stats,
                'last_transfer': self.knowledge_transfer_log[-1] if self.knowledge_transfer_log else None
            }
    
    def maintain_buffers(self) -> Dict[str, Any]:
        """Public method to check buffer health and perform maintenance if needed."""
        try:
            # Get stats before maintenance
            pre_stats = self.get_stats()
            
            # Perform buffer maintenance
            self._refill_buffers_if_needed()
            
            # Get stats after maintenance
            post_stats = self.get_stats()
            
            networks_added = post_stats['total_buffered_networks'] - pre_stats['total_buffered_networks']
            
            maintenance_result = {
                'maintenance_performed': networks_added > 0,
                'networks_added': networks_added,
                'total_buffered_before': pre_stats['total_buffered_networks'],
                'total_buffered_after': post_stats['total_buffered_networks'],
                'active_pools': len(self.network_pools),
                'buffer_health': 'good' if post_stats['total_buffered_networks'] > 20 else 'needs_attention'
            }
            
            if networks_added > 0:
                logger.info(f"🔧 Buffer maintenance completed: +{networks_added} networks, {post_stats['total_buffered_networks']} total buffered")
            
            return maintenance_result
            
        except Exception as e:
            logger.error(f"❌ Buffer maintenance failed: {e}")
            return {
                'maintenance_performed': False,
                'error': str(e)
            }
    
    def force_elite_network_assignment(self, agent_id: str) -> bool:
        """Force assign an elite network to an agent (for testing/debugging)."""
        if not self.elite_networks:
            logger.warning("No elite networks available for assignment")
            return False
        
        try:
            # Get best elite network
            best_network = self.get_best_network_for_agent(agent_id)
            if best_network:
                # Create a copy for the agent
                network_copy = AttentionDeepQLearning(
                    state_dim=best_network.state_dim,
                    action_dim=best_network.action_dim,
                    learning_rate=best_network.learning_rate
                )
                
                # Copy weights
                network_copy.q_network.load_state_dict(best_network.q_network.state_dict())
                network_copy.target_network.load_state_dict(best_network.target_network.state_dict())
                network_copy.epsilon = best_network.epsilon
                network_copy.steps_done = best_network.steps_done
                
                # Assign to agent
                network_id = str(uuid.uuid4())[:8]
                self.agent_networks[agent_id] = (network_copy, network_id)
                
                logger.info(f"🏆 Force assigned elite network to agent {agent_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to force assign elite network: {e}")
            return False
        
        return False 