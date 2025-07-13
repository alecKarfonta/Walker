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
            # Try to reuse an existing network with matching dimensions
            if self.available_networks:
                # Find network with matching dimensions
                for i, network_data in enumerate(self.available_networks):
                    network = network_data['network']
                    network_id = network_data['id']
                    is_prebuffered = network_data.get('prebuffered', False)
                    
                    # CRITICAL FIX: Validate network dimensions match request
                    if (hasattr(network, 'action_dim') and network.action_dim == action_dim and
                        hasattr(network, 'state_dim') and network.state_dim == state_dim):
                        
                        # Remove this network from available pool
                        del self.available_networks[i]
                        
                        # Reset target network sync for transferred networks
                        network.reset_target_network_sync()
                        
                        self.reuse_count += 1
                        
                        # Enhanced logging for pre-buffered networks
                        if is_prebuffered:
                            logger.info(f"⚡ Used pre-buffered network {network_id} (state={state_dim}, action={action_dim}) (pool: {len(self.available_networks)} remaining)")
                        else:
                            logger.info(f"♻️ Reused returned network {network_id} (state={state_dim}, action={action_dim}) (pool: {len(self.available_networks)} remaining)")
                        
                        return network, network_id
                
                # No compatible network found - log this important event
                logger.warning(f"🔍 No compatible network found for dimensions {state_dim}x{action_dim} in pool of {len(self.available_networks)} networks")
            
            # Create new network if pool is empty or no compatible network found
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
            logger.warning(f"🆘 Buffer depleted! Created emergency network {network_id} for state_size={state_dim}, action_size={action_dim}")
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
        
        # LAZY INITIALIZATION FLAG - prevents initialization order issues
        self._buffers_initialized = False
        self._buffer_initialization_attempted = False
        
        logger.info("🧠 Learning Manager initialized")
        print("🧠 Learning Manager initialized: neural network pooling enabled")
        
        # DO NOT pre-initialize here - use lazy initialization on first network request
    
    def _pre_initialize_network_buffers(self):
        """Pre-create a buffer of neural networks focused on one-limb robots first to eliminate 'pool empty' situations."""
        print("🔍 TESTING: _pre_initialize_network_buffers method called!")
        try:
            state_size = 29  # Fixed state size for all agents
            
            # STRATEGY: Focus on one-limb robots first (action_size=9) with 40 networks
            # Then create smaller amounts for multi-limb robots
            network_allocation = {
                9: 40,   # 1 limb × 3 segments = 3 joints → 40 networks (primary focus)
                11: 8,   # 2 limbs × 2 segments = 4 joints → 8 networks
                13: 6,   # 2 limbs × 2-3 segments = 5 joints → 6 networks  
                15: 5,   # 2 limbs × 3 segments = 6 joints → 5 networks
                17: 4,   # 3 limbs × 2 segments = 6 joints → 4 networks
                19: 3,   # 3 limbs × 2-3 segments = 7 joints → 3 networks
                21: 3,   # 3 limbs × 3 segments = 9 joints → 3 networks
                23: 2,   # 4 limbs × 2-3 segments = 9 joints → 2 networks
                25: 2,   # 4 limbs × 3 segments = 12 joints → 2 networks
                27: 2,   # 5 limbs × 2-3 segments = 12 joints → 2 networks
                29: 1,   # 5 limbs × 3 segments = 15 joints → 1 network
                31: 1,   # 6 limbs × 2-3 segments = 15 joints → 1 network
                33: 1,   # 6 limbs × 3 segments = 18 joints → 1 network
            }
            
            total_networks_created = 0
            successful_action_sizes = []
            failed_action_sizes = []
            
            print(f"🚀 Starting network pre-initialization with focus on one-limb robots...")
            
            for action_size, buffer_count in network_allocation.items():
                try:
                    print(f"🔧 Creating {buffer_count} networks for action_size={action_size}...")
                    
                    pool = self._get_or_create_pool(state_size, action_size)
                    
                    # Pre-create buffer networks for this action size
                    for i in range(buffer_count):
                        try:
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
                            
                        except Exception as network_error:
                            print(f"❌ Failed to create individual network {i} for action_size={action_size}: {network_error}")
                            failed_action_sizes.append(f"{action_size}_{i}")
                            continue
                    
                    successful_action_sizes.append(action_size)
                    print(f"✅ Successfully created {buffer_count} networks for action_size={action_size}")
                    
                except Exception as pool_error:
                    print(f"❌ Failed to create pool for action_size={action_size}: {pool_error}")
                    failed_action_sizes.append(str(action_size))
                    continue
            
            # Print comprehensive summary
            print(f"🚀 Pre-initialized {total_networks_created} neural networks across {len(successful_action_sizes)} action sizes")
            print(f"💾 Network allocation: action_size=9 (40 networks), others (1-8 networks each)")
            
            # Log buffer status for verification
            for action_size in [9, 11, 13, 15, 17, 19, 21]:  # Show most common sizes
                pool_key = self._get_pool_key(state_size, action_size)
                if pool_key in self.network_pools:
                    available = len(self.network_pools[pool_key].available_networks)
                    print(f"   📦 Pool {pool_key}: {available} networks ready")
            
            # Report any failures
            if failed_action_sizes:
                print(f"⚠️  Failed to create networks for: {failed_action_sizes}")
            
            # Log to standard logger as well
            logger.info(f"🚀 Pre-initialized {total_networks_created} neural networks (focus: {network_allocation[9]} networks for action_size=9)")
            
            return total_networks_created
                    
        except Exception as e:
            print(f"❌ CRITICAL: Failed to pre-initialize network buffers: {e}")
            import traceback
            traceback.print_exc()
            logger.error(f"❌ Failed to pre-initialize network buffers: {e}")
            # Don't fail initialization if buffer pre-creation fails
            return 0
    
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
        """Acquire a neural network for an agent with lazy buffer initialization."""
        try:
            # LAZY INITIALIZATION: Initialize buffers on first network request
            if not self._buffers_initialized and not self._buffer_initialization_attempted:
                self._buffer_initialization_attempted = True
                try:
                    print("🚀 LAZY INIT: Pre-initializing network buffers on first request...")
                    total_created = self._pre_initialize_network_buffers()
                    self._buffers_initialized = True
                    print(f"✅ LAZY INIT: Successfully pre-initialized {total_created} networks")
                except Exception as init_error:
                    print(f"❌ LAZY INIT: Pre-initialization failed: {init_error}")
                    import traceback
                    traceback.print_exc()
                    # Continue without pre-buffered networks
                    print("⚠️  Continuing without pre-buffered networks (degraded performance)")
            
            # Validate state dimensions (fixed)
            if state_size != 29:
                logger.error(f"Invalid state_size {state_size}, expected 29")
                return None
            
            # FIXED: Support variable action sizes for different agent morphologies
            # Action sizes can range from 5 (minimum viable) to 200+ (complex robots)
            if action_size < 5 or action_size > 200:
                logger.error(f"Invalid action_size {action_size}, must be between 5 and 200")
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
        """Proactively refill network buffers when they get low to prevent 'Buffer depleted!' messages."""
        try:
            state_size = 29
            min_buffer_threshold = 5  # Refill when below 5 networks (increased from 2)
            refill_count = 8  # Add 8 networks when refilling (increased from 3)
            
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
                            # Aggressively refill low pools
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
                                    'refilled': True  # Mark as refilled
                                })
                                pool.creation_count += 1
                                pool.network_history[network_id] = {
                                    'created_at': time.time(),
                                    'reuse_count': 0,
                                    'state_dim': state_size,
                                    'action_dim': action_size,
                                    'refilled': True
                                }
                                
                                total_refilled += 1
                            
                            logger.info(f"🔄 Refilled pool {pool_key}: {available_count} → {available_count + refill_count} networks")
                
            if total_refilled > 0:
                logger.info(f"📈 Buffer maintenance: Refilled {total_refilled} networks across pools to prevent depletion")
                
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