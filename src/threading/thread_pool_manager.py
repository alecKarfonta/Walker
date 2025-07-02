"""
🧵 Thread Pool Manager for Walker Training System
Optimizes workload distribution across 48 CPU cores
"""

import os
import time
import threading
import concurrent.futures
from typing import List, Dict, Any, Callable, Optional
from collections import defaultdict, deque
import psutil
import numpy as np


class ThreadPoolManager:
    """
    Centralized thread pool management for Walker training system.
    Distributes workload across multiple CPU cores for optimal performance.
    """
    
    def __init__(self, max_cores: int = None):
        """
        Initialize thread pool manager.
        
        Args:
            max_cores: Maximum CPU cores to use (default: auto-detect)
        """
        self.max_cores = max_cores or min(48, os.cpu_count() or 1)
        
        # Thread pool allocation strategy for 48-core system
        self.ai_threads = max(4, self.max_cores // 3)          # 16 threads for AI (33%)
        self.stats_threads = max(2, self.max_cores // 6)       # 8 threads for stats (17%)
        self.background_threads = max(4, self.max_cores // 4)  # 12 threads for background (25%)
        self.physics_threads = 4                               # 4 threads for physics helpers (8%)
        self.evaluation_threads = 8                            # 8 threads for evaluation (17%)
        
        print(f"🧵 Initializing ThreadPoolManager for {self.max_cores} cores")
        print(f"   🧠 AI Processing: {self.ai_threads} threads")
        print(f"   📊 Statistics: {self.stats_threads} threads")
        print(f"   🌍 Background: {self.background_threads} threads")
        print(f"   ⚡ Physics: {self.physics_threads} threads")
        print(f"   🔧 Evaluation: {self.evaluation_threads} threads")
        
        # Create thread pools
        self._create_thread_pools()
        
        # Performance monitoring
        self.performance_monitor = ThreadPoolPerformanceMonitor()
        self.load_balancer = ThreadPoolLoadBalancer(self)
        
        # Thread pool health monitoring
        self.health_monitor = ThreadPoolHealthMonitor(self)
        self.health_monitor.start()
        
        # Emergency shutdown handling
        self._shutdown_requested = False
        self._shutdown_lock = threading.Lock()
        
    def _create_thread_pools(self):
        """Create and configure thread pools."""
        try:
            # AI processing pool - highest priority for agent thinking
            self.ai_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.ai_threads,
                thread_name_prefix="ai_worker"
            )
            
            # Statistics collection pool 
            self.stats_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.stats_threads,
                thread_name_prefix="stats_worker"
            )
            
            # Background processing pool
            self.background_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.background_threads,
                thread_name_prefix="bg_worker"
            )
            
            # Physics helper pool (limited due to Box2D constraints)
            self.physics_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.physics_threads,
                thread_name_prefix="physics_worker"
            )
            
            # Evaluation processing pool
            self.evaluation_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.evaluation_threads,
                thread_name_prefix="eval_worker"
            )
            
            print("✅ Thread pools created successfully")
            
        except Exception as e:
            print(f"❌ Error creating thread pools: {e}")
            raise
    
    def parallel_agent_processing(self, agents: List[Any], dt: float) -> Dict[str, Any]:
        """
        Process agents in parallel across multiple threads.
        
        Args:
            agents: List of agent objects to process
            dt: Time delta for physics step
            
        Returns:
            Dictionary with processing results and performance metrics
        """
        start_time = time.time()
        
        if not agents:
            return {'processed': 0, 'time': 0, 'fps': 0}
        
        # Create agent batches for parallel processing
        batch_size = max(1, len(agents) // self.ai_threads)
        agent_batches = [agents[i:i + batch_size] for i in range(0, len(agents), batch_size)]
        
        # Submit processing jobs to thread pool
        futures = []
        for batch_idx, batch in enumerate(agent_batches):
            future = self.ai_pool.submit(self._process_agent_batch, batch, dt, batch_idx)
            futures.append(future)
        
        # Wait for all processing to complete with timeout
        results = []
        try:
            for future in concurrent.futures.as_completed(futures, timeout=5.0):
                result = future.result()
                results.append(result)
                
        except concurrent.futures.TimeoutError:
            print("⚠️ Agent processing timeout - cancelling remaining tasks")
            for future in futures:
                future.cancel()
        
        # Calculate performance metrics
        total_time = time.time() - start_time
        total_processed = sum(r['processed'] for r in results)
        processing_fps = total_processed / total_time if total_time > 0 else 0
        
        # Update performance monitoring
        self.performance_monitor.record_agent_processing(total_time, total_processed, len(agent_batches))
        
        return {
            'processed': total_processed,
            'time': total_time,
            'fps': processing_fps,
            'batches': len(agent_batches),
            'results': results
        }
    
    def _process_agent_batch(self, agent_batch: List[Any], dt: float, batch_idx: int) -> Dict[str, Any]:
        """
        Process a batch of agents in a single thread.
        
        Args:
            agent_batch: Batch of agents to process
            dt: Time delta
            batch_idx: Batch identifier for debugging
            
        Returns:
            Processing results for this batch
        """
        start_time = time.time()
        processed_count = 0
        errors = 0
        
        try:
            for agent in agent_batch:
                try:
                    if not getattr(agent, '_destroyed', False):
                        agent.step(dt)
                        processed_count += 1
                except Exception as e:
                    errors += 1
                    if errors < 5:  # Limit error logging
                        print(f"⚠️ Agent processing error in batch {batch_idx}: {e}")
            
        except Exception as e:
            print(f"❌ Batch processing error {batch_idx}: {e}")
            
        processing_time = time.time() - start_time
        
        return {
            'batch_idx': batch_idx,
            'processed': processed_count,
            'errors': errors,
            'time': processing_time,
            'thread_id': threading.current_thread().ident
        }
    
    def parallel_statistics_collection(self, agents: List[Any]) -> Dict[str, Any]:
        """
        Collect statistics from agents in parallel.
        
        Args:
            agents: List of agents to collect stats from
            
        Returns:
            Aggregated statistics
        """
        start_time = time.time()
        
        if not agents:
            return {'stats': {}, 'time': 0}
        
        # Create batches for parallel stats collection
        batch_size = max(1, len(agents) // self.stats_threads)
        agent_batches = [agents[i:i + batch_size] for i in range(0, len(agents), batch_size)]
        
        # Submit stats collection jobs
        futures = []
        for batch_idx, batch in enumerate(agent_batches):
            future = self.stats_pool.submit(self._collect_batch_statistics, batch, batch_idx)
            futures.append(future)
        
        # Collect results
        batch_stats = []
        for future in concurrent.futures.as_completed(futures, timeout=3.0):
            try:
                result = future.result()
                batch_stats.append(result)
            except Exception as e:
                print(f"⚠️ Stats collection error: {e}")
        
        # Aggregate statistics from all batches
        aggregated_stats = self._aggregate_statistics(batch_stats)
        
        total_time = time.time() - start_time
        self.performance_monitor.record_stats_collection(total_time, len(agents))
        
        return {
            'stats': aggregated_stats,
            'time': total_time,
            'batches_processed': len(batch_stats)
        }
    
    def _collect_batch_statistics(self, agent_batch: List[Any], batch_idx: int) -> Dict[str, Any]:
        """Collect statistics from a batch of agents."""
        stats = {
            'batch_idx': batch_idx,
            'agent_count': 0,
            'total_rewards': [],
            'positions': [],
            'fitness_scores': [],
            'learning_metrics': [],
            'errors': 0
        }
        
        for agent in agent_batch:
            try:
                if not getattr(agent, '_destroyed', False):
                    stats['agent_count'] += 1
                    stats['total_rewards'].append(getattr(agent, 'total_reward', 0.0))
                    
                    if hasattr(agent, 'body') and agent.body:
                        stats['positions'].append((agent.body.position.x, agent.body.position.y))
                    
                    if hasattr(agent, 'get_fitness_score'):
                        stats['fitness_scores'].append(agent.get_fitness_score())
                    
                    # Collect learning metrics if available
                    if hasattr(agent, 'get_current_performance'):
                        stats['learning_metrics'].append(agent.get_current_performance())
                        
            except Exception as e:
                stats['errors'] += 1
        
        return stats
    
    def _aggregate_statistics(self, batch_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate statistics from multiple batches."""
        aggregated = {
            'total_agents': 0,
            'avg_reward': 0.0,
            'max_reward': 0.0,
            'min_reward': 0.0,
            'positions': [],
            'fitness_scores': [],
            'learning_metrics': [],
            'total_errors': 0
        }
        
        all_rewards = []
        
        for batch in batch_stats:
            aggregated['total_agents'] += batch['agent_count']
            aggregated['total_errors'] += batch['errors']
            all_rewards.extend(batch['total_rewards'])
            aggregated['positions'].extend(batch['positions'])
            aggregated['fitness_scores'].extend(batch['fitness_scores'])
            aggregated['learning_metrics'].extend(batch['learning_metrics'])
        
        if all_rewards:
            aggregated['avg_reward'] = np.mean(all_rewards)
            aggregated['max_reward'] = np.max(all_rewards)
            aggregated['min_reward'] = np.min(all_rewards)
        
        return aggregated
    
    def submit_background_task(self, task: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit a task to the background processing pool."""
        return self.background_pool.submit(task, *args, **kwargs)
    
    def submit_evaluation_task(self, task: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit a task to the evaluation processing pool."""
        return self.evaluation_pool.submit(task, *args, **kwargs)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics for all thread pools."""
        return {
            'thread_pools': {
                'ai_pool': self._get_pool_metrics(self.ai_pool, 'ai'),
                'stats_pool': self._get_pool_metrics(self.stats_pool, 'stats'),
                'background_pool': self._get_pool_metrics(self.background_pool, 'background'),
                'physics_pool': self._get_pool_metrics(self.physics_pool, 'physics'),
                'evaluation_pool': self._get_pool_metrics(self.evaluation_pool, 'evaluation')
            },
            'performance_history': self.performance_monitor.get_metrics(),
            'health_status': self.health_monitor.get_health_status(),
            'load_balancing': self.load_balancer.get_status()
        }
    
    def _get_pool_metrics(self, pool: concurrent.futures.ThreadPoolExecutor, pool_name: str) -> Dict[str, Any]:
        """Get metrics for a specific thread pool."""
        return {
            'max_workers': pool._max_workers,
            'active_threads': getattr(pool, '_threads', 0),
            'pending_tasks': pool._work_queue.qsize() if hasattr(pool, '_work_queue') else 0,
            'pool_name': pool_name
        }
    
    def shutdown(self, wait: bool = True, timeout: float = 5.0):
        """Shutdown all thread pools gracefully."""
        with self._shutdown_lock:
            if self._shutdown_requested:
                return
            
            self._shutdown_requested = True
            print("🛑 Shutting down thread pools...")
            
            # Stop health monitoring
            self.health_monitor.stop()
            
            # Shutdown all pools
            pools = [
                ('AI', self.ai_pool),
                ('Stats', self.stats_pool),
                ('Background', self.background_pool),
                ('Physics', self.physics_pool),
                ('Evaluation', self.evaluation_pool)
            ]
            
            for name, pool in pools:
                try:
                    pool.shutdown(wait=wait, timeout=timeout)
                    print(f"   ✅ {name} pool shutdown complete")
                except Exception as e:
                    print(f"   ⚠️ {name} pool shutdown error: {e}")
            
            print("✅ Thread pool shutdown complete")


class ThreadPoolPerformanceMonitor:
    """Monitors thread pool performance and efficiency."""
    
    def __init__(self, history_length: int = 1000):
        self.history_length = history_length
        self.agent_processing_history = deque(maxlen=history_length)
        self.stats_collection_history = deque(maxlen=history_length)
        self.lock = threading.Lock()
    
    def record_agent_processing(self, time_taken: float, agents_processed: int, batches: int):
        """Record agent processing performance."""
        with self.lock:
            self.agent_processing_history.append({
                'timestamp': time.time(),
                'time_taken': time_taken,
                'agents_processed': agents_processed,
                'batches': batches,
                'agents_per_second': agents_processed / time_taken if time_taken > 0 else 0
            })
    
    def record_stats_collection(self, time_taken: float, agents_count: int):
        """Record statistics collection performance."""
        with self.lock:
            self.stats_collection_history.append({
                'timestamp': time.time(),
                'time_taken': time_taken,
                'agents_count': agents_count,
                'agents_per_second': agents_count / time_taken if time_taken > 0 else 0
            })
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        with self.lock:
            agent_metrics = self._analyze_history(self.agent_processing_history)
            stats_metrics = self._analyze_history(self.stats_collection_history)
            
            return {
                'agent_processing': agent_metrics,
                'stats_collection': stats_metrics,
                'timestamp': time.time()
            }
    
    def _analyze_history(self, history: deque) -> Dict[str, Any]:
        """Analyze performance history."""
        if not history:
            return {'avg_time': 0, 'avg_throughput': 0, 'samples': 0}
        
        times = [entry['time_taken'] for entry in history]
        throughputs = [entry['agents_per_second'] for entry in history]
        
        return {
            'avg_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'avg_throughput': np.mean(throughputs),
            'max_throughput': np.max(throughputs),
            'samples': len(history)
        }


class ThreadPoolLoadBalancer:
    """Dynamically balances load across thread pools."""
    
    def __init__(self, thread_manager: ThreadPoolManager):
        self.thread_manager = thread_manager
        self.load_history = deque(maxlen=100)
        self.last_balance_time = time.time()
        self.balance_interval = 30.0  # Rebalance every 30 seconds
    
    def monitor_and_balance(self):
        """Monitor thread pool performance and rebalance if needed."""
        current_time = time.time()
        if current_time - self.last_balance_time < self.balance_interval:
            return
        
        # Get current load metrics
        metrics = self.thread_manager.get_performance_metrics()
        
        # Analyze load distribution
        load_analysis = self._analyze_load_distribution(metrics)
        
        # Rebalance if needed
        if load_analysis['needs_rebalancing']:
            self._rebalance_pools(load_analysis)
        
        self.last_balance_time = current_time
    
    def _analyze_load_distribution(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current load distribution across pools."""
        # Implementation would analyze queue depths, processing times, etc.
        return {
            'needs_rebalancing': False,  # Placeholder
            'bottleneck_pools': [],
            'underutilized_pools': []
        }
    
    def _rebalance_pools(self, analysis: Dict[str, Any]):
        """Rebalance thread pools based on analysis."""
        # Implementation would dynamically adjust pool sizes
        print("🔄 Rebalancing thread pools...")
    
    def get_status(self) -> Dict[str, Any]:
        """Get load balancer status."""
        return {
            'last_balance_time': self.last_balance_time,
            'balance_interval': self.balance_interval,
            'load_samples': len(self.load_history)
        }


class ThreadPoolHealthMonitor:
    """Monitors thread pool health and detects issues."""
    
    def __init__(self, thread_manager: ThreadPoolManager):
        self.thread_manager = thread_manager
        self.monitoring = False
        self.monitor_thread = None
        self.health_status = {'status': 'healthy', 'issues': []}
    
    def start(self):
        """Start health monitoring."""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            print("🏥 Thread pool health monitoring started")
    
    def stop(self):
        """Stop health monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                self._check_thread_pool_health()
                time.sleep(5.0)  # Check every 5 seconds
            except Exception as e:
                print(f"⚠️ Health monitoring error: {e}")
    
    def _check_thread_pool_health(self):
        """Check health of all thread pools."""
        issues = []
        
        # Check each pool for issues
        pools = [
            ('AI', self.thread_manager.ai_pool),
            ('Stats', self.thread_manager.stats_pool),
            ('Background', self.thread_manager.background_pool),
            ('Physics', self.thread_manager.physics_pool),
            ('Evaluation', self.thread_manager.evaluation_pool)
        ]
        
        for name, pool in pools:
            if hasattr(pool, '_work_queue'):
                queue_size = pool._work_queue.qsize()
                if queue_size > 100:  # Large queue indicates bottleneck
                    issues.append(f"{name} pool has large queue: {queue_size}")
        
        # Update health status
        status = 'healthy' if not issues else 'warning' if len(issues) < 3 else 'critical'
        self.health_status = {
            'status': status,
            'issues': issues,
            'last_check': time.time()
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        return self.health_status.copy() 