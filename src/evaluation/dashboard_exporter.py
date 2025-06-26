"""
Dashboard exporter for Grafana integration.
Provides real-time metrics export and monitoring capabilities.
"""

import time
import json
import threading
from typing import Dict, List, Any, Optional
from pathlib import Path
from flask import Flask, jsonify, request
import sqlite3
import queue
from dataclasses import asdict


class DashboardExporter:
    """
    Exports metrics for Grafana dashboards and monitoring systems.
    Provides REST API endpoints and database storage for metrics.
    """
    
    def __init__(self, 
                 port: int = 2322,
                 db_path: str = "metrics.db",
                 enable_api: bool = True):
        """
        Initialize the dashboard exporter.
        
        Args:
            port: Port for REST API server
            db_path: Path to SQLite database for metrics storage
            enable_api: Whether to enable REST API
        """
        self.port = port
        self.db_path = db_path
        self.enable_api = enable_api
        
        # Initialize database
        self._init_database()
        
        # Metrics queue for async processing
        self.metrics_queue = queue.Queue()
        self.processing_thread = None
        self.is_running = False
        
        # Flask app for REST API
        if enable_api:
            self.app = Flask(__name__)
            self._setup_api_routes()
            self.api_thread = None
        
        # Current metrics cache
        self.current_metrics = {}
        self.metrics_lock = threading.Lock()
        
        print(f"🌐 Dashboard exporter initialized on port {port}")
    
    def _init_database(self):
        """Initialize SQLite database for metrics storage."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Population metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS population_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    generation INTEGER,
                    step_count INTEGER,
                    population_size INTEGER,
                    best_fitness REAL,
                    avg_fitness REAL,
                    genotypic_diversity REAL,
                    phenotypic_diversity REAL,
                    behavioral_diversity REAL,
                    extinction_risk REAL,
                    species_count INTEGER
                )
            ''')
            
            # Training metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    generation INTEGER,
                    step_count INTEGER,
                    convergence_speed REAL,
                    training_variance REAL,
                    cpu_usage REAL,
                    memory_usage REAL,
                    training_fps REAL,
                    plateau_detected BOOLEAN
                )
            ''')
            
            # Individual robot metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS robot_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    generation INTEGER,
                    step_count INTEGER,
                    agent_id TEXT,
                    q_learning_convergence REAL,
                    exploration_efficiency REAL,
                    action_diversity REAL,
                    motor_efficiency REAL,
                    policy_stability REAL,
                    sample_efficiency REAL
                )
            ''')
            
            # Q-learning metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS q_learning_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    generation INTEGER,
                    agent_id TEXT,
                    convergence_rate REAL,
                    q_table_size INTEGER,
                    exploration_balance REAL,
                    learning_plateau BOOLEAN
                )
            ''')
            
            # Reward signal quality metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS reward_signal_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    agent_id TEXT,
                    quality_score REAL,
                    signal_to_noise_ratio REAL,
                    reward_consistency REAL,
                    reward_sparsity REAL,
                    exploration_incentive REAL,
                    convergence_support REAL,
                    behavioral_alignment REAL,
                    reward_mean REAL,
                    reward_std REAL,
                    steps_analyzed INTEGER,
                    total_rewards_received INTEGER,
                    quality_issues TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
            print(f"✅ Database initialized at {self.db_path}")
            
        except Exception as e:
            print(f"❌ Error initializing database: {e}")
    
    def _setup_api_routes(self):
        """Setup REST API routes for Grafana integration."""
        
        @self.app.route('/api/health')
        def health_check():
            """Health check endpoint."""
            return jsonify({'status': 'healthy', 'timestamp': time.time()})
        
        @self.app.route('/api/metrics/population')
        def get_population_metrics():
            """Get current population metrics."""
            try:
                with self.metrics_lock:
                    metrics = self.current_metrics.get('population', {})
                    return jsonify(metrics)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/metrics/training')
        def get_training_metrics():
            """Get current training metrics."""
            try:
                with self.metrics_lock:
                    metrics = self.current_metrics.get('training', {})
                    return jsonify(metrics)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/metrics/robots')
        def get_robot_metrics():
            """Get current robot metrics."""
            try:
                with self.metrics_lock:
                    metrics = self.current_metrics.get('robots', [])
                    return jsonify(metrics)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/metrics/history/population')
        def get_population_history():
            """Get population metrics history."""
            try:
                hours = request.args.get('hours', 1, type=int)
                since = time.time() - (hours * 3600)
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT timestamp, generation, best_fitness, avg_fitness, 
                           genotypic_diversity, extinction_risk, species_count
                    FROM population_metrics 
                    WHERE timestamp > ?
                    ORDER BY timestamp
                ''', (since,))
                
                rows = cursor.fetchall()
                conn.close()
                
                history = []
                for row in rows:
                    history.append({
                        'timestamp': row[0],
                        'generation': row[1],
                        'best_fitness': row[2],
                        'avg_fitness': row[3],
                        'genotypic_diversity': row[4],
                        'extinction_risk': row[5],
                        'species_count': row[6]
                    })
                
                return jsonify(history)
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/metrics/history/training')
        def get_training_history():
            """Get training metrics history."""
            try:
                hours = request.args.get('hours', 1, type=int)
                since = time.time() - (hours * 3600)
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT timestamp, generation, convergence_speed, training_variance,
                           cpu_usage, memory_usage, training_fps, plateau_detected
                    FROM training_metrics 
                    WHERE timestamp > ?
                    ORDER BY timestamp
                ''', (since,))
                
                rows = cursor.fetchall()
                conn.close()
                
                history = []
                for row in rows:
                    history.append({
                        'timestamp': row[0],
                        'generation': row[1],
                        'convergence_speed': row[2],
                        'training_variance': row[3],
                        'cpu_usage': row[4],
                        'memory_usage': row[5],
                        'training_fps': row[6],
                        'plateau_detected': bool(row[7])
                    })
                
                return jsonify(history)
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/alerts')
        def get_alerts():
            """Get current system alerts."""
            try:
                alerts = self._generate_alerts()
                return jsonify(alerts)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/dashboards/config')
        def get_dashboard_config():
            """Get Grafana dashboard configuration."""
            try:
                config = self._generate_grafana_dashboard_config()
                return jsonify(config)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/metrics/reward_signals')
        def get_reward_signal_metrics():
            """Get current reward signal quality metrics."""
            try:
                with self.metrics_lock:
                    metrics = self.current_metrics.get('reward_signals', {})
                    return jsonify(metrics)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/metrics/history/reward_signals')
        def get_reward_signal_history():
            """Get reward signal metrics history."""
            try:
                hours = request.args.get('hours', 1, type=int)
                since = time.time() - (hours * 3600)
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT timestamp, agent_id, quality_score, signal_to_noise_ratio,
                           reward_consistency, reward_sparsity, exploration_incentive,
                           convergence_support, behavioral_alignment, reward_mean, reward_std
                    FROM reward_signal_metrics 
                    WHERE timestamp > ?
                    ORDER BY timestamp
                ''', (since,))
                
                rows = cursor.fetchall()
                conn.close()
                
                history = []
                for row in rows:
                    history.append({
                        'timestamp': row[0],
                        'agent_id': row[1],
                        'quality_score': row[2],
                        'signal_to_noise_ratio': row[3],
                        'reward_consistency': row[4],
                        'reward_sparsity': row[5],
                        'exploration_incentive': row[6],
                        'convergence_support': row[7],
                        'behavioral_alignment': row[8],
                        'reward_mean': row[9],
                        'reward_std': row[10]
                    })
                
                return jsonify(history)
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/metrics')
        def get_prometheus_metrics_endpoint():
            """Prometheus metrics endpoint."""
            try:
                metrics_text = self.get_prometheus_metrics()
                from flask import Response
                return Response(metrics_text, mimetype='text/plain')
            except Exception as e:
                return f"# Error generating metrics: {str(e)}", 500
    
    def start(self):
        """Start the dashboard exporter services."""
        try:
            self.is_running = True
            
            # Start metrics processing thread
            self.processing_thread = threading.Thread(target=self._process_metrics_queue)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            # Start API server
            if self.enable_api:
                self.api_thread = threading.Thread(
                    target=lambda: self.app.run(host='0.0.0.0', port=self.port, debug=False)
                )
                self.api_thread.daemon = True
                self.api_thread.start()
                print(f"🌐 REST API started on http://0.0.0.0:{self.port}")
            
            print("✅ Dashboard exporter services started")
            
        except Exception as e:
            print(f"❌ Error starting dashboard exporter: {e}")
    
    def stop(self):
        """Stop the dashboard exporter services."""
        self.is_running = False
        print("🛑 Dashboard exporter stopped")
    
    def export_metrics(self, comprehensive_metrics):
        """
        Export comprehensive metrics for dashboard consumption.
        
        Args:
            comprehensive_metrics: ComprehensiveMetrics object
        """
        try:
            # Add to processing queue
            self.metrics_queue.put(comprehensive_metrics)
            
        except Exception as e:
            print(f"⚠️  Error queuing metrics for export: {e}")
    
    def _process_metrics_queue(self):
        """Process metrics queue in background thread."""
        while self.is_running:
            try:
                # Get metrics from queue with timeout
                metrics = self.metrics_queue.get(timeout=1.0)
                
                # Store in database
                self._store_metrics_in_db(metrics)
                
                # Update current metrics cache
                self._update_current_metrics(metrics)
                
                # Mark task as done
                self.metrics_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️  Error processing metrics: {e}")
    
    def _store_metrics_in_db(self, comprehensive_metrics):
        """Store metrics in SQLite database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Store population metrics
            pop_metrics = comprehensive_metrics.population_metrics
            fitness_dist = pop_metrics.fitness_distribution_analysis
            
            cursor.execute('''
                INSERT INTO population_metrics (
                    timestamp, generation, step_count, population_size,
                    best_fitness, avg_fitness, genotypic_diversity,
                    phenotypic_diversity, behavioral_diversity, extinction_risk, species_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                comprehensive_metrics.timestamp,
                comprehensive_metrics.generation,
                comprehensive_metrics.step_count,
                pop_metrics.population_size,
                fitness_dist.get('max', 0.0),
                fitness_dist.get('mean', 0.0),
                pop_metrics.genotypic_diversity,
                pop_metrics.phenotypic_diversity,
                pop_metrics.behavioral_diversity,
                pop_metrics.extinction_risk,
                pop_metrics.speciation_dynamics.get('species_count', 1)
            ))
            
            # Store training metrics
            train_metrics = comprehensive_metrics.training_metrics
            
            cursor.execute('''
                INSERT INTO training_metrics (
                    timestamp, generation, step_count, convergence_speed,
                    training_variance, cpu_usage, memory_usage, training_fps, plateau_detected
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                comprehensive_metrics.timestamp,
                comprehensive_metrics.generation,
                comprehensive_metrics.step_count,
                train_metrics.convergence_speed,
                train_metrics.training_variance,
                train_metrics.cpu_usage,
                train_metrics.memory_usage,
                train_metrics.training_fps,
                train_metrics.plateau_detection
            ))
            
            # Store individual robot metrics (sample a few to avoid database bloat)
            sample_robot_ids = list(comprehensive_metrics.individual_metrics.keys())[:10]
            
            for robot_id in sample_robot_ids:
                individual = comprehensive_metrics.individual_metrics[robot_id]
                q_learning = comprehensive_metrics.q_learning_metrics.get(robot_id)
                
                cursor.execute('''
                    INSERT INTO robot_metrics (
                        timestamp, generation, step_count, agent_id,
                        q_learning_convergence, exploration_efficiency,
                        action_diversity, motor_efficiency, policy_stability, sample_efficiency
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    comprehensive_metrics.timestamp,
                    comprehensive_metrics.generation,
                    comprehensive_metrics.step_count,
                    robot_id,
                    individual.q_learning_convergence,
                    individual.exploration_efficiency,
                    individual.action_diversity_score,
                    individual.motor_efficiency_score,
                    q_learning.policy_stability if q_learning else 0.0,
                    q_learning.sample_efficiency if q_learning else 0.0
                ))
                
                # Store Q-learning specific metrics
                if q_learning:
                    cursor.execute('''
                        INSERT INTO q_learning_metrics (
                            timestamp, generation, agent_id, convergence_rate,
                            q_table_size, exploration_balance, learning_plateau
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        comprehensive_metrics.timestamp,
                        comprehensive_metrics.generation,
                        robot_id,
                        q_learning.convergence_rate,
                        q_learning.q_table_size,
                        q_learning.exploration_exploitation_balance,
                        q_learning.learning_plateau_detection
                    ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"⚠️  Error storing metrics in database: {e}")
    
    def _update_current_metrics(self, comprehensive_metrics):
        """Update current metrics cache for API endpoints."""
        try:
            with self.metrics_lock:
                # Population metrics
                pop_metrics = comprehensive_metrics.population_metrics
                fitness_dist = pop_metrics.fitness_distribution_analysis
                
                self.current_metrics['population'] = {
                    'timestamp': comprehensive_metrics.timestamp,
                    'generation': comprehensive_metrics.generation,
                    'step_count': comprehensive_metrics.step_count,
                    'population_size': pop_metrics.population_size,
                    'best_fitness': fitness_dist.get('max', 0.0),
                    'avg_fitness': fitness_dist.get('mean', 0.0),
                    'fitness_std': fitness_dist.get('std', 0.0),
                    'genotypic_diversity': pop_metrics.genotypic_diversity,
                    'phenotypic_diversity': pop_metrics.phenotypic_diversity,
                    'behavioral_diversity': pop_metrics.behavioral_diversity,
                    'extinction_risk': pop_metrics.extinction_risk,
                    'species_count': pop_metrics.speciation_dynamics.get('species_count', 1),
                    'pareto_front_size': pop_metrics.pareto_front_analysis.get('pareto_front_size', 0)
                }
                
                # Training metrics
                train_metrics = comprehensive_metrics.training_metrics
                
                self.current_metrics['training'] = {
                    'timestamp': comprehensive_metrics.timestamp,
                    'generation': comprehensive_metrics.generation,
                    'convergence_speed': train_metrics.convergence_speed,
                    'training_variance': train_metrics.training_variance,
                    'cpu_usage': train_metrics.cpu_usage,
                    'memory_usage': train_metrics.memory_usage,
                    'training_fps': train_metrics.training_fps,
                    'plateau_detected': train_metrics.plateau_detection,
                    'improvement_rate': train_metrics.improvement_rate,
                    'computational_efficiency': train_metrics.computational_efficiency
                }
                
                # Robot metrics summary
                robot_summaries = []
                for robot_id, individual in comprehensive_metrics.individual_metrics.items():
                    q_learning = comprehensive_metrics.q_learning_metrics.get(robot_id)
                    exploration = comprehensive_metrics.exploration_metrics.get(robot_id)
                    
                    summary = {
                        'agent_id': robot_id,
                        'q_learning_convergence': individual.q_learning_convergence,
                        'exploration_efficiency': individual.exploration_efficiency,
                        'action_diversity': individual.action_diversity_score,
                        'motor_efficiency': individual.motor_efficiency_score,
                        'policy_stability': q_learning.policy_stability if q_learning else 0.0,
                        'sample_efficiency': q_learning.sample_efficiency if q_learning else 0.0,
                        'state_coverage': exploration.state_space_coverage if exploration else 0.0,
                        'exploration_redundancy': exploration.exploration_redundancy if exploration else 0.0
                    }
                    robot_summaries.append(summary)
                
                self.current_metrics['robots'] = robot_summaries
                
                # Update reward signal metrics if available
                self._update_reward_signal_metrics()
                
        except Exception as e:
            print(f"⚠️  Error updating current metrics: {e}")
    
    def _update_reward_signal_metrics(self):
        """Update reward signal quality metrics in current metrics cache."""
        try:
            # Import reward signal adapter
            from .reward_signal_integration import reward_signal_adapter
            
            # Get all reward signal metrics
            all_reward_metrics = reward_signal_adapter.get_all_reward_metrics()
            reward_status = reward_signal_adapter.get_system_status()
            
            if all_reward_metrics:
                # Calculate aggregate statistics
                quality_scores = [m.quality_score for m in all_reward_metrics.values()]
                snr_values = [m.signal_to_noise_ratio for m in all_reward_metrics.values()]
                consistency_values = [m.reward_consistency for m in all_reward_metrics.values()]
                sparsity_values = [m.reward_sparsity for m in all_reward_metrics.values()]
                
                # Count agents by quality tier
                excellent = len([m for m in all_reward_metrics.values() if m.quality_score >= 0.8])
                good = len([m for m in all_reward_metrics.values() if 0.6 <= m.quality_score < 0.8])
                fair = len([m for m in all_reward_metrics.values() if 0.4 <= m.quality_score < 0.6])
                poor = len([m for m in all_reward_metrics.values() if 0.2 <= m.quality_score < 0.4])
                very_poor = len([m for m in all_reward_metrics.values() if m.quality_score < 0.2])
                
                # Count common issues
                all_issues = []
                for metrics in all_reward_metrics.values():
                    all_issues.extend([issue.value for issue in metrics.quality_issues])
                
                issue_counts = {}
                for issue in all_issues:
                    issue_counts[issue] = issue_counts.get(issue, 0) + 1
                
                self.current_metrics['reward_signals'] = {
                    'timestamp': time.time(),
                    'total_agents': len(all_reward_metrics),
                    'total_rewards_recorded': reward_status.get('total_rewards_recorded', 0),
                    'active': reward_status.get('active', False),
                    
                    # Aggregate quality metrics
                    'avg_quality_score': sum(quality_scores) / len(quality_scores),
                    'avg_signal_to_noise_ratio': sum(snr_values) / len(snr_values),
                    'avg_consistency': sum(consistency_values) / len(consistency_values),
                    'avg_sparsity': sum(sparsity_values) / len(sparsity_values),
                    'min_quality_score': min(quality_scores),
                    'max_quality_score': max(quality_scores),
                    
                    # Quality distribution
                    'agents_excellent': excellent,
                    'agents_good': good,
                    'agents_fair': fair,
                    'agents_poor': poor,
                    'agents_very_poor': very_poor,
                    
                    # Issue tracking
                    'agents_with_issues': len([m for m in all_reward_metrics.values() if m.quality_issues]),
                    'sparse_reward_agents': issue_counts.get('sparse_rewards', 0),
                    'noisy_reward_agents': issue_counts.get('noisy_rewards', 0),
                    'inconsistent_reward_agents': issue_counts.get('inconsistent_rewards', 0),
                    'poor_exploration_agents': issue_counts.get('poor_exploration_incentive', 0),
                    
                    # Performance indicators
                    'agents_with_good_rewards': excellent + good,
                    'percentage_good_quality': (excellent + good) / len(all_reward_metrics) * 100 if all_reward_metrics else 0
                }
                
                # Store reward signal metrics in database
                self._store_reward_signal_metrics_in_db(all_reward_metrics)
                
            else:
                self.current_metrics['reward_signals'] = {
                    'timestamp': time.time(),
                    'total_agents': 0,
                    'total_rewards_recorded': reward_status.get('total_rewards_recorded', 0),
                    'active': reward_status.get('active', False),
                    'status': 'no_data'
                }
                
        except Exception as e:
            print(f"⚠️  Error updating reward signal metrics: {e}")
            # Set fallback metrics
            self.current_metrics['reward_signals'] = {
                'timestamp': time.time(),
                'status': 'error',
                'error': str(e)
            }
    
    def _store_reward_signal_metrics_in_db(self, reward_metrics):
        """Store reward signal quality metrics in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            current_time = time.time()
            
            for agent_id, metrics in reward_metrics.items():
                quality_issues_str = ','.join([issue.value for issue in metrics.quality_issues])
                
                cursor.execute('''
                    INSERT INTO reward_signal_metrics (
                        timestamp, agent_id, quality_score, signal_to_noise_ratio,
                        reward_consistency, reward_sparsity, exploration_incentive,
                        convergence_support, behavioral_alignment, reward_mean,
                        reward_std, steps_analyzed, total_rewards_received, quality_issues
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    current_time,
                    agent_id,
                    metrics.quality_score,
                    metrics.signal_to_noise_ratio,
                    metrics.reward_consistency,
                    metrics.reward_sparsity,
                    metrics.exploration_incentive,
                    metrics.convergence_support,
                    metrics.behavioral_alignment,
                    metrics.reward_mean,
                    metrics.reward_std,
                    metrics.steps_analyzed,
                    metrics.total_rewards_received,
                    quality_issues_str
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"⚠️  Error storing reward signal metrics in database: {e}")
    
    def _generate_alerts(self) -> List[Dict[str, Any]]:
        """Generate system alerts for monitoring."""
        alerts = []
        
        try:
            with self.metrics_lock:
                current_time = time.time()
                
                # Training alerts
                training = self.current_metrics.get('training', {})
                
                if training.get('plateau_detected', False):
                    alerts.append({
                        'level': 'warning',
                        'type': 'training_plateau',
                        'message': 'Training plateau detected',
                        'timestamp': current_time,
                        'value': True
                    })
                
                if training.get('memory_usage', 0) > 2000:  # 2GB threshold
                    alerts.append({
                        'level': 'warning',
                        'type': 'high_memory_usage',
                        'message': f"High memory usage: {training.get('memory_usage', 0):.1f} MB",
                        'timestamp': current_time,
                        'value': training.get('memory_usage', 0)
                    })
                
                if training.get('cpu_usage', 0) > 90:
                    alerts.append({
                        'level': 'warning',
                        'type': 'high_cpu_usage',
                        'message': f"High CPU usage: {training.get('cpu_usage', 0):.1f}%",
                        'timestamp': current_time,
                        'value': training.get('cpu_usage', 0)
                    })
                
                # Population alerts
                population = self.current_metrics.get('population', {})
                
                if population.get('extinction_risk', 0) > 0.7:
                    alerts.append({
                        'level': 'critical',
                        'type': 'extinction_risk',
                        'message': f"High extinction risk: {population.get('extinction_risk', 0):.2f}",
                        'timestamp': current_time,
                        'value': population.get('extinction_risk', 0)
                    })
                
                if population.get('genotypic_diversity', 0) < 0.3:
                    alerts.append({
                        'level': 'warning',
                        'type': 'low_diversity',
                        'message': f"Low genetic diversity: {population.get('genotypic_diversity', 0):.2f}",
                        'timestamp': current_time,
                        'value': population.get('genotypic_diversity', 0)
                    })
                
                # Robot performance alerts
                robots = self.current_metrics.get('robots', [])
                poor_performers = [r for r in robots if r.get('q_learning_convergence', 0) < 0.2]
                
                if len(poor_performers) > len(robots) * 0.5:  # More than 50% poor performers
                    alerts.append({
                        'level': 'warning',
                        'type': 'poor_learning_performance',
                        'message': f"{len(poor_performers)} robots showing poor learning convergence",
                        'timestamp': current_time,
                        'value': len(poor_performers)
                    })
                
                # Reward signal quality alerts
                reward_signals = self.current_metrics.get('reward_signals', {})
                
                if reward_signals.get('avg_quality_score', 1.0) < 0.3:
                    alerts.append({
                        'level': 'critical',
                        'type': 'poor_reward_quality',
                        'message': f"Poor average reward quality: {reward_signals.get('avg_quality_score', 0):.3f}",
                        'timestamp': current_time,
                        'value': reward_signals.get('avg_quality_score', 0)
                    })
                
                if reward_signals.get('sparse_reward_agents', 0) > len(robots) * 0.3:
                    alerts.append({
                        'level': 'warning',
                        'type': 'sparse_rewards',
                        'message': f"{reward_signals.get('sparse_reward_agents', 0)} agents have sparse rewards",
                        'timestamp': current_time,
                        'value': reward_signals.get('sparse_reward_agents', 0)
                    })
                
                if reward_signals.get('noisy_reward_agents', 0) > len(robots) * 0.2:
                    alerts.append({
                        'level': 'warning',
                        'type': 'noisy_rewards',
                        'message': f"{reward_signals.get('noisy_reward_agents', 0)} agents have noisy rewards",
                        'timestamp': current_time,
                        'value': reward_signals.get('noisy_reward_agents', 0)
                    })
                
        except Exception as e:
            print(f"⚠️  Error generating alerts: {e}")
        
        return alerts
    
    def _generate_grafana_dashboard_config(self) -> Dict[str, Any]:
        """Generate Grafana dashboard configuration."""
        return {
            "dashboard": {
                "id": None,
                "title": "Walker Robot Training Dashboard",
                "tags": ["walker", "robotics", "ml"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "Population Fitness Over Time",
                        "type": "graph",
                        "targets": [
                            {
                                "target": "api/metrics/history/population",
                                "refId": "A"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "Fitness",
                                "min": None,
                                "max": None
                            }
                        ],
                        "xAxes": [
                            {
                                "mode": "time",
                                "name": "time",
                                "values": []
                            }
                        ],
                        "gridPos": {
                            "h": 8,
                            "w": 12,
                            "x": 0,
                            "y": 0
                        }
                    },
                    {
                        "id": 2,
                        "title": "Population Diversity",
                        "type": "graph",
                        "targets": [
                            {
                                "target": "api/metrics/history/population",
                                "refId": "B"
                            }
                        ],
                        "gridPos": {
                            "h": 8,
                            "w": 12,
                            "x": 12,
                            "y": 0
                        }
                    },
                    {
                        "id": 3,
                        "title": "System Resources",
                        "type": "graph",
                        "targets": [
                            {
                                "target": "api/metrics/history/training",
                                "refId": "C"
                            }
                        ],
                        "gridPos": {
                            "h": 8,
                            "w": 24,
                            "x": 0,
                            "y": 8
                        }
                    },
                    {
                        "id": 4,
                        "title": "Training Status",
                        "type": "stat",
                        "targets": [
                            {
                                "target": "api/metrics/training",
                                "refId": "D"
                            }
                        ],
                        "gridPos": {
                            "h": 4,
                            "w": 6,
                            "x": 0,
                            "y": 16
                        }
                    },
                    {
                        "id": 5,
                        "title": "Active Alerts",
                        "type": "alertlist",
                        "targets": [
                            {
                                "target": "api/alerts",
                                "refId": "E"
                            }
                        ],
                        "gridPos": {
                            "h": 4,
                            "w": 18,
                            "x": 6,
                            "y": 16
                        }
                    },
                    {
                        "id": 6,
                        "title": "Reward Signal Quality Score",
                        "type": "gauge",
                        "datasource": "Prometheus",
                        "targets": [
                            {
                                "expr": "walker_reward_signals_avg_quality_score",
                                "refId": "F"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "thresholds": {
                                    "steps": [
                                        {"color": "red", "value": 0},
                                        {"color": "yellow", "value": 0.4},
                                        {"color": "green", "value": 0.6}
                                    ]
                                },
                                "min": 0,
                                "max": 1,
                                "unit": "none"
                            }
                        },
                        "gridPos": {
                            "h": 8,
                            "w": 6,
                            "x": 0,
                            "y": 20
                        }
                    },
                    {
                        "id": 7,
                        "title": "Reward Signal Metrics",
                        "type": "timeseries",
                        "datasource": "Prometheus",
                        "targets": [
                            {
                                "expr": "walker_reward_signals_avg_signal_to_noise_ratio",
                                "legendFormat": "Signal-to-Noise Ratio",
                                "refId": "G"
                            },
                            {
                                "expr": "walker_reward_signals_avg_consistency",
                                "legendFormat": "Consistency",
                                "refId": "H"
                            },
                            {
                                "expr": "1 - walker_reward_signals_avg_sparsity",
                                "legendFormat": "Reward Density",
                                "refId": "I"
                            }
                        ],
                        "gridPos": {
                            "h": 8,
                            "w": 12,
                            "x": 6,
                            "y": 20
                        }
                    },
                    {
                        "id": 8,
                        "title": "Reward Quality Distribution",
                        "type": "piechart",
                        "datasource": "Prometheus",
                        "targets": [
                            {
                                "expr": "walker_reward_signals_agents_excellent",
                                "legendFormat": "Excellent",
                                "refId": "J"
                            },
                            {
                                "expr": "walker_reward_signals_agents_good",
                                "legendFormat": "Good",
                                "refId": "K"
                            },
                            {
                                "expr": "walker_reward_signals_agents_fair",
                                "legendFormat": "Fair",
                                "refId": "L"
                            },
                            {
                                "expr": "walker_reward_signals_agents_poor + walker_reward_signals_agents_very_poor",
                                "legendFormat": "Poor",
                                "refId": "M"
                            }
                        ],
                        "gridPos": {
                            "h": 8,
                            "w": 6,
                            "x": 18,
                            "y": 20
                        }
                    },
                    {
                        "id": 9,
                        "title": "Reward Signal Issues",
                        "type": "stat",
                        "datasource": "Prometheus",
                        "targets": [
                            {
                                "expr": "walker_reward_signals_sparse_reward_agents",
                                "legendFormat": "Sparse Rewards",
                                "refId": "N"
                            },
                            {
                                "expr": "walker_reward_signals_noisy_reward_agents",
                                "legendFormat": "Noisy Rewards",
                                "refId": "O"
                            },
                            {
                                "expr": "walker_reward_signals_inconsistent_reward_agents",
                                "legendFormat": "Inconsistent",
                                "refId": "P"
                            },
                            {
                                "expr": "walker_reward_signals_poor_exploration_agents",
                                "legendFormat": "Poor Exploration",
                                "refId": "Q"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "thresholds": {
                                    "steps": [
                                        {"color": "green", "value": 0},
                                        {"color": "yellow", "value": 3},
                                        {"color": "red", "value": 6}
                                    ]
                                }
                            }
                        },
                        "gridPos": {
                            "h": 6,
                            "w": 24,
                            "x": 0,
                            "y": 28
                        }
                    }
                ],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "5s"
            }
        }
    
    def export_grafana_dashboard(self, output_path: str = "grafana_dashboard.json"):
        """Export Grafana dashboard configuration to file."""
        try:
            config = self._generate_grafana_dashboard_config()
            
            with open(output_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"📊 Grafana dashboard configuration exported to {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Error exporting Grafana dashboard: {e}")
            return None
    
    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        try:
            with self.metrics_lock:
                metrics_lines = []
                current_time = int(time.time() * 1000)  # Prometheus uses milliseconds
                
                # Population metrics
                population = self.current_metrics.get('population', {})
                for key, value in population.items():
                    if isinstance(value, (int, float)):
                        metrics_lines.append(f"walker_population_{key} {value} {current_time}")
                
                # Training metrics
                training = self.current_metrics.get('training', {})
                for key, value in training.items():
                    if isinstance(value, (int, float)):
                        metrics_lines.append(f"walker_training_{key} {value} {current_time}")
                
                # Robot metrics aggregates
                robots = self.current_metrics.get('robots', [])
                if robots:
                    avg_convergence = sum(r.get('q_learning_convergence', 0) for r in robots) / len(robots)
                    avg_efficiency = sum(r.get('exploration_efficiency', 0) for r in robots) / len(robots)
                    
                    metrics_lines.append(f"walker_robots_avg_convergence {avg_convergence} {current_time}")
                    metrics_lines.append(f"walker_robots_avg_exploration_efficiency {avg_efficiency} {current_time}")
                    metrics_lines.append(f"walker_robots_total_count {len(robots)} {current_time}")
                
                # Reward signal quality metrics
                reward_signals = self.current_metrics.get('reward_signals', {})
                for key, value in reward_signals.items():
                    if isinstance(value, (int, float)) and key != 'timestamp':
                        metrics_lines.append(f"walker_reward_signals_{key} {value} {current_time}")
                
                return '\n'.join(metrics_lines)
                
        except Exception as e:
            print(f"⚠️  Error generating Prometheus metrics: {e}")
            return "" 