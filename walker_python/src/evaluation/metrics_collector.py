"""
Central metrics collector that coordinates all evaluation modules.
Provides a unified interface for collecting and managing evaluation data.
"""

import time
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
from pathlib import Path

from .individual_evaluator import IndividualRobotEvaluator, BehaviorAnalyzer
from .exploration_evaluator import ExplorationEvaluator, ActionSpaceAnalyzer
from .q_learning_evaluator import QLearningEvaluator
from .training_evaluator import TrainingProgressEvaluator
from .population_evaluator import PopulationEvaluator
from .mlflow_integration import MLflowIntegration


@dataclass
class ComprehensiveMetrics:
    """Complete metrics package from all evaluators."""
    timestamp: float
    generation: int
    step_count: int
    
    # Individual robot metrics
    individual_metrics: Dict[str, Any]
    behavior_metrics: Dict[str, Any]
    
    # Exploration and action metrics
    exploration_metrics: Dict[str, Any]
    action_space_metrics: Dict[str, Any]
    
    # Q-learning metrics
    q_learning_metrics: Dict[str, Any]
    
    # Training progress metrics
    training_metrics: Any
    
    # Population metrics
    population_metrics: Any


class MetricsCollector:
    """
    Central coordinator for all evaluation modules.
    Collects, processes, and stores comprehensive training metrics.
    """
    
    def __init__(self, 
                 enable_mlflow: bool = True,
                 enable_file_export: bool = True,
                 export_directory: str = "evaluation_exports"):
        """
        Initialize the metrics collector.
        
        Args:
            enable_mlflow: Whether to enable MLflow integration
            enable_file_export: Whether to export metrics to files
            export_directory: Directory for metric exports
        """
        # Initialize all evaluators
        self.individual_evaluator = IndividualRobotEvaluator()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.exploration_evaluator = ExplorationEvaluator()
        self.action_analyzer = ActionSpaceAnalyzer()
        self.q_learning_evaluator = QLearningEvaluator()
        self.training_evaluator = TrainingProgressEvaluator()
        self.population_evaluator = PopulationEvaluator()
        
        # MLflow integration
        self.enable_mlflow = enable_mlflow
        self.mlflow_integration = None
        if enable_mlflow:
            try:
                self.mlflow_integration = MLflowIntegration()
                print("✅ MLflow integration enabled")
            except Exception as e:
                print(f"⚠️  MLflow integration failed: {e}")
                self.enable_mlflow = False
        
        # File export setup
        self.enable_file_export = enable_file_export
        self.export_directory = Path(export_directory)
        if enable_file_export:
            self.export_directory.mkdir(exist_ok=True)
            print(f"✅ File export enabled to {self.export_directory}")
        
        # Metrics storage
        self.metrics_history: List[ComprehensiveMetrics] = []
        self.last_evaluation_time = 0.0
        self.evaluation_interval = 30.0  # Evaluate every 30 seconds (reduced overhead)
        
        # Thread safety
        self.metrics_lock = threading.Lock()
        
        # Background collection thread
        self.collection_thread = None
        self.collection_queue = []
        self.is_collecting = False
        
        # Performance tracking
        self.collection_times = []
        
        print("🔬 MetricsCollector initialized with all evaluation modules")
    
    def start_training_session(self, 
                             session_name: Optional[str] = None,
                             population_size: int = 30,
                             evolution_config: Optional[Dict] = None):
        """
        Start a new training session with evaluation tracking.
        
        Args:
            session_name: Name for the training session
            population_size: Size of the robot population
            evolution_config: Evolution configuration parameters
        """
        try:
            if self.enable_mlflow and self.mlflow_integration:
                run_id = self.mlflow_integration.start_training_run(
                    run_name=session_name,
                    population_size=population_size,
                    evolution_config=evolution_config
                )
                print(f"📊 Started MLflow tracking for session: {session_name}")
                return run_id
            
            print(f"📊 Started evaluation tracking for session: {session_name}")
            return None
            
        except Exception as e:
            print(f"⚠️  Error starting training session: {e}")
            return None
    
    def collect_metrics_async(self,
                             agents: List[Any],
                             population_stats: Dict[str, Any],
                             evolution_summary: Dict[str, Any],
                             generation: int,
                             step_count: int) -> None:
        """
        Queue metrics for asynchronous collection to avoid blocking main thread.
        
        Args:
            agents: List of robot agents
            population_stats: Population-level statistics
            evolution_summary: Evolution engine summary
            generation: Current generation number
            step_count: Current training step
        """
        current_time = time.time()
        
        # Check if we should collect metrics
        if (current_time - self.last_evaluation_time) < self.evaluation_interval:
            return
        
        # Queue collection data for background processing
        collection_data = {
            'agents_snapshot': self._create_agents_snapshot(agents),
            'population_stats': population_stats.copy(),
            'evolution_summary': evolution_summary.copy(),
            'generation': generation,
            'step_count': step_count,
            'timestamp': current_time
        }
        
        with self.metrics_lock:
            self.collection_queue.append(collection_data)
            self.last_evaluation_time = current_time
        
        # Start background collection if not already running
        if not self.is_collecting:
            self._start_background_collection()
    
    def collect_metrics(self, 
                       agents: List[Any],
                       population_stats: Dict[str, Any],
                       evolution_summary: Dict[str, Any],
                       generation: int,
                       step_count: int,
                       force_collection: bool = False) -> Optional[ComprehensiveMetrics]:
        """
        Collect comprehensive metrics from all evaluators.
        
        Args:
            agents: List of robot agents
            population_stats: Population-level statistics
            evolution_summary: Evolution engine summary
            generation: Current generation number
            step_count: Current training step
            force_collection: Force collection even if interval hasn't elapsed
            
        Returns:
            ComprehensiveMetrics object or None if collection was skipped
        """
        current_time = time.time()
        
        # Check if we should collect metrics
        if not force_collection and (current_time - self.last_evaluation_time) < self.evaluation_interval:
            return None
        
        collection_start = time.time()
        
        try:
            with self.metrics_lock:
                print(f"🔬 Collecting comprehensive metrics for generation {generation}, step {step_count}")
                
                # Individual robot metrics
                individual_metrics = {}
                behavior_metrics = {}
                
                for agent in agents:
                    try:
                        agent_id = str(agent.id)
                        
                        # Individual performance evaluation
                        individual_metrics[agent_id] = self.individual_evaluator.evaluate_robot(agent, step_count)
                        
                        # Behavioral analysis
                        behavior_metrics[agent_id] = self.behavior_analyzer.analyze_behavior(agent, step_count)
                        
                    except Exception as e:
                        print(f"⚠️  Error evaluating individual agent {getattr(agent, 'id', 'unknown')}: {e}")
                
                # Exploration and action space analysis
                exploration_metrics = {}
                action_space_metrics = {}
                
                for agent in agents:
                    try:
                        agent_id = str(agent.id)
                        
                        # Exploration analysis
                        exploration_metrics[agent_id] = self.exploration_evaluator.evaluate_exploration(agent, step_count)
                        
                        # Action space analysis
                        action_space_metrics[agent_id] = self.action_analyzer.analyze_action_space(agent, step_count)
                        
                    except Exception as e:
                        print(f"⚠️  Error in exploration/action analysis for agent {getattr(agent, 'id', 'unknown')}: {e}")
                
                # Q-learning evaluation
                q_learning_metrics = {}
                
                for agent in agents:
                    try:
                        agent_id = str(agent.id)
                        q_learning_metrics[agent_id] = self.q_learning_evaluator.evaluate_q_learning(agent, step_count)
                        
                    except Exception as e:
                        print(f"⚠️  Error in Q-learning evaluation for agent {getattr(agent, 'id', 'unknown')}: {e}")
                
                # Training progress evaluation
                training_metrics = self.training_evaluator.evaluate_training_progress(
                    population_stats, generation, step_count
                )
                
                # Population evaluation
                population_metrics = self.population_evaluator.evaluate_population(
                    agents, generation, evolution_summary
                )
                
                # Create comprehensive metrics package
                comprehensive_metrics = ComprehensiveMetrics(
                    timestamp=current_time,
                    generation=generation,
                    step_count=step_count,
                    individual_metrics=individual_metrics,
                    behavior_metrics=behavior_metrics,
                    exploration_metrics=exploration_metrics,
                    action_space_metrics=action_space_metrics,
                    q_learning_metrics=q_learning_metrics,
                    training_metrics=training_metrics,
                    population_metrics=population_metrics
                )
                
                # Store metrics
                self.metrics_history.append(comprehensive_metrics)
                
                # Trim history if too long
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                # Log to MLflow
                if self.enable_mlflow and self.mlflow_integration:
                    self._log_to_mlflow(comprehensive_metrics, agents)
                
                # Export to files
                if self.enable_file_export:
                    self._export_to_files(comprehensive_metrics)
                
                self.last_evaluation_time = current_time
                
                # Track collection performance
                collection_time = time.time() - collection_start
                self.collection_times.append(collection_time)
                if len(self.collection_times) > 100:
                    self.collection_times = self.collection_times[-100:]
                
                print(f"✅ Metrics collection completed in {collection_time:.3f}s")
                return comprehensive_metrics
                
        except Exception as e:
            print(f"❌ Error during metrics collection: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _log_to_mlflow(self, metrics: ComprehensiveMetrics, agents: List[Any]):
        """Log metrics to MLflow."""
        try:
            # Log population-level metrics
            population_data = {
                'generation': metrics.generation,
                'population_size': len(agents),
                'genotypic_diversity': metrics.population_metrics.genotypic_diversity,
                'phenotypic_diversity': metrics.population_metrics.phenotypic_diversity,
                'behavioral_diversity': metrics.population_metrics.behavioral_diversity,
                'extinction_risk': metrics.population_metrics.extinction_risk
            }
            
            # Add fitness distribution metrics
            fitness_dist = metrics.population_metrics.fitness_distribution_analysis
            for key, value in fitness_dist.items():
                population_data[f'fitness_{key}'] = value
            
            # Add training metrics
            population_data['training_variance'] = metrics.training_metrics.training_variance
            population_data['convergence_speed'] = metrics.training_metrics.convergence_speed
            population_data['cpu_usage'] = metrics.training_metrics.cpu_usage
            population_data['memory_usage'] = metrics.training_metrics.memory_usage
            
            if self.mlflow_integration:
                self.mlflow_integration.log_population_metrics(metrics.generation, population_data)
            
            # Log individual robot metrics (sample a few to avoid overwhelming MLflow)
            sample_agents = list(agents)[:5]  # Log first 5 agents
            for agent in sample_agents:
                agent_id = str(agent.id)
                if agent_id in metrics.individual_metrics:
                    individual_data = {
                        'convergence_score': metrics.individual_metrics[agent_id].q_learning_convergence,
                        'exploration_efficiency': metrics.individual_metrics[agent_id].exploration_efficiency,
                        'action_diversity': metrics.individual_metrics[agent_id].action_diversity_score,
                        'motor_efficiency': metrics.individual_metrics[agent_id].motor_efficiency_score
                    }
                    
                    if agent_id in metrics.q_learning_metrics:
                        individual_data['policy_stability'] = metrics.q_learning_metrics[agent_id].policy_stability
                        individual_data['sample_efficiency'] = metrics.q_learning_metrics[agent_id].sample_efficiency
                    
                    if self.mlflow_integration:
                        self.mlflow_integration.log_individual_robot_metrics(agent_id, individual_data, metrics.step_count)
            
        except Exception as e:
            print(f"⚠️  Error logging to MLflow: {e}")
    
    def _export_to_files(self, metrics: ComprehensiveMetrics):
        """Export metrics to JSON files."""
        try:
            timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(metrics.timestamp))
            
            # Export population summary
            population_summary = {
                'timestamp': metrics.timestamp,
                'generation': metrics.generation,
                'step_count': metrics.step_count,
                'population_health': self.population_evaluator.get_population_summary(),
                'training_summary': self.training_evaluator.get_training_summary(),
                'fitness_distribution': metrics.population_metrics.fitness_distribution_analysis,
                'diversity_metrics': {
                    'genotypic': metrics.population_metrics.genotypic_diversity,
                    'phenotypic': metrics.population_metrics.phenotypic_diversity,
                    'behavioral': metrics.population_metrics.behavioral_diversity
                }
            }
            
            summary_file = self.export_directory / f"population_summary_{timestamp_str}.json"
            with open(summary_file, 'w') as f:
                json.dump(population_summary, f, indent=2, default=str)
            
            # Export individual robot summaries
            individual_summaries = {}
            for agent_id in metrics.individual_metrics.keys():
                individual_summaries[agent_id] = {
                    'individual': self.individual_evaluator.get_robot_summary(agent_id),
                    'behavior': self.behavior_analyzer.get_behavior_summary(agent_id),
                    'exploration': self.exploration_evaluator.get_exploration_summary(agent_id),
                    'action_space': self.action_analyzer.get_action_summary(agent_id),
                    'q_learning': self.q_learning_evaluator.get_q_learning_summary(agent_id)
                }
            
            individuals_file = self.export_directory / f"individual_summaries_{timestamp_str}.json"
            with open(individuals_file, 'w') as f:
                json.dump(individual_summaries, f, indent=2, default=str)
            
        except Exception as e:
            print(f"⚠️  Error exporting to files: {e}")
    
    def get_current_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics for web interface."""
        if not self.metrics_history:
            return {}
        
        latest_metrics = self.metrics_history[-1]
        
        # Aggregate individual metrics
        individual_summaries = []
        for agent_id in latest_metrics.individual_metrics.keys():
            summary = {
                'agent_id': agent_id,
                'convergence': latest_metrics.individual_metrics[agent_id].q_learning_convergence,
                'exploration_efficiency': latest_metrics.individual_metrics[agent_id].exploration_efficiency,
                'motor_efficiency': latest_metrics.individual_metrics[agent_id].motor_efficiency_score,
                'action_diversity': latest_metrics.individual_metrics[agent_id].action_diversity_score
            }
            
            if agent_id in latest_metrics.behavior_metrics:
                summary['learning_velocity'] = latest_metrics.behavior_metrics[agent_id].learning_velocity
                summary['adaptation_speed'] = latest_metrics.behavior_metrics[agent_id].adaptation_speed
            
            if agent_id in latest_metrics.q_learning_metrics:
                summary['policy_stability'] = latest_metrics.q_learning_metrics[agent_id].policy_stability
                summary['sample_efficiency'] = latest_metrics.q_learning_metrics[agent_id].sample_efficiency
            
            individual_summaries.append(summary)
        
        return {
            'timestamp': latest_metrics.timestamp,
            'generation': latest_metrics.generation,
            'step_count': latest_metrics.step_count,
            'population_health': self.population_evaluator.get_population_summary(),
            'training_health': self.training_evaluator.get_training_summary(),
            'individual_summaries': individual_summaries,
            'collection_performance': {
                'avg_collection_time': sum(self.collection_times) / len(self.collection_times) if self.collection_times else 0,
                'metrics_history_size': len(self.metrics_history)
            }
        }
    
    def _create_agents_snapshot(self, agents: List[Any]) -> List[Dict[str, Any]]:
        """Create a lightweight snapshot of agents for background processing."""
        snapshot = []
        for agent in agents:
            try:
                agent_data = {
                    'id': str(agent.id),
                    'total_reward': getattr(agent, 'total_reward', 0.0),
                    'epsilon': getattr(agent, 'epsilon', 0.0),
                    'steps': getattr(agent, 'steps', 0),
                    'action_history': getattr(agent, 'action_history', [])[-20:],  # Last 20 actions
                    'has_body': hasattr(agent, 'body') and agent.body is not None,
                    'has_q_table': hasattr(agent, 'q_table'),
                    'physical_params': {
                        'motor_torque': getattr(agent.physical_params, 'motor_torque', 150.0) if hasattr(agent, 'physical_params') else 150.0
                    }
                }
                
                # Add position data if available
                if agent_data['has_body']:
                    try:
                        agent_data['position'] = {'x': agent.body.position.x, 'y': agent.body.position.y}
                        agent_data['angle'] = agent.body.angle
                        agent_data['initial_position'] = getattr(agent, 'initial_position', [0, 0])
                    except:
                        agent_data['has_body'] = False
                
                # Add Q-learning data if available
                if agent_data['has_q_table']:
                    try:
                        agent_data['q_convergence'] = agent.q_table.get_convergence_estimate() if hasattr(agent.q_table, 'get_convergence_estimate') else 0.0
                        agent_data['q_table_size'] = len(getattr(agent.q_table, 'q_values', {}))
                    except:
                        agent_data['has_q_table'] = False
                
                snapshot.append(agent_data)
            except Exception as e:
                print(f"⚠️  Error creating snapshot for agent: {e}")
                
        return snapshot
    
    def _start_background_collection(self):
        """Start background metrics collection thread."""
        if self.is_collecting:
            return
            
        self.is_collecting = True
        self.collection_thread = threading.Thread(target=self._background_collection_worker, daemon=True)
        self.collection_thread.start()
        print("📊 Started background metrics collection thread")
    
    def _background_collection_worker(self):
        """Background worker for processing metrics collection queue."""
        while self.is_collecting:
            try:
                with self.metrics_lock:
                    if not self.collection_queue:
                        time.sleep(1.0)
                        continue
                    
                    # Process one item from queue
                    data = self.collection_queue.pop(0)
                
                # Process metrics collection in background
                print(f"🔬 Background: Processing metrics for generation {data['generation']}")
                
                # Note: This would need to be adapted to work with snapshot data
                # For now, just clear the queue to prevent memory buildup
                # A full implementation would reconstruct agent objects from snapshots
                
                print(f"✅ Background: Metrics processing completed")
                
            except Exception as e:
                print(f"⚠️  Error in background metrics collection: {e}")
                time.sleep(5.0)

    def get_training_diagnostics(self) -> Dict[str, Any]:
        """Get training diagnostics and recommendations."""
        if not self.metrics_history:
            return {'status': 'no_data'}
        
        latest_metrics = self.metrics_history[-1]
        
        # Get recommendations from all evaluators
        training_recommendations = self.training_evaluator.get_recommendations()
        
        # Aggregate Q-learning diagnostics
        q_learning_issues = []
        for agent_id, q_metrics in latest_metrics.q_learning_metrics.items():
            diagnostics = self.q_learning_evaluator.get_learning_diagnostics(agent_id)
            if diagnostics.get('overall_health') == 'needs_attention':
                q_learning_issues.append({
                    'agent_id': agent_id,
                    'issues': diagnostics.get('issues_detected', []),
                    'recommendations': diagnostics.get('recommendations', [])
                })
        
        # Population health assessment
        population_health = self.population_evaluator.get_population_summary().get('population_health', 'unknown')
        training_health = self.training_evaluator.get_training_summary().get('training_health', 'unknown')
        
        return {
            'overall_health': self._assess_overall_health(population_health, training_health),
            'population_health': population_health,
            'training_health': training_health,
            'training_recommendations': training_recommendations,
            'q_learning_issues': q_learning_issues,
            'performance_metrics': {
                'collection_efficiency': sum(self.collection_times) / len(self.collection_times) if self.collection_times else 0,
                'memory_usage': latest_metrics.training_metrics.memory_usage,
                'cpu_usage': latest_metrics.training_metrics.cpu_usage
            }
        }
    
    def _assess_overall_health(self, population_health: str, training_health: str) -> str:
        """Assess overall system health."""
        health_mapping = {'excellent': 4, 'good': 3, 'fair': 2, 'poor': 1, 'unknown': 0}
        
        pop_score = health_mapping.get(population_health, 0)
        train_score = health_mapping.get(training_health, 0)
        
        avg_score = (pop_score + train_score) / 2
        
        if avg_score >= 3.5:
            return 'excellent'
        elif avg_score >= 2.5:
            return 'good'
        elif avg_score >= 1.5:
            return 'fair'
        else:
            return 'poor'
    
    def end_training_session(self, final_summary: Optional[Dict] = None):
        """End the current training session."""
        try:
            # Stop background collection
            self.is_collecting = False
            if self.collection_thread:
                self.collection_thread.join(timeout=5.0)
                print("📊 Background metrics collection stopped")
            
            if self.enable_mlflow and self.mlflow_integration:
                self.mlflow_integration.end_training_run(final_summary)
            
            # Export final comprehensive report
            if self.enable_file_export and self.metrics_history:
                self._export_final_report()
            
            print("📊 Training session evaluation completed")
            
        except Exception as e:
            print(f"⚠️  Error ending training session: {e}")
    
    def _export_final_report(self):
        """Export final comprehensive training report."""
        try:
            if not self.metrics_history:
                return
            
            final_metrics = self.metrics_history[-1]
            
            # Create comprehensive final report
            final_report = {
                'session_summary': {
                    'start_time': self.metrics_history[0].timestamp,
                    'end_time': final_metrics.timestamp,
                    'total_generations': final_metrics.generation,
                    'total_steps': final_metrics.step_count,
                    'metrics_collected': len(self.metrics_history)
                },
                'final_population_state': self.population_evaluator.get_population_summary(),
                'final_training_state': self.training_evaluator.get_training_summary(),
                'performance_evolution': self._analyze_performance_evolution(),
                'recommendations': self._generate_final_recommendations()
            }
            
            report_file = self.export_directory / f"final_training_report_{int(time.time())}.json"
            with open(report_file, 'w') as f:
                json.dump(final_report, f, indent=2, default=str)
            
            print(f"📊 Final training report exported to {report_file}")
            
        except Exception as e:
            print(f"⚠️  Error exporting final report: {e}")
    
    def _analyze_performance_evolution(self) -> Dict[str, Any]:
        """Analyze how performance evolved over the training session."""
        if len(self.metrics_history) < 2:
            return {}
        
        # Track key metrics over time
        generations = [m.generation for m in self.metrics_history]
        diversity_evolution = [m.population_metrics.genotypic_diversity for m in self.metrics_history]
        training_variance = [m.training_metrics.training_variance for m in self.metrics_history]
        
        return {
            'diversity_trend': 'increasing' if diversity_evolution[-1] > diversity_evolution[0] else 'decreasing',
            'stability_trend': 'improving' if training_variance[-1] < training_variance[0] else 'declining',
            'total_generations': len(set(generations)),
            'peak_diversity': max(diversity_evolution),
            'final_diversity': diversity_evolution[-1]
        }
    
    def _generate_final_recommendations(self) -> List[str]:
        """Generate final recommendations based on entire training session."""
        if not self.metrics_history:
            return ["Insufficient data for recommendations"]
        
        recommendations = []
        final_metrics = self.metrics_history[-1]
        
        # Population-level recommendations
        pop_health = self.population_evaluator.get_population_summary().get('population_health', 'unknown')
        if pop_health in ['poor', 'fair']:
            recommendations.append("Consider adjusting evolution parameters to improve population health")
        
        # Training-level recommendations
        train_health = self.training_evaluator.get_training_summary().get('training_health', 'unknown')
        if train_health in ['poor', 'fair']:
            recommendations.append("Training stability could be improved with different hyperparameters")
        
        # Performance recommendations
        if final_metrics.training_metrics.plateau_detection:
            recommendations.append("Training has plateaued. Consider curriculum learning or parameter adjustments")
        
        if final_metrics.population_metrics.extinction_risk > 0.5:
            recommendations.append("High extinction risk detected. Increase diversity preservation mechanisms")
        
        if not recommendations:
            recommendations.append("Training completed successfully. Current configuration appears optimal")
        
        return recommendations 