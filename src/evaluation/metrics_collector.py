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
import mlflow
import numpy as np

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
        
        # MLflow integration - Accept shared instance from training environment
        self.enable_mlflow = enable_mlflow  # Use the parameter value
        self.mlflow_integration = None  # Will be set by training environment if needed
        
        # File export setup
        self.enable_file_export = enable_file_export
        self.export_directory = Path(export_directory)
        if enable_file_export:
            self.export_directory.mkdir(exist_ok=True)
            print(f"✅ File export enabled to {self.export_directory}")
        
        # Metrics storage
        self.metrics_history: List[ComprehensiveMetrics] = []
        self.last_evaluation_time = 0.0
        self.evaluation_interval = 10.0  # Evaluate every 10 seconds (faster for better visibility)
        
        # Thread safety
        self.metrics_lock = threading.Lock()
        
        # Background collection thread
        self.collection_thread = None
        self.collection_queue = []
        self.is_collecting = False
        
        # Performance tracking
        self.collection_times = []
        
        print("🔬 MetricsCollector initialized with all evaluation modules")
    
    def set_mlflow_integration(self, mlflow_integration, enable: bool = True):
        """
        Set the shared MLflow integration instance from the training environment.
        
        Args:
            mlflow_integration: The MLflow integration instance from training environment
            enable: Whether to enable MLflow logging
        """
        try:
            self.mlflow_integration = mlflow_integration
            self.enable_mlflow = enable
            
            if mlflow_integration and enable:
                print(f"✅ MetricsCollector: MLflow integration enabled with shared instance")
            else:
                print(f"⚠️ MetricsCollector: MLflow integration disabled")
                
        except Exception as e:
            print(f"❌ Error setting MLflow integration: {e}")
            self.enable_mlflow = False
    
    def start_training_session(self, 
                             session_name: Optional[str] = None,
                             population_size: int = 30,
                             evolution_config: Optional[Dict] = None):
        """
        Prepare the metrics collector for a new training session.
        Note: MLflow run should be started by the TrainingEnvironment, not here.
        
        Args:
            session_name: Name for the training session
            population_size: Size of the robot population
            evolution_config: Evolution configuration parameters
        """
        try:
            # FIXED: Don't start MLflow run here - TrainingEnvironment handles that
            # This prevents duplicate run creation
            
            if self.enable_mlflow and self.mlflow_integration:
                # Just verify that there's an active run
                if hasattr(self.mlflow_integration, 'current_run') and self.mlflow_integration.current_run:
                    run_id = self.mlflow_integration.current_run.info.run_id
                    print(f"📊 Metrics collector connected to MLflow run: {run_id}")
                    return run_id
                else:
                    print(f"⚠️  No active MLflow run found. TrainingEnvironment should start the run first.")
            
            print(f"📊 Metrics collector ready for session: {session_name}")
            return None
            
        except Exception as e:
            print(f"⚠️  Error preparing training session: {e}")
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
        
        # NEW: Collect system metrics data for background logging
        system_metrics_data = self._collect_system_metrics_data()
        
        # Queue collection data for background processing
        collection_data = {
            'agents_snapshot': self._create_agents_snapshot(agents),
            'population_stats': population_stats.copy(),
            'evolution_summary': evolution_summary.copy(),
            'generation': generation,
            'step_count': step_count,
            'timestamp': current_time,
            # NEW: Add system metrics data for MLflow logging
            **system_metrics_data
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
                        # Use the correct method name that exists on QLearningEvaluator
                        q_learning_metrics[agent_id] = self.q_learning_evaluator._evaluate_agent_performance(agent)
                        
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
        """Log cleaned up metrics to MLflow - focus on aggregate learning performance with organized sections."""
        try:
            # Calculate proper step for time series visualization
            current_time = time.time()
            step_counter = int((current_time - 1752340000) / 10)  # 10-second intervals from baseline
            
            # SECTION 1: POPULATION HEALTH & FITNESS
            fitness_metrics = {
                'population_health/generation': metrics.generation,
                'population_health/population_size': len(agents),
                'population_health/genotypic_diversity': metrics.population_metrics.genotypic_diversity,
                'population_health/phenotypic_diversity': metrics.population_metrics.phenotypic_diversity,
                'population_health/behavioral_diversity': metrics.population_metrics.behavioral_diversity,
                'population_health/extinction_risk': metrics.population_metrics.extinction_risk,
            }
            
            # Add fitness distribution metrics to population health
            fitness_dist = metrics.population_metrics.fitness_distribution_analysis
            for key, value in fitness_dist.items():
                fitness_metrics[f'population_health/fitness_{key}'] = value
            
            # SECTION 2: LEARNING PROGRESS & TRAINING
            training_metrics = {
                'learning_progress/training_variance': metrics.training_metrics.training_variance,
                'learning_progress/convergence_speed': metrics.training_metrics.convergence_speed,
            }
            
            # SECTION 3: EXPLORATION & EXPLOITATION
            # Calculate population-level learning performance instead of individual robot details
            learning_performance = self._calculate_aggregate_learning_metrics(agents, metrics)
            exploration_metrics = {}
            for key, value in learning_performance.items():
                if 'epsilon' in key.lower() or 'exploration' in key.lower() or 'exploit' in key.lower():
                    exploration_metrics[f'exploration/{key}'] = value
                elif 'learning' in key.lower() or 'convergence' in key.lower():
                    training_metrics[f'learning_progress/{key}'] = value
            
            # SECTION 4: NETWORK PERFORMANCE
            # Calculate aggregate training statistics instead of logging each agent
            individual_loss_metrics = self._capture_individual_network_losses(agents)
            network_metrics = {}
            
            if individual_loss_metrics:
                # Aggregate network training statistics across all agents
                all_network_losses = []
                all_network_epsilons = []
                all_training_steps = []
                all_buffer_sizes = []
                all_q_values = []
                
                for agent_id, loss_data in individual_loss_metrics.items():
                    if 'network_loss' in loss_data:
                        all_network_losses.append(loss_data['network_loss'])
                    if 'network_epsilon' in loss_data:
                        all_network_epsilons.append(loss_data['network_epsilon'])
                    if 'network_training_steps' in loss_data:
                        all_training_steps.append(loss_data['network_training_steps'])
                    if 'network_experience_buffer_size' in loss_data:
                        all_buffer_sizes.append(loss_data['network_experience_buffer_size'])
                    if 'network_mean_q_value' in loss_data:
                        all_q_values.append(loss_data['network_mean_q_value'])
                
                # Log aggregate network metrics
                if all_network_losses:
                    import numpy as np
                    network_metrics.update({
                        'network_performance/avg_loss': float(np.mean(all_network_losses)),
                        'network_performance/loss_std': float(np.std(all_network_losses)),
                        'network_performance/networks_training': len(all_network_losses),
                    })
                
                if all_q_values:
                    network_metrics.update({
                        'network_performance/avg_q_value': float(np.mean(all_q_values)),
                        'network_performance/q_value_std': float(np.std(all_q_values)),
                    })
                
                if all_buffer_sizes:
                    network_metrics.update({
                        'network_performance/total_experience': sum(all_buffer_sizes),
                        'network_performance/avg_experience_per_agent': sum(all_buffer_sizes) / len(all_buffer_sizes),
                    })
            
            # SECTION 5: SYSTEM PERFORMANCE
            system_metrics = {
                'system_performance/cpu_usage': metrics.training_metrics.cpu_usage,
                'system_performance/memory_usage': metrics.training_metrics.memory_usage,
                'system_performance/generation': metrics.generation,
                'system_performance/step_count': metrics.step_count,
                'system_performance/timestamp': metrics.timestamp,
            }
            
            # Combine all organized metrics and log to MLflow
            all_metrics = {}
            all_metrics.update(fitness_metrics)
            all_metrics.update(training_metrics)
            all_metrics.update(exploration_metrics)
            all_metrics.update(network_metrics)
            all_metrics.update(system_metrics)
            
            if self.mlflow_integration:
                # Log all organized metrics with proper step
                for metric_name, value in all_metrics.items():
                    if isinstance(value, (int, float, np.number)) and not np.isnan(float(value)):
                        # Use the existing MLflow integration method for proper logging
                        self.mlflow_integration.log_population_metrics(metrics.generation, {metric_name.split('/')[-1]: value}, metrics.step_count)
            
            # Print success message with organized sections count
            section_counts = {
                'Population Health': len(fitness_metrics),
                'Learning Progress': len(training_metrics),
                'Exploration': len(exploration_metrics),
                'Network Performance': len(network_metrics),
                'System Performance': len(system_metrics)
            }
            
            total_metrics = sum(section_counts.values())
            print(f"📊 Logged {total_metrics} organized metrics to MLflow:")
            for section, count in section_counts.items():
                if count > 0:
                    print(f"   📈 {section}: {count} metrics")
            
        except Exception as e:
            print(f"⚠️ Error logging organized metrics to MLflow: {e}")
            import traceback
            traceback.print_exc()

    def _capture_individual_network_losses(self, agents: List[Any]) -> Dict[str, Dict[str, float]]:
        """Capture loss values from individual Q-learning networks for MLflow logging."""
        individual_losses = {}
        
        try:
            for agent in agents:
                if getattr(agent, '_destroyed', False):
                    continue
                    
                agent_id = str(agent.id)[:12]  # Truncate for readability
                
                # Extract loss and training metrics from the agent's learning system
                loss_metrics = {}
                
                if hasattr(agent, '_learning_system') and agent._learning_system:
                    learning_system = agent._learning_system
                    
                    # Get loss history from the agent if available
                    loss_found = False
                    
                    # Method 1: Check agent's loss history (from training stats)
                    if hasattr(agent, '_loss_history') and agent._loss_history:
                        current_loss = agent._loss_history[-1] if agent._loss_history else 0.0
                        loss_metrics['network_loss'] = float(current_loss)
                        loss_found = True
                        
                        # Calculate moving average loss for smoother visualization
                        if len(agent._loss_history) >= 5:
                            recent_losses = agent._loss_history[-5:]
                            loss_metrics['network_loss_avg_5'] = float(sum(recent_losses) / len(recent_losses))
                    
                    # Method 2: Check learning system's loss history if agent history is empty
                    if not loss_found and hasattr(learning_system, '_loss_history') and learning_system._loss_history:
                        current_loss = learning_system._loss_history[-1] if learning_system._loss_history else 0.0
                        loss_metrics['network_loss'] = float(current_loss)
                        loss_found = True
                        
                        # Calculate moving average loss for smoother visualization
                        if len(learning_system._loss_history) >= 5:
                            recent_losses = learning_system._loss_history[-5:]
                            loss_metrics['network_loss_avg_5'] = float(sum(recent_losses) / len(recent_losses))
                    
                    # Method 3: Default to 0.0 if no loss found (early training)
                    if not loss_found:
                        loss_metrics['network_loss'] = 0.0
                    
                    # Get Q-value history if available
                    q_value_found = False
                    
                    # Method 1: Check agent's Q-value history (from training stats)
                    if hasattr(agent, '_qval_history') and agent._qval_history:
                        current_qval = agent._qval_history[-1] if agent._qval_history else 0.0
                        loss_metrics['network_mean_q_value'] = float(current_qval)
                        q_value_found = True
                        
                        # Calculate moving average Q-value
                        if len(agent._qval_history) >= 5:
                            recent_qvals = agent._qval_history[-5:]
                            loss_metrics['network_mean_q_value_avg_5'] = float(sum(recent_qvals) / len(recent_qvals))
                    
                    # Method 2: Get Q-values directly from the learning system if agent history is empty
                    if not q_value_found and hasattr(learning_system, 'q_network'):
                        try:
                            # Get a sample Q-value from the current state
                            current_state = agent.get_state_representation() if hasattr(agent, 'get_state_representation') else None
                            if current_state is not None and len(current_state) == 29:  # Correct state size
                                import torch
                                learning_system.q_network.eval()
                                with torch.no_grad():
                                    state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(learning_system.device)
                                    q_values, _ = learning_system.q_network(state_tensor)
                                    current_qval = float(q_values.mean().item())
                                    loss_metrics['network_mean_q_value'] = current_qval
                                    q_value_found = True
                                learning_system.q_network.train()
                        except Exception as e:
                            # If direct Q-value extraction fails, continue without Q-values
                            pass
                    
                    # Method 3: Default to 0.0 if no Q-values found (early training)
                    if not q_value_found:
                        loss_metrics['network_mean_q_value'] = 0.0
                    
                    # Get epsilon value for exploration tracking
                    if hasattr(learning_system, 'epsilon'):
                        loss_metrics['network_epsilon'] = float(learning_system.epsilon)
                    
                    # Get learning rate if available
                    if hasattr(learning_system, 'learning_rate'):
                        loss_metrics['network_learning_rate'] = float(learning_system.learning_rate)
                    
                    # Get training step count
                    if hasattr(learning_system, 'steps_done'):
                        loss_metrics['network_training_steps'] = int(learning_system.steps_done)
                    
                    # NEW: Get training run count for avg_training_runs_per_agent metric
                    if hasattr(learning_system, 'training_runs'):
                        loss_metrics['network_training_runs'] = int(learning_system.training_runs)
                    
                    # NEW: Calculate training frequency (training runs per step)
                    if hasattr(learning_system, 'training_runs') and hasattr(learning_system, 'steps_done'):
                        steps = getattr(learning_system, 'steps_done', 1)
                        training_runs = getattr(learning_system, 'training_runs', 0)
                        if steps > 0:
                            loss_metrics['network_training_frequency'] = float(training_runs / steps)
                        else:
                            loss_metrics['network_training_frequency'] = 0.0
                    
                    # Get memory buffer size for monitoring experience collection
                    if (hasattr(learning_system, 'memory') and 
                        hasattr(learning_system.memory, 'buffer')):
                        buffer_size = len(learning_system.memory.buffer)
                        loss_metrics['network_experience_buffer_size'] = int(buffer_size)
                
                # Only include agents that have actual loss data
                if loss_metrics:
                    individual_losses[agent_id] = loss_metrics
                    
        except Exception as e:
            print(f"⚠️ Error capturing individual network losses: {e}")
            
        return individual_losses
    
    def _calculate_aggregate_learning_metrics(self, agents: List[Any], metrics: ComprehensiveMetrics) -> Dict[str, float]:
        """Calculate aggregate learning performance metrics for the population."""
        try:
            learning_metrics = {}
            
            # Collect individual metrics for aggregation
            individual_metrics = []
            q_learning_metrics = []
            exploration_metrics = []
            
            for agent in agents:
                agent_id = str(agent.id)
                if agent_id in metrics.individual_metrics:
                    individual_metrics.append(metrics.individual_metrics[agent_id])
                if agent_id in metrics.q_learning_metrics:
                    q_learning_metrics.append(metrics.q_learning_metrics[agent_id])
                if agent_id in metrics.exploration_metrics:
                    exploration_metrics.append(metrics.exploration_metrics[agent_id])
            
            # LEARNING CONVERGENCE METRICS
            if individual_metrics:
                convergence_scores = [m.q_learning_convergence for m in individual_metrics]
                learning_metrics['learning_convergence_mean'] = float(np.mean(convergence_scores))
                learning_metrics['learning_convergence_std'] = float(np.std(convergence_scores))
                learning_metrics['learning_convergence_min'] = float(np.min(convergence_scores))
                learning_metrics['learning_convergence_max'] = float(np.max(convergence_scores))
                
                # Learning efficiency
                efficiency_scores = [m.exploration_efficiency for m in individual_metrics]
                learning_metrics['exploration_efficiency_mean'] = float(np.mean(efficiency_scores))
                learning_metrics['exploration_efficiency_std'] = float(np.std(efficiency_scores))
                
                # Action diversity (population creativity)
                action_diversity = [m.action_diversity_score for m in individual_metrics]
                learning_metrics['action_diversity_mean'] = float(np.mean(action_diversity))
                learning_metrics['action_diversity_std'] = float(np.std(action_diversity))
                
                # Motor efficiency
                motor_efficiency = [m.motor_efficiency_score for m in individual_metrics]
                learning_metrics['motor_efficiency_mean'] = float(np.mean(motor_efficiency))
                learning_metrics['motor_efficiency_std'] = float(np.std(motor_efficiency))
            
            # Q-LEARNING SPECIFIC METRICS
            if q_learning_metrics:
                # Policy stability across population
                policy_stability = [m.policy_stability for m in q_learning_metrics]
                learning_metrics['policy_stability_mean'] = float(np.mean(policy_stability))
                learning_metrics['policy_stability_std'] = float(np.std(policy_stability))
                
                # Sample efficiency
                sample_efficiency = [m.sample_efficiency for m in q_learning_metrics]
                learning_metrics['sample_efficiency_mean'] = float(np.mean(sample_efficiency))
                learning_metrics['sample_efficiency_std'] = float(np.std(sample_efficiency))
                
                # Learning rate progression
                if hasattr(q_learning_metrics[0], 'learning_rate_progression'):
                    lr_progress = [m.learning_rate_progression for m in q_learning_metrics]
                    learning_metrics['learning_rate_progression_mean'] = float(np.mean(lr_progress))
                
                # Exploration vs exploitation balance
                if hasattr(q_learning_metrics[0], 'exploration_exploitation_ratio'):
                    exp_ratio = [m.exploration_exploitation_ratio for m in q_learning_metrics]
                    learning_metrics['exploration_exploitation_ratio_mean'] = float(np.mean(exp_ratio))
                    learning_metrics['exploration_exploitation_ratio_std'] = float(np.std(exp_ratio))
            
            # POPULATION LEARNING HEALTH INDICATORS
            # Calculate how many agents are learning effectively
            if individual_metrics:
                converging_agents = sum(1 for m in individual_metrics if m.q_learning_convergence > 0.5)
                learning_metrics['converging_agents_ratio'] = float(converging_agents / len(individual_metrics))
                
                efficient_explorers = sum(1 for m in individual_metrics if m.exploration_efficiency > 0.3)
                learning_metrics['efficient_explorers_ratio'] = float(efficient_explorers / len(individual_metrics))
                
                diverse_actors = sum(1 for m in individual_metrics if m.action_diversity_score > 0.4)
                learning_metrics['diverse_actors_ratio'] = float(diverse_actors / len(individual_metrics))
            
            # REWARD SIGNAL QUALITY (from agents)
            total_rewards = []
            epsilon_values = []
            for agent in agents:
                if hasattr(agent, 'total_reward'):
                    total_rewards.append(float(agent.total_reward))
                if hasattr(agent, 'epsilon'):
                    epsilon_values.append(float(agent.epsilon))
            
            if total_rewards:
                learning_metrics['reward_mean'] = float(np.mean(total_rewards))
                learning_metrics['reward_std'] = float(np.std(total_rewards))
                learning_metrics['reward_max'] = float(np.max(total_rewards))
                learning_metrics['reward_min'] = float(np.min(total_rewards))
                
                # Reward distribution health
                positive_rewards = sum(1 for r in total_rewards if r > 0)
                learning_metrics['positive_reward_ratio'] = float(positive_rewards / len(total_rewards))
            
            if epsilon_values:
                learning_metrics['epsilon_mean'] = float(np.mean(epsilon_values))
                learning_metrics['epsilon_std'] = float(np.std(epsilon_values))
                # Exploration decay health (higher std might indicate good adaptive exploration)
                learning_metrics['epsilon_range'] = float(np.max(epsilon_values) - np.min(epsilon_values))
            
            # NEURAL NETWORK HEALTH (for attention deep Q-learning agents)
            attention_agents = 0
            network_health_scores = []
            for agent in agents:
                if hasattr(agent, '_learning_system') and agent._learning_system:
                    attention_agents += 1
                    # Calculate network health based on gradient norms, loss stability, etc.
                    if hasattr(agent._learning_system, 'get_network_health'):
                        health = agent._learning_system.get_network_health()
                        network_health_scores.append(health)
            
            if attention_agents > 0:
                learning_metrics['attention_agents_count'] = float(attention_agents)
                learning_metrics['attention_agents_ratio'] = float(attention_agents / len(agents))
                
                if network_health_scores:
                    learning_metrics['network_health_mean'] = float(np.mean(network_health_scores))
                    learning_metrics['network_health_std'] = float(np.std(network_health_scores))
            
            return learning_metrics
            
        except Exception as e:
            print(f"⚠️  Error calculating aggregate learning metrics: {e}")
            return {}
    
    def _calculate_population_training_stats(self, individual_losses: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate population-level training run statistics from individual agent data."""
        try:
            if not individual_losses:
                return {
                    'total_training_runs': 0,
                    'avg_training_runs_per_agent': 0.0,
                    'total_agents_with_training': 0,
                    'training_runs_std': 0.0,
                    'avg_training_frequency': 0.0,
                    'total_experience_buffer_size': 0
                }
            
            # Extract training run counts and other statistics
            training_runs_list = []
            training_frequencies = []
            total_buffer_size = 0
            agents_with_training = 0
            
            for agent_id, stats in individual_losses.items():
                training_runs = stats.get('network_training_runs', 0)
                training_frequency = stats.get('network_training_frequency', 0.0)
                buffer_size = stats.get('network_experience_buffer_size', 0)
                
                if training_runs > 0:
                    training_runs_list.append(training_runs)
                    agents_with_training += 1
                
                if training_frequency > 0:
                    training_frequencies.append(training_frequency)
                    
                total_buffer_size += buffer_size
            
            # Calculate aggregate statistics
            total_training_runs = sum(training_runs_list)
            avg_training_runs = sum(training_runs_list) / len(training_runs_list) if training_runs_list else 0.0
            training_runs_std = 0.0
            if len(training_runs_list) > 1:
                import numpy as np
                training_runs_std = float(np.std(training_runs_list))
            
            avg_training_frequency = sum(training_frequencies) / len(training_frequencies) if training_frequencies else 0.0
            
            return {
                'total_training_runs': total_training_runs,
                'avg_training_runs_per_agent': avg_training_runs,
                'total_agents_with_training': agents_with_training,
                'training_runs_std': training_runs_std,
                'avg_training_frequency': avg_training_frequency,
                'total_experience_buffer_size': total_buffer_size
            }
            
        except Exception as e:
            print(f"⚠️ Error calculating population training stats: {e}")
            return {}

    def _calculate_snapshot_learning_metrics(self, agents_snapshot: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate aggregate learning metrics from lightweight agent snapshots."""
        try:
            learning_metrics = {}
            
            if not agents_snapshot:
                return learning_metrics
            
            # Extract basic metrics from snapshots
            total_rewards = []
            epsilon_values = []
            steps_list = []
            agents_with_qlearning = 0
            
            for agent_data in agents_snapshot:
                # Reward metrics
                if 'total_reward' in agent_data:
                    total_rewards.append(float(agent_data['total_reward']))
                
                # Exploration metrics  
                if 'epsilon' in agent_data:
                    epsilon_values.append(float(agent_data['epsilon']))
                
                # Training progress
                if 'steps' in agent_data:
                    steps_list.append(int(agent_data['steps']))
                
                # Q-learning capability
                if agent_data.get('has_q_table', False):
                    agents_with_qlearning += 1
            
            # REWARD SIGNAL QUALITY
            if total_rewards:
                learning_metrics['reward_mean'] = float(np.mean(total_rewards))
                learning_metrics['reward_std'] = float(np.std(total_rewards))
                learning_metrics['reward_max'] = float(np.max(total_rewards))
                learning_metrics['reward_min'] = float(np.min(total_rewards))
                
                # Learning health indicators
                positive_rewards = sum(1 for r in total_rewards if r > 0)
                learning_metrics['positive_reward_ratio'] = float(positive_rewards / len(total_rewards))
                
                improving_agents = sum(1 for r in total_rewards if r > 0.1)
                learning_metrics['improving_agents_ratio'] = float(improving_agents / len(total_rewards))
            
            # EXPLORATION HEALTH
            if epsilon_values:
                learning_metrics['epsilon_mean'] = float(np.mean(epsilon_values))
                learning_metrics['epsilon_std'] = float(np.std(epsilon_values))
                learning_metrics['epsilon_range'] = float(np.max(epsilon_values) - np.min(epsilon_values))
                
                # Exploration diversity (good if agents have varied epsilon values)
                exploring_agents = sum(1 for e in epsilon_values if e > 0.1)
                learning_metrics['exploring_agents_ratio'] = float(exploring_agents / len(epsilon_values))
                
                exploiting_agents = sum(1 for e in epsilon_values if e < 0.05)
                learning_metrics['exploiting_agents_ratio'] = float(exploiting_agents / len(epsilon_values))
            
            # TRAINING PROGRESS - Focus on agent maturity, not step statistics
            if steps_list:
                # Training maturity - what percentage of agents have enough experience
                experienced_agents = sum(1 for s in steps_list if s > 1000)
                learning_metrics['experienced_agents_ratio'] = float(experienced_agents / len(steps_list))
            
            # Q-LEARNING ADOPTION
            learning_metrics['qlearning_agents_count'] = float(agents_with_qlearning)
            learning_metrics['qlearning_agents_ratio'] = float(agents_with_qlearning / len(agents_snapshot))
            
            # POPULATION LEARNING HEALTH SCORE
            # Combine multiple indicators into a single health score
            health_indicators = []
            
            if total_rewards:
                # Reward health: positive ratio and improvement
                reward_health = learning_metrics.get('positive_reward_ratio', 0) * 0.5 + learning_metrics.get('improving_agents_ratio', 0) * 0.5
                health_indicators.append(reward_health)
            
            if epsilon_values:
                # Exploration health: balance between exploration and exploitation
                exploration_balance = min(learning_metrics.get('exploring_agents_ratio', 0), learning_metrics.get('exploiting_agents_ratio', 0)) * 2
                health_indicators.append(exploration_balance)
            
            if agents_with_qlearning > 0:
                # Learning capability health
                learning_capability = learning_metrics.get('qlearning_agents_ratio', 0)
                health_indicators.append(learning_capability)
            
            if health_indicators:
                learning_metrics['population_learning_health'] = float(np.mean(health_indicators))
            
            return learning_metrics
            
        except Exception as e:
            print(f"⚠️  Error calculating snapshot learning metrics: {e}")
            return {}
    
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
                    # Use the correct method - get latest metrics for the agent
                    'q_learning': self.q_learning_evaluator.get_agent_metrics(agent_id)
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
                    },
                    # NEW: Include learning system for training run tracking
                    'learning_system': getattr(agent, '_learning_system', None),
                    # NEW: Capture loss and Q-value histories from agent if available  
                    'loss_history': getattr(agent, '_loss_history', []),
                    'qval_history': getattr(agent, '_qval_history', [])
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
    
    def _collect_system_metrics_data(self) -> Dict[str, Any]:
        """Collect system metrics data for background logging to MLflow."""
        try:
            import psutil
            import os
            
            # Get basic system information
            system_data = {}
            
            # Physics world information (if available)
            try:
                # This should be called from training environment context, so we might not have direct access
                # We'll collect what we can and let the training environment provide the rest
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                
                system_data.update({
                    'process_memory_mb': memory_info.rss / (1024**2),
                    'process_memory_vms_mb': memory_info.vms / (1024**2),
                    'process_cpu_percent': process.cpu_percent(),
                    'system_memory_percent': psutil.virtual_memory().percent,
                    'system_cpu_percent': psutil.cpu_percent(interval=0.1),
                    'timestamp': time.time()
                })
            except Exception as e:
                print(f"⚠️ Error collecting basic system metrics: {e}")
            
            return system_data
            
        except ImportError:
            print("⚠️ psutil not available for system metrics collection")
            return {}
        except Exception as e:
            print(f"⚠️ Error in system metrics collection: {e}")
            return {}
    
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
                data = None
                with self.metrics_lock:
                    if not self.collection_queue:
                        pass
                    else:
                        data = self.collection_queue.pop(0)
                
                if data is None:
                    time.sleep(1.0)
                    continue
                
                # Process metrics collection in background
                print(f"🔬 Background: Processing metrics for generation {data['generation']}")
                
                # FIXED: Actually process the metrics instead of just clearing the queue
                try:
                    # Use the snapshot data to reconstruct minimal agent info for logging
                    agents_snapshot = data['agents_snapshot']
                    
                    # Create minimal comprehensive metrics for MLflow logging
                    if self.enable_mlflow and self.mlflow_integration:
                        try:
                            # CRITICAL FIX: Use the same run ID as the main thread to consolidate all metrics
                            # MLflow uses thread-local storage, so we need to explicitly continue the existing run
                            if hasattr(self.mlflow_integration, 'current_run') and self.mlflow_integration.current_run:
                                # Get the run ID from the main thread's MLflow integration
                                main_thread_run_id = self.mlflow_integration.current_run.info.run_id
                                
                                                                # FIXED: Don't start a new run context - use MLflow integration methods instead
                                # This prevents the "Run already active" error and ensures all metrics go to the same run
                                
                                # The step count will be passed directly to the MLflow integration method
                                
                                # Capture individual agent loss/training data
                                individual_losses = self._capture_loss_from_snapshots(agents_snapshot)
                                
                                # NEW: Calculate population-level training statistics
                                population_training_stats = self._calculate_population_training_stats(individual_losses)
                                
                                # Log basic population metrics with organized sections
                                population_data = {
                                    'generation': data['generation'],
                                    'step_count': data['step_count'],
                                }
                                
                                # SECTION 1: POPULATION HEALTH & FITNESS
                                fitness_metrics = {
                                    'population_health/avg_fitness': data['population_stats'].get('average_fitness', 0.0),
                                    'population_health/best_fitness': data['population_stats'].get('best_fitness', 0.0),
                                    'population_health/worst_fitness': data['population_stats'].get('worst_fitness', 0.0),
                                    'population_health/fitness_variance': data['population_stats'].get('fitness_variance', 0.0),
                                    'population_health/population_size': len(agents_snapshot),
                                    'population_health/diversity_score': data['population_stats'].get('diversity', 0.0),
                                }
                                
                                # SECTION 2: LEARNING PROGRESS & TRAINING
                                training_metrics = {}
                                if population_training_stats:
                                    training_metrics.update({
                                        'learning_progress/total_training_runs': population_training_stats.get('total_training_runs', 0),
                                        'learning_progress/avg_training_runs_per_agent': population_training_stats.get('avg_training_runs_per_agent', 0.0),
                                        'learning_progress/agents_actively_training': population_training_stats.get('total_agents_with_training', 0),
                                        'learning_progress/training_frequency_hz': population_training_stats.get('avg_training_frequency', 0.0),
                                        'learning_progress/experience_buffer_total': population_training_stats.get('total_experience_buffer_size', 0),
                                    })
                                
                                # NEW: EXPERIENCE BUFFER MONITORING - Track when training should occur
                                buffer_metrics = {}
                                if individual_losses:
                                    all_buffer_sizes = [loss_metrics.get('network_experience_buffer_size', 0) for loss_metrics in individual_losses.values()]
                                    if all_buffer_sizes:
                                        training_threshold = 32  # Minimum experiences needed to start training
                                        max_buffer_capacity = 2000  # Typical buffer capacity
                                        
                                        agents_ready_for_training = len([size for size in all_buffer_sizes if size >= training_threshold])
                                        agents_with_full_buffers = len([size for size in all_buffer_sizes if size >= max_buffer_capacity * 0.8])
                                        
                                        buffer_metrics.update({
                                            'learning_progress/buffer_avg_size': sum(all_buffer_sizes) / len(all_buffer_sizes),
                                            'learning_progress/buffer_max_size': max(all_buffer_sizes),
                                            'learning_progress/buffer_min_size': min(all_buffer_sizes),
                                            'learning_progress/agents_ready_for_training': agents_ready_for_training,
                                            'learning_progress/agents_ready_ratio': agents_ready_for_training / len(all_buffer_sizes),
                                            'learning_progress/agents_with_full_buffers': agents_with_full_buffers,
                                            'learning_progress/buffer_utilization_avg': (sum(all_buffer_sizes) / len(all_buffer_sizes)) / max_buffer_capacity,
                                            'learning_progress/training_threshold': training_threshold,
                                        })
                                
                                training_metrics.update(buffer_metrics)
                                
                                # SECTION 3: EXPLORATION & EXPLOITATION
                                exploration_metrics = {}
                                if individual_losses:
                                    all_epsilons = [loss_metrics.get('network_epsilon', 0) for loss_metrics in individual_losses.values() if 'network_epsilon' in loss_metrics]
                                    if all_epsilons:
                                        exploration_metrics.update({
                                            'exploration/epsilon_mean': sum(all_epsilons) / len(all_epsilons),
                                            'exploration/epsilon_std': float(np.std(all_epsilons)),
                                            'exploration/epsilon_min': min(all_epsilons),
                                            'exploration/epsilon_max': max(all_epsilons),
                                            'exploration/agents_exploring_ratio': len([e for e in all_epsilons if e > 0.1]) / len(all_epsilons),
                                            'exploration/agents_exploiting_ratio': len([e for e in all_epsilons if e <= 0.1]) / len(all_epsilons),
                                        })
                                
                                # SECTION 4: NETWORK PERFORMANCE
                                network_metrics = {}
                                if individual_losses:
                                    all_losses = [loss_metrics.get('network_loss', 0) for loss_metrics in individual_losses.values() if 'network_loss' in loss_metrics]
                                    all_q_values = [loss_metrics.get('network_mean_q_value', 0) for loss_metrics in individual_losses.values() if 'network_mean_q_value' in loss_metrics]
                                    
                                    if all_losses:
                                        network_metrics.update({
                                            'network_performance/avg_loss': sum(all_losses) / len(all_losses),
                                            'network_performance/loss_std': float(np.std(all_losses)),
                                            'network_performance/networks_training': len(all_losses),
                                        })
                                    
                                    if all_q_values:
                                        network_metrics.update({
                                            'network_performance/avg_q_value': sum(all_q_values) / len(all_q_values),
                                            'network_performance/q_value_std': float(np.std(all_q_values)),
                                        })
                                
                                # SECTION 5: SYSTEM PERFORMANCE
                                system_metrics = {
                                    'system_performance/simulation_step': data['step_count'],
                                    'system_performance/generation': data['generation'],
                                    'system_performance/timestamp': data['timestamp'],
                                }
                                
                                # Add system resource usage if available
                                if 'process_memory_mb' in data:
                                    system_metrics.update({
                                        'system_performance/process_memory_mb': data.get('process_memory_mb', 0),
                                        'system_performance/process_cpu_percent': data.get('process_cpu_percent', 0),
                                        'system_performance/system_memory_percent': data.get('system_memory_percent', 0),
                                    })
                                
                                    # Combine all organized metrics
                                    all_metrics = {}
                                    all_metrics.update(fitness_metrics)
                                    all_metrics.update(training_metrics)
                                    all_metrics.update(exploration_metrics)
                                    all_metrics.update(network_metrics)
                                    all_metrics.update(system_metrics)
                                    
                                    # FIXED: Use MLflow integration's log_population_metrics method to batch all metrics
                                    # This ensures all metrics appear in the same run without context conflicts
                                    valid_metrics = {}
                                    for metric_name, value in all_metrics.items():
                                        if isinstance(value, (int, float, np.number)) and not np.isnan(float(value)):
                                            valid_metrics[metric_name] = float(value)
                                    
                                    if valid_metrics:
                                        self.mlflow_integration.log_population_metrics(data['generation'], valid_metrics, data['step_count'])
                                    
                                    # Print success message with organized sections count
                                    section_counts = {
                                        'Population Health': len(fitness_metrics),
                                        'Learning Progress': len(training_metrics),
                                        'Exploration': len(exploration_metrics),
                                        'Network Performance': len(network_metrics),
                                        'System Performance': len(system_metrics)
                                    }
                                    
                                    total_metrics = sum(section_counts.values())
                                    print(f"📊 Logged {total_metrics} organized metrics to MLflow run {main_thread_run_id[:8]}:")
                                    for section, count in section_counts.items():
                                        if count > 0:
                                            print(f"   📈 {section}: {count} metrics")
                        except Exception as e:
                            print(f"⚠️ Error logging background metrics to MLflow: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    print(f"✅ Background: Metrics processing completed")
                    
                except Exception as e:
                    print(f"⚠️ Error in background metrics collection: {e}")
                    time.sleep(5.0)
            except Exception as e:
                print(f"⚠️ Error in background collection worker main loop: {e}")
                time.sleep(5.0)

    def _capture_loss_from_snapshots(self, agents_snapshot: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Capture loss values from agent snapshots."""
        try:
            individual_losses = {}
            
            # BUFFER DEBUG: Check if agents have learning systems and why buffers are empty
            agents_with_systems = 0
            agents_without_systems = 0
            total_buffer_size = 0
            max_buffer_size = 0
            
            for agent_data in agents_snapshot:
                try:
                    agent_id = agent_data.get('id')
                    learning_system = agent_data.get('learning_system')
                    
                    # DEBUG: Check learning system status
                    if learning_system:
                        agents_with_systems += 1
                        
                        # Check buffer size for debugging
                        buffer_size = 0
                        if hasattr(learning_system, 'memory') and hasattr(learning_system.memory, 'buffer'):
                            buffer_size = len(learning_system.memory.buffer)
                            total_buffer_size += buffer_size
                            max_buffer_size = max(max_buffer_size, buffer_size)
                        
                        # Debug first few agents with details
                        if buffer_size > 0 or agents_with_systems <= 3:
                            steps = agent_data.get('steps', 0)
                            #print(f"🔍 DEBUG Agent {str(agent_id)[:8]}: Buffer={buffer_size}, Steps={steps}, Learning_system_type={type(learning_system).__name__}")
                    else:
                        agents_without_systems += 1
                        if agents_without_systems <= 3:  # Log first few
                            steps = agent_data.get('steps', 0)
                            print(f"❌ DEBUG Agent {str(agent_id)[:8]}: NO LEARNING SYSTEM (steps={steps})")
                    
                    if learning_system and hasattr(learning_system, 'training_runs'):
                        # Extract current training statistics
                        training_runs = getattr(learning_system, 'training_runs', 0)
                        last_training_time = getattr(learning_system, 'last_training_time', 0.0)
                        
                        # Get recent training statistics if available
                        recent_loss = 0.0
                        recent_mean_q_value = 0.0
                        
                        # Try to get stats from recent training (if the agent has loss history)
                        if hasattr(learning_system, '_loss_history') and learning_system._loss_history:
                            recent_loss = learning_system._loss_history[-1]  # Most recent loss
                        else:
                            # If no loss history yet, keep default value of 0.0
                            recent_loss = 0.0
                        if hasattr(learning_system, '_qval_history') and learning_system._qval_history:
                            recent_mean_q_value = learning_system._qval_history[-1]  # Most recent Q-value
                        
                        # FIXED: If no Q-value history yet, try to get current Q-value from network
                        if recent_mean_q_value == 0.0 and hasattr(learning_system, 'q_network'):
                            try:
                                # Get a sample Q-value from a default state
                                import torch
                                learning_system.q_network.eval()
                                with torch.no_grad():
                                    # Use a simple default state for Q-value sampling
                                    default_state = torch.zeros(1, 29).to(learning_system.device)
                                    q_values, _ = learning_system.q_network(default_state)
                                    recent_mean_q_value = float(q_values.mean().item())
                                learning_system.q_network.train()
                            except Exception:
                                # If Q-value extraction fails, keep it as 0.0
                                recent_mean_q_value = 0.0
                        
                        # Calculate training frequency (runs per minute)
                        training_frequency = 0.0
                        if last_training_time > 0:
                            time_since_first = last_training_time - (last_training_time - (training_runs * 30))  # Rough estimate
                            if time_since_first > 0:
                                training_frequency = training_runs / (time_since_first / 60.0)  # runs per minute
                        
                        # FIXED: Correctly access experience buffer size
                        buffer_size = 0
                        if hasattr(learning_system, 'memory') and hasattr(learning_system.memory, 'buffer'):
                            buffer_size = len(learning_system.memory.buffer)
                        
                        individual_losses[agent_id] = {
                            'network_loss': recent_loss,
                            'network_mean_q_value': recent_mean_q_value,
                            'network_training_runs': training_runs,
                            'network_last_training_time': last_training_time,
                            'network_training_frequency': training_frequency,
                            'network_epsilon': getattr(learning_system, 'epsilon', 0.0),
                            'network_experience_buffer_size': buffer_size,
                        }
                        
                except Exception as e:
                    print(f"⚠️ Error capturing training data for agent {agent_data.get('id', 'unknown')}: {e}")
                    continue
            
            # Print debug summary
            print(f"📊 BUFFER DEBUG SUMMARY (from snapshots):")
            print(f"   Agents with learning systems: {agents_with_systems}")
            print(f"   Agents without learning systems: {agents_without_systems}")
            print(f"   Total buffer experiences: {total_buffer_size}")
            print(f"   Max buffer size: {max_buffer_size}")
            if agents_with_systems > 0:
                print(f"   Avg buffer per agent: {total_buffer_size / agents_with_systems:.1f}")
            
            return individual_losses
            
        except Exception as e:
            print(f"⚠️ Error capturing loss data from snapshots: {e}")
            return {} 