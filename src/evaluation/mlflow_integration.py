"""
MLflow integration for experiment tracking and comparison.
Provides comprehensive logging and experiment management capabilities.
"""

import mlflow
import mlflow.tracking
import numpy as np
import time
import json
import os
from typing import Dict, List, Any, Optional, Union
from dataclasses import asdict
import tempfile
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


class MLflowIntegration:
    """
    MLflow integration for tracking robot training experiments.
    Logs metrics, parameters, and artifacts for comprehensive experiment tracking.
    """
    
    def __init__(self, tracking_uri: Optional[str] = None, experiment_name: str = "walker_robot_training"):
        """
        Initialize MLflow integration.
        
        Args:
            tracking_uri: MLflow tracking URI (defaults to local SQLite)
            experiment_name: Name of the MLflow experiment
        """
        # Set up MLflow tracking
        if tracking_uri is None:
            # Create a local SQLite database in the project directory
            project_root = Path(__file__).parent.parent.parent
            db_path = project_root / "experiments" / "walker_experiments.db"
            db_path.parent.mkdir(exist_ok=True)
            tracking_uri = f"sqlite:///{db_path}"
        
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
        except Exception as e:
            print(f"⚠️  Error setting up MLflow experiment: {e}")
            experiment_id = mlflow.create_experiment(experiment_name)
        
        self.experiment_name = experiment_name
        self.experiment_id = experiment_id
        
        # Current run tracking
        self.current_run = None
        self.run_start_time = None
        
        print(f"✅ MLflow integration initialized")
        print(f"   Tracking URI: {tracking_uri}")
        print(f"   Experiment: {experiment_name}")
    
    def start_training_run(self, run_name: Optional[str] = None, 
                          population_size: int = 30, 
                          evolution_config: Optional[Dict] = None) -> str:
        """
        Start a new MLflow run for a training session.
        
        Args:
            run_name: Name for the run (auto-generated if None)
            population_size: Size of the robot population
            evolution_config: Evolution configuration parameters
            
        Returns:
            Run ID
        """
        try:
            if run_name is None:
                run_name = f"training_run_{int(time.time())}"
            
            # End current run if exists
            if self.current_run is not None:
                mlflow.end_run()
            
            # Start new run
            self.current_run = mlflow.start_run(run_name=run_name, experiment_id=self.experiment_id)
            self.run_start_time = time.time()
            
            # Log initial parameters
            mlflow.log_param("population_size", population_size)
            mlflow.log_param("start_time", time.strftime("%Y-%m-%d %H:%M:%S"))
            
            if evolution_config:
                for key, value in evolution_config.items():
                    mlflow.log_param(f"evolution_{key}", value)
            
            print(f"🔬 Started MLflow run: {run_name}")
            return self.current_run.info.run_id
            
        except Exception as e:
            print(f"❌ Error starting MLflow run: {e}")
            return ""
    
    def log_individual_robot_metrics(self, robot_id: str, metrics: Dict[str, Any], step: int):
        """
        Log metrics for an individual robot.
        
        Args:
            robot_id: Unique robot identifier
            metrics: Dictionary of metrics to log
            step: Training step number
        """
        try:
            if self.current_run is None:
                print("⚠️  No active MLflow run. Starting default run.")
                self.start_training_run()
            
            # Log individual metrics with robot prefix
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float, np.number)):
                    mlflow.log_metric(f"robot_{robot_id}_{metric_name}", float(value), step=step)
                elif isinstance(value, (list, np.ndarray)):
                    # Log list metrics as JSON or summary statistics
                    if len(value) > 0:
                        if all(isinstance(x, (int, float, np.number)) for x in value):
                            mlflow.log_metric(f"robot_{robot_id}_{metric_name}_mean", float(np.mean(value)), step=step)
                            mlflow.log_metric(f"robot_{robot_id}_{metric_name}_std", float(np.std(value)), step=step)
                            mlflow.log_metric(f"robot_{robot_id}_{metric_name}_max", float(np.max(value)), step=step)
                            mlflow.log_metric(f"robot_{robot_id}_{metric_name}_min", float(np.min(value)), step=step)
                
        except Exception as e:
            print(f"⚠️  Error logging individual robot metrics: {e}")
    
    def log_population_metrics(self, generation: int, metrics: Dict[str, Any]):
        """
        Log population-level metrics.
        
        Args:
            generation: Generation number
            metrics: Dictionary of population metrics
        """
        try:
            if self.current_run is None:
                print("⚠️  No active MLflow run. Starting default run.")
                self.start_training_run()
            
            # Log population metrics
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float, np.number)):
                    mlflow.log_metric(f"population_{metric_name}", float(value), step=generation)
                elif isinstance(value, dict):
                    # Log nested dictionary metrics
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float, np.number)):
                            mlflow.log_metric(f"population_{metric_name}_{sub_key}", float(sub_value), step=generation)
                
        except Exception as e:
            print(f"⚠️  Error logging population metrics: {e}")
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """
        Log hyperparameters for the experiment.
        
        Args:
            params: Dictionary of hyperparameters
        """
        try:
            if self.current_run is None:
                print("⚠️  No active MLflow run. Starting default run.")
                self.start_training_run()
            
            # Log parameters
            for param_name, value in params.items():
                # Convert to string for MLflow compatibility
                mlflow.log_param(param_name, str(value))
                
        except Exception as e:
            print(f"⚠️  Error logging hyperparameters: {e}")
    
    def log_training_artifacts(self, 
                             population_summary: Optional[Dict] = None,
                             best_robot_config: Optional[Dict] = None,
                             training_plots: Optional[Dict] = None):
        """
        Log training artifacts.
        
        Args:
            population_summary: Summary of population performance
            best_robot_config: Configuration of best performing robot
            training_plots: Dictionary of plot data for visualization
        """
        try:
            if self.current_run is None:
                print("⚠️  No active MLflow run. Starting default run.")
                self.start_training_run()
            
            # Create temporary directory for artifacts
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Save population summary
                if population_summary:
                    summary_file = temp_path / "population_summary.json"
                    with open(summary_file, 'w') as f:
                        json.dump(population_summary, f, indent=2, default=str)
                    mlflow.log_artifact(str(summary_file), "summaries")
                
                # Save best robot configuration
                if best_robot_config:
                    config_file = temp_path / "best_robot_config.json"
                    with open(config_file, 'w') as f:
                        json.dump(best_robot_config, f, indent=2, default=str)
                    mlflow.log_artifact(str(config_file), "configs")
                
                # Generate and save training plots
                if training_plots:
                    self._generate_training_plots(training_plots, temp_path)
                
        except Exception as e:
            print(f"⚠️  Error logging training artifacts: {e}")
    
    def _generate_training_plots(self, plot_data: Dict, output_dir: Path):
        """Generate training visualization plots."""
        try:
            # Fitness progression plot
            if 'fitness_history' in plot_data:
                plt.figure(figsize=(10, 6))
                fitness_data = plot_data['fitness_history']
                plt.plot(fitness_data.get('generations', []), fitness_data.get('best_fitness', []), 
                        label='Best Fitness', linewidth=2)
                plt.plot(fitness_data.get('generations', []), fitness_data.get('avg_fitness', []), 
                        label='Average Fitness', linewidth=2)
                plt.xlabel('Generation')
                plt.ylabel('Fitness')
                plt.title('Population Fitness Over Time')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plot_file = output_dir / "fitness_progression.png"
                plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                plt.close()
                mlflow.log_artifact(str(plot_file), "plots")
            
            # Diversity plot
            if 'diversity_history' in plot_data:
                plt.figure(figsize=(10, 6))
                diversity_data = plot_data['diversity_history']
                plt.plot(diversity_data.get('generations', []), diversity_data.get('diversity', []), 
                        color='green', linewidth=2)
                plt.xlabel('Generation')
                plt.ylabel('Population Diversity')
                plt.title('Population Diversity Over Time')
                plt.grid(True, alpha=0.3)
                
                plot_file = output_dir / "diversity_progression.png"
                plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                plt.close()
                mlflow.log_artifact(str(plot_file), "plots")
            
            # Q-learning convergence plot
            if 'q_learning_metrics' in plot_data:
                plt.figure(figsize=(12, 8))
                
                q_data = plot_data['q_learning_metrics']
                
                # Subplot 1: Convergence
                plt.subplot(2, 2, 1)
                if 'convergence_history' in q_data:
                    plt.plot(q_data['convergence_history'])
                    plt.title('Q-Learning Convergence')
                    plt.ylabel('Convergence Score')
                
                # Subplot 2: Exploration rate
                plt.subplot(2, 2, 2)
                if 'epsilon_history' in q_data:
                    plt.plot(q_data['epsilon_history'])
                    plt.title('Exploration Rate (Epsilon)')
                    plt.ylabel('Epsilon')
                
                # Subplot 3: Learning rate
                plt.subplot(2, 2, 3)
                if 'learning_rate_history' in q_data:
                    plt.plot(q_data['learning_rate_history'])
                    plt.title('Learning Rate')
                    plt.ylabel('Learning Rate')
                    plt.xlabel('Step')
                
                # Subplot 4: Q-table size
                plt.subplot(2, 2, 4)
                if 'q_table_size_history' in q_data:
                    plt.plot(q_data['q_table_size_history'])
                    plt.title('Q-Table Size Growth')
                    plt.ylabel('Number of States')
                    plt.xlabel('Step')
                
                plt.tight_layout()
                plot_file = output_dir / "q_learning_metrics.png"
                plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                plt.close()
                mlflow.log_artifact(str(plot_file), "plots")
                
        except Exception as e:
            print(f"⚠️  Error generating training plots: {e}")
    
    def end_training_run(self, final_summary: Optional[Dict] = None):
        """
        End the current training run.
        
        Args:
            final_summary: Final summary metrics
        """
        try:
            if self.current_run is None:
                print("⚠️  No active MLflow run to end.")
                return
            
            # Log final summary metrics
            if final_summary:
                for key, value in final_summary.items():
                    if isinstance(value, (int, float, np.number)):
                        mlflow.log_metric(f"final_{key}", float(value))
            
            # Log training duration
            if self.run_start_time:
                duration = time.time() - self.run_start_time
                mlflow.log_metric("training_duration_seconds", duration)
                mlflow.log_metric("training_duration_hours", duration / 3600)
            
            # End the run
            mlflow.end_run()
            self.current_run = None
            self.run_start_time = None
            
            print("✅ MLflow training run ended")
            
        except Exception as e:
            print(f"❌ Error ending MLflow run: {e}")
    
    def get_experiment_runs(self) -> List[Dict[str, Any]]:
        """
        Get all runs from the current experiment.
        
        Returns:
            List of run information dictionaries
        """
        try:
            client = mlflow.tracking.MlflowClient()
            runs = client.search_runs(experiment_ids=[self.experiment_id])
            
            run_info = []
            for run in runs:
                run_dict = {
                    'run_id': run.info.run_id,
                    'status': run.info.status,
                    'start_time': run.info.start_time,
                    'end_time': run.info.end_time,
                    'metrics': run.data.metrics,
                    'params': run.data.params,
                    'tags': run.data.tags
                }
                run_info.append(run_dict)
            
            return run_info
            
        except Exception as e:
            print(f"❌ Error getting experiment runs: {e}")
            return []
    
    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """
        Compare multiple runs.
        
        Args:
            run_ids: List of run IDs to compare
            
        Returns:
            DataFrame with comparison data
        """
        try:
            client = mlflow.tracking.MlflowClient()
            comparison_data = []
            
            for run_id in run_ids:
                run = client.get_run(run_id)
                
                run_data = {
                    'run_id': run_id,
                    'start_time': run.info.start_time,
                    'status': run.info.status
                }
                
                # Add parameters
                for param, value in run.data.params.items():
                    run_data[f"param_{param}"] = value
                
                # Add final metrics
                for metric, value in run.data.metrics.items():
                    run_data[f"metric_{metric}"] = value
                
                comparison_data.append(run_data)
            
            return pd.DataFrame(comparison_data)
            
        except Exception as e:
            print(f"❌ Error comparing runs: {e}")
            return pd.DataFrame()


class ExperimentComparator:
    """
    Compare and analyze different experiments and configurations.
    """
    
    def __init__(self, mlflow_integration: MLflowIntegration):
        """
        Initialize experiment comparator.
        
        Args:
            mlflow_integration: MLflow integration instance
        """
        self.mlflow = mlflow_integration
        self.client = mlflow.tracking.MlflowClient()
    
    def compare_learning_algorithms(self, algorithm_configs: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Compare different Q-learning algorithm configurations.
        
        Args:
            algorithm_configs: Dictionary mapping algorithm names to configurations
            
        Returns:
            Comparison results
        """
        try:
            comparison_results = {
                'algorithms': list(algorithm_configs.keys()),
                'convergence_rates': {},
                'sample_efficiency': {},
                'final_performance': {},
                'stability_scores': {}
            }
            
            # This would be implemented with actual experiment running
            # For now, return template structure
            for algo_name in algorithm_configs.keys():
                comparison_results['convergence_rates'][algo_name] = 0.0
                comparison_results['sample_efficiency'][algo_name] = 0.0
                comparison_results['final_performance'][algo_name] = 0.0
                comparison_results['stability_scores'][algo_name] = 0.0
            
            return comparison_results
            
        except Exception as e:
            print(f"❌ Error comparing learning algorithms: {e}")
            return {}
    
    def compare_evolution_strategies(self, evolution_configs: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Compare different evolutionary strategies.
        
        Args:
            evolution_configs: Dictionary mapping strategy names to configurations
            
        Returns:
            Comparison results
        """
        try:
            comparison_results = {
                'strategies': list(evolution_configs.keys()),
                'diversity_maintenance': {},
                'convergence_speed': {},
                'final_fitness': {},
                'parameter_exploration': {}
            }
            
            # Template implementation
            for strategy_name in evolution_configs.keys():
                comparison_results['diversity_maintenance'][strategy_name] = 0.0
                comparison_results['convergence_speed'][strategy_name] = 0.0
                comparison_results['final_fitness'][strategy_name] = 0.0
                comparison_results['parameter_exploration'][strategy_name] = 0.0
            
            return comparison_results
            
        except Exception as e:
            print(f"❌ Error comparing evolution strategies: {e}")
            return {}
    
    def parameter_sensitivity_analysis(self, 
                                     parameter_ranges: Dict[str, List],
                                     target_metric: str = 'final_fitness') -> Dict[str, Any]:
        """
        Analyze sensitivity of performance to different parameters.
        
        Args:
            parameter_ranges: Dictionary mapping parameter names to value ranges
            target_metric: Metric to analyze sensitivity for
            
        Returns:
            Sensitivity analysis results
        """
        try:
            sensitivity_results = {
                'parameters': list(parameter_ranges.keys()),
                'sensitivity_scores': {},
                'optimal_values': {},
                'parameter_interactions': {}
            }
            
            # Template implementation
            for param_name in parameter_ranges.keys():
                sensitivity_results['sensitivity_scores'][param_name] = 0.0
                sensitivity_results['optimal_values'][param_name] = 0.0
            
            return sensitivity_results
            
        except Exception as e:
            print(f"❌ Error in parameter sensitivity analysis: {e}")
            return {}
    
    def generate_experiment_report(self, experiment_id: str) -> str:
        """
        Generate a comprehensive experiment report.
        
        Args:
            experiment_id: MLflow experiment ID
            
        Returns:
            Path to generated report
        """
        try:
            runs = self.client.search_runs(experiment_ids=[experiment_id])
            
            if not runs:
                return "No runs found for experiment"
            
            # Generate report
            report_content = []
            report_content.append("# Robot Training Experiment Report\n")
            report_content.append(f"**Experiment ID:** {experiment_id}\n")
            report_content.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            report_content.append(f"**Total Runs:** {len(runs)}\n\n")
            
            # Best performing run
            best_run = max(runs, key=lambda x: x.data.metrics.get('final_fitness', 0))
            report_content.append("## Best Performing Run\n")
            report_content.append(f"**Run ID:** {best_run.info.run_id}\n")
            report_content.append(f"**Final Fitness:** {best_run.data.metrics.get('final_fitness', 'N/A')}\n")
            report_content.append(f"**Parameters:**\n")
            for param, value in best_run.data.params.items():
                report_content.append(f"- {param}: {value}\n")
            
            # Summary statistics
            report_content.append("\n## Summary Statistics\n")
            fitness_values = [run.data.metrics.get('final_fitness', 0) for run in runs]
            if fitness_values:
                report_content.append(f"**Average Fitness:** {np.mean(fitness_values):.3f}\n")
                report_content.append(f"**Best Fitness:** {np.max(fitness_values):.3f}\n")
                report_content.append(f"**Worst Fitness:** {np.min(fitness_values):.3f}\n")
                report_content.append(f"**Fitness Std:** {np.std(fitness_values):.3f}\n")
            
            report_text = "".join(report_content)
            
            # Save report
            report_path = f"experiment_report_{experiment_id}_{int(time.time())}.md"
            with open(report_path, 'w') as f:
                f.write(report_text)
            
            return report_path
            
        except Exception as e:
            print(f"❌ Error generating experiment report: {e}")
            return "" 