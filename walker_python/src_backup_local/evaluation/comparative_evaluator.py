"""
Comparative evaluation framework.
Compares different configurations, algorithms, and training strategies.
"""

import numpy as np
import pandas as pd
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import itertools


@dataclass
class ExperimentConfiguration:
    """Configuration for a comparative experiment."""
    name: str
    description: str
    parameters: Dict[str, Any]
    algorithm_type: str = "q_learning"
    evaluation_steps: int = 5000
    repetitions: int = 5


@dataclass
class ExperimentResult:
    """Results from a comparative experiment."""
    config_name: str
    metrics: Dict[str, float]
    time_series_data: Dict[str, List[float]]
    final_fitness: float
    convergence_time: Optional[float]
    stability_score: float
    efficiency_score: float
    statistical_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Results comparing multiple experiments."""
    winner: str
    confidence_level: float
    effect_size: float
    statistical_significance: float
    performance_gap: float
    detailed_comparison: Dict[str, Any]


class ComparativeEvaluator:
    """
    Compares different training configurations and algorithms.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize the comparative evaluator.
        
        Args:
            significance_level: Statistical significance level for comparisons
        """
        self.significance_level = significance_level
        self.experiments: Dict[str, ExperimentResult] = {}
        self.comparisons: Dict[Tuple[str, str], ComparisonResult] = {}
        
        # Predefined experiment configurations
        self.standard_configs = self._create_standard_configurations()
        
    def _create_standard_configurations(self) -> Dict[str, ExperimentConfiguration]:
        """Create standard configurations for comparison."""
        
        configs = {}
        
        # Baseline Q-learning configuration
        configs["baseline_q_learning"] = ExperimentConfiguration(
            name="baseline_q_learning",
            description="Standard Q-learning with default parameters",
            parameters={
                'learning_rate': 0.01,
                'epsilon': 0.3,
                'discount_factor': 0.9,
                'motor_torque': 150.0,
                'body_width': 1.5
            },
            algorithm_type="q_learning"
        )
        
        # High exploration configuration
        configs["high_exploration"] = ExperimentConfiguration(
            name="high_exploration",
            description="Q-learning with increased exploration",
            parameters={
                'learning_rate': 0.01,
                'epsilon': 0.6,
                'discount_factor': 0.9,
                'exploration_bonus': 0.3,
                'motor_torque': 150.0,
                'body_width': 1.5
            },
            algorithm_type="q_learning"
        )
        
        # Fast learning configuration
        configs["fast_learning"] = ExperimentConfiguration(
            name="fast_learning",
            description="Q-learning with higher learning rate",
            parameters={
                'learning_rate': 0.05,
                'epsilon': 0.2,
                'discount_factor': 0.95,
                'motor_torque': 150.0,
                'body_width': 1.5
            },
            algorithm_type="q_learning"
        )
        
        # Large body configuration
        configs["large_body"] = ExperimentConfiguration(
            name="large_body",
            description="Larger robot body configuration",
            parameters={
                'learning_rate': 0.01,
                'epsilon': 0.3,
                'discount_factor': 0.9,
                'motor_torque': 200.0,
                'body_width': 2.5,
                'body_height': 1.0
            },
            algorithm_type="q_learning"
        )
        
        # Small body configuration
        configs["small_body"] = ExperimentConfiguration(
            name="small_body", 
            description="Smaller robot body configuration",
            parameters={
                'learning_rate': 0.01,
                'epsilon': 0.3,
                'discount_factor': 0.9,
                'motor_torque': 100.0,
                'body_width': 0.8,
                'body_height': 0.5
            },
            algorithm_type="q_learning"
        )
        
        # High power configuration
        configs["high_power"] = ExperimentConfiguration(
            name="high_power",
            description="High motor power configuration",
            parameters={
                'learning_rate': 0.01,
                'epsilon': 0.3,
                'discount_factor': 0.9,
                'motor_torque': 300.0,
                'motor_speed': 6.0,
                'body_width': 1.5
            },
            algorithm_type="q_learning"
        )
        
        return configs
    
    def run_comparative_study(self,
                             agent_factory: Callable,
                             fitness_evaluator: Callable,
                             configs_to_compare: List[str] = None) -> Dict[str, ExperimentResult]:
        """
        Run a comparative study across multiple configurations.
        
        Args:
            agent_factory: Function to create agents with given parameters
            fitness_evaluator: Function to evaluate agent fitness
            configs_to_compare: List of configuration names to compare
            
        Returns:
            Dictionary of experiment results
        """
        if configs_to_compare is None:
            configs_to_compare = list(self.standard_configs.keys())
        
        print(f"🔬 Starting comparative study with {len(configs_to_compare)} configurations")
        
        results = {}
        
        for config_name in configs_to_compare:
            if config_name not in self.standard_configs:
                print(f"⚠️  Unknown configuration: {config_name}")
                continue
                
            config = self.standard_configs[config_name]
            
            print(f"🧪 Running experiment: {config.name}")
            try:
                result = self._run_single_experiment(config, agent_factory, fitness_evaluator)
                results[config_name] = result
                self.experiments[config_name] = result
                
            except Exception as e:
                print(f"❌ Error running experiment {config_name}: {e}")
                continue
        
        # Perform pairwise comparisons
        print("📊 Performing pairwise comparisons...")
        self._perform_pairwise_comparisons(list(results.keys()))
        
        print(f"✅ Comparative study complete. Evaluated {len(results)} configurations.")
        return results
    
    def _run_single_experiment(self,
                              config: ExperimentConfiguration,
                              agent_factory: Callable,
                              fitness_evaluator: Callable) -> ExperimentResult:
        """Run a single comparative experiment."""
        
        all_metrics = []
        time_series_collections = defaultdict(list)
        final_fitnesses = []
        convergence_times = []
        
        for rep in range(config.repetitions):
            print(f"  📈 Repetition {rep + 1}/{config.repetitions}")
            
            try:
                # Create agent with configuration parameters
                agent = agent_factory(**config.parameters)
                
                # Track metrics over time
                fitness_history = []
                reward_history = []
                convergence_history = []
                
                rep_start_time = time.time()
                convergence_detected = False
                convergence_time = None
                
                # Run evaluation steps
                for step in range(config.evaluation_steps):
                    # Evaluate current fitness
                    fitness = fitness_evaluator(agent)
                    fitness_history.append(fitness)
                    
                    # Track learning metrics
                    if hasattr(agent, 'total_reward'):
                        reward_history.append(agent.total_reward)
                    
                    if hasattr(agent, 'q_table') and hasattr(agent.q_table, 'get_convergence_estimate'):
                        convergence = agent.q_table.get_convergence_estimate()
                        convergence_history.append(convergence)
                        
                        # Detect convergence
                        if not convergence_detected and convergence > 0.8:
                            convergence_time = time.time() - rep_start_time
                            convergence_detected = True
                    
                    # Simulate learning step
                    if hasattr(agent, 'step'):
                        agent.step(1/60.0)  # Simulate 60Hz
                
                # Calculate metrics for this repetition
                final_fitness = fitness_history[-1] if fitness_history else 0.0
                final_fitnesses.append(final_fitness)
                
                if convergence_time is not None:
                    convergence_times.append(convergence_time)
                
                # Store time series data
                time_series_collections['fitness'].append(fitness_history)
                time_series_collections['reward'].append(reward_history)
                time_series_collections['convergence'].append(convergence_history)
                
                # Calculate repetition metrics
                rep_metrics = {
                    'final_fitness': final_fitness,
                    'max_fitness': max(fitness_history) if fitness_history else 0.0,
                    'avg_fitness': np.mean(fitness_history) if fitness_history else 0.0,
                    'fitness_std': np.std(fitness_history) if fitness_history else 0.0,
                    'convergence_time': convergence_time or config.evaluation_steps
                }
                all_metrics.append(rep_metrics)
                
            except Exception as e:
                print(f"    ❌ Error in repetition {rep}: {e}")
                continue
        
        # Aggregate results across repetitions
        if not all_metrics:
            # Return empty result if no successful repetitions
            return ExperimentResult(
                config_name=config.name,
                metrics={},
                time_series_data={},
                final_fitness=0.0,
                convergence_time=None,
                stability_score=0.0,
                efficiency_score=0.0
            )
        
        # Calculate aggregated metrics
        aggregated_metrics = {}
        for metric_name in all_metrics[0].keys():
            values = [m[metric_name] for m in all_metrics if m[metric_name] is not None]
            if values:
                aggregated_metrics[f"{metric_name}_mean"] = np.mean(values)
                aggregated_metrics[f"{metric_name}_std"] = np.std(values)
                aggregated_metrics[f"{metric_name}_min"] = np.min(values)
                aggregated_metrics[f"{metric_name}_max"] = np.max(values)
        
        # Average time series data
        averaged_time_series = {}
        for series_name, series_list in time_series_collections.items():
            if series_list:
                # Pad or truncate to same length
                min_length = min(len(s) for s in series_list)
                truncated_series = [s[:min_length] for s in series_list]
                averaged_time_series[series_name] = np.mean(truncated_series, axis=0).tolist()
        
        # Calculate derived metrics
        stability_score = self._calculate_stability_score(averaged_time_series.get('fitness', []))
        efficiency_score = self._calculate_efficiency_score(
            averaged_time_series.get('fitness', []),
            aggregated_metrics.get('convergence_time_mean', config.evaluation_steps)
        )
        
        return ExperimentResult(
            config_name=config.name,
            metrics=aggregated_metrics,
            time_series_data=averaged_time_series,
            final_fitness=aggregated_metrics.get('final_fitness_mean', 0.0),
            convergence_time=aggregated_metrics.get('convergence_time_mean'),
            stability_score=stability_score,
            efficiency_score=efficiency_score,
            statistical_data={
                'sample_size': len(all_metrics),
                'successful_repetitions': len(all_metrics),
                'total_repetitions': config.repetitions
            }
        )
    
    def _calculate_stability_score(self, fitness_history: List[float]) -> float:
        """Calculate stability score based on fitness variance."""
        if len(fitness_history) < 10:
            return 0.0
        
        # Look at last 20% of training
        final_portion = fitness_history[-len(fitness_history)//5:]
        if len(final_portion) < 2:
            return 0.0
        
        # Stability = 1 - normalized standard deviation
        mean_fitness = np.mean(final_portion)
        std_fitness = np.std(final_portion)
        
        if mean_fitness == 0:
            return 0.0
        
        normalized_std = std_fitness / abs(mean_fitness)
        stability = max(0.0, 1.0 - normalized_std)
        
        return float(stability)
    
    def _calculate_efficiency_score(self, fitness_history: List[float], convergence_time: float) -> float:
        """Calculate efficiency score based on learning speed and final performance."""
        if not fitness_history or convergence_time <= 0:
            return 0.0
        
        final_fitness = fitness_history[-1]
        max_fitness = max(fitness_history)
        
        # Efficiency = performance achieved / time taken
        performance_ratio = final_fitness / (max_fitness + 1e-8)
        time_efficiency = 1.0 / (1.0 + convergence_time / 1000.0)  # Normalize time
        
        efficiency = performance_ratio * time_efficiency
        return float(efficiency)
    
    def _perform_pairwise_comparisons(self, config_names: List[str]) -> None:
        """Perform statistical comparisons between all pairs of configurations."""
        
        for config1, config2 in itertools.combinations(config_names, 2):
            if config1 not in self.experiments or config2 not in self.experiments:
                continue
            
            try:
                comparison = self._compare_two_experiments(config1, config2)
                self.comparisons[(config1, config2)] = comparison
                
            except Exception as e:
                print(f"⚠️  Error comparing {config1} vs {config2}: {e}")
                continue
    
    def _compare_two_experiments(self, config1: str, config2: str) -> ComparisonResult:
        """Compare two experimental configurations statistically."""
        
        exp1 = self.experiments[config1]
        exp2 = self.experiments[config2]
        
        # Get final fitness values for statistical comparison
        fitness1 = exp1.final_fitness
        fitness2 = exp2.final_fitness
        
        # Determine winner
        winner = config1 if fitness1 > fitness2 else config2
        performance_gap = abs(fitness1 - fitness2)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((exp1.metrics.get('final_fitness_std', 0.1)**2 + 
                             exp2.metrics.get('final_fitness_std', 0.1)**2) / 2)
        effect_size = performance_gap / (pooled_std + 1e-8)
        
        # Statistical significance (using means and stds for t-test approximation)
        n1 = exp1.statistical_data.get('sample_size', 1)
        n2 = exp2.statistical_data.get('sample_size', 1)
        
        if n1 > 1 and n2 > 1:
            # Approximate t-test
            se_diff = np.sqrt(
                (exp1.metrics.get('final_fitness_std', 0.1)**2 / n1) +
                (exp2.metrics.get('final_fitness_std', 0.1)**2 / n2)
            )
            t_stat = performance_gap / (se_diff + 1e-8)
            df = n1 + n2 - 2
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        else:
            p_value = 1.0  # Cannot determine significance with small sample
        
        # Confidence level
        confidence_level = 1.0 - p_value
        
        # Detailed comparison
        detailed_comparison = {
            'config1_performance': {
                'name': config1,
                'final_fitness': fitness1,
                'stability': exp1.stability_score,
                'efficiency': exp1.efficiency_score,
                'convergence_time': exp1.convergence_time
            },
            'config2_performance': {
                'name': config2,
                'final_fitness': fitness2,
                'stability': exp2.stability_score,
                'efficiency': exp2.efficiency_score,
                'convergence_time': exp2.convergence_time
            },
            'statistical_tests': {
                'effect_size_interpretation': self._interpret_effect_size(effect_size),
                'significance_level': self.significance_level,
                'is_significant': p_value < self.significance_level
            }
        }
        
        return ComparisonResult(
            winner=winner,
            confidence_level=confidence_level,
            effect_size=effect_size,
            statistical_significance=p_value,
            performance_gap=performance_gap,
            detailed_comparison=detailed_comparison
        )
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_effect = abs(effect_size)
        
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"
    
    def get_ranking(self, metric: str = 'final_fitness') -> List[Tuple[str, float]]:
        """Get ranking of configurations by specified metric."""
        
        rankings = []
        for config_name, result in self.experiments.items():
            if metric == 'final_fitness':
                value = result.final_fitness
            elif metric == 'stability_score':
                value = result.stability_score
            elif metric == 'efficiency_score':
                value = result.efficiency_score
            elif metric in result.metrics:
                value = result.metrics[metric]
            else:
                value = 0.0
            
            rankings.append((config_name, value))
        
        # Sort by value (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings
    
    def generate_comparison_report(self) -> Dict[str, Any]:
        """Generate a comprehensive comparison report."""
        
        if not self.experiments:
            return {"error": "No experimental results available"}
        
        # Overall rankings
        fitness_ranking = self.get_ranking('final_fitness')
        stability_ranking = self.get_ranking('stability_score')
        efficiency_ranking = self.get_ranking('efficiency_score')
        
        # Best overall configuration
        best_config = fitness_ranking[0][0] if fitness_ranking else None
        
        # Significant differences
        significant_comparisons = [
            {
                'comparison': f"{comp[0]} vs {comp[1]}",
                'winner': result.winner,
                'performance_gap': result.performance_gap,
                'effect_size': result.effect_size,
                'p_value': result.statistical_significance
            }
            for comp, result in self.comparisons.items()
            if result.statistical_significance < self.significance_level
        ]
        
        # Configuration summaries
        config_summaries = []
        for config_name, result in self.experiments.items():
            config = self.standard_configs.get(config_name)
            summary = {
                'name': config_name,
                'description': config.description if config else "Custom configuration",
                'final_fitness': result.final_fitness,
                'stability_score': result.stability_score,
                'efficiency_score': result.efficiency_score,
                'convergence_time': result.convergence_time,
                'fitness_rank': next(i for i, (name, _) in enumerate(fitness_ranking, 1) if name == config_name),
                'stability_rank': next(i for i, (name, _) in enumerate(stability_ranking, 1) if name == config_name),
                'efficiency_rank': next(i for i, (name, _) in enumerate(efficiency_ranking, 1) if name == config_name)
            }
            config_summaries.append(summary)
        
        report = {
            'summary': {
                'total_configurations_tested': len(self.experiments),
                'best_overall_configuration': best_config,
                'significant_differences_found': len(significant_comparisons),
                'evaluation_timestamp': time.time()
            },
            'rankings': {
                'by_final_fitness': fitness_ranking,
                'by_stability': stability_ranking,
                'by_efficiency': efficiency_ranking
            },
            'configuration_summaries': config_summaries,
            'significant_comparisons': significant_comparisons,
            'recommendations': self._generate_recommendations(fitness_ranking, significant_comparisons)
        }
        
        return report
    
    def _generate_recommendations(self, 
                                 fitness_ranking: List[Tuple[str, float]],
                                 significant_comparisons: List[Dict]) -> List[str]:
        """Generate recommendations based on comparative analysis."""
        
        recommendations = []
        
        if not fitness_ranking:
            return ["No experimental data available for recommendations"]
        
        # Best configuration recommendation
        best_config, best_fitness = fitness_ranking[0]
        recommendations.append(
            f"Use '{best_config}' configuration for best performance (fitness: {best_fitness:.3f})"
        )
        
        # Significant differences
        if significant_comparisons:
            recommendations.append(
                f"Found {len(significant_comparisons)} statistically significant performance differences"
            )
            
            # Highlight biggest effect size
            biggest_effect = max(significant_comparisons, key=lambda x: abs(x['effect_size']))
            recommendations.append(
                f"Largest performance difference: {biggest_effect['comparison']} "
                f"(effect size: {biggest_effect['effect_size']:.2f})"
            )
        
        # Parameter insights
        if len(fitness_ranking) >= 3:
            top_3 = [name for name, _ in fitness_ranking[:3]]
            recommendations.append(
                f"Top 3 configurations for further tuning: {', '.join(top_3)}"
            )
        
        return recommendations
    
    def export_results(self, filepath: str) -> None:
        """Export comparative evaluation results to file."""
        
        export_data = {
            'evaluation_timestamp': time.time(),
            'experiments': {
                name: {
                    'config_name': result.config_name,
                    'metrics': result.metrics,
                    'time_series_data': result.time_series_data,
                    'final_fitness': result.final_fitness,
                    'convergence_time': result.convergence_time,
                    'stability_score': result.stability_score,
                    'efficiency_score': result.efficiency_score,
                    'statistical_data': result.statistical_data
                }
                for name, result in self.experiments.items()
            },
            'comparisons': {
                f"{comp[0]}_vs_{comp[1]}": {
                    'winner': result.winner,
                    'confidence_level': result.confidence_level,
                    'effect_size': result.effect_size,
                    'statistical_significance': result.statistical_significance,
                    'performance_gap': result.performance_gap,
                    'detailed_comparison': result.detailed_comparison
                }
                for comp, result in self.comparisons.items()
            },
            'report': self.generate_comparison_report()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            print(f"✅ Comparative evaluation results exported to {filepath}")
        except Exception as e:
            print(f"❌ Error exporting results: {e}")


# Mock classes for testing
class MockComparativeAgentFactory:
    """Mock agent factory for testing comparative evaluation."""
    
    @staticmethod
    def create_agent(**params):
        """Create a mock agent with specified parameters."""
        class MockAgent:
            def __init__(self, **kwargs):
                self.params = kwargs
                self.total_reward = 0.0
                self.steps = 0
                
            def step(self, dt):
                self.steps += 1
                # Simulate learning progress
                self.total_reward += np.random.normal(0.01, 0.005)
                
        return MockAgent(**params)


class MockComparativeFitnessEvaluator:
    """Mock fitness evaluator for testing comparative evaluation."""
    
    @staticmethod
    def evaluate_fitness(agent) -> float:
        """Evaluate mock fitness based on agent parameters."""
        
        # Simulate different configuration strengths
        lr = agent.params.get('learning_rate', 0.01)
        epsilon = agent.params.get('epsilon', 0.3)
        motor_torque = agent.params.get('motor_torque', 150.0)
        body_width = agent.params.get('body_width', 1.5)
        
        # Learning rate optimal around 0.02
        lr_fitness = 1.0 - abs(lr - 0.02) * 5
        
        # Epsilon should be moderate
        epsilon_fitness = 1.0 - abs(epsilon - 0.4) * 1.5
        
        # Higher motor torque generally better
        torque_fitness = min(1.0, motor_torque / 250.0)
        
        # Body width optimal around 1.2
        width_fitness = 1.0 - abs(body_width - 1.2) * 0.2
        
        # Combine factors
        base_fitness = (lr_fitness + epsilon_fitness + torque_fitness + width_fitness) / 4.0
        
        # Add noise and agent-specific variation
        noise = np.random.normal(0, 0.05)
        progress_bonus = agent.total_reward * 0.1  # Reward learning progress
        
        return max(0.0, base_fitness + noise + progress_bonus) 