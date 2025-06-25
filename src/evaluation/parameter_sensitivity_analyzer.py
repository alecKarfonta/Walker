"""
Parameter sensitivity analysis framework.
Analyzes how different physical and learning parameters affect robot performance.
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import json
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
import itertools


@dataclass
class ParameterSensitivityResult:
    """Results from parameter sensitivity analysis."""
    parameter_name: str
    sensitivity_score: float
    correlation_with_fitness: float
    statistical_significance: float
    optimal_range: Tuple[float, float]
    performance_curve: List[Tuple[float, float]]  # (parameter_value, fitness)
    interaction_effects: Dict[str, float] = field(default_factory=dict)


@dataclass
class SensitivityExperiment:
    """Configuration for a sensitivity experiment."""
    parameter_name: str
    test_values: List[float]
    fixed_parameters: Dict[str, Any]
    repetitions: int = 3
    evaluation_steps: int = 1000


class ParameterSensitivityAnalyzer:
    """
    Analyzes parameter sensitivity across the robot training system.
    """
    
    def __init__(self, significance_threshold: float = 0.05):
        """
        Initialize the parameter sensitivity analyzer.
        
        Args:
            significance_threshold: Statistical significance threshold for results
        """
        self.significance_threshold = significance_threshold
        self.experiment_history: List[Dict[str, Any]] = []
        self.sensitivity_results: Dict[str, ParameterSensitivityResult] = {}
        self.parameter_interactions: Dict[Tuple[str, str], float] = {}
        
        # Standard parameter ranges for analysis
        self.default_parameter_ranges = {
            # Physical parameters
            'body_width': (0.5, 3.0),
            'body_height': (0.3, 1.5),
            'wheel_radius': (0.1, 0.8),
            'motor_torque': (50.0, 300.0),
            'motor_speed': (1.0, 8.0),
            'leg_spread': (0.4, 4.0),
            'suspension': (0.3, 1.5),
            
            # Learning parameters
            'learning_rate': (0.001, 0.1),
            'epsilon': (0.01, 0.8),
            'discount_factor': (0.5, 0.99),
            'exploration_bonus': (0.01, 0.5),
            
            # Reward weights
            'speed_value_weight': (0.01, 0.2),
            'acceleration_value_weight': (0.01, 0.15),
            'stability_weight': (0.005, 0.1),
            'position_weight': (0.005, 0.05),
        }
        
    def analyze_parameter_sensitivity(self, 
                                    agent_factory: Callable,
                                    fitness_evaluator: Callable,
                                    parameters_to_test: List[str] = None,
                                    samples_per_parameter: int = 10) -> Dict[str, ParameterSensitivityResult]:
        """
        Perform comprehensive parameter sensitivity analysis.
        
        Args:
            agent_factory: Function to create agents with given parameters
            fitness_evaluator: Function to evaluate agent fitness
            parameters_to_test: List of parameter names to test
            samples_per_parameter: Number of samples per parameter
            
        Returns:
            Dictionary of parameter sensitivity results
        """
        if parameters_to_test is None:
            parameters_to_test = list(self.default_parameter_ranges.keys())
        
        print(f"🔬 Starting parameter sensitivity analysis for {len(parameters_to_test)} parameters")
        
        all_results = {}
        
        for param_name in parameters_to_test:
            print(f"🧪 Analyzing parameter: {param_name}")
            
            try:
                result = self._analyze_single_parameter(
                    param_name, agent_factory, fitness_evaluator, samples_per_parameter
                )
                all_results[param_name] = result
                self.sensitivity_results[param_name] = result
                
            except Exception as e:
                print(f"❌ Error analyzing parameter {param_name}: {e}")
                continue
        
        # Analyze parameter interactions
        print("🔗 Analyzing parameter interactions...")
        self._analyze_parameter_interactions(agent_factory, fitness_evaluator, parameters_to_test[:5])
        
        print(f"✅ Parameter sensitivity analysis complete. Analyzed {len(all_results)} parameters.")
        return all_results
    
    def _analyze_single_parameter(self, 
                                 parameter_name: str,
                                 agent_factory: Callable,
                                 fitness_evaluator: Callable,
                                 num_samples: int) -> ParameterSensitivityResult:
        """Analyze sensitivity of a single parameter."""
        
        if parameter_name not in self.default_parameter_ranges:
            raise ValueError(f"Unknown parameter: {parameter_name}")
        
        param_min, param_max = self.default_parameter_ranges[parameter_name]
        test_values = np.linspace(param_min, param_max, num_samples)
        
        fitness_results = []
        performance_curve = []
        
        for value in test_values:
            # Create multiple agents with this parameter value
            fitness_scores = []
            
            for _ in range(3):  # 3 repetitions per value
                try:
                    # Create agent with specific parameter value
                    agent = agent_factory(**{parameter_name: value})
                    fitness = fitness_evaluator(agent)
                    fitness_scores.append(fitness)
                    
                except Exception as e:
                    print(f"⚠️  Error evaluating {parameter_name}={value}: {e}")
                    continue
            
            if fitness_scores:
                avg_fitness = np.mean(fitness_scores)
                fitness_results.append(avg_fitness)
                performance_curve.append((float(value), float(avg_fitness)))
            else:
                fitness_results.append(0.0)
                performance_curve.append((float(value), 0.0))
        
        # Calculate sensitivity metrics
        sensitivity_score = self._calculate_sensitivity_score(test_values, fitness_results)
        correlation = self._calculate_correlation(test_values, fitness_results)
        significance = self._calculate_statistical_significance(test_values, fitness_results)
        optimal_range = self._find_optimal_range(test_values, fitness_results)
        
        return ParameterSensitivityResult(
            parameter_name=parameter_name,
            sensitivity_score=sensitivity_score,
            correlation_with_fitness=correlation,
            statistical_significance=significance,
            optimal_range=optimal_range,
            performance_curve=performance_curve
        )
    
    def _analyze_parameter_interactions(self,
                                       agent_factory: Callable,
                                       fitness_evaluator: Callable,
                                       parameters: List[str]) -> None:
        """Analyze interactions between parameters."""
        
        # Test pairs of parameters
        for param1, param2 in itertools.combinations(parameters, 2):
            try:
                interaction_strength = self._measure_parameter_interaction(
                    param1, param2, agent_factory, fitness_evaluator
                )
                self.parameter_interactions[(param1, param2)] = interaction_strength
                
                # Store interaction effects in both parameters
                if param1 in self.sensitivity_results:
                    self.sensitivity_results[param1].interaction_effects[param2] = interaction_strength
                if param2 in self.sensitivity_results:
                    self.sensitivity_results[param2].interaction_effects[param1] = interaction_strength
                    
            except Exception as e:
                print(f"⚠️  Error analyzing interaction {param1}-{param2}: {e}")
                continue
    
    def _measure_parameter_interaction(self,
                                      param1: str,
                                      param2: str,
                                      agent_factory: Callable,
                                      fitness_evaluator: Callable) -> float:
        """Measure interaction strength between two parameters."""
        
        # Sample parameter values
        range1 = self.default_parameter_ranges[param1]
        range2 = self.default_parameter_ranges[param2]
        
        values1 = np.linspace(range1[0], range1[1], 5)
        values2 = np.linspace(range2[0], range2[1], 5)
        
        fitness_matrix = np.zeros((len(values1), len(values2)))
        
        for i, val1 in enumerate(values1):
            for j, val2 in enumerate(values2):
                try:
                    agent = agent_factory(**{param1: val1, param2: val2})
                    fitness = fitness_evaluator(agent)
                    fitness_matrix[i, j] = fitness
                except:
                    fitness_matrix[i, j] = 0.0
        
        # Calculate interaction effect using variance analysis
        # High interaction = fitness depends on combination, not just individual values
        row_means = np.mean(fitness_matrix, axis=1)
        col_means = np.mean(fitness_matrix, axis=0)
        grand_mean = np.mean(fitness_matrix)
        
        # Expected values if no interaction
        expected = np.outer(row_means - grand_mean, col_means - grand_mean) / grand_mean + grand_mean
        
        # Interaction strength is deviation from expected
        interaction_strength = np.mean(np.abs(fitness_matrix - expected))
        
        return float(interaction_strength)
    
    def _calculate_sensitivity_score(self, values: np.ndarray, fitness: List[float]) -> float:
        """Calculate how sensitive fitness is to parameter changes."""
        if len(fitness) < 2:
            return 0.0
        
        # Normalize values and fitness
        values_norm = (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-8)
        fitness_norm = np.array(fitness)
        fitness_norm = (fitness_norm - np.min(fitness_norm)) / (np.max(fitness_norm) - np.min(fitness_norm) + 1e-8)
        
        # Calculate rate of change
        sensitivity = np.mean(np.abs(np.gradient(fitness_norm, values_norm)))
        return float(sensitivity)
    
    def _calculate_correlation(self, values: np.ndarray, fitness: List[float]) -> float:
        """Calculate correlation between parameter values and fitness."""
        if len(fitness) < 2:
            return 0.0
        
        try:
            correlation, _ = stats.pearsonr(values, fitness)
            return float(correlation) if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _calculate_statistical_significance(self, values: np.ndarray, fitness: List[float]) -> float:
        """Calculate statistical significance of the relationship."""
        if len(fitness) < 3:
            return 1.0
        
        try:
            # Perform linear regression test
            slope, intercept, r_value, p_value, std_err = stats.linregress(values, fitness)
            return float(p_value)
        except:
            return 1.0
    
    def _find_optimal_range(self, values: np.ndarray, fitness: List[float]) -> Tuple[float, float]:
        """Find the optimal range of parameter values."""
        if len(fitness) < 2:
            return (float(np.min(values)), float(np.max(values)))
        
        fitness_array = np.array(fitness)
        
        # Find values that produce fitness within 80% of maximum
        max_fitness = np.max(fitness_array)
        threshold = max_fitness * 0.8
        
        good_indices = np.where(fitness_array >= threshold)[0]
        
        if len(good_indices) == 0:
            # Return range around best single value
            best_idx = np.argmax(fitness_array)
            value = values[best_idx]
            range_size = (np.max(values) - np.min(values)) * 0.1
            return (float(value - range_size), float(value + range_size))
        
        optimal_min = float(np.min(values[good_indices]))
        optimal_max = float(np.max(values[good_indices]))
        
        return (optimal_min, optimal_max)
    
    def get_sensitivity_report(self) -> Dict[str, Any]:
        """Generate a comprehensive sensitivity analysis report."""
        
        if not self.sensitivity_results:
            return {"error": "No sensitivity analysis results available"}
        
        # Rank parameters by sensitivity
        ranked_params = sorted(
            self.sensitivity_results.items(),
            key=lambda x: x[1].sensitivity_score,
            reverse=True
        )
        
        # Find most significant parameters
        significant_params = [
            name for name, result in self.sensitivity_results.items()
            if result.statistical_significance < self.significance_threshold
        ]
        
        # Find strongest interactions
        top_interactions = sorted(
            self.parameter_interactions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        report = {
            'summary': {
                'total_parameters_analyzed': len(self.sensitivity_results),
                'statistically_significant': len(significant_params),
                'most_sensitive_parameter': ranked_params[0][0] if ranked_params else None,
                'least_sensitive_parameter': ranked_params[-1][0] if ranked_params else None,
            },
            'parameter_rankings': [
                {
                    'parameter': name,
                    'sensitivity_score': result.sensitivity_score,
                    'correlation': result.correlation_with_fitness,
                    'p_value': result.statistical_significance,
                    'optimal_range': result.optimal_range,
                    'significant': result.statistical_significance < self.significance_threshold
                }
                for name, result in ranked_params
            ],
            'significant_parameters': significant_params,
            'parameter_interactions': [
                {
                    'parameters': list(params),
                    'interaction_strength': strength
                }
                for params, strength in top_interactions
            ],
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on sensitivity analysis."""
        recommendations = []
        
        if not self.sensitivity_results:
            return ["No analysis results available for recommendations"]
        
        # Find most sensitive parameters
        ranked_params = sorted(
            self.sensitivity_results.items(),
            key=lambda x: x[1].sensitivity_score,
            reverse=True
        )
        
        # Recommend focusing on most sensitive parameters
        if ranked_params:
            top_3 = ranked_params[:3]
            recommendations.append(
                f"Focus optimization efforts on: {', '.join([p[0] for p in top_3])}"
            )
        
        # Recommend optimal ranges for significant parameters
        for name, result in self.sensitivity_results.items():
            if result.statistical_significance < self.significance_threshold:
                opt_min, opt_max = result.optimal_range
                recommendations.append(
                    f"Keep {name} between {opt_min:.3f} and {opt_max:.3f} for optimal performance"
                )
        
        # Recommend interaction considerations
        strong_interactions = [
            params for params, strength in self.parameter_interactions.items()
            if strength > 0.1  # Threshold for "strong" interaction
        ]
        
        if strong_interactions:
            recommendations.append(
                f"Consider parameter interactions when tuning: {strong_interactions[:3]}"
            )
        
        return recommendations
    
    def export_results(self, filepath: str) -> None:
        """Export sensitivity analysis results to file."""
        
        export_data = {
            'analysis_timestamp': time.time(),
            'sensitivity_results': {
                name: {
                    'parameter_name': result.parameter_name,
                    'sensitivity_score': result.sensitivity_score,
                    'correlation_with_fitness': result.correlation_with_fitness,
                    'statistical_significance': result.statistical_significance,
                    'optimal_range': result.optimal_range,
                    'performance_curve': result.performance_curve,
                    'interaction_effects': result.interaction_effects
                }
                for name, result in self.sensitivity_results.items()
            },
            'parameter_interactions': {
                f"{p1}_{p2}": strength 
                for (p1, p2), strength in self.parameter_interactions.items()
            },
            'report': self.get_sensitivity_report()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            print(f"✅ Sensitivity analysis results exported to {filepath}")
        except Exception as e:
            print(f"❌ Error exporting results: {e}")


class MockAgentFactory:
    """Mock agent factory for testing sensitivity analysis."""
    
    @staticmethod
    def create_agent(**params):
        """Create a mock agent with specified parameters."""
        class MockAgent:
            def __init__(self, **kwargs):
                self.params = kwargs
                
        return MockAgent(**params)


class MockFitnessEvaluator:
    """Mock fitness evaluator for testing sensitivity analysis."""
    
    @staticmethod
    def evaluate_fitness(agent) -> float:
        """Evaluate mock fitness based on agent parameters."""
        # Simulate realistic parameter relationships
        
        # Learning rate should be moderate
        lr = agent.params.get('learning_rate', 0.01)
        lr_fitness = 1.0 - abs(lr - 0.02) * 10  # Optimal around 0.02
        
        # Motor torque should be moderate to high
        torque = agent.params.get('motor_torque', 150)
        torque_fitness = min(1.0, torque / 200.0)  # Better with higher torque
        
        # Body width should be moderate
        width = agent.params.get('body_width', 1.5)
        width_fitness = 1.0 - abs(width - 1.2) * 0.3  # Optimal around 1.2
        
        # Combine with some noise
        base_fitness = (lr_fitness + torque_fitness + width_fitness) / 3.0
        noise = np.random.normal(0, 0.1)
        
        return max(0.0, base_fitness + noise) 