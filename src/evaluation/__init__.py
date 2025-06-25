"""
Evaluation framework for robot training system.
Provides comprehensive metrics and analysis for individual robots, 
population dynamics, and training effectiveness.

Phase 2 enhancements include:
- Parameter sensitivity analysis
- Comparative evaluation between configurations
- Performance prediction models based on early training metrics
- Advanced behavioral analysis
"""

from .individual_evaluator import IndividualRobotEvaluator, BehaviorAnalyzer
from .exploration_evaluator import ExplorationEvaluator, ActionSpaceAnalyzer
from .q_learning_evaluator import QLearningEvaluator
from .training_evaluator import TrainingProgressEvaluator
from .population_evaluator import PopulationEvaluator
from .mlflow_integration import MLflowIntegration, ExperimentComparator
from .metrics_collector import MetricsCollector
from .dashboard_exporter import DashboardExporter

# Phase 2 components
from .parameter_sensitivity_analyzer import ParameterSensitivityAnalyzer, ParameterSensitivityResult
from .comparative_evaluator import ComparativeEvaluator, ExperimentConfiguration, ExperimentResult
from .performance_predictor import PerformancePredictor, PredictionFeatures, PredictionTarget

__all__ = [
    # Phase 1 components
    'IndividualRobotEvaluator',
    'BehaviorAnalyzer', 
    'ExplorationEvaluator',
    'ActionSpaceAnalyzer',
    'QLearningEvaluator',
    'TrainingProgressEvaluator',
    'PopulationEvaluator',
    'MLflowIntegration',
    'ExperimentComparator',
    'MetricsCollector',
    'DashboardExporter',
    
    # Phase 2 components - Advanced Analytics
    'ParameterSensitivityAnalyzer',
    'ParameterSensitivityResult', 
    'ComparativeEvaluator',
    'ExperimentConfiguration',
    'ExperimentResult',
    'PerformancePredictor',
    'PredictionFeatures',
    'PredictionTarget'
] 