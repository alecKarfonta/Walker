"""
Tests for Phase 2 evaluation framework components.
Tests parameter sensitivity analysis, comparative evaluation, and performance prediction.
"""

import pytest
import numpy as np
from typing import Dict, List, Any

# Import the Phase 2 components
from src.evaluation.parameter_sensitivity_analyzer import ParameterSensitivityAnalyzer
from src.evaluation.comparative_evaluator import ComparativeEvaluator
from src.evaluation.performance_predictor import PerformancePredictor


class TestParameterSensitivityAnalyzer:
    """Test parameter sensitivity analysis."""
    
    @pytest.fixture
    def analyzer(self):
        return ParameterSensitivityAnalyzer()
    
    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer is not None
        assert hasattr(analyzer, 'default_parameter_ranges')
        assert hasattr(analyzer, 'sensitivity_results')
        assert hasattr(analyzer, 'parameter_interactions')
    
    def test_basic_analyzer_functionality(self, analyzer):
        """Test basic analyzer functionality."""
        # The analyzer should initialize with default parameter ranges
        assert hasattr(analyzer, 'default_parameter_ranges')
        assert 'learning_rate' in analyzer.default_parameter_ranges
        assert 'body_width' in analyzer.default_parameter_ranges
    
    def test_sensitivity_report_generation(self, analyzer):
        """Test sensitivity report generation."""
        # Should generate empty report when no results
        report = analyzer.get_sensitivity_report()
        assert isinstance(report, dict)
        assert 'error' in report or 'summary' in report


class TestComparativeEvaluator:
    """Test comparative evaluation framework."""
    
    @pytest.fixture
    def evaluator(self):
        return ComparativeEvaluator()
    
    def test_initialization(self, evaluator):
        """Test evaluator initialization."""
        assert evaluator is not None
        assert hasattr(evaluator, 'experiments')
        assert hasattr(evaluator, 'comparisons')
        assert hasattr(evaluator, 'standard_configs')
    
    def test_standard_configurations(self, evaluator):
        """Test that standard configurations are loaded."""
        assert len(evaluator.standard_configs) > 0
        assert 'baseline_q_learning' in evaluator.standard_configs
        
        # Check that configurations have required fields
        baseline = evaluator.standard_configs['baseline_q_learning']
        assert hasattr(baseline, 'name')
        assert hasattr(baseline, 'parameters')
        assert hasattr(baseline, 'description')
    
    def test_ranking_empty_experiments(self, evaluator):
        """Test ranking with no experiments."""
        ranking = evaluator.get_ranking('final_fitness')
        assert isinstance(ranking, list)
        assert len(ranking) == 0
    
    def test_comparison_report_generation(self, evaluator):
        """Test comparison report generation."""
        # Should generate error report when no experiments
        report = evaluator.generate_comparison_report()
        assert isinstance(report, dict)
        assert 'error' in report


class TestPerformancePredictor:
    """Test performance prediction framework."""
    
    @pytest.fixture
    def predictor(self):
        return PerformancePredictor()
    
    def test_initialization(self, predictor):
        """Test predictor initialization."""
        assert predictor is not None
        assert hasattr(predictor, 'models')
        assert hasattr(predictor, 'training_features')
        assert hasattr(predictor, 'training_targets')
        assert hasattr(predictor, 'scalers')
    
    def test_empty_prediction_data(self, predictor):
        """Test predictor with no training data."""
        # Should handle empty training data gracefully
        assert len(predictor.training_features) == 0
        assert len(predictor.training_targets) == 0
        
        # Should not crash when getting a prediction result with empty features
        from src.evaluation.performance_predictor import PredictionFeatures
        empty_features = PredictionFeatures()
        result = predictor.predict_performance(empty_features)
        assert hasattr(result, 'predicted_values')


def test_basic_functionality():
    """Test basic functionality of all components."""
    # Test parameter sensitivity analyzer
    analyzer = ParameterSensitivityAnalyzer()
    assert analyzer is not None
    
    # Test comparative evaluator
    evaluator = ComparativeEvaluator()
    assert evaluator is not None
    
    # Test performance prediction framework
    predictor = PerformancePredictor()
    assert predictor is not None
    
    print("✅ Basic functionality tests passed")


def test_integration_basic():
    """Test basic integration between components."""
    # Create all three components
    analyzer = ParameterSensitivityAnalyzer()
    evaluator = ComparativeEvaluator()
    predictor = PerformancePredictor()
    
    # They should all exist and be different objects
    assert analyzer is not None
    assert evaluator is not None 
    assert predictor is not None
    assert analyzer != evaluator
    assert evaluator != predictor
    
    print("✅ Basic integration tests passed")


if __name__ == "__main__":
    # Run tests directly
    test_basic_functionality()
    test_integration_basic()
    print("✅ All basic tests completed successfully") 