"""
Tests for the evaluation framework.
Ensures all evaluation modules can be imported and basic functionality works.
"""

import pytest
import sys
import os
import time
from unittest.mock import Mock, MagicMock

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


class TestEvaluationFramework:
    """Test the evaluation framework components."""
    
    def test_individual_evaluator_import(self):
        """Test that individual evaluator can be imported."""
        from evaluation.individual_evaluator import IndividualRobotEvaluator, BehaviorAnalyzer
        
        evaluator = IndividualRobotEvaluator()
        analyzer = BehaviorAnalyzer()
        
        assert evaluator is not None
        assert analyzer is not None
    
    def test_exploration_evaluator_import(self):
        """Test that exploration evaluator can be imported."""
        from evaluation.exploration_evaluator import ExplorationEvaluator, ActionSpaceAnalyzer
        
        evaluator = ExplorationEvaluator()
        analyzer = ActionSpaceAnalyzer()
        
        assert evaluator is not None
        assert analyzer is not None
    
    def test_q_learning_evaluator_import(self):
        """Test that Q-learning evaluator can be imported."""
        from evaluation.q_learning_evaluator import QLearningEvaluator
        
        evaluator = QLearningEvaluator()
        assert evaluator is not None
    
    def test_training_evaluator_import(self):
        """Test that training evaluator can be imported."""
        from evaluation.training_evaluator import TrainingProgressEvaluator
        
        evaluator = TrainingProgressEvaluator()
        assert evaluator is not None
    
    def test_population_evaluator_import(self):
        """Test that population evaluator can be imported."""
        from evaluation.population_evaluator import PopulationEvaluator
        
        evaluator = PopulationEvaluator()
        assert evaluator is not None
    
    def test_mlflow_integration_import(self):
        """Test that MLflow integration can be imported."""
        try:
            from evaluation.mlflow_integration import MLflowIntegration, ExperimentComparator
            # Don't actually initialize MLflow in tests unless specifically testing it
            assert MLflowIntegration is not None
            assert ExperimentComparator is not None
        except ImportError as e:
            pytest.skip(f"MLflow not available: {e}")
    
    def test_metrics_collector_import(self):
        """Test that metrics collector can be imported."""
        from evaluation.metrics_collector import MetricsCollector
        
        # Initialize with evaluation disabled for testing
        collector = MetricsCollector(enable_mlflow=False, enable_file_export=False)
        assert collector is not None
    
    def test_dashboard_exporter_import(self):
        """Test that dashboard exporter can be imported."""
        from evaluation.dashboard_exporter import DashboardExporter
        
        # Initialize with API disabled for testing
        exporter = DashboardExporter(enable_api=False)
        assert exporter is not None
    
    def test_individual_evaluator_basic_functionality(self):
        """Test basic functionality of individual evaluator."""
        from evaluation.individual_evaluator import IndividualRobotEvaluator
        
        evaluator = IndividualRobotEvaluator()
        
        # Create a mock agent
        mock_agent = Mock()
        mock_agent.id = "test_agent_1"
        mock_agent.q_table = Mock()
        mock_agent.q_table.get_convergence_estimate.return_value = 0.5
        mock_agent.q_table.q_values = {}
        mock_agent.q_table.q_value_history = []
        mock_agent.q_table.state_coverage = set()
        mock_agent.action_history = [(1, 0), (0, 1), (1, 1)]
        mock_agent.total_reward = 5.0
        mock_agent.epsilon = 0.3
        mock_agent.steps = 100
        mock_agent.physical_params = Mock()
        mock_agent.physical_params.motor_torque = 150.0
        
        # Test evaluation
        metrics = evaluator.evaluate_robot(mock_agent, 100)
        
        assert metrics.robot_id == "test_agent_1"
        assert metrics.timestamp > 0
        assert metrics.q_learning_convergence == 0.5
        assert isinstance(metrics.action_diversity_score, float)
    
    def test_behavior_analyzer_basic_functionality(self):
        """Test basic functionality of behavior analyzer."""
        from evaluation.individual_evaluator import BehaviorAnalyzer
        
        analyzer = BehaviorAnalyzer()
        
        # Create a mock agent
        mock_agent = Mock()
        mock_agent.id = "test_agent_1"
        mock_agent.action_history = [(1, 0), (0, 1), (1, 1), (1, 0)]
        mock_agent.body = Mock()
        mock_agent.body.position = Mock()
        mock_agent.body.position.x = 10.0
        mock_agent.body.position.y = 5.0
        mock_agent.initial_position = [0, 0]
        mock_agent.total_reward = 3.0
        mock_agent.steps = 50
        mock_agent.recent_rewards = [0.1, 0.2, 0.15, 0.3]
        mock_agent.time_since_good_value = 10.0
        
        # Test analysis
        metrics = analyzer.analyze_behavior(mock_agent, 50)
        
        assert metrics.robot_id == "test_agent_1"
        assert metrics.timestamp > 0
        assert isinstance(metrics.movement_efficiency, float)
        assert isinstance(metrics.learning_velocity, float)
    
    def test_metrics_collector_basic_functionality(self):
        """Test basic functionality of metrics collector."""
        from evaluation.metrics_collector import MetricsCollector
        
        # Initialize with evaluation disabled for testing
        collector = MetricsCollector(enable_mlflow=False, enable_file_export=False)
        
        # Create mock agents
        mock_agents = []
        for i in range(3):
            agent = Mock()
            agent.id = f"test_agent_{i}"
            agent.q_table = Mock()
            agent.q_table.get_convergence_estimate.return_value = 0.5
            agent.q_table.q_values = {}
            agent.q_table.q_value_history = []
            agent.q_table.state_coverage = set()
            agent.action_history = [(1, 0), (0, 1)]
            agent.total_reward = float(i + 1)
            agent.epsilon = 0.3
            agent.steps = 100
            agent.physical_params = Mock()
            agent.physical_params.motor_torque = 150.0
            agent.body = Mock()
            agent.body.position = Mock()
            agent.body.position.x = 10.0
            agent.body.position.y = 5.0
            agent.body.angle = 0.1
            agent.initial_position = [0, 0]
            agent.recent_rewards = [0.1, 0.2]
            agent.time_since_good_value = 5.0
            agent.current_state = (1, 2)
            agent.current_action_tuple = (1, 0)
            agent.immediate_reward = 0.1
            mock_agents.append(agent)
        
        # Mock population stats
        population_stats = {
            'generation': 1,
            'best_fitness': 3.0,
            'average_fitness': 2.0,
            'diversity': 0.5,
            'total_agents': 3
        }
        
        # Mock evolution summary
        evolution_summary = {
            'generation': 1,
            'species_count': 2,
            'diversity': 0.5,
            'mutation_rate': 0.1
        }
        
        # Test metrics collection
        metrics = collector.collect_metrics(
            agents=mock_agents,
            population_stats=population_stats,
            evolution_summary=evolution_summary,
            generation=1,
            step_count=100,
            force_collection=True
        )
        
        assert metrics is not None
        assert metrics.generation == 1
        assert metrics.step_count == 100
        assert len(metrics.individual_metrics) == 3
        assert len(metrics.behavior_metrics) == 3
    
    def test_evaluation_framework_end_to_end(self):
        """Test end-to-end functionality of the evaluation framework."""
        from evaluation.metrics_collector import MetricsCollector
        
        # Initialize collector without external dependencies
        collector = MetricsCollector(enable_mlflow=False, enable_file_export=False)
        
        # Start a mock training session
        session_id = collector.start_training_session(
            session_name="test_session",
            population_size=2,
            evolution_config={'mutation_rate': 0.1}
        )
        
        # Create mock agents and collect metrics multiple times
        for generation in range(3):
            mock_agents = []
            for i in range(2):
                agent = Mock()
                agent.id = f"test_agent_{i}"
                # Set up all required mock attributes
                agent.q_table = Mock()
                agent.q_table.get_convergence_estimate.return_value = 0.5 + generation * 0.1
                agent.q_table.q_values = {}
                agent.q_table.q_value_history = []
                agent.q_table.state_coverage = set()
                agent.action_history = [(1, 0), (0, 1)]
                agent.total_reward = float(i + 1 + generation)
                agent.epsilon = 0.3 - generation * 0.05
                agent.steps = 100 + generation * 50
                agent.physical_params = Mock()
                agent.physical_params.motor_torque = 150.0
                agent.body = Mock()
                agent.body.position = Mock()
                agent.body.position.x = 10.0 + generation
                agent.body.position.y = 5.0
                agent.body.angle = 0.1
                agent.initial_position = [0, 0]
                agent.recent_rewards = [0.1, 0.2, 0.15]
                agent.time_since_good_value = 5.0
                agent.current_state = (1, 2)
                agent.current_action_tuple = (1, 0)
                agent.immediate_reward = 0.1
                mock_agents.append(agent)
            
            population_stats = {
                'generation': generation + 1,
                'best_fitness': 3.0 + generation,
                'average_fitness': 2.0 + generation,
                'diversity': 0.5,
                'total_agents': 2
            }
            
            evolution_summary = {
                'generation': generation + 1,
                'species_count': 2,
                'diversity': 0.5,
                'mutation_rate': 0.1
            }
            
            metrics = collector.collect_metrics(
                agents=mock_agents,
                population_stats=population_stats,
                evolution_summary=evolution_summary,
                generation=generation + 1,
                step_count=100 + generation * 50,
                force_collection=True
            )
            
            assert metrics is not None
            assert metrics.generation == generation + 1
        
        # Test getting current metrics summary
        summary = collector.get_current_metrics_summary()
        assert summary is not None
        assert 'generation' in summary
        assert 'individual_summaries' in summary
        
        # Test getting diagnostics
        diagnostics = collector.get_training_diagnostics()
        assert diagnostics is not None
        assert 'overall_health' in diagnostics
        
        # End training session
        collector.end_training_session({'final_test': True})
        
        print("✅ End-to-end evaluation framework test completed successfully")


if __name__ == "__main__":
    # Run a simple test if executed directly
    test = TestEvaluationFramework()
    test.test_evaluation_framework_end_to_end()
    print("✅ All basic tests passed!") 