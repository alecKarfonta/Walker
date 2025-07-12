#!/usr/bin/env python3
"""
Test script to verify the evaluation framework is working correctly.
Tests all evaluation components and generates sample reports.
"""

import sys
import os
import time
import json
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.evaluation.metrics_collector import MetricsCollector
from src.evaluation.training_evaluator import TrainingProgressEvaluator
from src.evaluation.individual_evaluator import IndividualRobotEvaluator
from src.evaluation.q_learning_evaluator import QLearningEvaluator
from src.evaluation.population_evaluator import PopulationEvaluator
from src.evaluation.dashboard_exporter import DashboardExporter

def test_evaluation_components():
    """Test individual evaluation components."""
    print("🧪 Testing Evaluation Framework Components")
    print("=" * 50)
    
    # Test MetricsCollector
    print("📊 Testing MetricsCollector...")
    try:
        metrics_collector = MetricsCollector(
            enable_mlflow=False,  # Skip MLflow for testing
            enable_file_export=True,
            export_directory="evaluation_exports"
        )
        print("✅ MetricsCollector initialized successfully")
    except Exception as e:
        print(f"❌ MetricsCollector failed: {e}")
        return False
    
    # Test TrainingProgressEvaluator
    print("🎯 Testing TrainingProgressEvaluator...")
    try:
        training_evaluator = TrainingProgressEvaluator()
        
        # Create mock population stats
        mock_population_stats = {
            'generation': 1,
            'best_fitness': 150.0,
            'average_fitness': 75.0,
            'diversity': 0.6,
            'total_agents': 30
        }
        
        training_metrics = training_evaluator.evaluate_training_progress(
            population_stats=mock_population_stats,
            generation=1,
            training_step=1000
        )
        
        print(f"✅ TrainingProgressEvaluator working - generated {len(training_evaluator.training_metrics)} metrics")
    except Exception as e:
        print(f"❌ TrainingProgressEvaluator failed: {e}")
        return False
    
    # Test IndividualRobotEvaluator
    print("🤖 Testing IndividualRobotEvaluator...")
    try:
        individual_evaluator = IndividualRobotEvaluator()
        print("✅ IndividualRobotEvaluator initialized successfully")
    except Exception as e:
        print(f"❌ IndividualRobotEvaluator failed: {e}")
        return False
    
    # Test QLearningEvaluator
    print("🧠 Testing QLearningEvaluator...")
    try:
        q_learning_evaluator = QLearningEvaluator()
        print("✅ QLearningEvaluator initialized successfully")
    except Exception as e:
        print(f"❌ QLearningEvaluator failed: {e}")
        return False
    
    # Test PopulationEvaluator
    print("👥 Testing PopulationEvaluator...")
    try:
        population_evaluator = PopulationEvaluator()
        print("✅ PopulationEvaluator initialized successfully")
    except Exception as e:
        print(f"❌ PopulationEvaluator failed: {e}")
        return False
    
    # Test DashboardExporter
    print("📈 Testing DashboardExporter...")
    try:
        dashboard_exporter = DashboardExporter(
            port=2323,  # Different port for testing
            enable_api=False  # Don't start server for testing
        )
        print("✅ DashboardExporter initialized successfully")
    except Exception as e:
        print(f"❌ DashboardExporter failed: {e}")
        return False
    
    print("\n🎉 All evaluation components passed basic tests!")
    return True

def test_evaluation_exports():
    """Test that evaluation exports are working."""
    print("\n📁 Testing Evaluation Export System")
    print("=" * 40)
    
    export_dir = Path("evaluation_exports")
    
    # Ensure export directory exists
    export_dir.mkdir(exist_ok=True)
    print(f"✅ Export directory ready: {export_dir}")
    
    # Test MetricsCollector with file export
    try:
        metrics_collector = MetricsCollector(
            enable_mlflow=False,
            enable_file_export=True,
            export_directory=str(export_dir)
        )
        
        # Create mock comprehensive metrics
        from src.evaluation.metrics_collector import ComprehensiveMetrics
        
        mock_metrics = ComprehensiveMetrics(
            timestamp=time.time(),
            generation=1,
            step_count=1000,
            individual_metrics={},
            behavior_metrics={},
            exploration_metrics={},
            action_space_metrics={},
            q_learning_metrics={},
            training_metrics=None,
            population_metrics=None
        )
        
        # Test export functionality
        metrics_collector._export_to_files(mock_metrics)
        
        # Check if files were created
        json_files = list(export_dir.glob("*.json"))
        if json_files:
            print(f"✅ Export test successful - {len(json_files)} files created")
            for file in json_files:
                print(f"   📄 {file.name}")
        else:
            print("⚠️ No export files found")
            
    except Exception as e:
        print(f"❌ Export test failed: {e}")
        return False
    
    return True

def test_integration_with_training():
    """Test integration with training environment (if running)."""
    print("\n🔗 Testing Integration with Training System")
    print("=" * 45)
    
    try:
        import requests
        
        # Try to connect to training system
        response = requests.get("http://localhost:2322/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Training system is running and accessible")
            print(f"   Agents: {data.get('statistics', {}).get('total_agents', 0)}")
            print(f"   Generation: {data.get('statistics', {}).get('generation', 0)}")
            
            # Test evaluation endpoints if they exist
            try:
                eval_response = requests.get("http://localhost:2322/evaluation/status", timeout=3)
                if eval_response.status_code == 200:
                    print("✅ Evaluation endpoints accessible")
                else:
                    print("⚠️ Evaluation endpoints not available (may not be implemented yet)")
            except:
                print("⚠️ Evaluation endpoints not accessible")
            
            return True
        else:
            print(f"⚠️ Training system responded with status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"⚠️ Training system not accessible: {e}")
        print("   This is expected if training is not currently running")
        return False

def generate_test_report():
    """Generate a test evaluation report."""
    print("\n📊 Generating Test Evaluation Report")
    print("=" * 35)
    
    try:
        from scripts.generate_evaluation_report import EvaluationReportGenerator
        
        generator = EvaluationReportGenerator("evaluation_exports")
        
        # Try to generate historical report (should work with existing data)
        historical_report = generator.generate_historical_report()
        
        if historical_report:
            print("✅ Test historical report generated successfully")
            
            # Print some key findings
            data_sources = historical_report.get('data_sources', {})
            insights = historical_report.get('insights', [])
            
            print(f"   📊 Data sources found: {list(data_sources.keys())}")
            if insights:
                print(f"   💡 Key insight: {insights[0]}")
            
            return True
        else:
            print("⚠️ Could not generate test report (may be missing historical data)")
            return False
            
    except Exception as e:
        print(f"❌ Test report generation failed: {e}")
        return False

def main():
    print("🤖 Walker Robot Training - Evaluation Framework Test")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    test_results.append(("Component Tests", test_evaluation_components()))
    test_results.append(("Export Tests", test_evaluation_exports()))
    test_results.append(("Integration Tests", test_integration_with_training()))
    test_results.append(("Report Generation", generate_test_report()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 Test Summary")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Evaluation framework is working correctly.")
        print("\nNext steps:")
        print("1. Start training with evaluation enabled: python train_robots_web_visual.py")
        print("2. Generate live reports: python scripts/generate_evaluation_report.py --live")
        print("3. View evaluation dashboard: http://localhost:2322/dashboard")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Check the errors above.")
        
    # Show what was created
    export_dir = Path("evaluation_exports")
    if export_dir.exists():
        files = list(export_dir.glob("*"))
        if files:
            print(f"\n📁 Files created in {export_dir}:")
            for file in files:
                print(f"   📄 {file.name}")

if __name__ == "__main__":
    main() 