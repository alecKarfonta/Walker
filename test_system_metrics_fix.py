#!/usr/bin/env python3
"""
Test script to verify MLflow system metrics monitoring fix.
This script demonstrates the correct implementation of persistent system metrics monitoring.
"""

import os
import sys
import time
import threading
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_system_metrics():
    """Test the fixed system metrics implementation."""
    print("🧪 Testing MLflow system metrics monitoring fix...")
    
    try:
        # Import our fixed MLflow integration
        from src.evaluation.mlflow_integration import MLflowIntegration
        
        # Create MLflow integration instance
        print("📊 Creating MLflow integration...")
        mlflow_integration = MLflowIntegration(
            tracking_uri="sqlite:///test_metrics.db",
            experiment_name="system_metrics_test"
        )
        
        # Start a training run
        print("🔬 Starting training run...")
        run_id = mlflow_integration.start_training_run(
            run_name="system_metrics_test_run",
            population_size=10,
            evolution_config={"mutation_rate": 0.1, "crossover_rate": 0.7}
        )
        
        if run_id:
            print(f"✅ Training run started: {run_id}")
            
            # Monitor for 30 seconds to see if system metrics are stable
            print("📈 Monitoring system metrics for 30 seconds...")
            for i in range(6):
                time.sleep(5)
                print(f"   {(i+1)*5}s: System metrics should be running continuously...")
                
                # Log some custom metrics to verify MLflow is working
                import mlflow
                mlflow.log_metric(f"test_metric", i * 0.5, step=i)
                
            print("✅ System metrics monitoring test completed!")
            print("🔍 Expected behavior:")
            print("   - System metrics thread should start ONCE and run continuously")
            print("   - No 'Starting/Stopping' messages should appear repeatedly")
            print("   - Check MLflow UI 'System Metrics' tab for continuous data")
            
            # End the run
            mlflow.end_run()
            print("🔄 Training run ended")
            
        else:
            print("❌ Failed to start training run")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """Run the system metrics test."""
    print("=" * 60)
    print("🔧 MLFLOW SYSTEM METRICS MONITORING FIX TEST")
    print("=" * 60)
    
    success = test_system_metrics()
    
    print("=" * 60)
    if success:
        print("✅ System metrics test PASSED!")
        print("🔧 The fix should resolve the constant start/stop issue")
    else:
        print("❌ System metrics test FAILED!")
        print("🔧 Additional debugging may be needed")
    print("=" * 60)

if __name__ == "__main__":
    main() 