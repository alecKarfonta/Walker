#!/usr/bin/env python3
"""
Test script to verify MLflow PostgreSQL integration is working.
"""

import os
import mlflow
import mlflow.tracking
import time

def test_mlflow_postgres():
    """Test MLflow PostgreSQL connection and basic operations."""
    print("🚀 Testing MLflow PostgreSQL Integration")
    print("=" * 50)
    
    # Test connection
    tracking_uri = "postgresql://walker_user:walker_secure_2024@walker-postgres:5432/mlflow_db"
    mlflow.set_tracking_uri(tracking_uri)
    
    print(f"📡 Connecting to: {tracking_uri}")
    
    try:
        # List experiments
        experiments = mlflow.search_experiments()
        print(f"✅ Connection successful!")
        print(f"📊 Found {len(experiments)} experiments:")
        for exp in experiments:
            print(f"   - {exp.name} (ID: {exp.experiment_id})")
        
        # Test creating a run
        experiment_name = "walker_robot_training"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment:
            print(f"\n🧪 Testing run creation in experiment '{experiment_name}'...")
            
            with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="postgres_test_run"):
                # Log some test metrics
                mlflow.log_param("test_param", "postgres_migration_test")
                mlflow.log_metric("test_metric", 42.0)
                mlflow.log_metric("test_fitness", 123.45)
                
                # Log multiple steps
                for step in range(5):
                    mlflow.log_metric("training_progress", step * 10, step=step)
                
                run_id = mlflow.active_run().info.run_id
                print(f"✅ Successfully created test run: {run_id}")
            
            # Verify the run was saved
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], max_results=1)
            if len(runs) > 0:
                latest_run = runs.iloc[0]
                print(f"📊 Latest run metrics:")
                print(f"   - test_metric: {latest_run.get('metrics.test_metric', 'N/A')}")
                print(f"   - test_fitness: {latest_run.get('metrics.test_fitness', 'N/A')}")
                
        else:
            print(f"❌ Experiment '{experiment_name}' not found")
            return False
        
        print(f"\n🎉 All tests passed! PostgreSQL MLflow integration is working correctly.")
        print(f"🌐 MLflow UI available at: http://localhost:5002")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mlflow_postgres()
    exit(0 if success else 1) 