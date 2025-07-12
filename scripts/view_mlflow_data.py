#!/usr/bin/env python3
"""
View MLflow experiment data for Walker robot training.
Shows recent runs and metrics without needing the MLflow UI.
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    
    # Set up MLflow client
    db_path = "experiments/walker_experiments.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    client = MlflowClient()
    
    print("🔬 MLflow Walker Robot Training Data")
    print("=" * 50)
    
    # Get experiments
    experiments = client.search_experiments()
    
    for exp in experiments:
        print(f"\n📊 Experiment: {exp.name}")
        print(f"   ID: {exp.experiment_id}")
        
        # Get recent runs (last 10)
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id], 
            max_results=10,
            order_by=["start_time DESC"]
        )
        
        print(f"   Recent runs: {len(runs)} (showing last 10)")
        
        if runs:
            print("\n   📈 Recent Run Metrics:")
            print("   " + "-" * 70)
            print(f"   {'Run ID':<12} {'Status':<10} {'Generation':<5} {'Avg Fitness':<12} {'Best Fitness':<12} {'Population':<6}")
            print("   " + "-" * 70)
            
            for run in runs:
                run_id = run.info.run_id[:8]
                status = run.info.status
                metrics = run.data.metrics
                
                generation = metrics.get('generation', 'N/A')
                avg_fitness = f"{metrics.get('avg_fitness', 0):.2f}" if metrics.get('avg_fitness') else 'N/A'
                best_fitness = f"{metrics.get('best_fitness', 0):.2f}" if metrics.get('best_fitness') else 'N/A'
                population = int(metrics.get('population_size', 0)) if metrics.get('population_size') else 'N/A'
                
                print(f"   {run_id:<12} {status:<10} {generation!s:<5} {avg_fitness:<12} {best_fitness:<12} {population!s:<6}")
            
            # Show metrics from the most recent run
            if runs:
                latest_run = runs[0]
                print(f"\n   🎯 Latest Run Details ({latest_run.info.run_id[:8]}):")
                print("   " + "-" * 40)
                
                # Show all metrics
                metrics = latest_run.data.metrics
                if metrics:
                    for metric_name, value in sorted(metrics.items()):
                        if isinstance(value, (int, float)):
                            print(f"     {metric_name}: {value:.3f}")
                        else:
                            print(f"     {metric_name}: {value}")
                
                # Show parameters
                params = latest_run.data.params
                if params:
                    print(f"\n   ⚙️  Run Parameters:")
                    for param_name, value in sorted(params.items()):
                        print(f"     {param_name}: {value}")
        
        print(f"\n   💾 Total runs in experiment: {len(client.search_runs(exp.experiment_id))}")
    
    print(f"\n🌐 To view full MLflow UI:")
    print(f"   1. MLflow UI available at: http://localhost:5002 (using PostgreSQL backend)")
    print(f"   2. Open: http://localhost:5000")
    print(f"   3. Or try: http://localhost:5000 (if running in Docker)")
    
    print(f"\n📂 Database location: {db_path}")
    if os.path.exists(db_path):
        size_mb = os.path.getsize(db_path) / 1024 / 1024
        print(f"   Size: {size_mb:.1f} MB")

except Exception as e:
    print(f"❌ Error accessing MLflow data: {e}")
    import traceback
    traceback.print_exc() 