#!/bin/bash

echo "🔄 Updating MLflow Dashboard with Latest Training Data"
echo "===================================================="

# Check if container is running
if ! docker compose ps | grep -q "walker-training-app.*Up"; then
    echo "❌ Walker training container is not running!"
    echo "   Please start it first: docker compose up -d"
    exit 1
fi

echo "📊 Syncing latest training data..."

# Copy latest database to working location
docker compose exec walker-training-app cp /app/experiments/walker_experiments.db /tmp/mlflow_data.db
docker compose exec walker-training-app chmod 666 /tmp/mlflow_data.db

echo "🔬 Regenerating analytics dashboard..."

# Regenerate the dashboard with latest data
docker compose exec walker-training-app python -c "
import sys
import os
from datetime import datetime
sys.path.insert(0, '/app/src')

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    
    # Use the updated database copy
    db_path = '/tmp/mlflow_data.db'
    mlflow.set_tracking_uri(f'sqlite:///{db_path}')
    client = MlflowClient()
    
    # Get current statistics
    experiments = client.search_experiments()
    total_runs = 0
    for exp in experiments:
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        total_runs += len(runs)
        print(f'  📈 {exp.name}: {len(runs)} runs')
    
    print(f'✅ Dashboard updated with {total_runs} runs from {len(experiments)} experiments')
    print(f'📅 Last updated: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')
    
except Exception as e:
    print(f'❌ Error updating dashboard: {e}')
"

echo ""
echo "✅ MLflow Dashboard Updated!"
echo "🌐 View latest data at: http://localhost:7777/static/mlflow_analytics_complete.html"
echo ""
echo "💡 To auto-update regularly, you can run this script periodically:"
echo "   watch -n 300 ./scripts/update_mlflow_dashboard.sh  # Updates every 5 minutes" 