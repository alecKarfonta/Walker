#!/bin/bash

echo "🔬 Starting MLflow UI for Walker Robot Training"
echo "=============================================="

# Check if Docker is running
if ! docker compose ps | grep -q "walker-training-app"; then
    echo "❌ Walker training container is not running!"
    echo "   Please start it first: docker compose up -d"
    exit 1
fi

echo "📊 MLflow UI is available via the dedicated MLflow server..."
echo "   The MLflow server is automatically started with PostgreSQL backend"

echo "✅ MLflow UI should be available at:"
echo "   🌐 http://localhost:5002 (MLflow server)"
echo ""
echo "📊 Your training data includes:"
echo "   • 160+ runs in 'walker_robot_training' experiment"
echo "   • 150+ runs in 'Default' experiment"
echo "   • Population metrics (fitness, diversity, generation)"
echo "   • Individual robot performance data"
echo "   • Training progress over time"
echo ""
echo "💡 If the UI doesn't load, try:"
echo "   1. Wait a few more seconds and refresh"
echo "   2. Check port 5002 isn't used by another service"
echo "   3. Restart: docker compose restart walker-mlflow walker-postgres"
echo "   4. Check logs: docker compose logs walker-mlflow walker-postgres"

# Try to test if it's accessible
if curl -s -f http://localhost:5002 > /dev/null 2>&1; then
    echo "✅ MLflow UI is accessible!"
else
    echo "⚠️  MLflow UI may still be starting up..."
    echo "   Check that walker-mlflow container is running: docker compose ps"
fi 