#!/bin/bash

echo "🚀 Starting Walker MLflow with PostgreSQL Backend"
echo "================================================"

# Check if Docker Compose is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed or not in PATH"
    exit 1
fi

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker compose down

# Start PostgreSQL first
echo "🐘 Starting PostgreSQL database..."
docker compose up -d walker-postgres

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL to be ready..."
sleep 10

# Check PostgreSQL health
for i in {1..30}; do
    if docker compose exec walker-postgres pg_isready -U walker_user -d walker_metrics > /dev/null 2>&1; then
        echo "✅ PostgreSQL is ready!"
        break
    else
        echo "⏳ Waiting for PostgreSQL... (attempt $i/30)"
        sleep 2
    fi
    
    if [ $i -eq 30 ]; then
        echo "❌ PostgreSQL failed to start after 60 seconds"
        docker compose logs walker-postgres
        exit 1
    fi
done

# Initialize MLflow database schema
echo "🔧 Initializing MLflow database schema..."
docker compose run --rm walker-training-app python scripts/init_mlflow_postgres.py

if [ $? -ne 0 ]; then
    echo "❌ Failed to initialize MLflow database"
    exit 1
fi

# Start MLflow server
echo "🔬 Starting MLflow server..."
docker compose up -d walker-mlflow

# Wait for MLflow to be ready
echo "⏳ Waiting for MLflow server to be ready..."
sleep 5

for i in {1..30}; do
    if curl -s -f http://localhost:5002/health > /dev/null 2>&1; then
        echo "✅ MLflow server is ready!"
        break
    else
        echo "⏳ Waiting for MLflow server... (attempt $i/30)"
        sleep 2
    fi
    
    if [ $i -eq 30 ]; then
        echo "❌ MLflow server failed to start after 60 seconds"
        docker compose logs walker-mlflow
        exit 1
    fi
done

# Start the full application
echo "🤖 Starting Walker training application..."
docker compose up -d

echo ""
echo "🎉 Walker MLflow with PostgreSQL is ready!"
echo "================================="
echo "📊 MLflow UI: http://localhost:5002"
echo "🤖 Training App: http://localhost:7777"
echo "📈 Prometheus: http://localhost:9889"
echo "📊 Grafana: http://localhost:3009"
echo ""
echo "📋 Useful commands:"
echo "   View logs: docker compose logs -f walker-mlflow"
echo "   Check status: docker compose ps"
echo "   Stop all: docker compose down"
echo ""
echo "🔧 Database details:"
echo "   Host: localhost:5434"
echo "   Database: walker_metrics"
echo "   User: walker_user"
echo "   Connection: postgresql://walker_user:walker_secure_2024@localhost:5434/walker_metrics" 