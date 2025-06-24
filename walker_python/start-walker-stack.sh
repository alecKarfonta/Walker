#!/bin/bash

# Walker Robot Training Stack Startup Script
# This script starts the complete evaluation framework with Docker Compose

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🤖 Walker Robot Training Stack${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Function to print status messages
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if ports are available
check_port() {
    local port=$1
    local service=$2
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        print_error "Port $port is already in use (needed for $service)"
        echo "Please stop the service using port $port or change the port in docker-compose.yml"
        return 1
    fi
}

echo "🔍 Checking port availability..."
check_port 8080 "Walker App" || exit 1
check_port 5001 "MLflow" || exit 1
check_port 9091 "Prometheus" || exit 1
check_port 3001 "Grafana" || exit 1
check_port 6380 "Redis" || exit 1
check_port 5433 "PostgreSQL" || exit 1
print_status "All required ports are available"

# Create necessary directories if they don't exist
echo "📁 Creating necessary directories..."
mkdir -p evaluation_exports
mkdir -p experiments
mkdir -p mlruns
mkdir -p config/prometheus
mkdir -p config/grafana/datasources
mkdir -p config/grafana/dashboards
mkdir -p config/grafana/dashboard-configs
print_status "Directories created"

# Function to start services
start_services() {
    echo "🚀 Starting Walker Training Stack..."
    
    # Start infrastructure services first
    echo "  📊 Starting infrastructure services (MLflow, Prometheus, Grafana)..."
    docker-compose up -d mlflow prometheus grafana redis postgres
    
    # Wait for infrastructure to be ready
    echo "  ⏳ Waiting for infrastructure services to be ready..."
    sleep 10
    
    # Check if MLflow is ready
    echo "  🔍 Checking MLflow health..."
    for i in {1..30}; do
        if curl -f -s http://localhost:5001/health > /dev/null 2>&1; then
            print_status "MLflow is ready"
            break
        fi
        if [ $i -eq 30 ]; then
            print_warning "MLflow might not be ready yet, continuing anyway..."
        fi
        sleep 2
    done
    
    # Check if Prometheus is ready
    echo "  🔍 Checking Prometheus health..."
    for i in {1..30}; do
        if curl -f -s http://localhost:9091/-/healthy > /dev/null 2>&1; then
            print_status "Prometheus is ready"
            break
        fi
        if [ $i -eq 30 ]; then
            print_warning "Prometheus might not be ready yet, continuing anyway..."
        fi
        sleep 2
    done
    
    # Start the main application
    echo "  🤖 Starting Walker training application..."
    docker-compose up -d walker-app
    
    echo ""
    print_status "Walker Training Stack is starting up!"
    echo ""
}

# Function to show service URLs
show_urls() {
    echo -e "${BLUE}🌐 Service URLs:${NC}"
    echo -e "  🤖 Walker Training Interface: ${GREEN}http://localhost:8080${NC}"
    echo -e "  📊 MLflow Tracking UI:        ${GREEN}http://localhost:5001${NC}"
    echo -e "  📈 Prometheus:               ${GREEN}http://localhost:9091${NC}"
    echo -e "  📊 Grafana Dashboard:        ${GREEN}http://localhost:3001${NC}"
    echo -e "      └─ Username: ${YELLOW}admin${NC}"
    echo -e "      └─ Password: ${YELLOW}walker-admin-2024${NC}"
    echo -e "  🔧 Dashboard API:            ${GREEN}http://localhost:2322${NC}"
    echo ""
}

# Function to show logs
show_logs() {
    echo -e "${BLUE}📋 Viewing logs for all services...${NC}"
    echo -e "Press ${YELLOW}Ctrl+C${NC} to stop viewing logs (services will continue running)"
    echo ""
    docker-compose logs -f
}

# Function to stop services
stop_services() {
    echo "🛑 Stopping Walker Training Stack..."
    docker-compose down
    print_status "All services stopped"
}

# Function to clean up everything
cleanup() {
    echo "🧹 Cleaning up Walker Training Stack..."
    docker-compose down -v --remove-orphans
    print_status "All services and volumes removed"
}

# Main menu
case "${1:-start}" in
    "start")
        start_services
        show_urls
        echo -e "${YELLOW}💡 Tip: Use './start-walker-stack.sh logs' to view service logs${NC}"
        echo -e "${YELLOW}💡 Tip: Use './start-walker-stack.sh stop' to stop all services${NC}"
        ;;
    "logs")
        show_logs
        ;;
    "stop")
        stop_services
        ;;
    "restart")
        stop_services
        sleep 2
        start_services
        show_urls
        ;;
    "cleanup")
        cleanup
        ;;
    "status")
        echo "📊 Service Status:"
        docker-compose ps
        echo ""
        show_urls
        ;;
    "help"|"-h"|"--help")
        echo "Walker Training Stack Management Script"
        echo ""
        echo "Usage: $0 [COMMAND]"
        echo ""
        echo "Commands:"
        echo "  start     Start all services (default)"
        echo "  stop      Stop all services"
        echo "  restart   Restart all services"
        echo "  logs      View logs from all services"
        echo "  status    Show service status and URLs"
        echo "  cleanup   Stop services and remove all volumes"
        echo "  help      Show this help message"
        echo ""
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac 