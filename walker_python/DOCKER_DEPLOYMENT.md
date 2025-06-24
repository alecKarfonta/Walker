# Walker Robot Training Stack - Docker Deployment

This guide provides instructions for deploying the complete Walker Robot Training system with comprehensive monitoring and evaluation using Docker Compose.

## 🚀 Quick Start

### Prerequisites

- Docker (version 20.10 or higher)
- Docker Compose (version 2.0 or higher)
- 8GB+ RAM (recommended)
- 10GB+ free disk space

### One-Command Deployment

```bash
./start-walker-stack.sh
```

This script will:
- Check port availability 
- Create necessary directories
- Start all services in the correct order
- Provide health checks
- Display service URLs

## 📊 Services Overview

The stack includes the following services with **non-conflicting ports**:

| Service | Port | Purpose | URL |
|---------|------|---------|-----|
| **Walker App** | 8080 | Main training interface | http://localhost:8080 |
| **Dashboard API** | 2322 | Evaluation metrics API | http://localhost:2322 |
| **MLflow** | 5001 | ML experiment tracking | http://localhost:5001 |
| **Prometheus** | 9091 | Metrics collection | http://localhost:9091 |
| **Grafana** | 3001 | Visualization dashboards | http://localhost:3001 |
| **Redis** | 6380 | Caching & session storage | - |
| **PostgreSQL** | 5433 | Advanced metrics storage | - |

### Default Credentials

- **Grafana**: 
  - Username: `admin`
  - Password: `walker-admin-2024`

- **PostgreSQL**:
  - Database: `walker_metrics`
  - Username: `walker_user`
  - Password: `walker_secure_2024`

## 🛠️ Management Commands

### Start Services
```bash
./start-walker-stack.sh start    # Start all services
./start-walker-stack.sh          # Same as above (default)
```

### Monitor Services
```bash
./start-walker-stack.sh status   # Show service status and URLs
./start-walker-stack.sh logs     # View real-time logs
```

### Control Services
```bash
./start-walker-stack.sh stop     # Stop all services
./start-walker-stack.sh restart  # Restart all services
./start-walker-stack.sh cleanup  # Stop and remove all data
```

### Help
```bash
./start-walker-stack.sh help     # Show all available commands
```

## 📈 Accessing the Applications

### 1. Walker Training Interface (Port 8080)
- **Main training visualization**: http://localhost:8080
- Real-time robot simulation and training progress
- Interactive robot control and parameter adjustment
- Live leaderboards and statistics

### 2. MLflow Tracking (Port 5001)
- **ML experiment tracking**: http://localhost:5001
- Compare different training runs
- Track model performance and parameters
- Download trained models and artifacts

### 3. Grafana Dashboards (Port 3001)
- **Comprehensive monitoring**: http://localhost:3001
- Pre-configured dashboards for walker training
- Real-time metrics visualization
- Custom dashboard creation

### 4. Prometheus Metrics (Port 9091)
- **Raw metrics**: http://localhost:9091
- Query training metrics directly
- Set up custom alerts
- Advanced metric analysis

### 5. Dashboard API (Port 2322)
- **Evaluation metrics API**: http://localhost:2322/api/metrics
- Programmatic access to training data
- Integration with external systems
- Custom metric queries

## 🔧 Configuration

### Environment Variables

The Docker Compose setup uses these key environment variables:

```yaml
# Application Configuration
PYTHONPATH: /app
MLFLOW_TRACKING_URI: http://mlflow:5000
PROMETHEUS_MULTIPROC_DIR: /tmp/prometheus_metrics
GRAFANA_URL: http://grafana:3000

# Grafana Configuration  
GF_SECURITY_ADMIN_USER: admin
GF_SECURITY_ADMIN_PASSWORD: walker-admin-2024
GF_SERVER_ROOT_URL: http://localhost:3001

# Database Configuration
POSTGRES_DB: walker_metrics
POSTGRES_USER: walker_user
POSTGRES_PASSWORD: walker_secure_2024
```

### Custom Ports

To change ports, edit `docker-compose.yml`:

```yaml
services:
  walker-app:
    ports:
      - "YOUR_PORT:8080"  # Change YOUR_PORT to desired port
```

### Volume Mounts

The following directories are mounted for data persistence:

- `./evaluation_exports` → Application evaluation data
- `./experiments` → Walker experiment data  
- `./mlruns` → MLflow experiment tracking
- `mlflow-data` → MLflow database and artifacts
- `prometheus-data` → Prometheus time-series data
- `grafana-data` → Grafana dashboards and settings
- `postgres-data` → PostgreSQL database
- `redis-data` → Redis cache data

## 📊 Monitoring Features

### Real-Time Metrics

The stack automatically collects and visualizes:

- **Training Progress**: Fitness trends, learning curves, convergence rates
- **Population Dynamics**: Generation statistics, diversity metrics, evolution progress  
- **Individual Robots**: Performance rankings, Q-learning metrics, behavior analysis
- **System Performance**: CPU/memory usage, training speed, resource utilization
- **Parameter Sensitivity**: Which parameters most affect performance
- **Performance Prediction**: Early stopping recommendations based on ML models

### Pre-Built Dashboards

Grafana includes pre-configured dashboards:

1. **Walker Training Overview** - High-level training progress
2. **Individual Robot Metrics** - Detailed robot performance
3. **Parameter Sensitivity Analysis** - Parameter optimization insights
4. **System Performance** - Resource usage and system health

## 🔄 Data Flow

```
Walker App → Prometheus → Grafana (Visualization)
    ↓
MLflow (Experiment Tracking)
    ↓
PostgreSQL (Advanced Metrics)
    ↓
Redis (Caching)
```

## 🐛 Troubleshooting

### Common Issues

#### Port Conflicts
```bash
# Check what's using a port
lsof -i :8080

# The script will automatically check for conflicts
./start-walker-stack.sh
```

#### Service Not Starting
```bash
# Check service logs
./start-walker-stack.sh logs

# Restart specific service
docker-compose restart walker-app
```

#### Out of Memory
```bash
# Check memory usage
docker stats

# Reduce number of robots in training
# Edit train_robots_web_visual.py and change num_agents
```

#### Permission Issues
```bash
# Fix permissions for mounted volumes
sudo chown -R $USER:$USER evaluation_exports experiments mlruns
```

### Health Checks

All services include health checks. Check service health:

```bash
# Overall status
./start-walker-stack.sh status

# Detailed health info
docker-compose ps
```

### Reset Everything

To completely reset the stack:

```bash
./start-walker-stack.sh cleanup
docker system prune -f
```

## 🧪 Development Mode

For development, you can run services individually:

```bash
# Start only infrastructure
docker-compose up -d mlflow prometheus grafana

# Run walker app locally (with virtual env)
source venv/bin/activate
python train_robots_web_visual.py

# Start specific services
docker-compose up -d postgres redis
```

## 📚 Advanced Usage

### Custom Metrics

Add custom metrics to Prometheus by modifying the walker application to expose additional metrics at `/metrics` endpoint.

### Custom Dashboards

1. Access Grafana at http://localhost:3001
2. Login with admin/walker-admin-2024
3. Create new dashboards using Prometheus as data source
4. Export dashboard JSON and save to `config/grafana/dashboard-configs/`

### Database Access

Connect to PostgreSQL for advanced analytics:

```bash
# Connect to database
docker exec -it walker-postgres psql -U walker_user -d walker_metrics

# Or use external tool with:
# Host: localhost
# Port: 5433
# Database: walker_metrics  
# Username: walker_user
# Password: walker_secure_2024
```

## 🔒 Security Notes

- Change default passwords in production
- Use environment files for sensitive configuration
- Consider using Docker secrets for production deployments
- Limit network access to required ports only

## 📦 Production Deployment

For production use:

1. Use Docker Swarm or Kubernetes
2. Set up proper backup strategies for volumes
3. Implement log rotation
4. Use external databases for persistence
5. Set up SSL/TLS termination
6. Implement proper monitoring and alerting

## 🆘 Support

For issues with the Docker deployment:

1. Check the logs: `./start-walker-stack.sh logs`
2. Verify service status: `./start-walker-stack.sh status`
3. Review this documentation
4. Check the main Walker project README for application-specific issues

---

**Happy Training! 🤖** 