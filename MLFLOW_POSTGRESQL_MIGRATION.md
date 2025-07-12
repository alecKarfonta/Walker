# MLflow PostgreSQL Migration Guide

## Overview

The Walker robot training system has been migrated from SQLite to PostgreSQL for MLflow experiment tracking. This resolves database access issues and provides better performance and reliability.

## What Changed

### 🔄 Database Backend
- **Before**: SQLite database (`experiments/walker_experiments.db`)
- **After**: PostgreSQL database (`walker_metrics` database)

### 🚀 Quick Start

1. **Start the new PostgreSQL-based system**:
   ```bash
   ./scripts/start_mlflow_postgres.sh
   ```

2. **Access MLflow UI**:
   - Visit: http://localhost:5002

### 📋 Detailed Changes

#### Docker Compose (`docker-compose.yml`)
- ✅ Updated MLflow service to use PostgreSQL backend
- ✅ Added dependency on PostgreSQL service
- ✅ Changed environment variables to use PostgreSQL connection string

#### Python Code
- ✅ Updated `src/training_environment.py` MLflow tracking URI
- ✅ Updated `src/evaluation/mlflow_integration.py` to use environment variables
- ✅ Enhanced error handling and fallback mechanisms

#### Scripts
- ✅ Updated `scripts/start_mlflow_ui.sh` to point to new MLflow server
- ✅ Updated `scripts/view_mlflow_data.py` for PostgreSQL backend
- ✅ Updated `scripts/export_mlflow_report.py` for new UI location
- ✅ Created `scripts/init_mlflow_postgres.py` for database initialization
- ✅ Created `scripts/start_mlflow_postgres.sh` for complete startup

#### Dependencies
- ✅ Added `psycopg2-binary>=2.9.0` to `requirements.txt`

## 🔧 Configuration Details

### Environment Variables
```bash
MLFLOW_TRACKING_URI=postgresql://walker_user:walker_secure_2024@walker-postgres:5432/walker_metrics
```

### Database Connection
- **Host**: walker-postgres (internal), localhost:5434 (external)
- **Database**: walker_metrics
- **User**: walker_user
- **Password**: walker_secure_2024

### Service Ports
- **MLflow UI**: http://localhost:5002
- **PostgreSQL**: localhost:5434
- **Training App**: http://localhost:7777

## 🚀 Migration Steps

### For New Users
Simply run:
```bash
./scripts/start_mlflow_postgres.sh
```

### For Existing Users (Data Migration)

1. **Stop existing containers**:
   ```bash
   docker compose down
   ```

2. **Start with PostgreSQL**:
   ```bash
   ./scripts/start_mlflow_postgres.sh
   ```

3. **Your existing SQLite data** remains in `experiments/walker_experiments.db` if you need it for reference.

### Optional: Export SQLite Data
If you want to export your existing SQLite data:
```bash
# Export existing data to CSV/JSON
python scripts/export_mlflow_report.py
```

## 🔍 Troubleshooting

### MLflow UI Not Loading
1. Check PostgreSQL is running:
   ```bash
   docker compose ps walker-postgres
   ```

2. Check MLflow server logs:
   ```bash
   docker compose logs walker-mlflow
   ```

3. Verify database connection:
   ```bash
   docker compose exec walker-postgres pg_isready -U walker_user -d walker_metrics
   ```

### Database Connection Issues
1. Restart PostgreSQL and MLflow:
   ```bash
   docker compose restart walker-postgres walker-mlflow
   ```

2. Check PostgreSQL logs:
   ```bash
   docker compose logs walker-postgres
   ```

3. Reinitialize database:
   ```bash
   docker compose run --rm walker-training-app python scripts/init_mlflow_postgres.py
   ```

### Port Conflicts
If port 5434 (PostgreSQL) or 5002 (MLflow) are in use:
1. Change ports in `docker-compose.yml`
2. Update connection strings accordingly

## 📊 Benefits of PostgreSQL

1. **Better Concurrency**: Multiple processes can access the database simultaneously
2. **No File Permission Issues**: No more SQLite file access problems in Docker
3. **Better Performance**: PostgreSQL is optimized for multi-user scenarios
4. **ACID Compliance**: Better data integrity and consistency
5. **Scalability**: Can handle larger datasets and more experiments

## 🔙 Rollback (If Needed)

To revert to SQLite (not recommended):
1. Change tracking URI back to SQLite in:
   - `docker-compose.yml`
   - `src/training_environment.py`
   - `src/evaluation/mlflow_integration.py`
2. Remove PostgreSQL dependency

## 📞 Support

If you encounter issues:
1. Check this migration guide
2. Review Docker logs: `docker compose logs`
3. Verify all services are running: `docker compose ps`
4. Try the troubleshooting steps above 