#!/usr/bin/env python3
"""
Initialize MLflow database schema in PostgreSQL.
This script creates the necessary MLflow tables in the PostgreSQL database.
"""

import os
import sys
import time
import psycopg2
from psycopg2 import sql
import mlflow
from mlflow.store.db.utils import create_sqlalchemy_engine_with_retry


def wait_for_postgres(connection_string, max_retries=30, delay=2):
    """Wait for PostgreSQL to be ready."""
    print("⏳ Waiting for PostgreSQL to be ready...")
    
    for attempt in range(max_retries):
        try:
            # Parse connection string
            # postgresql://walker_user:walker_secure_2024@walker-postgres:5432/walker_metrics
            parts = connection_string.replace('postgresql://', '').split('@')
            user_pass = parts[0].split(':')
            host_db = parts[1].split('/')
            host_port = host_db[0].split(':')
            
            user = user_pass[0]
            password = user_pass[1]
            host = host_port[0]
            port = int(host_port[1])
            database = host_db[1]
            
            # Test connection
            conn = psycopg2.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database
            )
            conn.close()
            print("✅ PostgreSQL is ready!")
            return True
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⏳ Attempt {attempt + 1}/{max_retries}: PostgreSQL not ready yet ({e})")
                time.sleep(delay)
            else:
                print(f"❌ Failed to connect to PostgreSQL after {max_retries} attempts: {e}")
                return False
    
    return False


def initialize_mlflow_database(tracking_uri):
    """Initialize MLflow database schema."""
    print(f"🔧 Initializing MLflow database schema...")
    print(f"   Tracking URI: {tracking_uri}")
    
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        
        # This will create the MLflow database schema if it doesn't exist
        mlflow.get_experiment_by_name("Default")
        
        # Create the walker experiment
        try:
            experiment = mlflow.get_experiment_by_name("walker_robot_training")
            if experiment is None:
                experiment_id = mlflow.create_experiment("walker_robot_training")
                print(f"✅ Created experiment 'walker_robot_training' with ID: {experiment_id}")
            else:
                print(f"✅ Experiment 'walker_robot_training' already exists with ID: {experiment.experiment_id}")
        except Exception as e:
            print(f"⚠️  Note: Could not create/verify experiment: {e}")
        
        print("✅ MLflow database schema initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize MLflow database: {e}")
        return False


def main():
    """Main initialization function."""
    print("🚀 MLflow PostgreSQL Database Initialization")
    print("=" * 50)
    
    # Get connection details from environment or use defaults
    tracking_uri = os.environ.get(
        'MLFLOW_TRACKING_URI', 
        'postgresql://walker_user:walker_secure_2024@walker-postgres:5432/walker_metrics'
    )
    
    # Wait for PostgreSQL to be ready
    if not wait_for_postgres(tracking_uri):
        print("❌ Could not connect to PostgreSQL. Exiting.")
        sys.exit(1)
    
    # Initialize MLflow database
    if not initialize_mlflow_database(tracking_uri):
        print("❌ Failed to initialize MLflow database. Exiting.")
        sys.exit(1)
    
    print("\n🎉 MLflow PostgreSQL initialization completed successfully!")
    print("   You can now start the MLflow UI and training application.")


if __name__ == "__main__":
    main() 