FROM ghcr.io/mlflow/mlflow:latest

# Install PostgreSQL adapter
RUN pip install psycopg2-binary

# Set working directory
WORKDIR /mlflow

# Create artifacts directory
RUN mkdir -p /mlflow/artifacts

# Expose MLflow port
EXPOSE 5000

# Default command (will be overridden by docker-compose)
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"] 