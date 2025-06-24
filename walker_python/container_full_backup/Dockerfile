# Use official PyTorch image with CUDA 12.1 support
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    pkg-config \
    swig \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional monitoring dependencies
RUN pip install --no-cache-dir \
    prometheus-client==0.19.0 \
    prometheus-flask-exporter==0.22.4 \
    psutil==5.9.6

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/evaluation_exports \
    /app/experiments \
    /app/mlruns \
    /app/data \
    /tmp/prometheus_metrics

# Set Python path and environment variables
ENV PYTHONPATH=/app
ENV PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_metrics
# Set CUDA device to GPU 1 by default
ENV CUDA_VISIBLE_DEVICES=1
# Enable CUDA memory fraction to prevent OOM
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Expose ports
EXPOSE 8080 2322

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/status || exit 1

# Start the application
CMD ["python", "train_robots_web_visual.py"] 