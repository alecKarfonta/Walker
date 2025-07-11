#!/usr/bin/env python3
"""
Web-based training visualization with actual physics world rendering.
Shows the real robots, arms, and physics simulation in the browser.
Enhanced with comprehensive evaluation framework.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import threading
import time
import json
import logging
import random
import math
from collections import deque
from typing import Dict, Any, List, Optional
from flask import Flask, jsonify, request
import numpy as np
import Box2D as b2
from src.agents.physical_parameters import PhysicalParameters
from src.population.enhanced_evolution import EnhancedEvolutionEngine, EvolutionConfig, TournamentSelection
from src.population.population_controller import PopulationController
from flask_socketio import SocketIO
from typing import List

# Import evaluation framework

from src.agents.robot_memory_pool import RobotMemoryPool
from src.evaluation.metrics_collector import MetricsCollector
from src.evaluation.dashboard_exporter import DashboardExporter

from src.training_environment import TrainingEnvironment

# Import ecosystem dynamics for enhanced visualization
from src.ecosystem_dynamics import EcosystemDynamics, EcosystemRole
from src.environment_challenges import EnvironmentalSystem

# Import survival Q-learning integration
# EcosystemInterface removed - was part of learning manager system
# Learning manager removed - agents handle their own learning

# Import elite robot management
from src.persistence import EliteManager, StorageManager

# Import realistic terrain generation
from src.terrain_generation import generate_robot_scale_terrain

# Import routes
from routes import register_routes

# WebGL is the only rendering mode - high performance rendering always enabled

# Configure logging - set debug level for Deep Q-Learning GPU training logs
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

# Suppress Flask logging for status endpoint
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# Create logger for this module
logger = logging.getLogger(__name__)


# Create Flask app and SocketIO instance
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Create training environment instance
env = TrainingEnvironment()

# Register all routes
register_routes(app, env)

def main():
    # Set a different port for the web server to avoid conflicts
    web_port = 8080
    
    # Start the training loop
    env.start()
    
    # Start the web server in a separate thread
    server_thread = threading.Thread(
        target=lambda: socketio.run(app, host='0.0.0.0', port=web_port, allow_unsafe_werkzeug=True),
        daemon=True
    )
    server_thread.start()
    
    print(f"✅ Web server started on http://localhost:{web_port}")
    print(f"🧬 Evolutionary training running indefinitely...")
    print(f"   🤖 Population: {len(env.agents)} diverse crawling robots")
    print(f"   🧬 Auto-evolution every {env.evolution_interval/60:.1f} minutes")
    print(f"   🌐 Web interface: http://localhost:{web_port}")
    print(f"   ⏹️  Press Ctrl+C to stop")
    
    # Keep the main thread alive indefinitely to allow background threads to run
    try:
        while True:
            time.sleep(5)  # Check every 5 seconds
            # Optional: Print periodic status (FIXED: Prevent infinite logging at multiples of 18000)
            if hasattr(env, 'step_count'):
                # Only log at exact intervals, not every time we're at a multiple
                if env.step_count > 0 and env.step_count % 18000 == 0:
                    if not hasattr(env, '_last_logged_step') or getattr(env, '_last_logged_step', 0) != env.step_count:
                        print(f"🔄 System running: Step {env.step_count}, Generation {env.evolution_engine.generation}")
                        setattr(env, '_last_logged_step', env.step_count)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down training environment...")
        env.stop()
        print("✅ Training stopped.")

if __name__ == "__main__":
    main()

# When the script exits, ensure the environment is stopped
import atexit
atexit.register(lambda: env.stop()) 