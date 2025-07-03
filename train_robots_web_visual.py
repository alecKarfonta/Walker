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
from flask import Flask, render_template_string, jsonify, request
import numpy as np
import Box2D as b2
from src.agents.crawling_agent import CrawlingAgent
from src.agents.physical_parameters import PhysicalParameters
from src.population.enhanced_evolution import EnhancedEvolutionEngine, EvolutionConfig, TournamentSelection
from src.population.population_controller import PopulationController
from flask_socketio import SocketIO
from typing import List

# Import evaluation framework
from src.evaluation.metrics_collector import MetricsCollector
from src.evaluation.dashboard_exporter import DashboardExporter

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

# Import WebGL renderer
from src.rendering.webgl_renderer import get_webgl_template

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

class TrainingEnvironment:
    """
    Enhanced training environment with evolutionary physical parameters.
    Manages physics simulation and evolution of diverse crawling robots.
    Enhanced with comprehensive evaluation framework.
    """
    def __init__(self, num_agents=30, enable_evaluation=False):  # PERFORMANCE FIX: Reduced from 60 to 30 to improve FPS
        self.num_agents = num_agents
        # CRITICAL FIX: Disable sleeping at the world level to prevent static bodies from going to sleep
        self.world = b2.b2World(gravity=(0, -9.8), doSleep=False)
        self.dt = 1.0 / 60.0  # 60 FPS

        # EXPANDED WORLD BOUNDS: 4x larger world for strategic food zone system
        self.world_bounds_x_min = -200.0  # Expanded from -60 to match ecosystem
        self.world_bounds_x_max = 200.0   # Expanded from 60 to match ecosystem  
        self.world_bounds_y = -20.0       # Keep same for fallen agents
        
        # Collision filtering setup
        self.GROUND_CATEGORY = 0x0001
        self.AGENT_CATEGORY = 0x0002
        self.OBSTACLE_CATEGORY = 0x0004  # NEW: Category for obstacles

        # Create the ground
        self._create_ground()
        
        # Initialize evaluation framework
        self.enable_evaluation = enable_evaluation
        self.metrics_collector = None
        self.dashboard_exporter = None
        self.mlflow_integration = None
        
        if enable_evaluation:
            try:
                # Initialize MLflow integration first
                from src.evaluation.mlflow_integration import MLflowIntegration
                self.mlflow_integration = MLflowIntegration(
                    tracking_uri="sqlite:///experiments/walker_experiments.db",
                    experiment_name="walker_robot_training"
                )
                
                self.metrics_collector = MetricsCollector(
                    enable_mlflow=True,
                    enable_file_export=True,
                    export_directory="evaluation_exports"
                )
                
                self.dashboard_exporter = DashboardExporter(
                    port=2322,
                    enable_api=True
                )
                
                print("✅ Evaluation framework initialized")
            except Exception as e:
                print(f"⚠️  Evaluation framework initialization failed: {e}")
                self.enable_evaluation = False
                self.mlflow_integration = None

        # Enhanced evolution configuration
        self.evolution_config = EvolutionConfig(
            population_size=num_agents,
            elite_size=max(2, num_agents // 10),  # 10% elite
            max_generations=100,
            tournament_size=3,
            crossover_rate=0.7,
            mutation_rate=0.15,
            clone_rate=0.1,
            immigration_rate=0.05,
            target_diversity=0.3,
            adaptive_mutation=True,
            use_speciation=True,
            use_hall_of_fame=True
        )

        # Initialize enhanced evolution engine
        self.evolution_engine = EnhancedEvolutionEngine(
            world=self.world,
            config=self.evolution_config,
            mlflow_integration=self.mlflow_integration,
            logger=None
        )

        # Initialize reward signal evaluation system BEFORE creating agents
        self.reward_signal_evaluator = None
        self.reward_signal_adapter = None
        try:
            from src.evaluation.reward_signal_integration import get_reward_signal_adapter
            self.reward_signal_adapter = get_reward_signal_adapter()
            print("📊 Reward signal evaluation system initialized (before agent creation)")
        except ImportError as e:
            print(f"⚠️ Reward signal evaluation not available: {e}")
        except Exception as e:
            print(f"⚠️ Reward signal evaluation initialization failed: {e}")

        # Create initial diverse population (agents can now use training env's adapter)
        self.agents = self.evolution_engine.initialize_population()

        # Add training environment reference to all newly created agents
        if self.reward_signal_adapter:
            for agent in self.agents:
                if not getattr(agent, '_destroyed', False):
                    agent._training_env = self
                    # Register agent with reward signal adapter
                    try:
                        agent_type = getattr(agent, 'learning_approach', 'evolutionary')
                        self.reward_signal_adapter.register_agent(
                            agent.id,
                            agent_type,
                            metadata={
                                'physical_params': str(agent.physical_params) if hasattr(agent, 'physical_params') else None,
                                'created_at': time.time()
                            }
                        )
                    except Exception as e:
                        print(f"⚠️ Failed to register agent {agent.id}: {e}")
            print(f"🔗 Added training environment reference to {len(self.agents)} agents")

        # Statistics and state
        self.step_count = 0
        self.robot_stats = {}
        self.population_stats = {
            'generation': 1,
            'best_distance': 0,
            'average_distance': 0,
            'diversity': 0,
            'total_agents': len(self.agents),
            'q_learning_stats': {
                'avg_epsilon': 0,
                'avg_learning_rate': 0,
                'total_q_updates': 0,
                'avg_q_value': 0
            }
        }
        self.is_running = False
        self.thread = None
        # MORPHOLOGY-AWARE EPISODE LENGTHS: Complex robots need more time to learn
        # Base episode length depends on robot complexity (number of joints to coordinate)
        self.base_episode_length = 3600  # 60 seconds for simple 2-joint robots
        self.episode_length_multipliers = {
            'simple': 1.0,    # 2-4 joints: 60 seconds (3600 steps)
            'medium': 5.0,    # 5-8 joints: 5 minutes (18000 steps) 
            'complex': 15.0,  # 9-12 joints: 15 minutes (54000 steps)
            'very_complex': 30.0  # 13+ joints: 30 minutes (108000 steps)
        }
        
        # Learning preservation settings
        self.preserve_learning_on_reset = True  # Use reset_position() instead of full reset()
        self.learning_progress_threshold = 0.1  # Only full reset if no learning progress
        
        # Backward compatibility: default episode length for simple robots
        self.episode_length = self.base_episode_length
        
        # Enhanced thread safety for Box2D operations
        import threading
        self._physics_lock = threading.RLock()  # Use RLock for re-entrant locking
        self._evolution_lock = threading.Lock()  # Separate lock for evolution state
        self._is_evolving = False  # Flag to prevent concurrent evolution
        self._agents_pending_destruction = []  # Safe destruction queue
        
        # LIMB DISTRIBUTION TRACKING: Monitor multi-limb robot survival (FIXED: Limited size)
        from collections import deque
        self.limb_distribution_history = deque(maxlen=100)  # FIXED: Limit to 100 entries
        self.last_limb_distribution_log = 0
        self.limb_distribution_interval = 10.0  # Log every 10 seconds
        self.agent_creation_log = {}  # Track when agents are created
        self.agent_death_log = {}    # Track when agents die
        
        # Statistics update timing
        self.stats_update_interval = 2.0
        self.last_stats_update = 0
        
        # Evolution timing with safety (FIXED: Disabled auto-evolution for learning)
        self.evolution_interval = 1800.0  # INCREASED: 30 minutes between generations (was 3 minutes)
        self.last_evolution_time = time.time()
        self.auto_evolution_enabled = False  # KEPT DISABLED: Let Q-learning work without interference
        self._evolution_requested = False  # Flag for requested evolution
        
        # Settle the world
        for _ in range(10):
            self.world.Step(self.dt, 8, 3)

        # Camera and focus system
        self.focused_agent = None
        self.camera_target = (0, 0)
        self.camera_position = (0, 0)
        self.camera_zoom = 1.0
        self.target_zoom = 1.0
        self.follow_speed = 0.05
        self.zoom_speed = 0.05
        
        # User zoom tracking
        self.user_zoom_level = 1.0
        self.user_has_manually_zoomed = False
        
        # Periodic learning system
        self.last_learning_time = time.time()
        self.learning_interval = 90.0  # 1.5 minutes between learning events
        self.learning_rate = 0.3

        # Enhanced visualization systems
        self.ecosystem_dynamics = EcosystemDynamics()
        self.environmental_system = EnvironmentalSystem()
        
        # Dynamic world generation system for expanding exploration
        try:
            from src.world.dynamic_world_manager import DynamicWorldManager
            self.dynamic_world_manager = DynamicWorldManager(
                box2d_world=self.world,
                ecosystem_dynamics=self.ecosystem_dynamics
            )
            print("🌍 Dynamic world generation system initialized")
        except Exception as e:
            print(f"⚠️ Dynamic world manager initialization failed: {e}")
            self.dynamic_world_manager = None
        self.agent_health = {}  # Track agent health/energy for visualization
        self.agent_statuses = {}  # Track agent statuses (hunting, feeding, etc.)
        self.predation_events = []  # Track recent predation events for visualization
        self.last_ecosystem_update = time.time()
        self.ecosystem_update_interval = 20.0  # Update ecosystem every 20 seconds for better responsiveness
        
        # Resource generation system - BALANCED FREQUENCY for survival
        self.last_resource_generation = time.time()
        self.resource_generation_interval = 60.0  # Generate strategic resources every 1 minute for better survival balance
        self.agent_energy_levels = {}  # Track agent energy levels for resource consumption
        
        # Death and survival system (FIXED: Limited size)
        self.death_events = deque(maxlen=50)  # FIXED: Limit to 50 death events for visualization
        self.agents_pending_replacement = []  # Queue for dead agents needing replacement
        
        # Food consumption animation system (FIXED: Limited size)
        self.consumption_events = deque(maxlen=100)  # FIXED: Limit to 100 consumption events for animation
        
        # Ray casting visualization removed for performance
        self.survival_stats = {
            'total_deaths': 0,
            'deaths_by_starvation': 0,
            'average_lifespan': 0,
            'lifespan': 0,  # Add missing lifespan key
            'agent_birth_times': {}  # Track when each agent was born/created
        }
        
        # Initialize ecosystem roles for existing agents
        self._initialize_ecosystem_roles()

        # Initialize realistic terrain generation system (replaces dynamic obstacle spawning)
        self.terrain_style = 'mixed'  # Default terrain style
        self.terrain_mesh = None  # Store generated terrain mesh
        self.terrain_collision_bodies = []  # Store terrain collision bodies
        self.obstacle_bodies = {}  # Track obstacle ID -> Box2D body mapping
        
        # 🌍 DISABLE static terrain generation - let dynamic world handle everything
        # Generate realistic terrain at startup
        # self._generate_realistic_terrain()  # DISABLED: Using dynamic world biome system instead

        # Initialize elite robot management system
        self.elite_manager = EliteManager(
            storage_directory="robot_storage",
            elite_per_generation=3,  # Preserve top 3 robots each generation
            max_elite_storage=150,   # Maximum 150 elite robots in storage
            min_fitness_threshold=1.0  # Only preserve robots with fitness > 1.0
        )
        
        # Flag to track if we should restore elites on startup
        self.restore_elites_on_start = True
        
        # Auto-save elite robots every generation
        self.auto_save_elites = True
        self.last_elite_save_generation = 0
        
        # Storage manager for periodic elite saves
        self.storage_manager = StorageManager("robot_storage")
        self.storage_manager.enable_auto_save(interval_seconds=600)  # Auto-save every 10 minutes

        # All agents now handle their own learning - no learning manager needed
        print("🧠 All agents using standalone attention deep Q-learning - no learning manager required")

        # ✨ INITIALIZE RANDOM LEARNING APPROACHES FOR ALL AGENTS (after learning_manager is initialized)
        self._initialize_random_learning_approaches()
        
        # Performance optimization tracking with AGGRESSIVE cleanup
        self.last_performance_cleanup = time.time()
        self.performance_cleanup_interval = 10.0  # MUCH more frequent - every 10 seconds
        
        # Attention network specific cleanup
        self.last_attention_cleanup = time.time()
        self.attention_cleanup_interval = 5.0  # MUCH more frequent - every 5 seconds
        
        # Web interface throttling
        self.last_web_interface_update = time.time()
        self.web_interface_update_interval = 0.05  # 20 FPS instead of 60 FPS
        self.web_data_cache = {}
        self.web_cache_valid = False
        
        # Health bar rendering toggle (PERFORMANCE OPTIMIZATION)
        self.show_health_bars = False  # Default: enabled, can be toggled off for performance
        
        # Simulation speed control
        self.simulation_speed_multiplier = 10.0  # 10x speed by default for faster training
        self.max_speed_multiplier = 500.0  # Maximum 500x speed for high-speed training
        
        # AI Processing optimization settings
        self.ai_optimization_enabled = True  # Enable AI processing optimizations
        self.ai_batch_percentage = 0.15  # PERFORMANCE FIX: Process only 15% of agents per frame (was 25%)
        self.ai_spatial_culling_enabled = False  # Only update AI for agents near camera
        self.ai_spatial_culling_distance = 50.0  # Distance from camera to update AI
        
        # Physics FPS tracking
        self.physics_fps_counter = 0
        self.physics_fps_start_time = time.time()
        self.last_physics_fps_update = time.time()
        self.current_physics_fps = 0
        
        # Background processing
        import threading
        self._background_processing_active = False
        self._background_lock = threading.Lock()
        
        # Emergency shutdown tracking (for performance monitoring)
        self._emergency_shutdown_state = {
            'max_consecutive_slow_frames': 10,
            'consecutive_slow_frames': 0,
            'max_physics_bodies': 1000,
            'max_memory_mb': 2000,
            'last_emergency_check': 0
        }
    
    @property
    def slow_frame_count(self):
        """Get current slow frame count for performance monitoring."""
        return self._emergency_shutdown_state.get('consecutive_slow_frames', 0)

        # Initialize Robot Memory Pool for efficient agent reuse with learning preservation
        try:
            from src.agents.robot_memory_pool import RobotMemoryPool
            self.robot_memory_pool = RobotMemoryPool(
                world=self.world,
                min_pool_size=max(5, num_agents // 4),  # 25% of population as minimum pool
                max_pool_size=num_agents * 2,  # 2x population as maximum pool
                category_bits=self.AGENT_CATEGORY,
                mask_bits=self.GROUND_CATEGORY | self.OBSTACLE_CATEGORY  # Collide with ground AND obstacles
            )
            
            # Agents handle their own learning - no learning manager needed
            
            print(f"🏊 Robot Memory Pool initialized: {self.robot_memory_pool.min_pool_size}-{self.robot_memory_pool.max_pool_size} robots")
        except Exception as e:
            print(f"⚠️ Robot Memory Pool initialization failed: {e}")
            self.robot_memory_pool = None

        # Q-learning evaluation system
        self.q_learning_evaluator = None
        self.q_learning_adapter = None
        try:
            from src.evaluation.q_learning_integration import create_evaluator_for_training_environment
            self.q_learning_evaluator = create_evaluator_for_training_environment(self)
            print("🧠 Q-Learning evaluation system initialized")
        except ImportError as e:
            print(f"⚠️ Q-learning evaluation not available: {e}")

        # Reward signal evaluation system already initialized earlier before agent creation

        print(f"🧬 Enhanced Training Environment initialized:")
        print(f"   Population: {len(self.agents)} diverse agents")
        print(f"   Evolution: {self.evolution_config.population_size} agents, {self.evolution_config.elite_size} elite")
        print(f"   Diversity target: {self.evolution_config.target_diversity}")
        print(f"   Auto-evolution every {self.evolution_interval}s")
        print(f"🌿 Ecosystem dynamics and visualization systems active")
        print(f"🏆 Elite preservation: {self.elite_manager.elite_per_generation} per generation, max {self.elite_manager.max_elite_storage} stored")
        print(f"🏞️ Realistic terrain generated: {len(self.terrain_collision_bodies)} terrain bodies using '{self.terrain_style}' style")
        
        # Load elite robots on startup if requested
        if self.restore_elites_on_start:
            self._load_elite_robots_on_startup()
        
        # Log the morphology-aware learning time improvements
        self.log_morphology_aware_learning_times()

    def _load_elite_robots_on_startup(self):
        """Load elite robots from storage and integrate them into the population on startup."""
        try:
            print("🔄 Loading elite robots from storage...")
            
            # Get elite statistics
            elite_stats = self.elite_manager.get_elite_statistics()
            total_elites = elite_stats.get('total_elites_stored', 0)
            
            if total_elites == 0:
                print("📂 No elite robots found in storage")
                return
            
            # Load top 10 elite robots
            elite_robots = self.elite_manager.restore_elite_robots(
                world=self.world,
                count=min(10, total_elites),
                min_generation=max(0, self.evolution_engine.generation - 5)  # Only recent elites
            )
            
            if not elite_robots:
                print("📂 No elite robots could be loaded")
                return
            
            # Replace some random agents with elite robots to maintain population size
            agents_to_replace = min(len(elite_robots), len(self.agents) // 3)  # Replace up to 1/3 
            
            if agents_to_replace > 0:
                # Remove random agents
                for _ in range(agents_to_replace):
                    if self.agents:
                        removed_agent = self.agents.pop(random.randint(0, len(self.agents) - 1))
                        self._safe_destroy_agent(removed_agent)
                
                # Add elite robots
                for i, elite_robot in enumerate(elite_robots[:agents_to_replace]):
                    self.agents.append(elite_robot)
                    # Initialize ecosystem data for elite robot
                    self._initialize_single_agent_ecosystem(elite_robot)
                
                print(f"🏆 Loaded {agents_to_replace} elite robots into population")
                print(f"   💫 Elite fitness range: {min(getattr(r, 'total_reward', 0) for r in elite_robots[:agents_to_replace]):.2f} - {max(getattr(r, 'total_reward', 0) for r in elite_robots[:agents_to_replace]):.2f}")
            
        except Exception as e:
            print(f"⚠️ Error loading elite robots on startup: {e}")

    def _save_elite_robots_periodically(self):
        """Save elite robots periodically (called from training loop)."""
        try:
            current_generation = self.evolution_engine.generation
            
            # Save elites every generation (but not the same generation twice)
            if (self.auto_save_elites and 
                current_generation > self.last_elite_save_generation):
                
                # Get top performing agents
                top_agents = sorted(
                    [a for a in self.agents if not getattr(a, '_destroyed', False)],
                    key=lambda a: getattr(a, 'total_reward', 0.0),
                    reverse=True
                )[:5]  # Top 5 agents
                
                if top_agents and max(getattr(a, 'total_reward', 0.0) for a in top_agents) > 1.0:
                    # Save using storage manager for periodic backup
                    self.storage_manager.save_elite_robots(top_agents, top_n=3)
                    self.last_elite_save_generation = current_generation
                    
                    print(f"💾 Auto-saved top 3 elite robots from generation {current_generation}")
            
            # Check for auto-save checkpoint
            self.storage_manager.check_auto_save(self.agents)
            
        except Exception as e:
            print(f"⚠️ Error in periodic elite saving: {e}")

    def get_morphology_aware_episode_length(self, agent) -> int:
        """Calculate episode length based on robot morphology complexity."""
        try:
            # Determine robot complexity based on joint count
            if hasattr(agent, 'physical_params'):
                total_joints = agent.physical_params.num_arms * agent.physical_params.segments_per_limb
            elif hasattr(agent, 'get_actual_joint_count'):
                total_joints = agent.get_actual_joint_count()
            else:
                total_joints = 2  # Default for basic robots
            
            # Classify complexity
            if total_joints <= 4:
                complexity = 'simple'
            elif total_joints <= 8:
                complexity = 'medium'
            elif total_joints <= 12:
                complexity = 'complex'
            else:
                complexity = 'very_complex'
            
            # Calculate episode length
            multiplier = self.episode_length_multipliers[complexity]
            episode_length = int(self.base_episode_length * multiplier)
            
            return episode_length
            
        except Exception as e:
            print(f"⚠️ Error calculating episode length for agent {getattr(agent, 'id', 'unknown')}: {e}")
            return self.base_episode_length
    
    def should_preserve_learning_on_reset(self, agent) -> bool:
        """Determine if agent's learning should be preserved during reset."""
        try:
            if not self.preserve_learning_on_reset:
                return False
            
            # Always preserve learning for complex robots
            if hasattr(agent, 'physical_params'):
                total_joints = agent.physical_params.num_arms * agent.physical_params.segments_per_limb
                if total_joints > 4:  # Medium+ complexity robots
                    return True
            
            # Check learning progress for simple robots
            if hasattr(agent, 'total_reward') and agent.total_reward > self.learning_progress_threshold:
                return True  # Robot is making progress, preserve learning
            
            # Check for recent reward improvements
            if (hasattr(agent, 'recent_displacements') and 
                agent.recent_displacements and 
                len(agent.recent_displacements) >= 3):
                recent_avg = sum(agent.recent_displacements[-3:]) / 3
                if recent_avg > 0.001:  # Recent positive movement
                    return True
            
            return False
            
        except Exception as e:
            print(f"⚠️ Error checking learning preservation for agent {getattr(agent, 'id', 'unknown')}: {e}")
            return True  # Default to preserving learning when in doubt

    def log_morphology_aware_learning_times(self):
        """Log the learning time improvements for different robot complexities."""
        try:
            complexity_stats = {
                'simple': {'count': 0, 'total_time': 0, 'avg_joints': 0},
                'medium': {'count': 0, 'total_time': 0, 'avg_joints': 0},
                'complex': {'count': 0, 'total_time': 0, 'avg_joints': 0},
                'very_complex': {'count': 0, 'total_time': 0, 'avg_joints': 0}
            }
            
            for agent in self.agents:
                if getattr(agent, '_destroyed', False):
                    continue
                    
                try:
                    # Get joint count
                    if hasattr(agent, 'physical_params'):
                        total_joints = agent.physical_params.num_arms * agent.physical_params.segments_per_limb
                    else:
                        total_joints = 2
                    
                    # Classify complexity
                    if total_joints <= 4:
                        complexity = 'simple'
                    elif total_joints <= 8:
                        complexity = 'medium'
                    elif total_joints <= 12:
                        complexity = 'complex'
                    else:
                        complexity = 'very_complex'
                    
                    # Get learning time
                    episode_length = self.get_morphology_aware_episode_length(agent)
                    learning_time_minutes = episode_length / (60 * 60)  # Convert steps to minutes
                    
                    # Update stats
                    complexity_stats[complexity]['count'] += 1
                    complexity_stats[complexity]['total_time'] += learning_time_minutes
                    complexity_stats[complexity]['avg_joints'] += total_joints
                    
                except Exception as e:
                    continue
            
            print(f"\n🧠 === MORPHOLOGY-AWARE LEARNING TIME REPORT ===")
            total_robots = sum(stats['count'] for stats in complexity_stats.values())
            print(f"📊 Population: {total_robots} robots with adaptive learning times")
            
            for complexity, stats in complexity_stats.items():
                if stats['count'] > 0:
                    avg_time = stats['total_time'] / stats['count']
                    avg_joints = stats['avg_joints'] / stats['count']
                    multiplier = self.episode_length_multipliers[complexity]
                    
                    print(f"   {complexity.upper():12} ({stats['count']:2} robots): "
                          f"{avg_joints:.1f} joints avg, "
                          f"{avg_time:.1f} min learning time "
                          f"({multiplier}x multiplier)")
            
            # Calculate total learning capacity improvement
            old_total_time = total_robots * (self.base_episode_length / (60 * 60))  # All robots at base time
            new_total_time = sum(stats['total_time'] for stats in complexity_stats.values())
            improvement_factor = new_total_time / old_total_time if old_total_time > 0 else 1.0
            
            print(f"📈 Learning capacity improvement: {improvement_factor:.1f}x total learning time")
            print(f"   Previous system: {old_total_time:.1f} total robot-minutes")
            print(f"   New system: {new_total_time:.1f} total robot-minutes")
            print(f"🎯 Complex robots now get up to 30x more learning time!")
            
        except Exception as e:
            print(f"❌ Error logging morphology-aware learning times: {e}")
 
    def _create_ground(self):
        """Creates a static ground body for the EXPANDED strategic food zone world."""
        ground_body = self.world.CreateStaticBody(position=(0, -1))
        
        # STRATEGIC FOOD ZONES: Ground must support 400m wide world (-200 to +200)
        # This matches the ecosystem_dynamics world bounds for strategic food placement
        ground_width = 450  # Wide enough for -200 to +200 world with margin
        
        print(f"🌍 Creating expanded ground for strategic food zones: {ground_width}m wide")
        
        # The ground's mask is set to collide with the agent category
        ground_fixture = ground_body.CreateFixture(
            shape=b2.b2PolygonShape(box=(ground_width, 1)),
            density=0.0,
            friction=0.9,
            filter=b2.b2Filter(
                categoryBits=self.GROUND_CATEGORY,
                maskBits=self.AGENT_CATEGORY  # Collide with all agents
            )
        )
        print(f"🔧 Expanded ground setup complete with width {ground_width}m for strategic food zones.")

    def _generate_realistic_terrain(self):
        """Generate robot-scale terrain with navigable features appropriate for 1.5m robots."""
        try:
            # Calculate world bounds based on robot scale and distribution  
            # Robot-scale terrain needs to match the ground width for consistency
            min_spacing = 12  # Must match evolution engine's min_spacing
            calculated_spawn_width = max(800, self.num_agents * min_spacing * 1.5)
            ground_width = calculated_spawn_width + 200  # Match ground width calculation
            world_bounds = (-ground_width // 2, 0, ground_width // 2, 30)  # Lower height for robots
            
            print(f"🤖 Generating robot-scale terrain with style '{self.terrain_style}'...")
            print(f"🌍 World bounds: {world_bounds}")
            
            # Generate robot-scale terrain using the new terrain generation system
            self.terrain_mesh, self.terrain_collision_bodies = generate_robot_scale_terrain(
                style=self.terrain_style,
                bounds=world_bounds,
                resolution=0.25  # High resolution (25cm) for robot-scale details
            )
            
            # Create physics bodies for all terrain features
            self._create_terrain_physics_bodies()
            
            # Clear any existing environmental system obstacles since we're using terrain generation
            if hasattr(self, 'environmental_system'):
                self.environmental_system.obstacles = []
            
            # Clear any evolution engine obstacles since we're using terrain generation
            if hasattr(self, 'evolution_engine') and hasattr(self.evolution_engine, 'environment_obstacles'):
                self.evolution_engine.environment_obstacles = []
            
            print(f"✅ Realistic terrain generation complete!")
            print(f"   🏔️ {len(self.terrain_collision_bodies)} terrain collision bodies")
            print(f"   🏗️  {len(self.obstacle_bodies)} physics bodies created")
            
            # Show terrain statistics
            if self.terrain_mesh:
                import numpy as np
                min_elevation = np.min(self.terrain_mesh.elevation)
                max_elevation = np.max(self.terrain_mesh.elevation)
                mean_elevation = np.mean(self.terrain_mesh.elevation)
                
                print(f"   📊 Terrain elevation:")
                print(f"      Range: {min_elevation:.1f}m to {max_elevation:.1f}m")
                print(f"      Average: {mean_elevation:.1f}m")
                print(f"      Resolution: {self.terrain_mesh.resolution}m per grid cell")
                
        except Exception as e:
            print(f"❌ Error generating realistic terrain: {e}")
            # Fallback to empty terrain
            self.terrain_collision_bodies = []
            self.terrain_mesh = None

    def _create_terrain_physics_bodies(self):
        """Create Box2D physics bodies for all terrain features."""
        try:
            bodies_created = 0
            
            for i, terrain_body_data in enumerate(self.terrain_collision_bodies):
                try:
                    # Create unique ID for each terrain body
                    terrain_id = f"terrain_{i}_{terrain_body_data['type']}"
                    
                    if terrain_id not in self.obstacle_bodies:
                        body = self._create_single_terrain_body({
                            'id': terrain_id,
                            'type': terrain_body_data['type'],
                            'position': terrain_body_data['position'],
                            'size': terrain_body_data['size'],
                            'height': terrain_body_data.get('height', terrain_body_data['size']),
                            'friction': terrain_body_data.get('friction', 0.7),
                            'properties': terrain_body_data.get('properties', {}),
                            'source': 'terrain_generation'
                        })
                        
                        if body:
                            self.obstacle_bodies[terrain_id] = body
                            bodies_created += 1
                            
                except Exception as e:
                    print(f"⚠️ Error creating physics body for terrain segment {i}: {e}")
            
            if bodies_created > 0:
                print(f"🏗️ Created {bodies_created} terrain physics bodies")
            
        except Exception as e:
            print(f"❌ Error creating terrain physics bodies: {e}")
    
    def _create_single_terrain_body(self, terrain_data):
        """Create a Box2D physics body for a single terrain segment."""
        try:
            position = terrain_data['position']
            size = terrain_data['size']
            height = terrain_data.get('height', size)
            friction = terrain_data.get('friction', 0.7)
            
            # Create static body for terrain
            terrain_body = self.world.CreateStaticBody(position=position)
            
            # Create terrain as a box (representing elevated ground)
            fixture = terrain_body.CreateFixture(
                shape=b2.b2PolygonShape(box=(size/2, height/2)),
                density=0.0,  # Static body
                friction=friction,
                restitution=0.1,  # Slight bounce for natural feel
                filter=b2.b2Filter(
                    categoryBits=self.OBSTACLE_CATEGORY,
                    maskBits=self.AGENT_CATEGORY  # ONLY collide with agents, NOT other obstacles (performance optimization)
                )
            )
            
            # Store terrain type on the body for identification
            terrain_body.userData = {
                'type': 'terrain',
                'terrain_type': terrain_data['type'],
                'terrain_id': terrain_data['id'],
                'elevation': height,
                'properties': terrain_data.get('properties', {})
            }
            
            return terrain_body
            
        except Exception as e:
            print(f"⚠️ Error creating terrain body: {e}")
            return None

    def change_terrain_style(self, new_style: str):
        """Change the terrain style and regenerate the terrain."""
        # Robot-scale terrain styles
        robot_terrain_styles = [
            'flat', 'gentle_hills', 'obstacle_course', 'slopes_and_ramps', 
            'rough_terrain', 'varied', 'mixed'
        ]
        
        if new_style not in robot_terrain_styles:
            print(f"⚠️ Unknown terrain style '{new_style}'. Available styles: {robot_terrain_styles}")
            return False
        
        print(f"🏞️ Changing terrain from '{self.terrain_style}' to '{new_style}'...")
        
        # Clean up existing terrain
        self._cleanup_all_terrain()
        
        # 🌍 DISABLE static terrain generation - let dynamic world handle everything
        # Update style and regenerate
        self.terrain_style = new_style
        # self._generate_realistic_terrain()  # DISABLED: Using dynamic world biome system instead
        
        print(f"✅ Terrain changed to '{new_style}' style")
        return True

    def _cleanup_all_terrain(self):
        """Remove all existing terrain physics bodies."""
        try:
            for terrain_id, body in self.obstacle_bodies.items():
                try:
                    if body:
                        self.world.DestroyBody(body)
                except Exception as e:
                    print(f"⚠️ Error destroying terrain body {terrain_id}: {e}")
            
            self.obstacle_bodies.clear()
            self.terrain_collision_bodies.clear()
            self.terrain_mesh = None
            
            print(f"🧹 Cleaned up all existing terrain")
            
        except Exception as e:
            print(f"❌ Error cleaning up terrain: {e}")

    def _initialize_ecosystem_roles(self):
        """Initialize ecosystem roles for all agents based on their characteristics."""
        for agent in self.agents:
            if not getattr(agent, '_destroyed', False):
                # Extract and properly normalize fitness traits from agent's physical parameters
                motor_speed = getattr(agent.physical_params, 'motor_speed', 5.0)
                motor_torque = getattr(agent.physical_params, 'motor_torque', 50.0)
                learning_rate = getattr(agent.physical_params, 'learning_rate', 0.1)
                
                fitness_traits = {
                    'speed': min(1.0, motor_speed / 15.0),  # Normalize motor_speed (typically 3-12) to 0-1
                    'strength': min(1.0, motor_torque / 200.0),  # Normalize motor_torque (typically 30-180) to 0-1  
                    'cooperation': min(1.0, learning_rate * 50.0)  # Boost learning_rate (typically 0.005-0.02) to meaningful range
                }
                
                # Assign ecosystem role
                role = self.ecosystem_dynamics.assign_ecosystem_role(agent.id, fitness_traits)
                
                # Initialize agent status tracking
                self.agent_statuses[agent.id] = {
                    'role': role.value,
                    'status': 'idle',  # idle, hunting, feeding, eating, fleeing, territorial
                    'last_status_change': time.time(),
                    'energy': 1.0,  # 0.0 to 1.0
                    'speed_factor': 1.0,
                                # Alliances and territories removed
                }
                
                self.agent_health[agent.id] = {
                    'health': 1.0,  # 0.0 to 1.0
                    'energy': 1.0,  # 0.0 to 1.0
                    'last_updated': time.time()
                }
                
                # Initialize energy level for resource consumption
                self.agent_energy_levels[agent.id] = 1.0
                
                # Track birth time for survival statistics
                self.survival_stats['agent_birth_times'][agent.id] = time.time()
        
        print(f"🦎 Initialized ecosystem roles for {len(self.agents)} agents")

    def _initialize_random_learning_approaches(self):
        """Initialize learning approaches for all agents (simplified - no learning manager needed)."""
        successful_assignments = 0
        
        print(f"🎯 All agents use standalone attention deep Q-learning...")
        
        for i, agent in enumerate(self.agents):
            if getattr(agent, '_destroyed', False):
                continue
                
            try:
                # All agents use attention deep Q-learning (handled in their own __init__)
                setattr(agent, 'learning_approach', 'attention_deep_q_learning')
                successful_assignments += 1
                
                # Log first few assignments for verification
                if i < 5:
                    print(f"   Agent {str(agent.id)[:8]}: standalone attention_deep_q_learning")
                    
            except Exception as e:
                print(f"   ❌ Error setting up agent {agent.id}: {e}")
        
        print(f"🧠 Learning Approach Distribution:")
        print(f"   🧠 Standalone Attention Deep Q-Learning: {successful_assignments} agents (100%)")
        print(f"✅ All agents handle their own learning - no external dependencies")

    def _assign_random_learning_approach_single(self, agent):
        """Assign learning approach to a single agent (for replacement agents)."""
        try:
            # All agents use standalone attention deep Q-learning
            setattr(agent, 'learning_approach', 'attention_deep_q_learning')
            print(f"   🎯 Replacement agent {str(agent.id)[:8]} using standalone attention deep Q-learning")
                
        except Exception as e:
            print(f"   ❌ Error setting up replacement agent {agent.id}: {e}")

    def _update_ecosystem_dynamics(self):
        """Update ecosystem dynamics including agent interactions and predation."""
        try:
            current_time = time.time()
            
            # Update ecosystem state
            self.ecosystem_dynamics.update_ecosystem(
                generation=self.evolution_engine.generation,
                population_size=len([a for a in self.agents if not getattr(a, '_destroyed', False)])
            )
            
            # Update environmental challenges
            self.environmental_system.update_environment(self.evolution_engine.generation)
            
            # Create physics bodies for any new obstacles
            self._create_obstacle_physics_bodies()
            
            # Update agent health and energy based on ecosystem effects
            for agent in self.agents:
                if getattr(agent, '_destroyed', False) or not agent.body:
                    continue
                    
                agent_id = agent.id
                position = (agent.body.position.x, agent.body.position.y)
                
                # Get ecosystem effects for this agent
                ecosystem_effects = self.ecosystem_dynamics.get_ecosystem_effects(agent_id, position)
                environmental_effects = self.environmental_system.get_effects(position)
                
                # Update agent health/energy
                if agent_id in self.agent_health:
                    health_data = self.agent_health[agent_id]
                    
                    # RESTORED: Normal energy drain since we're reusing networks
                    energy_drain = 0.005 + environmental_effects.get('energy_cost', 0.0)  # Moderate rate
                    health_data['energy'] = max(0.0, health_data['energy'] - energy_drain)
                    
                    # Resource access affects energy recovery
                    resource_access = ecosystem_effects.get('resource_access', 1.0)
                    if resource_access > 1.0:
                        health_data['energy'] = min(1.0, health_data['energy'] + 0.005)
                    
                    # Health is affected by territory bonuses and competition
                    territory_bonus = ecosystem_effects.get('territory_bonus', 0.0)
                    competition_penalty = ecosystem_effects.get('competition_penalty', 0.0)
                    
                    health_change = (territory_bonus - competition_penalty) * 0.01
                    health_data['health'] = max(0.0, min(1.0, health_data['health'] + health_change))
                    
                    health_data['last_updated'] = current_time
            
            # Update agent statuses based on their behaviors and interactions
            self._update_agent_statuses()
            
            
            # Clean up old predation events (keep only last 10 seconds)
            self.predation_events = [
                event for event in self.predation_events 
                if current_time - event['timestamp'] < 10.0
            ]
            
            # Clean up old death events (keep only last 5 seconds - shorter since animations are now brief)
            self.death_events = [
                event for event in self.death_events 
                if current_time - event['timestamp'] < 5.0
            ]
            
            # Clean up old consumption events (keep only active animations)
            self.consumption_events = [
                event for event in self.consumption_events
                if current_time - event['timestamp'] < event['duration']
            ]
            
            # Ray casting events cleanup removed for performance
            
            # Update dynamic world generation based on robot positions
            if self.dynamic_world_manager:
                try:
                    # PERFORMANCE MONITOR: Track dynamic world update time
                    world_update_start = time.time()
                    
                    # Get robot positions for world expansion
                    robot_positions = []
                    for agent in self.agents:
                        if not getattr(agent, '_destroyed', False) and agent.body:
                            robot_positions.append((agent.id, (agent.body.position.x, agent.body.position.y)))
                    
                    # SAFEGUARD: Skip update if too many robots (performance protection)
                    if len(robot_positions) > 100:
                        print(f"⚠️ PERFORMANCE: Skipping dynamic world update ({len(robot_positions)} robots)")
                        return
                    
                    # Update dynamic world (generates new tiles as robots progress)
                    self.dynamic_world_manager.update(robot_positions)
                    
                    # PERFORMANCE MONITOR: Warn if update takes too long
                    world_update_time = time.time() - world_update_start
                    if world_update_time > 0.5:  # More than 500ms is problematic
                        print(f"🐌 SLOW: Dynamic world update took {world_update_time:.2f}s")
                        
                        # Get world status for debugging
                        status = self.dynamic_world_manager.get_world_status()
                        print(f"   📊 World status: {status.get('active_tiles', 0)} tiles, {status.get('total_obstacles', 0)} obstacles")
                    
                except Exception as e:
                    print(f"❌ Error updating dynamic world: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # EMERGENCY SAFEGUARD: Disable dynamic world if it keeps failing
                    if not hasattr(self, '_dynamic_world_errors'):
                        self._dynamic_world_errors = 0
                    self._dynamic_world_errors += 1
                    
                    if self._dynamic_world_errors >= 5:
                        print(f"🚨 EMERGENCY: Disabling dynamic world after {self._dynamic_world_errors} consecutive errors")
                        self.dynamic_world_manager = None
            
        except Exception as e:
            print(f"⚠️ Error updating ecosystem dynamics: {e}")
    
    def _update_agent_statuses(self):
        """Update agent statuses - SIMPLIFIED to only hunting and eating."""
        current_time = time.time()
        
        for agent in self.agents:
            if getattr(agent, '_destroyed', False) or not agent.body:
                continue
                
            agent_id = agent.id
            if agent_id not in self.agent_statuses:
                continue
                
            status_data = self.agent_statuses[agent_id]
            
            # Update speed factor based on velocity
            velocity = agent.body.linearVelocity
            speed = (velocity.x ** 2 + velocity.y ** 2) ** 0.5
            status_data['speed_factor'] = min(2.0, speed / 2.0)  # Normalize and cap at 2x
            
            # Check if agent is currently eating and if enough time has passed
            if status_data['status'] == 'eating':
                # Keep eating status for 3 seconds after consumption
                if current_time - status_data['last_status_change'] < 3.0:
                    continue  # Keep eating status
                # Otherwise, switch back to hunting
                status_data['status'] = 'hunting'
                status_data['last_status_change'] = current_time
            
            # SIMPLIFIED: If not eating, then always hunting
            elif status_data['status'] != 'hunting':
                status_data['status'] = 'hunting'
                status_data['last_status_change'] = current_time
    
    def _generate_resources_between_agents(self):
        """Generate resources strategically between agents."""
        try:
            # Get positions of all active agents
            agent_positions = []
            for agent in self.agents:
                if not getattr(agent, '_destroyed', False) and agent.body:
                    agent_positions.append((agent.id, (agent.body.position.x, agent.body.position.y)))
            
            if len(agent_positions) >= 2:
                # Sort agents by x position for systematic resource placement
                agent_positions.sort(key=lambda x: x[1][0])
                
                # Generate resources between agents using ecosystem dynamics with enhanced parameters
                self.ecosystem_dynamics.generate_resources_between_agents(agent_positions)
                
                # Post-process resources to ensure they're consumable (within 3.0m of agents)
                self._validate_resource_positions(agent_positions)
                
                # Only log occasionally to reduce spam
                if len(self.ecosystem_dynamics.food_sources) % 10 == 0:  # Log every 10th resource milestone
                    print(f"🌱 Resource generation cycle completed. Total resources: {len(self.ecosystem_dynamics.food_sources)}")
        
        except Exception as e:
            print(f"⚠️ Error generating resources: {e}")
    
    def _validate_resource_positions(self, agent_positions):
        """Validate that resources maintain proper distance from agents and are at reachable heights."""
        try:
            minimum_safe_distance = 6.0  # Must be at least 6m from any agent (matches ecosystem_dynamics)
            minimum_height = 0.0   # FIXED: Minimum Y coordinate (ground at Y=-1, so food at Y=0+)
            maximum_height = 8.0   # Maximum Y coordinate robots can reasonably reach
            resources_to_remove = []
            
            for food_source in self.ecosystem_dynamics.food_sources:
                food_pos = food_source.position
                should_remove = False
                removal_reason = ""
                
                # Check if this resource is too close to ANY agent
                too_close_to_agent = False
                for agent_id, agent_pos in agent_positions:
                    distance = ((food_pos[0] - agent_pos[0])**2 + (food_pos[1] - agent_pos[1])**2)**0.5
                    if distance < minimum_safe_distance:
                        too_close_to_agent = True
                        break
                
                if too_close_to_agent:
                    should_remove = True
                    removal_reason = f"too close to agents (<{minimum_safe_distance}m)"
                
                # NEW: Check if resource is at unreachable height
                if food_pos[1] < minimum_height:
                    should_remove = True
                    removal_reason = f"too low (Y={food_pos[1]:.1f} < {minimum_height})"
                elif food_pos[1] > maximum_height:
                    should_remove = True
                    removal_reason = f"too high (Y={food_pos[1]:.1f} > {maximum_height})"
                
                # CRITICAL: Remove resources that are invalid instead of moving them
                # This prevents random rewards and ensures fair competition
                if should_remove:
                    resources_to_remove.append((food_source, removal_reason))
            
            # Remove invalid resources and log reasons
            if resources_to_remove:
                height_issues = 0
                distance_issues = 0
                
                for food_source, reason in resources_to_remove:
                    self.ecosystem_dynamics.food_sources.remove(food_source)
                    if "too high" in reason or "too low" in reason:
                        height_issues += 1
                    else:
                        distance_issues += 1
                
                print(f"🚫 Removed {len(resources_to_remove)} invalid food sources:")
                if height_issues > 0:
                    print(f"   📏 {height_issues} resources at unreachable heights (must be {minimum_height}m to {maximum_height}m)")
                if distance_issues > 0:
                    print(f"   📍 {distance_issues} resources too close to agents (<{minimum_safe_distance}m)")
                print(f"   ✅ This ensures all resources are reachable and fairly positioned")
                
        except Exception as e:
            print(f"⚠️ Error validating resource positions: {e}")
    
    def _update_resource_consumption(self):
        """Update agent energy levels through resource consumption and handle death."""
        try:
            agents_to_replace = []  # Track agents that died from starvation
            
            for agent in self.agents:
                if getattr(agent, '_destroyed', False) or not agent.body:
                    continue
                
                agent_id = agent.id
                agent_position = (agent.body.position.x, agent.body.position.y)
                
                # Get current energy level
                current_energy = self.agent_energy_levels.get(agent_id, 1.0)
                
                # Try to consume nearby resources FIRST (before energy decay)
                energy_gain, consumed_food_type, consumed_food_position = self.ecosystem_dynamics.consume_resource(agent_id, agent_position)
                # Energy gain is now properly scaled in the consume_resource function
                
                # Apply consistent energy decay (not paused during eating) - DRASTICALLY REDUCED for much better survival
                base_decay = 0.000005  # Reduced from 0.00005 to 0.000005 (90% reduction - 10x longer survival)
                
                # Minimal additional decay based on movement 
                velocity = agent.body.linearVelocity
                speed = (velocity.x ** 2 + velocity.y ** 2) ** 0.5
                movement_cost = speed * 0.00005  # Reduced movement cost by 90%
                
                # Role-based energy costs (minimal differences)
                role = self.agent_statuses.get(agent_id, {}).get('role', 'omnivore')
                role_multipliers = {
                    'carnivore': 1.02,  # Slightly more energy needed for hunting
                    'herbivore': 0.98,  # Slightly more efficient
                    'omnivore': 1.0,    # Balanced
                    'scavenger': 0.99,  # Slightly efficient
                    'symbiont': 0.97    # More efficient in groups
                }
                role_multiplier = role_multipliers.get(role, 1.0)
                
                total_decay = (base_decay + movement_cost) * role_multiplier
                current_energy = max(0.0, current_energy - total_decay)
                
                # Health management based on energy levels
                if agent_id in self.agent_health:
                    current_health = self.agent_health[agent_id]['health']
                    
                    if current_energy > 0.7:
                        # HIGH ENERGY: Recover health gradually
                        health_recovery = 0.001  # Slow health recovery when well-fed
                        self.agent_health[agent_id]['health'] = min(1.0, current_health + health_recovery)
                        if current_health < 0.9 and current_health + health_recovery >= 0.9:
                            print(f"💚 {str(agent_id)[:8]} is recovering health (energy: {current_energy:.2f})")
                    
                    elif current_energy < 0.1:
                        # DRASTICALLY REDUCED: Much slower health degradation for 10x longer survival
                        health_degradation = 0.0001  # Reduced from 0.001 to 0.0001 (90% reduction)
                        self.agent_health[agent_id]['health'] = max(0.0, current_health - health_degradation)
                        if current_health - health_degradation <= 0.0:
                            print(f"💀 {str(agent_id)[:8]} is starving to death (health: {current_health:.3f})")
                
                # Apply energy gain from eating
                current_energy = min(1.0, current_energy + energy_gain)
                
                                            # IMPROVED: Update status when consuming food and create animation
                if energy_gain > 0 and agent_id in self.agent_statuses:
                    self.agent_statuses[agent_id]['status'] = 'eating'
                    self.agent_statuses[agent_id]['last_status_change'] = time.time()
                    
                    # STORE EATING TARGET for robot details display
                    self.agent_statuses[agent_id]['eating_target'] = {
                        'type': 'environmental',
                        'food_type': consumed_food_type,
                        'position': consumed_food_position,
                        'energy_gained': energy_gain,
                        'timestamp': time.time()
                    }
                    
                    # CLEAR ROLE-BASED CONSUMPTION MESSAGES
                    role_food_descriptions = {
                        'herbivore': {'plants': 'grazing on plants', 'seeds': 'eating seeds', 'insects': 'reluctantly eating insects', 'meat': 'eating meat'},
                        'carnivore': {},  # CARNIVORES CANNOT EAT ANY ENVIRONMENTAL FOOD - this should never be reached
                        'omnivore': {'plants': 'foraging plants', 'seeds': 'gathering seeds', 'insects': 'catching insects', 'meat': 'eating meat'},
                        'scavenger': {},  # SCAVENGERS CANNOT EAT ANY ENVIRONMENTAL FOOD - this should never be reached
                        'symbiont': {'plants': 'symbiotically feeding on plants', 'seeds': 'collecting seeds', 'insects': 'eating insects', 'meat': 'eating meat'}
                    }
                    
                    role = self.agent_statuses.get(agent_id, {}).get('role', 'omnivore')
                    food_description = role_food_descriptions.get(role, {}).get(consumed_food_type, f'eating {consumed_food_type}')
                    
                    print(f"🍴 {str(agent_id)[:8]} is {food_description} (energy: +{energy_gain:.2f}) [decay paused]")
                    
                    # FOOD ANIMATION: Create consumption event for visual line animation
                    if consumed_food_position:
                        consumption_event = {
                            'agent_id': agent_id,
                            'agent_position': [agent_position[0], agent_position[1]],
                            'food_position': [consumed_food_position[0], consumed_food_position[1]],
                            'food_type': f'{consumed_food_type}_{role}',  # Include role for clear visual distinction
                            'energy_gained': energy_gain,
                            'timestamp': time.time(),
                            'duration': 2.0  # Animation duration in seconds
                        }
                        self.consumption_events.append(consumption_event)
                
                # Track food consumption for leaderboard
                if energy_gain > 0 and agent_id in self.robot_stats:
                    if 'food_consumed' not in self.robot_stats[agent_id]:
                        self.robot_stats[agent_id]['food_consumed'] = 0.0
                    self.robot_stats[agent_id]['food_consumed'] += energy_gain
                
                # ROBOT CONSUMPTION: Use the fixed consume_robot function from ecosystem dynamics
                role = self.agent_statuses.get(agent_id, {}).get('role', 'omnivore')
                if role in ['carnivore', 'omnivore', 'scavenger'] and current_energy < 0.8:  # Hunt when hungry
                    robot_energy_gain, consumed_robot_id, consumed_robot_position = self.ecosystem_dynamics.consume_robot(
                        agent_id, agent_position, self.agents, self.agent_energy_levels, self.agent_health
                    )
                    
                    if robot_energy_gain > 0:
                        # Apply energy gain
                        current_energy = min(1.0, current_energy + robot_energy_gain)
                        
                        # Update predator status to show they are eating robots
                        if agent_id in self.agent_statuses:
                            self.agent_statuses[agent_id]['status'] = 'eating'
                            self.agent_statuses[agent_id]['last_status_change'] = time.time()
                            
                            # Store what they're eating for robot details
                            self.agent_statuses[agent_id]['eating_target'] = {
                                'type': 'robot',
                                'target_id': consumed_robot_id,
                                'energy_gained': robot_energy_gain,
                                'timestamp': time.time()
                            }
                            
                            # FOOD ANIMATION: Create consumption event for robot feeding
                            if consumed_robot_position:
                                consumption_event = {
                                    'agent_id': agent_id,
                                    'agent_position': [agent_position[0], agent_position[1]],
                                    'food_position': [consumed_robot_position[0], consumed_robot_position[1]],
                                    'food_type': f'robot_{role}',  # Special type showing predator role
                                    'energy_gained': robot_energy_gain,
                                    'timestamp': time.time(),
                                    'duration': 2.0  # Animation duration in seconds
                                }
                                self.consumption_events.append(consumption_event)
                        
                        # Track robot consumption for leaderboard
                        if agent_id in self.robot_stats:
                            if 'food_consumed' not in self.robot_stats[agent_id]:
                                self.robot_stats[agent_id]['food_consumed'] = 0.0
                            self.robot_stats[agent_id]['food_consumed'] += robot_energy_gain
                
                # Update energy level
                self.agent_energy_levels[agent_id] = current_energy
                
                # Check for death by health loss ONLY - robots should only die when health reaches 0.0
                current_health = self.agent_health.get(agent_id, {'health': 1.0})['health']
                # FIXED: Allow death but reuse networks on replacement
                if current_health <= 0.0:
                    # Only process death once per agent
                    if agent not in agents_to_replace:
                        cause = 'predation' if current_energy > 0.1 else 'starvation'
                        print(f"💀 Agent {agent_id} died from {cause}! (Role: {role}, Health: {current_health:.2f}, Energy: {current_energy:.2f})")
                        agents_to_replace.append(agent)
                        self._record_death_event(agent, cause)
                        # Mark agent as destroyed to prevent further processing
                        setattr(agent, '_destroyed', True)
                    continue
                
                # Update agent health data for visualization
                if agent_id in self.agent_health:
                    self.agent_health[agent_id]['energy'] = current_energy
                
                # Update agent status data for visualization
                if agent_id in self.agent_statuses:
                    self.agent_statuses[agent_id]['energy'] = current_energy
                    
                    # SIMPLIFIED: Don't override eating status - let it persist from consumption
                    current_status = self.agent_statuses[agent_id]['status']
                    current_health = self.agent_health.get(agent_id, {'health': 1.0})['health']
                    
                    # Only update status if not currently eating (preserves eating status)
                    if current_status != 'eating':
                        if current_health < 0.1:  # Critical health triggers dying
                            self.agent_statuses[agent_id]['status'] = 'dying'  # Critical health
                        elif current_energy < 0.15:  # Very low energy = immobilized
                            self.agent_statuses[agent_id]['status'] = 'immobilized'  # Can't move
                        elif current_energy < 0.4:  # Low energy = seeking food
                            self.agent_statuses[agent_id]['status'] = 'hunting'  # Seeking food
                        else:
                            self.agent_statuses[agent_id]['status'] = 'hunting'  # Always hunting when not eating
            
            # FIXED: Replace dead agents while reusing their neural networks
            if agents_to_replace:
                self._replace_dead_agents(agents_to_replace)
        
        except Exception as e:
            print(f"⚠️ Error updating resource consumption: {e}")
            import traceback
            traceback.print_exc()
    
    def _record_death_event(self, agent, cause='starvation'):
        """Record a death event for visualization and statistics."""
        try:
            current_time = time.time()
            agent_position = (agent.body.position.x, agent.body.position.y) if agent.body else (0, 0)
            
            # Calculate lifespan
            birth_time = self.survival_stats['agent_birth_times'].get(agent.id, current_time)
            lifespan = current_time - birth_time
            
            # Record death event for visualization
            self.death_events.append({
                'agent_id': agent.id,
                'position': agent_position,
                'timestamp': current_time,
                'cause': cause,
                'lifespan': lifespan,
                'role': self.agent_statuses.get(agent.id, {}).get('role', 'unknown')
            })
            
            # Update survival statistics
            self.survival_stats['total_deaths'] += 1
            if cause == 'starvation':
                self.survival_stats['deaths_by_starvation'] += 1
            
                            # Calculate average lifespan (only for events that have lifespan data)
            if self.survival_stats['total_deaths'] > 0:
                # FIXED: Convert deque to list before slicing
                events_with_lifespan = [event for event in list(self.death_events) if 'lifespan' in event]
                if events_with_lifespan:
                    total_lifespan = sum(event['lifespan'] for event in events_with_lifespan)
                    self.survival_stats['average_lifespan'] = total_lifespan / len(events_with_lifespan)
                
                # Clean up old tracking data
                if agent.id in self.survival_stats['agent_birth_times']:
                    del self.survival_stats['agent_birth_times'][agent.id]
                if agent.id in self.agent_energy_levels:
                    del self.agent_energy_levels[agent.id]
                if agent.id in self.agent_health:
                    del self.agent_health[agent.id]
                if agent.id in self.agent_statuses:
                    del self.agent_statuses[agent.id]
        
        except Exception as e:
            print(f"🔍 DEBUG: Error in _record_death_event: {e}")
            import traceback
            traceback.print_exc()
    
    def _replace_dead_agents(self, dead_agents):
        """Replace dead agents while transferring their neural networks to prevent network explosion."""
        with self._physics_lock:
            try:
                for dead_agent in dead_agents:
                    # Agents handle their own networks - no extraction needed
                    
                    # CRITICAL: Return dead agent to memory pool IMMEDIATELY so it can be reused
                    if self.robot_memory_pool:
                        try:
                            self.robot_memory_pool.return_robot(dead_agent)
                            print(f"♻️ Returned dead agent {dead_agent.id} to memory pool immediately")
                        except Exception as e:
                            print(f"⚠️ Failed to return dead agent {dead_agent.id} to memory pool: {e}")
                            # Fallback: add to destruction queue
                            self._agents_pending_destruction.append(dead_agent)
                    else:
                        # No memory pool available, use destruction queue
                        self._agents_pending_destruction.append(dead_agent)
                    
                    # Remove from active agents list
                    if dead_agent in self.agents:
                        self.agents.remove(dead_agent)
                    
                    # Create replacement agent
                    replacement_agent = self._create_replacement_agent()
                    if replacement_agent:
                        self.agents.append(replacement_agent)
                        
                        # Initialize new agent's ecosystem data
                        self._initialize_single_agent_ecosystem(replacement_agent)
                        
                        # Agents create their own networks - no transfer needed
                        self._assign_random_learning_approach_single(replacement_agent)
                        
                        print(f"🐣 Spawned replacement agent {replacement_agent.id} for dead agent {dead_agent.id}")
                
            except Exception as e:
                print(f"❌ Error replacing dead agents: {e}")
    
    def _get_current_spawn_position(self):
        """Get the current spawn position near the leftmost (beginning) boundary of the world."""
        try:
            # Check if we have a dynamic world manager to track current boundaries
            if hasattr(self, 'dynamic_world_manager') and self.dynamic_world_manager:
                # Get current leftmost boundary from dynamic world
                leftmost_boundary = self.dynamic_world_manager.left_wall_position
                
                # Spawn robots a safe distance from the left wall (10-30m to the right)
                spawn_x = leftmost_boundary + random.uniform(10.0, 30.0)
                spawn_y = random.uniform(4.0, 8.0)  # Slight height variation
                
                print(f"🌍 Dynamic world spawn: left wall at {leftmost_boundary:.1f}m, spawning at x={spawn_x:.1f}m")
                return (spawn_x, spawn_y)
            else:
                # Fallback: Use traditional central spawn area if no dynamic world
                spawn_x = random.uniform(-30.0, 30.0)
                spawn_y = random.uniform(4.0, 8.0)
                
                print(f"🏠 Static world spawn: spawning at x={spawn_x:.1f}m (no dynamic world)")
                return (spawn_x, spawn_y)
                
        except Exception as e:
            print(f"⚠️ Error getting spawn position: {e}, using fallback")
            # Safe fallback position
            return (0.0, 5.0)
    
    def _create_replacement_agent(self):
        """Create a new agent to replace a dead one using memory pool if available."""
        try:
            # 🌍 DYNAMIC WORLD: Spawn near current leftmost (beginning) position
            spawn_position = self._get_current_spawn_position()
            
            print(f"🐣 Respawning dead agent at current world beginning: ({spawn_position[0]:.1f}, {spawn_position[1]:.1f})")
            
            # Create random physical parameters (fresh genetics)
            from src.agents.physical_parameters import PhysicalParameters
            random_params = PhysicalParameters.random_parameters()
            
            # Use memory pool if available for efficient reuse with size mutations
            if self.robot_memory_pool:
                new_agent = self.robot_memory_pool.acquire_robot(
                    position=spawn_position,
                    physical_params=random_params,
                    apply_size_mutations=True  # Enable size mutations during respawning
                )
                logger.debug(f"♻️ Acquired replacement agent {new_agent.id} from memory pool with size mutations")
            else:
                # Fallback: Create new agent directly
                new_agent = CrawlingAgent(
                    world=self.world,
                    agent_id=None,  # Generate new UUID automatically
                    position=spawn_position,
                    category_bits=self.AGENT_CATEGORY,
                    mask_bits=self.GROUND_CATEGORY | self.OBSTACLE_CATEGORY,  # Collide with ground AND obstacles
                    physical_params=random_params
                )
                logger.warning(f"🆕 Created new replacement agent {new_agent.id} (no memory pool)")
            
            # Add training environment reference and register with reward signal adapter
            if hasattr(self, 'reward_signal_adapter') and self.reward_signal_adapter:
                try:
                    # Add training environment reference to agent
                    new_agent._training_env = self
                    
                    agent_type = getattr(new_agent, 'learning_approach', 'evolutionary')
                    self.reward_signal_adapter.register_agent(
                        new_agent.id,
                        agent_type,
                        metadata={
                            'physical_params': str(new_agent.physical_params) if hasattr(new_agent, 'physical_params') else None,
                            'created_at': time.time(),
                            'source': 'replacement'
                        }
                    )
                except Exception as e:
                    print(f"⚠️ Failed to register replacement agent {new_agent.id} with reward signal adapter: {e}")
            
            # Agents are standalone - no training environment injection needed
            
            return new_agent
            
        except Exception as e:
            print(f"❌ Error creating replacement agent: {e}")
            return None
    
    def _initialize_single_agent_ecosystem(self, agent):
        """Initialize ecosystem data for a single new agent."""
        try:
            agent_id = agent.id
            
            # Extract and properly normalize fitness traits from agent's physical parameters
            motor_speed = getattr(agent.physical_params, 'motor_speed', 5.0)
            motor_torque = getattr(agent.physical_params, 'motor_torque', 50.0)
            learning_rate = getattr(agent.physical_params, 'learning_rate', 0.1)
            
            fitness_traits = {
                'speed': min(1.0, motor_speed / 15.0),  # Normalize motor_speed (typically 3-12) to 0-1
                'strength': min(1.0, motor_torque / 200.0),  # Normalize motor_torque (typically 30-180) to 0-1  
                'cooperation': min(1.0, learning_rate * 50.0)  # Boost learning_rate (typically 0.005-0.02) to meaningful range
            }
            
            # Assign ecosystem role
            role = self.ecosystem_dynamics.assign_ecosystem_role(agent_id, fitness_traits)
            
            # Initialize agent status tracking
            self.agent_statuses[agent_id] = {
                'role': role.value,
                'status': 'idle',
                'last_status_change': time.time(),
                'energy': 1.0,
                'speed_factor': 1.0,
                # Alliances and territories removed
            }
            
            self.agent_health[agent_id] = {
                'health': 1.0,
                'energy': 1.0,
                'last_updated': time.time()
            }
            
            # Initialize energy level for resource consumption
            self.agent_energy_levels[agent_id] = 1.0
            
            # Track birth time for survival statistics
            self.survival_stats['agent_birth_times'][agent_id] = time.time()
            
        except Exception as e:
            print(f"❌ Error initializing ecosystem data for agent {agent_id}: {e}")

    def _track_limb_distribution(self):
        """Track and log the distribution of limb counts among agents."""
        try:
            current_time = time.time()
            if current_time - self.last_limb_distribution_log < self.limb_distribution_interval:
                return
            
            # Count limbs for all active agents
            limb_counts = {}
            multi_limb_details = []
            total_agents = 0
            
            for agent in self.agents:
                if getattr(agent, '_destroyed', False) or not hasattr(agent, 'physical_params'):
                    continue
                    
                total_agents += 1
                num_limbs = getattr(agent.physical_params, 'num_arms', 1)
                num_segments = getattr(agent.physical_params, 'segments_per_limb', 2)
                
                if num_limbs not in limb_counts:
                    limb_counts[num_limbs] = 0
                limb_counts[num_limbs] += 1
                
                # Track multi-limb robot details
                if num_limbs > 1:
                    multi_limb_details.append({
                        'id': agent.id,
                        'limbs': num_limbs,
                        'segments': num_segments,
                        'total_joints': num_limbs * num_segments,
                        'position': (agent.body.position.x, agent.body.position.y) if agent.body else (0, 0),
                        'reward': getattr(agent, 'total_reward', 0)
                    })
            
            # Calculate percentages
            limb_percentages = {}
            for limbs, count in limb_counts.items():
                limb_percentages[limbs] = (count / total_agents * 100) if total_agents > 0 else 0
            
            # Store history
            distribution_entry = {
                'timestamp': current_time,
                'total_agents': total_agents,
                'limb_counts': limb_counts.copy(),
                'limb_percentages': limb_percentages.copy(),
                'multi_limb_count': sum(count for limbs, count in limb_counts.items() if limbs > 1),
                'multi_limb_details': multi_limb_details.copy()
            }
            self.limb_distribution_history.append(distribution_entry)
            
            # Keep only last 100 entries (FIXED: deque maxlen=100 handles this automatically)
            
            # Log current distribution
            print(f"\n🦾 === LIMB DISTRIBUTION REPORT (Step {self.step_count}) ===")
            print(f"👥 Total Active Agents: {total_agents}")
            
            for limbs in sorted(limb_counts.keys()):
                count = limb_counts[limbs]
                percentage = limb_percentages[limbs]
                limb_emoji = "🦾" if limbs == 1 else "🕷️" if limbs <= 3 else "🐙" if limbs <= 5 else "👾"
                print(f"   {limb_emoji} {limbs}-limb robots: {count:2d} ({percentage:5.1f}%)")
            
            # Detailed multi-limb robot info
            if multi_limb_details:
                print(f"\n🔍 Multi-Limb Robot Details ({len(multi_limb_details)} robots):")
                for detail in sorted(multi_limb_details, key=lambda x: x['limbs'], reverse=True):
                    print(f"   🤖 Agent {str(detail['id'])[:8]}: {detail['limbs']} limbs × {detail['segments']} segments = {detail['total_joints']} joints")
                    print(f"      📍 Position: ({detail['position'][0]:.1f}, {detail['position'][1]:.1f}), Reward: {detail['reward']:.2f}")
            else:
                print("❌ NO MULTI-LIMB ROBOTS FOUND!")
                
            # Show trend if we have history
            if len(self.limb_distribution_history) >= 2:
                prev_multi = self.limb_distribution_history[-2]['multi_limb_count']
                curr_multi = distribution_entry['multi_limb_count']
                change = curr_multi - prev_multi
                
                if change > 0:
                    print(f"📈 Multi-limb robots INCREASED by {change} (+{change/max(1,prev_multi)*100:.1f}%)")
                elif change < 0:
                    print(f"📉 Multi-limb robots DECREASED by {abs(change)} ({change/max(1,prev_multi)*100:.1f}%)")
                else:
                    print(f"➡️ Multi-limb robot count STABLE")
            
            print(f"🦾 === END LIMB DISTRIBUTION REPORT ===\n")
            self.last_limb_distribution_log = current_time
            
        except Exception as e:
            print(f"❌ Error tracking limb distribution: {e}")
            import traceback
            traceback.print_exc()

    def _update_statistics(self):
        """Update population statistics with enhanced safety checks."""
        if not self.agents:
            return
        
        # Track limb distribution
        self._track_limb_distribution()
        
        # Use agent ID as key instead of list index to avoid evolution issues
        for agent in self.agents:
            # Skip destroyed agents or agents without bodies
            if getattr(agent, '_destroyed', False) or not agent.body:
                continue
                
            agent_id = agent.id
            
            # Safety check for all body parts before accessing
            try:
                # Get safe positions and angles with fallbacks
                current_position = tuple(agent.body.position) if agent.body else (0, 0)
                current_velocity = tuple(agent.body.linearVelocity) if agent.body else (0, 0)
                shoulder_angle = agent.upper_arm.angle if agent.upper_arm else 0.0
                elbow_angle = agent.lower_arm.angle if agent.lower_arm else 0.0
                total_distance = (agent.body.position.x - agent.initial_position[0]) if agent.body else 0.0
                
                if agent_id not in self.robot_stats:
                    self.robot_stats[agent_id] = {
                        'id': agent.id,
                        'current_position': current_position,
                        'velocity': current_velocity,
                        'arm_angles': {'shoulder': shoulder_angle, 'elbow': elbow_angle},
                        'steps_alive': 0,
                        'total_distance': total_distance,
                        'fitness': 0.0,
                        'q_updates': 0,
                        'episode_reward': 0.0,
                        'action_history': []
                    }
                
                # Update all stats with safety checks
                self.robot_stats[agent_id]['current_position'] = current_position
                self.robot_stats[agent_id]['velocity'] = current_velocity
                self.robot_stats[agent_id]['arm_angles']['shoulder'] = shoulder_angle
                self.robot_stats[agent_id]['arm_angles']['elbow'] = elbow_angle
                self.robot_stats[agent_id]['steps_alive'] += 1
                self.robot_stats[agent_id]['total_distance'] = total_distance
                self.robot_stats[agent_id]['fitness'] = total_distance
                
                # 🌊 INFINITE EXPLORATION: Use rolling rewards instead of total_reward
                if hasattr(agent, 'get_current_performance'):
                    performance = agent.get_current_performance()
                    self.robot_stats[agent_id]['reward_rate'] = performance['reward_rate']
                    self.robot_stats[agent_id]['short_term_avg'] = performance['short_term_avg'] 
                    self.robot_stats[agent_id]['medium_term_avg'] = performance['medium_term_avg']
                    self.robot_stats[agent_id]['performance_trend'] = performance['performance_trend']
                    self.robot_stats[agent_id]['achievement_count'] = performance['achievement_count']
                    self.robot_stats[agent_id]['fitness_score'] = agent.get_fitness_score() if hasattr(agent, 'get_fitness_score') else total_distance
                else:
                    # Fallback for backward compatibility
                    self.robot_stats[agent_id]['reward_rate'] = getattr(agent, 'total_reward', 0.0) / max(1, getattr(agent, 'steps', 1))
                    self.robot_stats[agent_id]['fitness_score'] = total_distance
                
                # Keep episode_reward for backward compatibility but mark as deprecated
                self.robot_stats[agent_id]['episode_reward'] = getattr(agent, 'total_reward', 0.0)  # DEPRECATED
                self.robot_stats[agent_id]['action_history'] = getattr(agent, 'action_history', [])
                
            except Exception as e:
                print(f"⚠️  Error updating stats for agent {agent_id}: {e}")
                # Remove the problematic agent from stats
                if agent_id in self.robot_stats:
                    del self.robot_stats[agent_id]
        
        # Update population statistics with safety checks
        if self.robot_stats:
            try:
                distances = [stats['total_distance'] for stats in self.robot_stats.values() if 'total_distance' in stats]
                
                # Get fitnesses only from agents that are not destroyed and have bodies
                valid_agents = [agent for agent in self.agents if not getattr(agent, '_destroyed', False) and agent.body]
                fitnesses = []
                for agent in valid_agents:
                    try:
                        fitness = agent.get_evolutionary_fitness()
                        fitnesses.append(fitness)
                    except Exception as e:
                        print(f"⚠️  Error getting fitness for agent {agent.id}: {e}")
                
                # Get evolution summary
                evolution_summary = self.evolution_engine.get_evolution_summary()
                
                # Calculate safe statistics
                avg_distance = sum(distances) / len(distances) if distances else 0
                best_distance = max(distances) if distances else 0
                worst_distance = min(distances) if distances else 0
                
                avg_fitness = sum(fitnesses) / len(fitnesses) if fitnesses else 0
                best_fitness = max(fitnesses) if fitnesses else 0
                
                # Calculate epsilon only from valid agents
                valid_epsilons = []
                for agent in valid_agents:
                    try:
                        if hasattr(agent, 'epsilon'):
                            valid_epsilons.append(agent.epsilon)
                    except:
                        pass
                avg_epsilon = sum(valid_epsilons) / len(valid_epsilons) if valid_epsilons else 0
                
                # Calculate total food consumed
                total_food_consumed = sum(stats.get('food_consumed', 0) for stats in self.robot_stats.values())
                
                self.population_stats = {
                    'generation': evolution_summary['generation'],
                    'best_distance': best_distance,
                    'average_distance': avg_distance,
                    'worst_distance': worst_distance,
                    'best_fitness': best_fitness,
                    'average_fitness': avg_fitness,
                    'diversity': evolution_summary['diversity'],
                    'total_agents': len(self.robot_stats),
                    'total_food_consumed': total_food_consumed,
                    'species_count': evolution_summary.get('species_count', 1),
                    'hall_of_fame_size': evolution_summary.get('hall_of_fame_size', 0),
                    'mutation_rate': evolution_summary['mutation_rate'],
                    'q_learning_stats': {
                        'avg_epsilon': avg_epsilon,
                        'total_q_updates': sum(stats.get('q_updates', 0) for stats in self.robot_stats.values())
                    }
                }
                
            except Exception as e:
                print(f"⚠️  Error updating population statistics: {e}")
                # Fallback to minimal stats
                self.population_stats = {
                    'generation': 1,
                    'best_distance': 0,
                    'average_distance': 0,
                    'worst_distance': 0,
                    'best_fitness': 0,
                    'average_fitness': 0,
                    'diversity': 0,
                    'total_agents': len(self.agents),
                    'species_count': 1,
                    'hall_of_fame_size': 0,
                    'mutation_rate': 0.1,
                    'q_learning_stats': {
                        'avg_epsilon': 0.3,
                        'total_q_updates': 0
                    }
                }

    def _safe_destroy_agent(self, agent):
        """Safely destroy an agent with proper error handling, using memory pool if available."""
        if not agent or getattr(agent, '_destroyed', False):
            return  # Already destroyed
            
        try:
            # Return agent to memory pool if available (preserves learning)
            if self.robot_memory_pool:
                try:
                    self.robot_memory_pool.return_robot(agent)
                    print(f"♻️ Returned agent {agent.id} to memory pool")
                    return
                except Exception as e:
                    print(f"⚠️ Failed to return agent {agent.id} to memory pool: {e}")
                    # Fall through to manual destruction
            
            # Manual destruction (fallback when no memory pool)
            # Mark as destroyed first to prevent further operations
            agent._destroyed = True
            
            # Disable all motors to prevent issues during destruction
            if hasattr(agent, 'upper_arm_joint') and agent.upper_arm_joint:
                try:
                    agent.upper_arm_joint.enableMotor = False
                    agent.upper_arm_joint.motorSpeed = 0
                except:
                    pass
                    
            if hasattr(agent, 'lower_arm_joint') and agent.lower_arm_joint:
                try:
                    agent.lower_arm_joint.enableMotor = False  
                    agent.lower_arm_joint.motorSpeed = 0
                except:
                    pass
            
            # Clear all references to Box2D objects before destruction
            bodies_to_destroy = []
            
            # Collect all bodies
            if hasattr(agent, 'wheels') and agent.wheels:
                bodies_to_destroy.extend([w for w in agent.wheels if w])
            if hasattr(agent, 'lower_arm') and agent.lower_arm:
                bodies_to_destroy.append(agent.lower_arm)
            if hasattr(agent, 'upper_arm') and agent.upper_arm:
                bodies_to_destroy.append(agent.upper_arm)
            if hasattr(agent, 'body') and agent.body:
                bodies_to_destroy.append(agent.body)
            
            # Destroy bodies (Box2D automatically destroys associated joints)
            for body in bodies_to_destroy:
                if body:
                    try:
                        self.world.DestroyBody(body)
                    except Exception as e:
                        print(f"⚠️  Error destroying body for agent {agent.id}: {e}")
            
            # Clear references to prevent access to destroyed objects
            agent.wheels = []
            agent.upper_arm = None
            agent.lower_arm = None
            agent.body = None
            agent.upper_arm_joint = None
            agent.lower_arm_joint = None
            agent.wheel_joints = []
            
            print(f"✅ Successfully destroyed agent {agent.id} (manual destruction)")
            
        except Exception as e:
            print(f"❌ Critical error destroying agent {getattr(agent, 'id', 'unknown')}: {e}")
            import traceback
            traceback.print_exc()

    def _process_destruction_queue(self):
        """Process pending agent destructions safely."""
        if not self._agents_pending_destruction:
            return
            
        with self._physics_lock:
            try:
                agents_to_process = self._agents_pending_destruction.copy()
                self._agents_pending_destruction.clear()
                
                for agent in agents_to_process:
                    self._safe_destroy_agent(agent)
                    
            except Exception as e:
                print(f"❌ Error processing destruction queue: {e}")

    def training_loop(self):
        """Main simulation loop with enhanced safety."""
        self.is_running = True
        
        # Timing parameters for a fixed-step loop
        fps = 60
        self.dt = 1.0 / fps
        accumulator = 0.0
        
        last_time = time.time()
        
        # Stats and debug timers
        last_stats_time = time.time()
        last_debug_time = time.time()
        last_health_check = time.time()
        last_mlflow_log = time.time()
        
        step_count = 0
        
        # CRITICAL SAFEGUARD: Use instance emergency shutdown state
        self._emergency_shutdown_state['last_emergency_check'] = last_time

        while self.is_running:
            frame_start_time = time.time()
            current_time = frame_start_time
            frame_time = current_time - last_time
            last_time = current_time
            
            # Add frame time to the accumulator
            accumulator += frame_time
            
            # CRITICAL SAFEGUARD: Emergency shutdown detection
            if current_time - self._emergency_shutdown_state['last_emergency_check'] >= 5.0:  # Check every 5 seconds
                try:
                    import psutil
                    import os
                    process = psutil.Process(os.getpid())
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    
                    # Count physics bodies
                    physics_body_count = 0
                    for agent in self.agents:
                        if hasattr(agent, 'body') and agent.body:
                            physics_body_count += 1
                        if hasattr(agent, 'upper_arm') and agent.upper_arm:
                            physics_body_count += 1
                        if hasattr(agent, 'lower_arm') and agent.lower_arm:
                            physics_body_count += 1
                        if hasattr(agent, 'wheels') and agent.wheels:
                            physics_body_count += len(agent.wheels)
                    
                    # Add dynamic world bodies
                    if self.dynamic_world_manager:
                        status = self.dynamic_world_manager.get_world_status()
                        physics_body_count += status.get('total_obstacles', 0)
                    
                    # Check emergency conditions
                    emergency_triggered = False
                    
                    if memory_mb > self._emergency_shutdown_state['max_memory_mb']:
                        print(f"🚨 EMERGENCY: Memory usage {memory_mb:.1f}MB exceeds limit {self._emergency_shutdown_state['max_memory_mb']}MB")
                        emergency_triggered = True
                    
                    if physics_body_count > self._emergency_shutdown_state['max_physics_bodies']:
                        print(f"🚨 EMERGENCY: Physics bodies {physics_body_count} exceeds limit {self._emergency_shutdown_state['max_physics_bodies']}")
                        emergency_triggered = True
                    
                    if frame_time > 0.5:  # Frame took more than 500ms
                        self._emergency_shutdown_state['consecutive_slow_frames'] += 1
                        print(f"🐌 SLOW FRAME: {frame_time:.2f}s ({self._emergency_shutdown_state['consecutive_slow_frames']}/{self._emergency_shutdown_state['max_consecutive_slow_frames']})")
                        
                        if self._emergency_shutdown_state['consecutive_slow_frames'] >= self._emergency_shutdown_state['max_consecutive_slow_frames']:
                            print(f"🚨 EMERGENCY: {self._emergency_shutdown_state['consecutive_slow_frames']} consecutive slow frames")
                            emergency_triggered = True
                    else:
                        self._emergency_shutdown_state['consecutive_slow_frames'] = 0
                    
                    if emergency_triggered:
                        print("🚨 TRIGGERING EMERGENCY SHUTDOWN TO PREVENT SYSTEM FREEZE")
                        if self.dynamic_world_manager:
                            print("🧹 Cleaning up dynamic world...")
                            self.dynamic_world_manager.cleanup_all_tiles()
                            self.dynamic_world_manager = None
                        
                        self.is_running = False
                        break
                    
                    self._emergency_shutdown_state['last_emergency_check'] = current_time
                    
                except Exception as e:
                    print(f"⚠️ Error in emergency check: {e}")
                    self._emergency_shutdown_state['last_emergency_check'] = current_time
            
            # Fixed-step physics updates with enhanced thread safety
            while accumulator >= self.dt:
                physics_start = time.time()
                with self._physics_lock:  # Protect ALL Box2D operations
                    try:
                        # Process any pending destructions first
                        self._process_destruction_queue()
                        
                        # Step the physics world only if not evolving
                        if not self._is_evolving:
                            # Run multiple physics steps based on simulation speed multiplier
                            physics_step_start = time.time()
                            physics_steps = max(1, int(self.simulation_speed_multiplier))
                            for _ in range(physics_steps):
                                self.world.Step(self.dt, 8, 3)
                            physics_step_time = time.time() - physics_step_start
                            
                            # Update all agents (copy list to avoid iteration issues)
                            ai_processing_start = time.time()
                            current_agents = self.agents.copy()
                            agents_to_reset = []
                            
                            # AI OPTIMIZATION: Intelligent AI update scheduling
                            ai_agents_this_frame = []
                            
                            if self.ai_optimization_enabled and len(current_agents) > 0:
                                # Batch processing: Only update AI for a subset of agents each frame
                                ai_batch_size = max(5, int(len(current_agents) * self.ai_batch_percentage))
                                ai_start_index = (self.step_count * ai_batch_size) % len(current_agents)
                                
                                for i, agent in enumerate(current_agents):
                                    should_update_ai = False
                                    
                                    # Always update AI for focused agent
                                    if self.focused_agent and agent.id == self.focused_agent.id:
                                        should_update_ai = True
                                    else:
                                        # Batch scheduling for other agents
                                        agent_index = (ai_start_index + i) % len(current_agents)
                                        if agent_index < ai_batch_size:
                                            # Additional spatial culling check
                                            if self.ai_spatial_culling_enabled:
                                                agent_pos = agent.body.position
                                                camera_pos = self.camera_position
                                                distance = ((agent_pos.x - camera_pos[0])**2 + (agent_pos.y - camera_pos[1])**2)**0.5
                                                should_update_ai = distance <= self.ai_spatial_culling_distance
                                            else:
                                                should_update_ai = True
                                    
                                    ai_agents_this_frame.append((agent, should_update_ai))
                            else:
                                # No optimization: update all agents (original behavior)
                                for agent in current_agents:
                                    ai_agents_this_frame.append((agent, True))
                            
                            for agent, should_update_ai in ai_agents_this_frame:
                                # Skip destroyed agents or agents without bodies
                                if getattr(agent, '_destroyed', False) or not agent.body:
                                    continue
                                    
                                try:
                                    # Check if agent is immobilized (very low energy)
                                    agent_energy = self.agent_energy_levels.get(agent.id, 1.0)
                                    is_immobilized = agent_energy < 0.15
                                    
                                    if is_immobilized:
                                        # Disable movement for immobilized agents
                                        if hasattr(agent, 'upper_arm_joint') and agent.upper_arm_joint:
                                            agent.upper_arm_joint.enableMotor = False
                                            agent.upper_arm_joint.motorSpeed = 0
                                        if hasattr(agent, 'lower_arm_joint') and agent.lower_arm_joint:
                                            agent.lower_arm_joint.enableMotor = False
                                            agent.lower_arm_joint.motorSpeed = 0
                                        # Don't call step() for immobilized agents
                                        continue
                                    else:
                                        # Re-enable movement for healthy agents
                                        if hasattr(agent, 'upper_arm_joint') and agent.upper_arm_joint:
                                            agent.upper_arm_joint.enableMotor = True
                                        if hasattr(agent, 'lower_arm_joint') and agent.lower_arm_joint:
                                            agent.lower_arm_joint.enableMotor = True
                                    
                                    # AI OPTIMIZATION: Use different step modes based on AI update schedule
                                    if should_update_ai:
                                        # Full AI update with learning
                                        agent.step(self.dt)
                                    else:
                                        # Physics-only update - continue previous action without AI decision
                                        if hasattr(agent, 'step_physics_only'):
                                            agent.step_physics_only(self.dt)
                                        else:
                                            # Fallback: continue previous action
                                            if hasattr(agent, 'current_action_tuple') and agent.current_action_tuple is not None:
                                                agent.apply_action(agent.current_action_tuple)
                                            agent.steps += 1

                                    # Check for reset conditions but don't reset immediately
                                    # 🌍 INFINITE WORLD: Only reset if agents fall off the world, no episode limits!
                                    if agent.body and agent.body.position.y < self.world_bounds_y:
                                        agents_to_reset.append(('world_bounds', agent))
                                    # REMOVED: Episode-based resets for infinite world exploration
                                    # Robots can now explore forever without being teleported back!
                                except Exception as e:
                                    print(f"⚠️  Error updating agent {agent.id}: {e}")
                                    import traceback
                                    traceback.print_exc()
                                    # DO NOT automatically destroy agents on errors - let them continue
                                    # Only health-based death (health <= 0.0) should trigger replacement
                            
                            # Process resets after physics step to avoid corruption
                            for reset_type, agent in agents_to_reset:
                                if getattr(agent, '_destroyed', False):
                                    continue  # Skip destroyed agents
                                    
                                try:
                                    if reset_type == 'world_bounds':
                                        # 🌍 INFINITE WORLD: Only reset position when agent falls off world
                                        # Preserve learning and continue exploration from spawn point
                                        agent.reset_position()
                                        print(f"🌍 Agent {str(agent.id)[:8]} fell off world - respawned to continue infinite exploration!")
                                    # REMOVED: episode_end handling - no more episode limits!
                                            
                                except Exception as e:
                                    print(f"⚠️  Error resetting agent {agent.id}: {e}")
                            
                            ai_processing_time = time.time() - ai_processing_start
                            
                            # Store performance timings and AI optimization stats
                            if not hasattr(self, 'performance_timings'):
                                self.performance_timings = {
                                    'physics_simulation': [],
                                    'agent_ai_processing': [],
                                    'ecosystem_updates': [],
                                    'statistics_updates': [],
                                    'data_serialization': [],
                                    'total_frame_time': []
                                }
                                self.ai_optimization_stats = {
                                    'total_agents': [],
                                    'ai_updated_agents': [],
                                    'ai_optimization_ratio': []
                                }
                            
                            self.performance_timings['physics_simulation'].append(physics_step_time)
                            self.performance_timings['agent_ai_processing'].append(ai_processing_time)
                            
                            # Track AI optimization effectiveness
                            ai_updated_count = sum(1 for _, should_update in ai_agents_this_frame if should_update)
                            total_agents = len(current_agents)
                            ai_optimization_ratio = 1.0 - (ai_updated_count / max(1, total_agents))
                            
                            self.ai_optimization_stats['total_agents'].append(total_agents)
                            self.ai_optimization_stats['ai_updated_agents'].append(ai_updated_count)
                            self.ai_optimization_stats['ai_optimization_ratio'].append(ai_optimization_ratio)
                            
                            # Keep only last 100 measurements
                            for key in self.performance_timings:
                                if len(self.performance_timings[key]) > 100:
                                    self.performance_timings[key] = self.performance_timings[key][-100:]
                            
                            for key in self.ai_optimization_stats:
                                if len(self.ai_optimization_stats[key]) > 100:
                                    self.ai_optimization_stats[key] = self.ai_optimization_stats[key][-100:]
                                    
                    except Exception as e:
                        print(f"❌ Critical error in physics loop: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Decrement accumulator
                accumulator -= self.dt
                self.step_count += 1
                
                # Update physics FPS tracking (account for speed multiplier)
                physics_steps_this_frame = max(1, int(self.simulation_speed_multiplier))
                self.physics_fps_counter += physics_steps_this_frame
                if current_time - self.last_physics_fps_update >= 1.0:  # Update every second
                    self.current_physics_fps = round(self.physics_fps_counter / (current_time - self.physics_fps_start_time))
                    self.physics_fps_counter = 0
                    self.physics_fps_start_time = current_time
                    self.last_physics_fps_update = current_time

            # Update camera and statistics (can be done once per frame)
            self.update_camera(frame_time)
            
            # Update ecosystem dynamics periodically
            if current_time - self.last_ecosystem_update > self.ecosystem_update_interval:
                ecosystem_start = time.time()
                self._update_ecosystem_dynamics()
                ecosystem_time = time.time() - ecosystem_start
                if hasattr(self, 'performance_timings'):
                    self.performance_timings['ecosystem_updates'].append(ecosystem_time)
                self.last_ecosystem_update = current_time
            
            # Create obstacle physics bodies periodically (every 10 seconds)
            if not hasattr(self, 'last_obstacle_update'):
                self.last_obstacle_update = current_time
            if current_time - self.last_obstacle_update > 10.0:
                self._create_obstacle_physics_bodies()
                self.last_obstacle_update = current_time
            
            # Generate resources between agents periodically
            if current_time - self.last_resource_generation > self.resource_generation_interval:
                self._generate_resources_between_agents()
                self.last_resource_generation = current_time
            
            # Update agent energy levels through resource consumption
            self._update_resource_consumption()
            
            # Health check logging every 30 seconds with DETAILED MEMORY TRACKING
            if current_time - last_health_check > 30.0:
                try:
                    import psutil
                    import os
                    process = psutil.Process(os.getpid())
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    cpu_percent = process.cpu_percent()
                    active_agents = len([a for a in self.agents if not getattr(a, '_destroyed', False)])
                    
                    # DETAILED OBJECT COUNTING in health check
                    total_buffers = 0
                    max_buffer_size = 0
                    agents_with_learning = 0
                    total_neural_networks = 0
                    total_attention_records = 0
                    total_physics_bodies = 0
                    total_joints = 0
                    total_food_sources = 0
                    total_obstacles = 0
                    
                    for agent in self.agents:
                        if getattr(agent, '_destroyed', False):
                            continue
                        
                        # Count physics bodies per agent
                        if hasattr(agent, 'body') and agent.body:
                            total_physics_bodies += 1
                        if hasattr(agent, 'upper_arm') and agent.upper_arm:
                            total_physics_bodies += 1
                        if hasattr(agent, 'lower_arm') and agent.lower_arm:
                            total_physics_bodies += 1
                        if hasattr(agent, 'wheels') and agent.wheels:
                            total_physics_bodies += len(agent.wheels)
                        if hasattr(agent, 'joints') and agent.joints:
                            total_joints += len(agent.joints)
                            
                        if hasattr(agent, '_learning_system') and agent._learning_system:
                            agents_with_learning += 1
                            total_neural_networks += 1
                            
                            # Count memory buffer
                            if hasattr(agent._learning_system, 'memory') and hasattr(agent._learning_system.memory, 'buffer'):
                                buffer_size = len(agent._learning_system.memory.buffer)
                                total_buffers += buffer_size
                                max_buffer_size = max(max_buffer_size, buffer_size)
                            
                            # Count attention records  
                            if hasattr(agent._learning_system, 'attention_history'):
                                attention_size = len(agent._learning_system.attention_history)
                                total_attention_records += attention_size
                    
                    # Count ecosystem objects
                    if hasattr(self, 'ecosystem_dynamics') and hasattr(self.ecosystem_dynamics, 'food_sources'):
                        total_food_sources = len(self.ecosystem_dynamics.food_sources)
                    
                    # Count environmental objects
                    if hasattr(self, 'environmental_system') and hasattr(self.environmental_system, 'obstacles'):
                        total_obstacles = len(self.environmental_system.obstacles)
                    
                    print(f"💚 HEALTH CHECK: Memory={memory_mb:.1f}MB, CPU={cpu_percent:.1f}%, Agents={active_agents}, Step={self.step_count}")
                    print(f"   🧠 Learning agents: {agents_with_learning}, Neural networks: {total_neural_networks}")
                    print(f"   💾 Total buffer entries: {total_buffers}, Max buffer: {max_buffer_size}")
                    print(f"   🎯 Total attention records: {total_attention_records}")
                    print(f"   🌍 Physics bodies: {total_physics_bodies}, Joints: {total_joints}")
                    print(f"   🍽️ Food sources: {total_food_sources}, Obstacles: {total_obstacles}")
                    print(f"   📊 Robot stats entries: {len(self.robot_stats)}")
                    print(f"   🌿 Ecosystem events: {len(getattr(self, 'consumption_events', []))}, Death events: {len(getattr(self, 'death_events', []))}")
                    # Ray casting logging removed for performance
                    
                    # Alert if memory is too high
                    if memory_mb > 800:
                        print(f"⚠️ HIGH MEMORY USAGE: {memory_mb:.1f}MB - triggering aggressive cleanup")
                        self._cleanup_performance_data()
                        self._cleanup_attention_networks()
                        
                except Exception as e:
                    print(f"💚 HEALTH CHECK: Step={self.step_count}, Agents={len(self.agents)} (Error getting system stats: {e})")
                last_health_check = current_time
            
            # Performance cleanup every minute (increased frequency)
            if current_time - self.last_performance_cleanup > self.performance_cleanup_interval:
                self._cleanup_performance_data()
                self.last_performance_cleanup = current_time
            
            # PERFORMANCE: Frequent attention network cleanup every 30 seconds
            if current_time - self.last_attention_cleanup > self.attention_cleanup_interval:
                self._cleanup_attention_networks()
                self.last_attention_cleanup = current_time
                
            # Invalidate web cache periodically
            if current_time - self.last_web_interface_update > 0.1:  # Invalidate after 100ms
                self.web_cache_valid = False
            
            # Debug logging every 15 seconds (increased from 10 for less spam)
            if current_time - last_debug_time > 15.0:
                print(f"🔧 Physics step {self.step_count}: {len(self.agents)} agents active, Gen={self.evolution_engine.generation}")
                if self.agents:
                    # Debug output for the first agent
                    try:
                        first_agent = self.agents[0]
                        if not getattr(first_agent, '_destroyed', False) and first_agent.body:
                            print(f"   Agent sample: pos=({first_agent.body.position.x:.2f}, {first_agent.body.position.y:.2f}), "
                                  f"vel=({first_agent.body.linearVelocity.x:.2f}, {first_agent.body.linearVelocity.y:.2f}), "
                                  f"reward={first_agent.total_reward:.2f}, steps={first_agent.steps}")
                    except Exception as e:
                        print(f"⚠️  Error in debug output: {e}")
                        
                last_debug_time = current_time
            
            if current_time - last_stats_time > self.stats_update_interval:
                stats_start = time.time()
                self._update_statistics()
                stats_time = time.time() - stats_start
                if hasattr(self, 'performance_timings'):
                    self.performance_timings['statistics_updates'].append(stats_time)
                
                # Queue evaluation metrics for background collection (non-blocking)
                if self.enable_evaluation and self.metrics_collector:
                    try:
                        evolution_summary = self.evolution_engine.get_evolution_summary()
                        self.metrics_collector.collect_metrics_async(
                            agents=self.agents,
                            population_stats=self.population_stats,
                            evolution_summary=evolution_summary,
                            generation=evolution_summary.get('generation', 1),
                            step_count=self.step_count
                        )
                    except Exception as e:
                        print(f"⚠️  Error queuing evaluation metrics: {e}")
                
                last_stats_time = current_time
            
            # MLflow logging every 60 seconds
            if current_time - last_mlflow_log > 60.0:
                if self.enable_evaluation and hasattr(self, 'mlflow_integration') and self.mlflow_integration:
                    try:
                        # Log population metrics
                        generation = self.evolution_engine.generation
                        population_metrics = {
                            'generation': generation,
                            'population_size': len(self.agents),
                            'avg_fitness': sum(a.get_evolutionary_fitness() for a in self.agents) / len(self.agents) if self.agents else 0,
                            'best_fitness': max(a.get_evolutionary_fitness() for a in self.agents) if self.agents else 0,
                            'diversity': self.evolution_engine.diversity_history[-1] if self.evolution_engine.diversity_history else 0,
                            'step_count': self.step_count
                        }
                        self.mlflow_integration.log_population_metrics(generation, population_metrics)
                        print(f"📊 Logged population metrics to MLflow (Gen {generation})")
                        
                        # Log individual robot metrics for top 3 performers
                        if self.agents:
                            sorted_agents = sorted(self.agents, key=lambda a: a.get_evolutionary_fitness(), reverse=True)
                            for i, agent in enumerate(sorted_agents[:3]):
                                individual_metrics = {
                                    'fitness': agent.get_evolutionary_fitness(),
                                    'total_reward': agent.total_reward,
                                    'steps': agent.steps,
                                    'position_x': agent.body.position.x if agent.body else 0                                }
                                self.mlflow_integration.log_individual_robot_metrics(
                                    f"top_{i+1}_{str(agent.id)[:8]}", individual_metrics, self.step_count
                                )
                        
                    except Exception as e:
                        print(f"⚠️  Error logging to MLflow: {e}")
                
                # Periodic elite robot saving
                try:
                    self._save_elite_robots_periodically()
                except Exception as e:
                    print(f"⚠️  Error in periodic elite saving: {e}")
                
                last_mlflow_log = current_time
                        
            # Check for automatic evolution (with safety)
            if (self.auto_evolution_enabled and 
                current_time - self.last_evolution_time >= self.evolution_interval and
                not self._is_evolving):
                self._evolution_requested = True
                self.last_evolution_time = current_time
            
            # Process evolution request if safe
            if self._evolution_requested and not self._is_evolving:
                self._evolution_requested = False
                try:
                    self.trigger_evolution()
                except Exception as e:
                    print(f"❌ Evolution failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Calculate total frame time and store performance data
            total_frame_time = time.time() - frame_start_time
            if hasattr(self, 'performance_timings'):
                self.performance_timings['total_frame_time'].append(total_frame_time)
                
                # Print performance report every 30 seconds
                if not hasattr(self, 'last_performance_report'):
                    self.last_performance_report = time.time()
                
                if current_time - self.last_performance_report > 30.0:
                    self._print_performance_report()
                    self.last_performance_report = current_time
            
            # Sleep to maintain target FPS
            time.sleep(max(0, self.dt - (time.time() - current_time)))

    def update_agent_params(self, params, target_agent_id=None):
        """Update parameters for specific agent or all agents with safety checks."""
        if target_agent_id is not None:
            # Update only the focused agent
            target_agent = next((agent for agent in self.agents 
                               if agent.id == target_agent_id and not getattr(agent, '_destroyed', False)), None)
            if not target_agent:
                print(f"❌ Agent {target_agent_id} not found or destroyed")
                return False
            
            agents_to_update = [target_agent]
        else:
            # Update all valid agents
            agents_to_update = [agent for agent in self.agents if not getattr(agent, '_destroyed', False)]
        
        for agent in agents_to_update:
            # Skip agents without bodies
            if not agent.body:
                continue
                
            try:
                for key, value in params.items():
                    # Handle special physical properties with safety checks
                    if key == 'friction':
                        # Get all valid parts
                        parts = [p for p in [agent.body, agent.upper_arm, agent.lower_arm] + (agent.wheels or []) if p]
                        for part in parts:
                            try:
                                for fixture in part.fixtures:
                                    fixture.friction = value
                            except Exception as e:
                                print(f"⚠️  Error setting friction for agent {agent.id}: {e}")
                                
                    elif key == 'density':
                        # Get all valid parts
                        parts = [p for p in [agent.body, agent.upper_arm, agent.lower_arm] + (agent.wheels or []) if p]
                        for part in parts:
                            try:
                                for fixture in part.fixtures:
                                    fixture.density = value
                            except Exception as e:
                                print(f"⚠️  Error setting density for agent {agent.id}: {e}")
                        
                        # Reset mass data for all valid parts
                        try:
                            if agent.body:
                                agent.body.ResetMassData()
                            if agent.upper_arm:
                                agent.upper_arm.ResetMassData()
                            if agent.lower_arm:
                                agent.lower_arm.ResetMassData()
                            for wheel in (agent.wheels or []):
                                if wheel:
                                    wheel.ResetMassData()
                        except Exception as e:
                            print(f"⚠️  Error resetting mass data for agent {agent.id}: {e}")
                            
                    elif key == 'linear_damping':
                        # Get all valid parts
                        parts = [p for p in [agent.body, agent.upper_arm, agent.lower_arm] + (agent.wheels or []) if p]
                        for part in parts:
                            try:
                                part.linearDamping = value
                            except Exception as e:
                                print(f"⚠️  Error setting linear damping for agent {agent.id}: {e}")
                                
                    # Handle generic agent attributes
                    elif hasattr(agent, key):
                        setattr(agent, key, value)
                        
            except Exception as e:
                print(f"⚠️  Error updating parameters for agent {agent.id}: {e}")
        
        target_desc = f"agent {target_agent_id}" if target_agent_id else "all agents"
        print(f"✅ Updated {target_desc} parameters: {params}")
        return True

    def _print_performance_report(self):
        """Print a detailed performance analysis report with enhanced debugging."""
        try:
            if not hasattr(self, 'performance_timings') or not self.performance_timings:
                return
            
            current_time = time.time()
            print(f"\n📊 === ENHANCED PERFORMANCE ANALYSIS REPORT ===")
            print(f"🕐 Report Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))}")
            print(f"📏 Sample size: {len(self.performance_timings.get('total_frame_time', []))} frames")
            print(f"🔢 Step count: {self.step_count}")
            print(f"⚡ Simulation speed: {self.simulation_speed_multiplier}x")
            
            # ENHANCED: Memory and system resource tracking
            print(f"\n🧠 === SYSTEM RESOURCES ===")
            try:
                import psutil
                import os
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                cpu_percent = process.cpu_percent()
                print(f"💾 Memory Usage: {memory_mb:.1f} MB")
                print(f"⚙️  CPU Usage: {cpu_percent:.1f}%")
                
                # Memory trend analysis
                if not hasattr(self, 'memory_history'):
                    self.memory_history = []
                self.memory_history.append((current_time, memory_mb))
                
                # Keep last 20 measurements (10 minutes of history)
                if len(self.memory_history) > 20:
                    self.memory_history = self.memory_history[-20:]
                
                if len(self.memory_history) >= 2:
                    memory_trend = self.memory_history[-1][1] - self.memory_history[0][1]
                    memory_rate = memory_trend / max(1, (self.memory_history[-1][0] - self.memory_history[0][0]) / 60)  # MB per minute
                    print(f"📈 Memory Trend: {memory_trend:+.1f} MB ({memory_rate:+.1f} MB/min)")
                    
                    if memory_rate > 5:
                        print(f"⚠️  WARNING: High memory growth rate detected!")
                    elif memory_mb > 1500:
                        print(f"⚠️  WARNING: High memory usage detected!")
                        
            except ImportError:
                print(f"⚠️  psutil not available for system monitoring")
            except Exception as e:
                print(f"⚠️  Error getting system info: {e}")
            
            # ENHANCED: Agent and simulation health metrics
            print(f"\n🤖 === SIMULATION HEALTH ===")
            active_agents = len([a for a in self.agents if not getattr(a, '_destroyed', False)])
            print(f"👥 Active Agents: {active_agents}")
            print(f"🌍 Physics Bodies: {len(self.world.bodies)}")
            print(f"🔗 Physics Joints: {len(self.world.joints)}")
            
            if hasattr(self, 'ecosystem_dynamics'):
                food_sources = len(self.ecosystem_dynamics.food_sources)
                print(f"🍽️ Food Sources: {food_sources}")
            
            if hasattr(self, 'slow_frame_count'):
                print(f"🐌 Slow Frames: {self.slow_frame_count}/10")
                if self.slow_frame_count >= 7:
                    print(f"🚨 CRITICAL: Approaching emergency shutdown threshold!")
            
            # ENHANCED: Performance timing analysis with percentiles
            print(f"\n⏱️  === DETAILED TIMING ANALYSIS ===")
            for category, timings in self.performance_timings.items():
                if timings:
                    sorted_times = sorted(timings)
                    avg_time = sum(timings) / len(timings) * 1000  # Convert to milliseconds
                    max_time = max(timings) * 1000
                    min_time = min(timings) * 1000
                    p50_time = sorted_times[len(sorted_times)//2] * 1000  # Median
                    p95_time = sorted_times[int(len(sorted_times)*0.95)] * 1000  # 95th percentile
                    p99_time = sorted_times[int(len(sorted_times)*0.99)] * 1000  # 99th percentile
                    
                    # Calculate percentage of total frame time
                    if category != 'total_frame_time' and self.performance_timings.get('total_frame_time'):
                        total_avg = sum(self.performance_timings['total_frame_time']) / len(self.performance_timings['total_frame_time'])
                        percentage = (avg_time / 1000) / total_avg * 100 if total_avg > 0 else 0
                        print(f"⏱️  {category.replace('_', ' ').title()}: {avg_time:.2f}ms avg ({percentage:.1f}% of frame)")
                        print(f"    📊 Distribution: min={min_time:.1f}ms p50={p50_time:.1f}ms p95={p95_time:.1f}ms p99={p99_time:.1f}ms max={max_time:.1f}ms")
                        
                        # Performance warnings
                        if p99_time > 100:  # 99th percentile over 100ms
                            print(f"    ⚠️  99th percentile shows occasional severe delays!")
                        elif p95_time > 50:  # 95th percentile over 50ms
                            print(f"    ⚠️  95th percentile shows regular delays")
                    else:
                        print(f"⏱️  {category.replace('_', ' ').title()}: {avg_time:.2f}ms avg, {min_time:.2f}-{max_time:.2f}ms range")
                        
                        # Total frame time analysis
                        if category == 'total_frame_time':
                            slow_frames = sum(1 for t in timings if t > 1.0)  # Frames over 1 second
                            very_slow_frames = sum(1 for t in timings if t > 5.0)  # Frames over 5 seconds
                            print(f"    🐌 Slow frames (>1s): {slow_frames}/{len(timings)} ({slow_frames/len(timings)*100:.1f}%)")
                            print(f"    🚨 Very slow frames (>5s): {very_slow_frames}/{len(timings)} ({very_slow_frames/len(timings)*100:.1f}%)")
            
            # ENHANCED: Bottleneck analysis with severity assessment
            print(f"\n🚩 === BOTTLENECK ANALYSIS ===")
            bottlenecks = []
            for category, timings in self.performance_timings.items():
                if category != 'total_frame_time' and timings:
                    avg_time = sum(timings) / len(timings)
                    max_time = max(timings)
                    bottlenecks.append((category, avg_time, max_time))
            
            if bottlenecks:
                bottlenecks.sort(key=lambda x: x[1], reverse=True)
                print(f"🚩 Primary bottleneck: {bottlenecks[0][0].replace('_', ' ').title()}")
                print(f"   Average: {bottlenecks[0][1]*1000:.2f}ms, Peak: {bottlenecks[0][2]*1000:.2f}ms")
                
                # Severity assessment
                severity = "LOW"
                if bottlenecks[0][1] > 0.050:  # > 50ms average
                    severity = "HIGH"
                elif bottlenecks[0][1] > 0.020:  # > 20ms average
                    severity = "MEDIUM"
                    
                print(f"   Severity: {severity}")
                
                # Enhanced recommendations with specific thresholds
                category = bottlenecks[0][0]
                avg_ms = bottlenecks[0][1] * 1000
                if category == 'physics_simulation':
                    if avg_ms > 10:
                        print(f"💡 URGENT: Physics simulation averaging {avg_ms:.1f}ms - reduce agent count or simulation complexity")
                    else:
                        print(f"💡 Recommendation: Physics simulation is the bottleneck - consider reducing simulation frequency or agent count")
                elif category == 'agent_ai_processing':
                    if avg_ms > 20:
                        print(f"💡 URGENT: AI processing averaging {avg_ms:.1f}ms - enable AI optimization or reduce learning frequency")
                    else:
                        print(f"💡 Recommendation: AI processing is the bottleneck - consider optimizing learning algorithms or reducing update frequency")
                elif category == 'ecosystem_updates':
                    print(f"💡 Recommendation: Ecosystem updates are the bottleneck - consider reducing update frequency or optimizing ecosystem calculations")
                elif category == 'statistics_updates':
                    print(f"💡 Recommendation: Statistics updates are the bottleneck - consider reducing statistics calculation frequency")
            
            # ENHANCED: Frame rate analysis with performance grade
            print(f"\n🎯 === FRAME RATE ANALYSIS ===")
            if self.performance_timings.get('total_frame_time'):
                total_times = self.performance_timings['total_frame_time']
                target_frame_time = 1.0 / 60.0  # 60 FPS
                
                frames_60fps = sum(1 for t in total_times if t <= target_frame_time)
                frames_30fps = sum(1 for t in total_times if t <= 1.0/30.0)
                frames_15fps = sum(1 for t in total_times if t <= 1.0/15.0)
                frames_1fps = sum(1 for t in total_times if t <= 1.0)
                
                perf_60 = frames_60fps / len(total_times) * 100
                perf_30 = frames_30fps / len(total_times) * 100
                perf_15 = frames_15fps / len(total_times) * 100
                perf_1 = frames_1fps / len(total_times) * 100
                
                print(f"🎯 60 FPS target: {perf_60:.1f}% of frames")
                print(f"🎯 30 FPS target: {perf_30:.1f}% of frames") 
                print(f"🎯 15 FPS target: {perf_15:.1f}% of frames")
                print(f"🎯 1 FPS minimum: {perf_1:.1f}% of frames")
                
                # Performance grade
                if perf_60 >= 95:
                    grade = "A+ (Excellent)"
                elif perf_60 >= 80:
                    grade = "A (Good)"
                elif perf_30 >= 90:
                    grade = "B (Acceptable)"
                elif perf_15 >= 80:
                    grade = "C (Poor)"
                else:
                    grade = "F (Critical)"
                    
                print(f"📊 Performance Grade: {grade}")
                
                # Critical warnings
                if perf_1 < 90:
                    print(f"🚨 CRITICAL: {100-perf_1:.1f}% of frames are slower than 1 FPS!")
                elif perf_15 < 80:
                    print(f"⚠️  WARNING: {100-perf_15:.1f}% of frames are slower than 15 FPS!")
                elif perf_60 < 80:
                    print(f"⚠️  Performance Warning: Less than 80% of frames meet the 60 FPS target!")
            
            # ENHANCED: AI Optimization with efficiency metrics
            print(f"\n🧠 === AI OPTIMIZATION ANALYSIS ===")
            if hasattr(self, 'ai_optimization_stats') and self.ai_optimization_stats.get('ai_optimization_ratio'):
                avg_optimization_ratio = sum(self.ai_optimization_stats['ai_optimization_ratio']) / len(self.ai_optimization_stats['ai_optimization_ratio'])
                avg_ai_updated = sum(self.ai_optimization_stats['ai_updated_agents']) / len(self.ai_optimization_stats['ai_updated_agents'])
                avg_total_agents = sum(self.ai_optimization_stats['total_agents']) / len(self.ai_optimization_stats['total_agents'])
                
                print(f"🧠 AI Optimization: {avg_optimization_ratio:.1%} AI processing reduction")
                print(f"🔄 AI Updates: {avg_ai_updated:.1f}/{avg_total_agents:.1f} agents per frame ({(avg_ai_updated/avg_total_agents):.1%})")
                
                # Efficiency assessment
                if avg_optimization_ratio > 0.7:
                    print(f"✅ AI optimization is highly effective - over 70% reduction in AI processing!")
                elif avg_optimization_ratio > 0.5:
                    print(f"✅ AI optimization is highly effective - over 50% reduction in AI processing!")
                elif avg_optimization_ratio > 0.2:
                    print(f"👍 AI optimization is working - {avg_optimization_ratio:.0%} reduction achieved")
                else:
                    print(f"⚠️  AI optimization is minimal - consider adjusting ai_batch_percentage or ai_spatial_culling_distance")
                    
                # Specific recommendations based on efficiency
                efficiency = avg_ai_updated / avg_total_agents
                if efficiency > 0.8:
                    print(f"💡 Consider reducing ai_batch_percentage to improve efficiency")
                elif efficiency < 0.1:
                    print(f"💡 AI optimization may be too aggressive - consider increasing ai_batch_percentage")
            else:
                print(f"⚠️  AI optimization statistics not available")
            
            # ENHANCED: Emergency shutdown proximity warning
            print(f"\n🚨 === STABILITY MONITORING ===")
            if hasattr(self, 'slow_frame_count'):
                emergency_proximity = self.slow_frame_count / 10.0 * 100
                print(f"🐌 Slow frame count: {self.slow_frame_count}/10 ({emergency_proximity:.0f}% to emergency shutdown)")
                
                if self.slow_frame_count >= 8:
                    print(f"🚨 CRITICAL: Very close to emergency shutdown! System will stop at 10 slow frames.")
                elif self.slow_frame_count >= 5:
                    print(f"⚠️  WARNING: Multiple slow frames detected. Monitor system closely.")
                elif self.slow_frame_count >= 3:
                    print(f"⚠️  CAUTION: Some slow frames detected. Performance degrading.")
                else:
                    print(f"✅ Stability: System running normally")
            
            # Performance trend analysis
            if len(self.performance_timings.get('total_frame_time', [])) >= 50:
                recent_frames = self.performance_timings['total_frame_time'][-20:]  # Last 20 frames
                older_frames = self.performance_timings['total_frame_time'][-50:-30]  # 30-50 frames ago
                
                recent_avg = sum(recent_frames) / len(recent_frames) * 1000
                older_avg = sum(older_frames) / len(older_frames) * 1000
                trend = recent_avg - older_avg
                
                print(f"📈 Performance Trend: Recent frames {recent_avg:.1f}ms vs older {older_avg:.1f}ms ({trend:+.1f}ms change)")
                
                if trend > 10:
                    print(f"📉 DEGRADING: Performance is getting worse over time!")
                elif trend > 5:
                    print(f"⚠️  DECLINING: Performance showing degradation")
                elif trend < -5:
                    print(f"📈 IMPROVING: Performance is getting better")
                else:
                    print(f"✅ STABLE: Performance is stable")
            
            print(f"📊 === END ENHANCED PERFORMANCE REPORT ===\n")
            
        except Exception as e:
            print(f"⚠️ Error generating enhanced performance report: {e}")
            import traceback
            traceback.print_exc()

    def get_status(self, canvas_width=1200, canvas_height=800, viewport_culling=True, camera_x=0.0, camera_y=0.0):
        """
        Returns the current state of the simulation for rendering with enhanced safety and optional viewport culling.
        
        VIEWPORT CULLING BEHAVIOR:
        - 'agents': Viewport-filtered agents for rendering performance (only visible agents)
        - 'all_agents': ALL agents for UI panels (population summary, robot details, leaderboard)
        - 'leaderboard': Always shows ALL agents for fair ranking
        - 'robots': Always shows ALL agents for complete population view
        - 'statistics': Always calculated from ALL agents
        
        This ensures viewport culling only affects visual rendering, not data integrity.
        """
        serialization_start = time.time()
        
        if not self.is_running:
            return {'shapes': {}, 'leaderboard': [], 'robots': [], 'agents': [], 'all_agents': [], 'statistics': {}, 'camera': self.get_camera_state(), 'focused_agent_id': None}

        # Use a read lock to safely access agents
        with self._physics_lock:
            try:
                current_time = time.time()  # For death event age calculations
                
                # Create a safe copy of agents list
                current_agents = [agent for agent in self.agents if not getattr(agent, '_destroyed', False)]
                
                if viewport_culling:
                    # Calculate viewport bounds for culling using actual canvas dimensions and frontend camera position
                    viewport_bounds = self._calculate_viewport_bounds(canvas_width, canvas_height, camera_x, camera_y)
                    # Filter agents by viewport for performance optimization
                    viewport_agents = self._filter_agents_by_viewport(current_agents, viewport_bounds)
                else:
                    # No viewport culling - render all agents
                    viewport_bounds = {'left': -9999, 'right': 9999, 'bottom': -9999, 'top': 9999}
                    viewport_agents = current_agents
                
                # Store viewport culling statistics for monitoring
                zoom_used = getattr(self, 'user_zoom_level', 1.0)
                
                
                # Log viewport culling effectiveness occasionally for monitoring
                if viewport_culling and self.step_count % 10000 == 0:  # Every 5 seconds
                    viewport_stats = {
                        'enabled': viewport_culling,
                        'total_agents': len(current_agents),
                        'visible_agents': len(viewport_agents),
                        'culled_agents': len(current_agents) - len(viewport_agents),
                        'culling_ratio': (len(current_agents) - len(viewport_agents)) / max(1, len(current_agents)),
                        'viewport_bounds': viewport_bounds,
                        'zoom_level': zoom_used,
                        'camera_position': self.camera_position,
                        'canvas_size': [canvas_width, canvas_height],
                        'performance_note': 'Viewport culling only affects frontend rendering - backend simulation still processes all agents'
                    }
                    if viewport_stats['culled_agents'] > 0:
                        print(f"🔍 Viewport culling active: {viewport_stats['visible_agents']}/{viewport_stats['total_agents']} agents visible ({viewport_stats['culling_ratio']:.1%} culled)")
                        print(f"   📐 Zoom: {zoom_used:.2f}x, Camera: ({self.camera_position[0]:.1f}, {self.camera_position[1]:.1f})")
                        print(f"   📏 Viewport: {viewport_bounds['left']:.1f} to {viewport_bounds['right']:.1f} (width: {viewport_bounds['right'] - viewport_bounds['left']:.1f})")
                    else:
                        print(f"🔍 Viewport culling: ALL {viewport_stats['total_agents']} agents visible (zoom: {zoom_used:.2f}x)")
                
                # 1. Get agent shapes for drawing (only for viewport-visible agents)
                robot_shapes = []
                for agent in viewport_agents:
                    try:
                        if not agent.body:  # Skip agents without bodies
                            continue
                            
                        body_parts = []
                        # Chassis, Arms, Wheels
                        body_list = [agent.body] + (agent.wheels or [])
                        
                        # MULTI-LIMB RENDERING FIX: Render ALL limbs, not just first 2
                        if hasattr(agent, 'limbs') and agent.limbs:
                            # Multi-limb robot: render all limb segments
                            for limb_segments in agent.limbs:
                                for segment in limb_segments:
                                    if segment:
                                        body_list.append(segment)
                        else:
                            # Legacy single-limb robot: render upper_arm and lower_arm
                            if hasattr(agent, 'upper_arm') and agent.upper_arm:
                                body_list.append(agent.upper_arm)
                            if hasattr(agent, 'lower_arm') and agent.lower_arm:
                                body_list.append(agent.lower_arm)
                            
                        for part in body_list:
                            if not part:  # Skip None bodies
                                continue
                                
                            try:
                                for fixture in part.fixtures:
                                    shape = fixture.shape
                                    if isinstance(shape, b2.b2PolygonShape):
                                        body_parts.append({
                                            'type': 'polygon',
                                            'vertices': [tuple(part.GetWorldPoint(v)) for v in shape.vertices]
                                        })
                                    elif isinstance(shape, b2.b2CircleShape):
                                        body_parts.append({
                                            'type': 'circle',
                                            'center': tuple(part.GetWorldPoint(shape.pos)),
                                            'radius': shape.radius
                                        })
                            except Exception as e:
                                print(f"⚠️  Error getting shape data for agent {agent.id}: {e}")
                                
                        if body_parts:  # Only add if we have valid parts
                            robot_shapes.append({'id': agent.id, 'body_parts': body_parts})
                            
                    except Exception as e:
                        print(f"⚠️  Error processing agent {agent.id} for rendering: {e}")

                # 2. Get ground shapes for drawing
                ground_shapes = []
                try:
                    for body in self.world.bodies:
                        if body.type == b2.b2_staticBody:
                            for fixture in body.fixtures:
                                shape = fixture.shape
                                if isinstance(shape, b2.b2PolygonShape):
                                    ground_shapes.append({
                                        'type': 'polygon',
                                        'vertices': [tuple(body.GetWorldPoint(v)) for v in shape.vertices]
                                    })
                except Exception as e:
                    print(f"⚠️  Error getting ground shapes: {e}")
                
                # 3. Get leaderboard data (top 10 robots) - use all agents for leaderboard fairness
                # NOTE: Leaderboard should show all agents regardless of viewport for fair ranking
                try:
                    valid_stats = {k: v for k, v in self.robot_stats.items() 
                                  if k in {agent.id for agent in current_agents}}
                    sorted_robots = sorted(valid_stats.values(), 
                                         key=lambda r: r.get('food_consumed', 0), reverse=True)
                    leaderboard_data = []
                    for r in sorted_robots[:10]:
                        # Get learning approach icon for visual indication
                        approach_name = self._get_agent_learning_approach_name(r['id'])
                        approach_icon = '⚡'  # Default
                        if 'basic' in approach_name.lower():
                            approach_icon = '🔤'
                        elif 'enhanced' in approach_name.lower():
                            approach_icon = '⚡'
                        elif 'survival' in approach_name.lower():
                            approach_icon = '🍃'
                        elif 'deep' in approach_name.lower():
                            approach_icon = '🧠'
                        
                        leaderboard_data.append({
                            'id': r['id'], 
                            'name': f"{approach_icon} Robot {r['id']}", 
                            'food_consumed': r.get('food_consumed', 0)
                        })
                except Exception as e:
                    print(f"⚠️  Error creating leaderboard: {e}")
                    leaderboard_data = []
                
                # 4. Get detailed stats for side panel (top 10)
                robot_details = []
                try:
                    for i, r_stat in enumerate(sorted_robots[:10]):
                        # Find agent by ID instead of assuming ID matches list index
                        agent = next((a for a in current_agents if a.id == r_stat['id']), None)
                        if agent:
                            robot_details.append({
                                'id': r_stat['id'],
                                'name': f"Robot {r_stat['id']}",
                                'rank': i + 1,
                                'distance': r_stat.get('total_distance', 0),
                                'position': r_stat.get('current_position', (0,0)),
                                'episode_reward': r_stat.get('episode_reward', 0)
                            })
                except Exception as e:
                    print(f"⚠️  Error creating robot details: {e}")
                    robot_details = []

                # 5. Get minimal agent data for rendering + detailed data for focused agent only
                # SEPARATE: rendering agents (viewport-filtered) vs all agents data (for UI panels)
                agents_data = []  # For rendering - viewport filtered
                all_agents_data = []  # For UI panels - ALL agents
                
                try:
                    # Rendering agents: Use viewport_agents for rendering, but always include focused agent
                    visible_agents_set = set(viewport_agents)
                    if self.focused_agent and not getattr(self.focused_agent, '_destroyed', False):
                        visible_agents_set.add(self.focused_agent)
                    
                    for agent in visible_agents_set:
                        if not agent.body:  # Skip agents without bodies
                            continue
                            
                        agent_id = agent.id
                        is_focused = (self.focused_agent and not getattr(self.focused_agent, '_destroyed', False) and self.focused_agent.id == agent_id)
                        
                        # Basic data for all agents (needed for rendering)
                        basic_agent_data = {
                            'id': agent.id,
                            'body': {
                                'x': float(agent.body.position.x),
                                'y': float(agent.body.position.y),
                                'velocity': {
                                    'x': float(agent.body.linearVelocity.x),
                                    'y': float(agent.body.linearVelocity.y)
                                }
                            },
                            'total_reward': float(agent.total_reward),
                        }
                        
                        # MULTI-LIMB DATA FIX: Include all limbs for rendering
                        if hasattr(agent, 'limbs') and agent.limbs:
                            # Multi-limb robot: include all limb segment positions
                            limbs_data = []
                            for limb_idx, limb_segments in enumerate(agent.limbs):
                                segments_data = []
                                for segment_idx, segment in enumerate(limb_segments):
                                    if segment:
                                        segments_data.append({
                                            'x': float(segment.position.x),
                                            'y': float(segment.position.y),
                                            'angle': float(segment.angle)
                                        })
                                    else:
                                        segments_data.append({'x': 0, 'y': 0, 'angle': 0})
                                limbs_data.append(segments_data)
                            basic_agent_data['limbs'] = limbs_data
                            basic_agent_data['num_limbs'] = len(agent.limbs)
                            basic_agent_data['segments_per_limb'] = len(agent.limbs[0]) if agent.limbs else 0
                        else:
                            # Legacy single-limb robot: use upper_arm and lower_arm
                            basic_agent_data['upper_arm'] = {
                                'x': float(agent.upper_arm.position.x) if agent.upper_arm else 0,
                                'y': float(agent.upper_arm.position.y) if agent.upper_arm else 0
                            }
                            basic_agent_data['lower_arm'] = {
                                'x': float(agent.lower_arm.position.x) if agent.lower_arm else 0,
                                'y': float(agent.lower_arm.position.y) if agent.lower_arm else 0
                            }
                        
                        # Add ecosystem data
                        ecosystem_data = {
                            'role': self.agent_statuses.get(agent_id, {}).get('role', 'omnivore'),
                            'status': self.agent_statuses.get(agent_id, {}).get('status', 'idle'),
                            'speed': float((agent.body.linearVelocity.x ** 2 + agent.body.linearVelocity.y ** 2) ** 0.5)
                        }
                        
                        # PERFORMANCE OPTIMIZATION: Only include health data if health bars are enabled
                        if self.show_health_bars:
                            ecosystem_data.update({
                                'health': float(self.agent_health.get(agent_id, {'health': 1.0})['health']),
                                'energy': float(self.agent_health.get(agent_id, {'energy': 1.0})['energy'])
                            })
                        
                        basic_agent_data.update({'ecosystem': ecosystem_data})
                        agents_data.append(basic_agent_data)
                        
                        # Add detailed data for focused agent
                        if is_focused:
                            agent_status = self.agent_statuses.get(agent_id, {})
                            closest_food_info = self._get_closest_food_distance_for_agent(agent)
                            
                            # Get recent reward information - use the last reward from current step
                            recent_reward = float(getattr(agent, 'last_reward', 
                                                 getattr(agent, 'immediate_reward', 0.0)))
                            
                            # Add detailed data to the basic agent data
                            basic_agent_data.update({
                                'steps': int(agent.steps),
                                'action_history': [int(x) for x in agent.action_history[-10:]] if hasattr(agent, 'action_history') and agent.action_history else [],
                                'best_reward': float(getattr(agent, 'best_reward_received', 0.0)),
                                'worst_reward': float(getattr(agent, 'worst_reward_received', 0.0)),
                                'recent_reward': recent_reward,
                                'learning_approach': str(getattr(agent, 'learning_approach', 'basic_q_learning')),
                            })
                            
                            # Add detailed ecosystem data
                            basic_agent_data['ecosystem'].update({
                                'speed_factor': float(agent_status.get('speed_factor', 1.0)),
                                                # Alliances and territories removed
                                'closest_food_distance': float(closest_food_info['distance']),
                                'closest_food_signed_x_distance': float(closest_food_info.get('signed_x_distance', closest_food_info['distance'])),
                                'closest_food_type': closest_food_info['food_type'],
                                'closest_food_source': closest_food_info.get('source_type', 'environment'),
                                'closest_food_position': [float(closest_food_info['food_position'][0]), float(closest_food_info['food_position'][1])] if closest_food_info.get('food_position') is not None else None
                            })
                            
                    # Generate ALL agents data (for UI panels - not affected by viewport culling)
                    for agent in current_agents:
                        if not agent.body:  # Skip agents without bodies
                            continue
                            
                        agent_id = agent.id
                        is_focused = (self.focused_agent and not getattr(self.focused_agent, '_destroyed', False) and self.focused_agent.id == agent_id)
                        
                        # Basic data for all agents
                        ecosystem_data_all = {
                            'role': self.agent_statuses.get(agent_id, {}).get('role', 'omnivore'),
                            'status': self.agent_statuses.get(agent_id, {}).get('status', 'idle'),
                            'speed': float((agent.body.linearVelocity.x ** 2 + agent.body.linearVelocity.y ** 2) ** 0.5)
                        }
                        
                        # PERFORMANCE OPTIMIZATION: Only include health data if health bars are enabled
                        if self.show_health_bars:
                            ecosystem_data_all.update({
                                'health': float(self.agent_health.get(agent_id, {'health': 1.0})['health']),
                                'energy': float(self.agent_health.get(agent_id, {'energy': 1.0})['energy'])
                            })
                        
                        basic_all_agent_data = {
                            'id': agent.id,
                            'body': {
                                'x': float(agent.body.position.x),
                                'y': float(agent.body.position.y),
                                'velocity': {
                                    'x': float(agent.body.linearVelocity.x),
                                    'y': float(agent.body.linearVelocity.y)
                                }
                            },
                            'total_reward': float(agent.total_reward),
                            # Basic ecosystem data
                            'ecosystem': ecosystem_data_all
                        }
                        
                        # CRITICAL FIX: Add detailed info for focused agent even if outside viewport
                        if is_focused:
                            try:
                                # Add detailed food info
                                closest_food_info = self._get_closest_food_distance_for_agent(agent)
                                basic_all_agent_data['ecosystem'].update({
                                    'closest_food_distance': float(closest_food_info['distance']),
                                    'closest_food_signed_x_distance': float(closest_food_info.get('signed_x_distance', closest_food_info['distance'])),
                                    'closest_food_type': closest_food_info['food_type'],
                                    'closest_food_source': closest_food_info.get('source_type', 'environment'),
                                    'closest_food_position': [float(closest_food_info['food_position'][0]), float(closest_food_info['food_position'][1])] if closest_food_info.get('food_position') is not None else None
                                })
                                
                                # Add detailed reward and step info 
                                recent_reward = float(getattr(agent, 'last_reward', getattr(agent, 'immediate_reward', 0.0)))
                                basic_all_agent_data.update({
                                    'steps': int(agent.steps) if hasattr(agent, 'steps') else 0,
                                    'recent_reward': recent_reward,
                                    'best_reward': float(getattr(agent, 'best_reward_received', 0.0)),
                                    'worst_reward': float(getattr(agent, 'worst_reward_received', 0.0)),
                                    'learning_approach': str(getattr(agent, 'learning_approach', 'basic_q_learning')),
                                })
                                
                                # Add reward components if available
                                reward_components = getattr(agent, 'reward_components', None)
                                if reward_components:
                                    basic_all_agent_data['reward_components'] = reward_components
                                    
                                # Add arm positions for focused agent (for angle calculations)
                                if hasattr(agent, 'upper_arm') and agent.upper_arm:
                                    basic_all_agent_data['upper_arm'] = {
                                        'x': float(agent.upper_arm.position.x),
                                        'y': float(agent.upper_arm.position.y)
                                    }
                                if hasattr(agent, 'lower_arm') and agent.lower_arm:
                                    basic_all_agent_data['lower_arm'] = {
                                        'x': float(agent.lower_arm.position.x), 
                                        'y': float(agent.lower_arm.position.y)
                                    }
                                    
                            except Exception as e:
                                print(f"⚠️ Error getting detailed info for focused agent {agent_id}: {e}")
                        
                        all_agents_data.append(basic_all_agent_data)
                        
                except Exception as e:
                    print(f"⚠️  Error creating agents data: {e}")
                    agents_data = []
                    all_agents_data = []

                # 6. Get focused agent ID safely
                focused_agent_id = None
                if self.focused_agent and not getattr(self.focused_agent, '_destroyed', False):
                    focused_agent_id = self.focused_agent.id

                # 7. Get ecosystem and environmental data with viewport filtering
                ecosystem_status = self.ecosystem_dynamics.get_ecosystem_status()
                environmental_status = self.environmental_system.get_status()
                
                # Calculate filtered counts for viewport statistics
                all_food_sources = self.ecosystem_dynamics.food_sources
                all_obstacles = self._get_obstacle_data_for_ui()
                
                if viewport_culling:
                    visible_food_sources = self._filter_food_sources_by_viewport(all_food_sources, viewport_bounds)
                    visible_obstacles = self._filter_obstacles_by_viewport(all_obstacles, viewport_bounds)
                else:
                    visible_food_sources = all_food_sources
                    visible_obstacles = all_obstacles
                
                # Calculate viewport culling statistics for frontend debugging
                viewport_culling_stats = {
                    'enabled': viewport_culling,
                    'total_agents': len(current_agents),
                    'visible_agents': len(viewport_agents),
                    'culled_agents': len(current_agents) - len(viewport_agents),
                    'culling_ratio': (len(current_agents) - len(viewport_agents)) / max(1, len(current_agents)),
                    'total_food_sources': len(all_food_sources),
                    'visible_food_sources': len(visible_food_sources),
                    'culled_food_sources': len(all_food_sources) - len(visible_food_sources),
                    'total_obstacles': len(all_obstacles),
                    'visible_obstacles': len(visible_obstacles),
                    'culled_obstacles': len(all_obstacles) - len(visible_obstacles),
                    'viewport_bounds': viewport_bounds if viewport_culling else None
                }
                
                # 8. Get recent predation events for visualization
                recent_predation_events = [
                    {
                        'predator_id': event['predator_id'],
                        'prey_id': event['prey_id'],
                        'position': event['position'],
                        'age': time.time() - event['timestamp'],
                        'success': event['success']
                    }
                    for event in self.predation_events[-10:]  # Last 10 events
                ]

                return {
                    'shapes': {'robots': robot_shapes, 'ground': ground_shapes},
                    'leaderboard': leaderboard_data,
                    'robots': robot_details,
                    'agents': agents_data,  # Viewport-filtered agents for rendering
                    'all_agents': all_agents_data,  # ALL agents for UI panels (not affected by viewport culling)
                    'statistics': self.population_stats,
                    'camera': self.get_camera_state(),
                    'focused_agent_id': focused_agent_id,
                    # Enhanced visualization data
                    'ecosystem': {
                        'status': ecosystem_status,
                        # Territories removed
                        'food_sources': [
                            {
                                'position': f.position,
                                'type': f.food_type,
                                'amount': f.amount,
                                'max_capacity': f.max_capacity
                            }
                            for f in visible_food_sources
                        ],
                        # Dynamic world status
                        'dynamic_world': self.dynamic_world_manager.get_world_status() if self.dynamic_world_manager else None,
                        'predation_events': recent_predation_events,
                        'death_events': [
                            {
                                'agent_id': d['agent_id'],
                                'position': d['position'],
                                'cause': d['cause'],
                                'role': d['role'],
                                'age': current_time - d['timestamp']
                            }
                            for d in self.death_events if current_time - d['timestamp'] < 3.0  # Show deaths for 3 seconds only
                        ],
                        'consumption_events': [
                            {
                                'agent_id': c['agent_id'],
                                'agent_position': c['agent_position'],
                                'food_position': c['food_position'],
                                'food_type': c['food_type'],
                                'energy_gained': c['energy_gained'],
                                'age': current_time - c['timestamp'],
                                'progress': min(1.0, (current_time - c['timestamp']) / c['duration'])  # 0.0 to 1.0 animation progress
                            }
                            for c in self.consumption_events  # Include all active consumption events
                        ],
                        # ray_casting_events removed for performance
                        'survival_stats': {
                            'total_deaths': self.survival_stats['total_deaths'],
                            'deaths_by_starvation': self.survival_stats['deaths_by_starvation'],
                            'average_lifespan': self.survival_stats['average_lifespan'],
                            'current_population': len([a for a in self.agents if not getattr(a, '_destroyed', False)]),
                            'starvation_rate': (self.survival_stats['deaths_by_starvation'] / max(1, self.survival_stats['total_deaths'])) * 100
                        }
                    },
                    'environment': {
                        'status': environmental_status,
                        'obstacles': visible_obstacles
                    },
                    'viewport_culling': viewport_culling_stats,
                    'physics_fps': getattr(self, 'current_physics_fps', 0),
                    'simulation_speed': self.simulation_speed_multiplier,
                    'show_health_bars': self.show_health_bars  # PERFORMANCE: Tell frontend if health data is included
                }
                
                # Track data serialization time for performance analysis
                serialization_time = time.time() - serialization_start
                if hasattr(self, 'performance_timings'):
                    self.performance_timings['data_serialization'].append(serialization_time)
                
            except Exception as e:
                print(f"❌ Critical error in get_status: {e}")
                import traceback
                traceback.print_exc()
                return {'shapes': {}, 'leaderboard': [], 'robots': [], 'agents': [], 'all_agents': [], 'statistics': {}, 'camera': self.get_camera_state(), 'focused_agent_id': None}

    def start(self):
        """Starts the training loop in a separate thread."""
        if not self.is_running:
            print("🔄 Starting training loop thread...")
            
            # Start evaluation services
            if self.enable_evaluation:
                try:
                    # Start MLflow tracking session (only via our direct integration)
                    if self.mlflow_integration:
                        session_name = f"training_session_{int(time.time())}"
                        evolution_config = {
                            'population_size': self.num_agents,
                            'elite_size': self.evolution_config.elite_size,
                            'mutation_rate': self.evolution_config.mutation_rate,
                            'crossover_rate': self.evolution_config.crossover_rate
                        }
                        self.mlflow_integration.start_training_run(
                            run_name=session_name,
                            population_size=self.num_agents,
                            evolution_config=evolution_config
                        )
                        print(f"🔬 Started MLflow run: {session_name}")
                    
                    # Start metrics collector WITHOUT MLflow (we handle MLflow separately)
                    if self.metrics_collector:
                        session_name = f"training_session_{int(time.time())}"
                        evolution_config = {
                            'population_size': self.num_agents,
                            'elite_size': self.evolution_config.elite_size,
                            'mutation_rate': self.evolution_config.mutation_rate,
                            'crossover_rate': self.evolution_config.crossover_rate
                        }
                        # Note: Not calling start_training_session to avoid MLflow conflict
                        print(f"📊 Metrics collector ready for session: {session_name}")
                    
                    if self.dashboard_exporter:
                        self.dashboard_exporter.start()
                        
                    print("📊 Evaluation services started")
                except Exception as e:
                    print(f"⚠️  Error starting evaluation services: {e}")
            
            self.thread = threading.Thread(target=self.training_loop)
            self.thread.daemon = True
            self.thread.start()
            print("✅ Training loop thread started successfully")
        else:
            print("⚠️  Training loop is already running")

    def stop(self):
        """Stops the training loop."""
        print("🛑 Stopping training loop...")
        self.is_running = False
        
        # Stop evaluation services
        if self.enable_evaluation:
            try:
                if self.metrics_collector:
                    final_summary = self.get_training_summary()
                    self.metrics_collector.end_training_session(final_summary)
                
                if self.dashboard_exporter:
                    self.dashboard_exporter.stop()
                    
                print("📊 Evaluation services stopped")
            except Exception as e:
                print(f"⚠️  Error stopping evaluation services: {e}")
        
        # Cleanup dynamic world manager
        if hasattr(self, 'dynamic_world_manager') and self.dynamic_world_manager:
            try:
                self.dynamic_world_manager.cleanup_all_tiles()
            except Exception as e:
                print(f"⚠️ Error cleaning up dynamic world: {e}")
        
        if self.thread:
            self.thread.join()
            print("✅ Training loop stopped")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get final training summary for evaluation."""
        try:
            if self.agents:
                best_agent = max(self.agents, key=lambda a: getattr(a, 'total_reward', 0))
                return {
                    'final_generation': self.evolution_engine.generation,
                    'final_population_size': len(self.agents),
                    'best_final_fitness': getattr(best_agent, 'total_reward', 0),
                    'average_final_fitness': sum(getattr(a, 'total_reward', 0) for a in self.agents) / len(self.agents),
                    'total_training_time': time.time() - getattr(self, 'training_start_time', time.time()),
                    'evolution_summary': self.evolution_engine.get_evolution_summary()
                }
            return {}
        except:
            return {}

    def get_best_agent(self):
        """Utility to get the best agent based on evolutionary fitness."""
        if not self.agents:
            return None
        return max(self.agents, key=lambda agent: agent.get_evolutionary_fitness())
    
    def find_leader(self):
        """Find the robot with the best performance (highest distance traveled) with safety checks."""
        if not self.agents:
            return None
        
        best_distance = -999999
        leader = None
        
        for agent in self.agents:
            # Skip destroyed agents or agents without bodies
            if getattr(agent, '_destroyed', False) or not agent.body:
                continue
                
            try:
                distance = agent.body.position.x - agent.initial_position[0]
                if distance > best_distance:
                    best_distance = distance
                    leader = agent
            except Exception as e:
                print(f"⚠️  Error calculating distance for agent {agent.id}: {e}")
                continue
        
        return leader
    
    

    def trigger_evolution(self):
        """Trigger evolutionary generation advancement with comprehensive safety."""
        # Check if evolution is already in progress
        with self._evolution_lock:
            if self._is_evolving:
                print("⚠️  Evolution already in progress, skipping...")
                return
            self._is_evolving = True
        
        try:
            print(f"\n🧬 === EVOLUTION TRIGGER ===")
            print(f"🔄 Evolving generation {self.evolution_engine.generation} -> {self.evolution_engine.generation + 1}")
            
            # Monitor memory before evolution
            try:
                import psutil
                import os
                process = psutil.Process(os.getpid())
                memory_before = process.memory_info().rss / 1024 / 1024
                print(f"🔍 Memory before evolution: {memory_before:.1f} MB")
            except:
                pass
            
            # Use physics lock to prevent race conditions during evolution
            with self._physics_lock:
                # Store old agents for safe cleanup
                old_agents = self.agents.copy()
                old_agent_ids = {agent.id for agent in old_agents}
                
                evolution_start_time = time.time()
                
                # Clear focus if it will be invalid after evolution
                if self.focused_agent and self.focused_agent in old_agents:
                    self.focused_agent = None
                
                try:
                    # Preserve elite robots BEFORE evolution
                    current_generation = self.evolution_engine.generation
                    elite_preservation_result = self.elite_manager.preserve_generation_elites(
                        old_agents, current_generation
                    )
                    
                    if elite_preservation_result['success']:
                        print(f"🏆 Preserved {elite_preservation_result['elites_preserved']} elite robots from generation {current_generation}")
                    else:
                        print(f"⚠️  Elite preservation failed: {elite_preservation_result.get('error', 'Unknown error')}")
                    
                    # Evolve to next generation (returns tuple of new_population, agents_to_destroy)
                    new_population, agents_to_destroy = self.evolution_engine.evolve_generation()
                    
                    # Update agents list FIRST, before any cleanup
                    self.agents = new_population
                    
                    # Add agents to destruction queue instead of immediate destruction
                    self._agents_pending_destruction.extend(agents_to_destroy)
                    
                    print(f"🧹 Queued {len(agents_to_destroy)} old agents for cleanup")
                    
                except Exception as e:
                    print(f"❌ Error during evolution generation: {e}")
                    # Don't let evolution failure corrupt the system
                    import traceback
                    traceback.print_exc()
                    return
            
            # Clean up robot stats - remove entries for agents that no longer exist (outside physics lock)
            try:
                current_agent_ids = {agent.id for agent in self.agents}
                old_stats_keys = list(self.robot_stats.keys())
                stats_cleaned = 0
                for old_id in old_stats_keys:
                    if old_id not in current_agent_ids:
                        del self.robot_stats[old_id]
                        stats_cleaned += 1
                
                print(f"📊 Cleaned up {stats_cleaned} old stat entries")
            except Exception as e:
                print(f"⚠️  Error cleaning up stats: {e}")
            
            # Update population stats
            try:
                self._update_statistics()
            except Exception as e:
                print(f"⚠️  Error updating statistics: {e}")
            
            evolution_time = time.time() - evolution_start_time
            print(f"✅ Evolution complete! New generation has {len(self.agents)} agents")
            print(f"⏱️  Evolution took {evolution_time:.2f} seconds")
            
            # Monitor memory after evolution
            try:
                import psutil
                import os
                process = psutil.Process(os.getpid())
                memory_after = process.memory_info().rss / 1024 / 1024
                print(f"🔍 Memory after evolution: {memory_after:.1f} MB")
            except:
                pass
                
            print(f"🧬 === EVOLUTION COMPLETE ===\n")
            
        except Exception as e:
            print(f"❌ Evolution failed: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Always release the evolution lock
            with self._evolution_lock:
                self._is_evolving = False

    def get_evolution_status(self):
        """Get current evolution status."""
        return {
            'generation': self.evolution_engine.generation,
            'auto_evolution_enabled': self.auto_evolution_enabled,
            'time_until_next_evolution': max(0, self.evolution_interval - (time.time() - self.last_evolution_time)),
            'evolution_summary': self.evolution_engine.get_evolution_summary()
        }
    
    def toggle_auto_evolution(self):
        """Toggle automatic evolution on/off."""
        self.auto_evolution_enabled = not self.auto_evolution_enabled
        status = "enabled" if self.auto_evolution_enabled else "disabled"
        print(f"🧬 Auto-evolution {status}")
        return self.auto_evolution_enabled

    def spawn_agent(self):
        """Adds a new, random agent to the simulation."""
        spacing = 8 if len(self.agents) > 20 else 15
        position = (len(self.agents) * spacing, 6)
        
        # Create random physical parameters for diversity
        random_params = PhysicalParameters.random_parameters()
        
        new_agent = CrawlingAgent(
            world=self.world,
            agent_id=None,  # Generate new UUID automatically
            position=position,
            category_bits=self.AGENT_CATEGORY,
            mask_bits=self.GROUND_CATEGORY | self.OBSTACLE_CATEGORY,  # Collide with ground AND obstacles
            physical_params=random_params
        )
        self.agents.append(new_agent)
        
        # Register new agent with reward signal adapter
        if hasattr(self, 'reward_signal_adapter') and self.reward_signal_adapter:
            try:
                agent_type = getattr(new_agent, 'learning_approach', 'evolutionary')
                self.reward_signal_adapter.register_agent(
                    new_agent.id,
                    agent_type,
                    metadata={
                        'physical_params': str(new_agent.physical_params) if hasattr(new_agent, 'physical_params') else None,
                        'created_at': time.time(),
                        'source': 'spawned'
                    }
                )
            except Exception as e:
                print(f"⚠️ Failed to register spawned agent {new_agent.id} with reward signal adapter: {e}")
        
        print(f"🐣 Spawned new agent {new_agent.id} with random parameters. Total agents: {len(self.agents)}")

    def clone_best_agent(self):
        """Clones the best performing agent using evolutionary methods."""
        best_agent = self.evolution_engine.get_best_agent()
        if not best_agent:
            print("No agents to clone.")
            return

        spacing = 8 if len(self.agents) > 20 else 15
        position = (len(self.agents) * spacing, 6)

        # Use the evolutionary clone method with slight mutation
        cloned_agent = best_agent.clone_with_mutation(mutation_rate=0.05)
        
        # Update position for the clone (ID is already generated)
        cloned_agent.initial_position = position
        cloned_agent.reset_position()
        
        self.agents.append(cloned_agent)
        print(f"👯 Cloned best agent {best_agent.id} to new agent {cloned_agent.id} (fitness: {best_agent.get_evolutionary_fitness():.3f}). Total agents: {len(self.agents)}")

    def evolve_population(self):
        """Legacy method - use trigger_evolution() instead."""
        print("⚠️  evolve_population() is deprecated. Using trigger_evolution() instead.")
        self.trigger_evolution()

    def _init_robot_stats(self):
        """Initialize robot statistics with safety checks."""
        self.robot_stats = {}
        for agent in self.agents:
            # Skip destroyed agents or agents without bodies
            if getattr(agent, '_destroyed', False) or not agent.body:
                continue
                
            try:
                current_position = tuple(agent.body.position) if agent.body else (0, 0)
                self.robot_stats[agent.id] = {
                    'id': agent.id,
                    'initial_position': tuple(agent.initial_position),
                    'current_position': current_position,
                    'total_distance': 0,
                    'velocity': (0, 0),
                    'arm_angles': {'shoulder': 0, 'elbow': 0},
                    'fitness': 0,
                    'steps_alive': 0,
                    'last_position': current_position,
                    'steps_tilted': 0,  # Track how long robot has been tilted
                    'episode_reward': 0,
                    'q_updates': 0,
                    'action_history': [],  # Track last actions taken
                    'food_consumed': 0.0  # Track total food consumed for leaderboard
                }
            except Exception as e:
                print(f"⚠️  Error initializing stats for agent {agent.id}: {e}")
            
    def update_camera(self, delta_time):
        """Smoothly moves the camera towards the focused agent with safety checks."""
        if (self.focused_agent and 
            not getattr(self.focused_agent, '_destroyed', False) and 
            self.focused_agent.body):
            try:
                self.camera_target = self.focused_agent.body.position
            except Exception as e:
                print(f"⚠️  Error getting focused agent position: {e}")
                self.focused_agent = None  # Clear invalid focus
                # Keep camera at current position instead of resetting to origin
                self.camera_target = self.camera_position
        else:
            # If no agent is focused, keep camera at current position instead of returning to origin
            if self.focused_agent and (getattr(self.focused_agent, '_destroyed', False) or not self.focused_agent.body):
                self.focused_agent = None  # Clear invalid focus
            # Don't reset to (0, 0) - keep current position when focus is cleared
            self.camera_target = self.camera_position
            
        # Smoothly interpolate camera position and zoom
        self.camera_position = (
            self.camera_position[0] + (self.camera_target[0] - self.camera_position[0]) * self.follow_speed,
            self.camera_position[1] + (self.camera_target[1] - self.camera_position[1]) * self.follow_speed
        )
        self.camera_zoom += (self.target_zoom - self.camera_zoom) * self.zoom_speed

    def focus_on_agent(self, agent):
        """Sets the given agent as the camera focus with safety checks."""
        with self._physics_lock:
            if agent and agent in self.agents and not getattr(agent, '_destroyed', False):
                self.focused_agent = agent
                print(f"🎯 SERVER: Focusing on agent {agent.id}")
                # Only set zoom if user hasn't manually adjusted it
                if not self.user_has_manually_zoomed:
                    self.user_zoom_level = 1.5
                    self._zoom_override = 1.5  # Send zoom override to frontend
                # If user has manually zoomed, don't override their preference
            else:
                self.focused_agent = None
                print("🎯 SERVER: Camera focus cleared.")
                # Don't change zoom when just clearing focus

    def get_agent_at_position(self, world_x, world_y):
        """Finds an agent at a given world coordinate with safety checks."""
        with self._physics_lock:
            try:
                for agent in self.agents:
                    if getattr(agent, '_destroyed', False) or not agent.body:
                        continue
                        
                    # Check if click is near the agent's body
                    agent_pos = agent.body.position
                    distance = ((world_x - agent_pos.x) ** 2 + (world_y - agent_pos.y) ** 2) ** 0.5
                    if distance < 2.0:  # Click radius
                        return agent
                return None
            except Exception as e:
                print(f"❌ Error finding agent at position: {e}")
                return None

    def move_agent(self, agent_id, x, y):
        """Move an agent to the specified world coordinates with enhanced safety."""
        with self._physics_lock:
            try:
                agent = next((a for a in self.agents if str(a.id) == str(agent_id) and not getattr(a, '_destroyed', False)), None)
                if not agent or not agent.body:
                    print(f"❌ Agent {agent_id} not found or destroyed for moving")
                    return False
                
                # Set the agent's position
                agent.body.position = (x, y)
                
                # Reset velocity to prevent physics issues
                agent.body.linearVelocity = (0, 0)
                agent.body.angularVelocity = 0
                
                print(f"🤖 Moved agent {agent_id} to ({x:.2f}, {y:.2f})")
                return True
                
            except Exception as e:
                print(f"❌ Error moving agent {agent_id}: {e}")
                return False

    def handle_click(self, screen_x, screen_y, canvas_width, canvas_height):
        """Handles a click event from the frontend."""
        data = request.get_json()
        agent_id = data.get('agent_id')
        print(f"🖱️ SERVER: Received click for agent_id: {agent_id} (type: {type(agent_id)})")

        if agent_id is not None:
            # DEBUG: Log actual agent IDs to see the type mismatch
            print(f"🔍 Available agent IDs: {[(agent.id, type(agent.id)) for agent in self.agents[:3]]}")
            
            # Find the agent by ID - convert both to strings for comparison
            agent_to_focus = next((agent for agent in self.agents if str(agent.id) == str(agent_id)), None)
            if agent_to_focus:
                self.focus_on_agent(agent_to_focus)
                return jsonify({'status': 'success', 'message': f'Focused on agent {agent_id}', 'agent_id': agent_id})
            else:
                self.focus_on_agent(None) # Clear focus if agent not found
                return jsonify({'status': 'error', 'message': f'Agent {agent_id} not found', 'agent_id': None})
        else:
            # Clear focus if no agent_id provided
            env.focus_on_agent(None)
            return jsonify({'status': 'success', 'message': 'Focus cleared', 'agent_id': None})

    def get_camera_state(self):
        """Get current camera state for rendering with safety checks."""
        # Get focused agent ID safely
        focused_agent_id = None
        if (self.focused_agent and 
            not getattr(self.focused_agent, '_destroyed', False) and 
            hasattr(self.focused_agent, 'id')):
            focused_agent_id = self.focused_agent.id
        
        return {
            'position': self.camera_position,
            'zoom': self.camera_zoom,
            'focused_agent_id': focused_agent_id,
            'zoom_override': getattr(self, '_zoom_override', None)  # Only send zoom when we want to override
        }

    def _calculate_viewport_bounds(self, canvas_width=1200, canvas_height=800, camera_x=None, camera_y=None):
        """Calculate the world-space bounds of the current viewport using ACTUAL frontend zoom level and camera position."""
        # Get camera parameters - use provided camera position if available (from frontend)
        if camera_x is not None and camera_y is not None:
            cam_x, cam_y = camera_x, camera_y  # Use frontend camera position
        else:
            cam_x, cam_y = self.camera_position  # Fallback to backend camera position
        # CRITICAL FIX: Use actual frontend zoom level, not backend camera zoom
        zoom = getattr(self, 'user_zoom_level', 1.0)  # Use the actual frontend zoom level
        
        # Calculate half-dimensions of the viewport in world units
        half_width_world = (canvas_width / 2) / zoom
        half_height_world = (canvas_height / 2) / zoom
        
        # Calculate viewport bounds in world coordinates
        viewport_bounds = {
            'left': cam_x - half_width_world,
            'right': cam_x + half_width_world,
            'bottom': cam_y - half_height_world,
            'top': cam_y + half_height_world
        }
        
        # Add a small margin for objects partially visible
        margin = max(5.0, min(half_width_world, half_height_world) * 0.1)
        viewport_bounds['left'] -= margin
        viewport_bounds['right'] += margin
        viewport_bounds['bottom'] -= margin
        viewport_bounds['top'] += margin
        
        return viewport_bounds

    def _is_object_in_viewport(self, position, size, viewport_bounds):
        """Check if an object is visible within the viewport bounds."""
        x, y = position
        
        # Object bounds
        half_size = size / 2
        obj_left = x - half_size
        obj_right = x + half_size
        obj_bottom = y - half_size
        obj_top = y + half_size
        
        # Check if object overlaps with viewport
        return not (obj_right < viewport_bounds['left'] or 
                   obj_left > viewport_bounds['right'] or
                   obj_top < viewport_bounds['bottom'] or 
                   obj_bottom > viewport_bounds['top'])

    def _filter_agents_by_viewport(self, agents, viewport_bounds):
        """Filter agents to only include those visible in the viewport."""
        visible_agents = []
        
        for agent in agents:
            if getattr(agent, '_destroyed', False) or not agent.body:
                continue
                
            # Agent position and approximate size
            position = (agent.body.position.x, agent.body.position.y)
            agent_size = 4.0  # Approximate robot size including arms
            
            if self._is_object_in_viewport(position, agent_size, viewport_bounds):
                visible_agents.append(agent)
                
        return visible_agents

    def _filter_food_sources_by_viewport(self, food_sources, viewport_bounds):
        """Filter food sources to only include those visible in the viewport."""
        visible_food = []
        
        for food in food_sources:
            position = food.position
            food_size = 3.0  # Approximate food source size
            
            if self._is_object_in_viewport(position, food_size, viewport_bounds):
                visible_food.append(food)
                
        return visible_food

    def _filter_obstacles_by_viewport(self, obstacles, viewport_bounds):
        """Filter obstacles to only include those visible in the viewport."""
        visible_obstacles = []
        
        for obstacle in obstacles:
            position = obstacle.get('position', (0, 0))
            obstacle_size = obstacle.get('size', 2.0)
            
            if self._is_object_in_viewport(position, obstacle_size, viewport_bounds):
                visible_obstacles.append(obstacle)
                
        return visible_obstacles

    def update_user_zoom(self, zoom_level):
        """Update the user's zoom level preference."""
        self.user_zoom_level = max(0.01, min(20, zoom_level))  # Clamp to reasonable bounds
        self.user_has_manually_zoomed = True
        # Don't update target_zoom - let frontend handle zoom locally
        print(f"🔍 SERVER: User zoom updated to {self.user_zoom_level:.2f}")
    
    def reset_user_zoom(self):
        """Reset user zoom preferences (called by Reset View)."""
        self.user_zoom_level = 1.0
        self.user_has_manually_zoomed = False
        self._zoom_override = 1.0  # Send reset zoom to frontend
        print("🔍 SERVER: User zoom preferences reset")
    
    def reset_camera_position(self):
        """Reset camera position to origin (called by Reset View)."""
        self.camera_position = (0, 0)
        self.camera_target = (0, 0)
        print("🎯 SERVER: Camera position reset to origin")
    
    def clear_zoom_override(self):
        """Clear the zoom override flag after it's been sent."""
        if hasattr(self, '_zoom_override'):
            delattr(self, '_zoom_override')

    def _get_agent_learning_approach_name(self, agent_id: str) -> str:
        """Get the learning approach name for an agent."""
        agent = next((a for a in self.agents if str(a.id) == str(agent_id) and not getattr(a, '_destroyed', False)), None)
        if agent:
            return getattr(agent, 'learning_approach', 'basic_q_learning')
        return 'basic_q_learning'
    
    def _get_closest_food_distance_for_agent(self, agent) -> Dict[str, Any]:
        """
        Get distance to closest food for a specific agent based on their ecosystem role.
        Similar to SurvivalStateProcessor._find_nearest_food but returns just distance info.
        """
        try:
            if getattr(agent, '_destroyed', False) or not agent.body:
                return {'distance': 999999, 'food_type': 'unknown', 'source_type': 'none', 'food_position': None, 'signed_x_distance': 999999}
            
            agent_pos = (agent.body.position.x, agent.body.position.y)
            agent_id = str(agent.id)
            
            # Get agent's ecosystem role
            agent_status = self.agent_statuses.get(agent_id, {})
            agent_role = agent_status.get('role', 'omnivore')
            
            # Get all available food sources
            food_sources = self.ecosystem_dynamics.food_sources
            potential_food_sources = []
            
            # Add environmental food sources ONLY for non-carnivore and non-scavenger agents
            if agent_role not in ['carnivore', 'scavenger']:  # CARNIVORES AND SCAVENGERS ARE PURE ROBOT CONSUMERS - NO environmental food
                for food in food_sources:
                    if food.amount > 0.1:  # Only consider food that can actually be consumed
                        food_pos = food.position
                        distance = ((agent_pos[0] - food_pos[0])**2 + (agent_pos[1] - food_pos[1])**2)**0.5
                        potential_food_sources.append({
                            'position': food.position,
                            'type': food.food_type,
                            'source': 'environment',
                            'amount': food.amount,
                            'distance': distance
                        })
            
            # For carnivores and scavengers, add other agents as potential prey
            if agent_role in ['carnivore', 'scavenger']:
                for other_agent in self.agents:
                    if (getattr(other_agent, '_destroyed', False) or not other_agent.body or 
                        other_agent.id == agent.id):
                        continue
                    
                    other_pos = (other_agent.body.position.x, other_agent.body.position.y)
                    other_energy = self.agent_energy_levels.get(other_agent.id, 1.0)
                    other_health = self.agent_health.get(other_agent.id, {'health': 1.0})['health']
                    
                    # SCAVENGER RESTRICTION: Only target robots with energy < 0.3 (any robot type when weakened)
                    if agent_role == 'scavenger' and other_energy >= 0.3:
                        continue  # Scavengers can only target weakened robots
                    
                    # CARNIVORE RESTRICTIONS: Can ONLY hunt herbivore, scavenger, and omnivore robots
                    other_role = self.agent_statuses.get(other_agent.id, {}).get('role', 'omnivore')
                    if agent_role == 'carnivore':
                        valid_prey_roles = ['herbivore', 'scavenger', 'omnivore']  # Match ecosystem_dynamics.py rules
                        if other_role not in valid_prey_roles:
                            continue  # Carnivores can only hunt specific prey types
                        if other_health <= 0.1:
                            continue  # Don't target dying robots
                    
                    # Calculate distance for all potential prey
                    distance = ((agent_pos[0] - other_pos[0])**2 + (agent_pos[1] - other_pos[1])**2)**0.5
                    
                    potential_food_sources.append({
                        'position': other_pos,
                        'type': 'robot',  # Other agents are robots, not "meat"
                        'source': 'prey',
                        'prey_id': other_agent.id,
                        'prey_energy': other_energy,
                        'prey_health': other_health,
                        'distance': distance
                    })
            
            if not potential_food_sources:
                # ROLE-SPECIFIC MESSAGES: Differentiate between no food and no valid targets
                if agent_role == 'carnivore':
                    return {'distance': 999999, 'food_type': 'no valid prey (herbivore/scavenger/omnivore) found', 'source_type': 'prey', 'food_position': None, 'signed_x_distance': 999999}
                elif agent_role == 'scavenger':
                    return {'distance': 999999, 'food_type': 'no weakened robots (energy < 30%) found', 'source_type': 'prey', 'food_position': None, 'signed_x_distance': 999999}
                else:
                    return {'distance': 999999, 'food_type': 'no environmental food sources found', 'source_type': 'environment', 'food_position': None, 'signed_x_distance': 999999}
            
            # Find the nearest food source for this agent type
            best_target = None
            best_distance = 999999
            
            for target in potential_food_sources:
                distance = target['distance']
                
                if distance < best_distance:
                    best_distance = distance
                    best_target = target
            
            if best_target is None:
                return {'distance': 999999, 'food_type': 'none found', 'source_type': 'none', 'food_position': None, 'signed_x_distance': 999999}
            
            # Calculate signed x-axis distance (positive = right, negative = left)
            target_pos = best_target['position']
            signed_x_distance = target_pos[0] - agent_pos[0]
            
            # Determine food type description based on target
            if best_target.get('source') == 'prey':
                prey_id = best_target.get('prey_id', 'unknown')
                prey_energy = best_target.get('prey_energy', 0.0)
                food_type_desc = f"robot prey {str(prey_id)[:8]} (energy: {prey_energy:.2f})"
            else:
                food_type_desc = best_target.get('type', 'unknown')
            
            return {
                'distance': best_distance,
                'signed_x_distance': signed_x_distance,  # New: signed x-axis distance
                'food_type': food_type_desc,
                'source_type': best_target.get('source', 'environment'),
                'food_position': target_pos  # New: position for line drawing
            }
            
        except Exception as e:
            print(f"⚠️ Error calculating closest food distance for agent {getattr(agent, 'id', 'unknown')}: {e}")
            return {'distance': 999999, 'food_type': 'error calculating', 'food_position': None, 'signed_x_distance': 999999}

    def _create_obstacle_physics_bodies(self):
        """Create Box2D physics bodies for obstacles that don't have them yet. 
        NOTE: This is kept for backward compatibility but static world generation is now preferred."""
        try:
            if not hasattr(self, 'obstacle_bodies'):
                self.obstacle_bodies = {}  # Track obstacle ID -> Box2D body mapping
            
            # MODIFIED: Reduced dynamic obstacle creation since we now use static world generation
            # Only create physics bodies for any remaining dynamic obstacles (minimal)
            obstacles_to_create = []
            
            # NOTE: Environmental system and evolution engine obstacle spawning has been disabled
            # Static obstacles are created once during initialization via _generate_static_world()
            
            # Only process existing moving obstacles or special dynamic obstacles if any exist
            if hasattr(self, 'environmental_system') and self.environmental_system.obstacles:
                for i, obstacle in enumerate(self.environmental_system.obstacles):
                    # Only create physics bodies for moving obstacles that weren't created statically
                    if hasattr(obstacle, 'movement_pattern') and obstacle.movement_pattern:
                        obstacle_id = f"moving_{i}_{obstacle.type.value if hasattr(obstacle, 'type') else 'unknown'}"
                        if obstacle_id not in self.obstacle_bodies:
                            obstacles_to_create.append({
                                'id': obstacle_id,
                                'type': obstacle.type.value if hasattr(obstacle, 'type') else 'boulder',
                                'position': obstacle.position,
                                'size': obstacle.size,
                                'source': 'environmental_moving'
                            })
            
            # Create physics bodies for any remaining dynamic obstacles
            bodies_created = 0
            for obstacle_data in obstacles_to_create:
                try:
                    body = self._create_single_obstacle_body(obstacle_data)
                    if body:
                        self.obstacle_bodies[obstacle_data['id']] = body
                        bodies_created += 1
                except Exception as e:
                    print(f"⚠️ Error creating physics body for dynamic obstacle {obstacle_data['id']}: {e}")
            
            if bodies_created > 0:
                print(f"🏃 Created {bodies_created} dynamic obstacle physics bodies. Total active: {len(self.obstacle_bodies)}")
            
            # Clean up bodies for obstacles that no longer exist
            self._cleanup_removed_obstacles()
            
        except Exception as e:
            print(f"⚠️ Error in obstacle physics body creation: {e}")
    
    def _create_single_obstacle_body(self, obstacle_data):
        """Create a Box2D physics body for a single obstacle."""
        try:
            obstacle_type = obstacle_data['type']
            position = obstacle_data['position']
            size = obstacle_data.get('size', 2.0)
            
            # Create static body for obstacle
            obstacle_body = self.world.CreateStaticBody(position=position)
            
            # Choose shape based on obstacle type
            if obstacle_type in ['boulder', 'wall']:
                # Rectangular obstacles
                width = size if obstacle_type == 'boulder' else min(size, 1.0)  # Walls are thinner
                height = size if obstacle_type == 'boulder' else max(size, 3.0)  # Walls are taller
                
                fixture = obstacle_body.CreateFixture(
                    shape=b2.b2PolygonShape(box=(width/2, height/2)),
                    density=0.0,  # Static body
                    friction=0.7,
                    restitution=0.2,
                                    filter=b2.b2Filter(
                    categoryBits=self.OBSTACLE_CATEGORY,
                    maskBits=self.AGENT_CATEGORY  # ONLY collide with agents, NOT other obstacles (performance optimization)
                )
                )
                
            elif obstacle_type == 'pit':
                # Create pit as a low rectangular obstacle
                fixture = obstacle_body.CreateFixture(
                    shape=b2.b2PolygonShape(box=(size/2, 0.5)),  # Low height for pit
                    density=0.0,
                    friction=0.3,  # Slippery
                    restitution=0.0,
                    filter=b2.b2Filter(
                        categoryBits=self.OBSTACLE_CATEGORY,
                        maskBits=self.AGENT_CATEGORY
                    )
                )
                
            else:
                # Default: circular obstacle for other types
                fixture = obstacle_body.CreateFixture(
                    shape=b2.b2CircleShape(radius=size/2),
                    density=0.0,
                    friction=0.5,
                    restitution=0.3,
                    filter=b2.b2Filter(
                        categoryBits=self.OBSTACLE_CATEGORY,
                        maskBits=self.AGENT_CATEGORY
                    )
                )
            
            # Store obstacle type on the body for identification
            obstacle_body.userData = {
                'type': 'obstacle',
                'obstacle_type': obstacle_type,
                'obstacle_id': obstacle_data['id']
            }
            
            return obstacle_body
            
        except Exception as e:
            print(f"⚠️ Error creating obstacle body: {e}")
            return None
    
    def _cleanup_removed_obstacles(self):
        """Remove physics bodies for obstacles that no longer exist."""
        try:
            if not hasattr(self, 'obstacle_bodies'):
                return
            
            # Get current obstacle IDs
            current_obstacle_ids = set()
            
            # Environmental obstacles
            if hasattr(self, 'environmental_system') and self.environmental_system.obstacles:
                for i, obstacle in enumerate(self.environmental_system.obstacles):
                    current_obstacle_ids.add(f"env_{i}_{obstacle['type']}")
            
            # Evolution obstacles
            if hasattr(self, 'evolution_engine') and hasattr(self.evolution_engine, 'environment_obstacles'):
                for i, obstacle in enumerate(self.evolution_engine.environment_obstacles):
                    if obstacle.get('active', True):
                        current_obstacle_ids.add(f"evo_{i}_{obstacle['type']}")
            
            # Remove bodies for obstacles that no longer exist
            bodies_to_remove = []
            for obstacle_id, body in self.obstacle_bodies.items():
                if obstacle_id not in current_obstacle_ids:
                    bodies_to_remove.append(obstacle_id)
            
            removed_count = 0
            for obstacle_id in bodies_to_remove:
                try:
                    body = self.obstacle_bodies.pop(obstacle_id)
                    self.world.DestroyBody(body)
                    removed_count += 1
                except Exception as e:
                    print(f"⚠️ Error removing obstacle body {obstacle_id}: {e}")
            
            if removed_count > 0:
                print(f"🗑️ Removed {removed_count} obsolete obstacle bodies")
                
        except Exception as e:
            print(f"⚠️ Error cleaning up obstacle bodies: {e}")
    
    def _get_obstacle_data_for_ui(self):
        """Get obstacle data for the web UI visualization."""
        try:
            obstacles_for_ui = []
            
            # Add terrain segments from the generated terrain
            if hasattr(self, 'terrain_collision_bodies'):
                for terrain_body in self.terrain_collision_bodies:
                    terrain_type = terrain_body.get('type', 'terrain_segment')
                    position = terrain_body.get('position', (0, 0))
                    size = terrain_body.get('size', 2.0)
                    height = terrain_body.get('height', size)
                    
                    # Convert position tuple to array for JavaScript
                    position_array = [position[0], position[1]]
                    
                    # Add terrain to UI data
                    obstacles_for_ui.append({
                        'type': terrain_type,
                        'position': position_array,
                        'size': size,
                        'height': height,
                        'source': 'terrain_generation',
                        'danger_level': 0.2,  # Terrain is natural, less dangerous
                        'active': True
                    })
            
            # 🌍 ADD DYNAMIC WORLD TERRAIN with biome colors
            if hasattr(self, 'dynamic_world_manager') and self.dynamic_world_manager:
                try:
                    # Get all active tiles from dynamic world
                    for tile in self.dynamic_world_manager.tiles.values():
                        if tile.active:
                            biome_config = self.dynamic_world_manager._get_biome_config(tile.biome)
                            biome_color = biome_config.get('color', (0.5, 0.5, 0.5))  # Default gray
                            
                            # Add ground bodies from this tile
                            for ground_body in tile.ground_bodies:
                                if hasattr(ground_body, 'position'):
                                    position_array = [ground_body.position.x, ground_body.position.y]
                                    
                                    # Determine size from fixtures
                                    size = 4.0  # Default
                                    if ground_body.fixtures:
                                        fixture = ground_body.fixtures[0]
                                        shape = fixture.shape
                                        if hasattr(shape, 'vertices'):
                                            vertices = shape.vertices
                                            if vertices:
                                                x_coords = [v[0] for v in vertices]
                                                y_coords = [v[1] for v in vertices]
                                                size = max(max(x_coords) - min(x_coords), max(y_coords) - min(y_coords))
                                    
                                    terrain_data = {
                                        'type': ground_body.userData.get('type', 'dynamic_terrain') if hasattr(ground_body, 'userData') and ground_body.userData else 'dynamic_terrain',
                                        'position': position_array,
                                        'size': size,
                                        'height': size,
                                        'source': 'dynamic_world',
                                        'danger_level': 0.1,  # Ground is safe
                                        'active': True,
                                        'biome_color': list(biome_color),  # Convert tuple to list for JSON
                                        'biome_type': tile.biome.value,
                                        'is_dynamic_world': True
                                    }
                                    
                                    obstacles_for_ui.append(terrain_data)
                                
                            # 🎯 ADD ACTUAL DYNAMIC WORLD OBSTACLES from tile.obstacle_bodies
                            for obstacle_body in tile.obstacle_bodies:
                                try:
                                    if obstacle_body and hasattr(obstacle_body, 'userData') and obstacle_body.userData:
                                        obstacle_type = obstacle_body.userData.get('type', 'dynamic_obstacle')
                                        position = (obstacle_body.position.x, obstacle_body.position.y)
                                        
                                        # Get size from fixture
                                        size = 2.0  # Default
                                        if obstacle_body.fixtures:
                                            fixture = obstacle_body.fixtures[0]
                                            shape = fixture.shape
                                            if hasattr(shape, 'radius'):  # Circle
                                                size = shape.radius * 2
                                            elif hasattr(shape, 'vertices'):  # Polygon box
                                                # Get the box half-extents and convert to full size
                                                vertices = shape.vertices
                                                if vertices and len(vertices) >= 2:
                                                    width = abs(vertices[1][0] - vertices[0][0])
                                                    height = abs(vertices[2][1] - vertices[1][1]) if len(vertices) >= 3 else width
                                                    size = max(width, height)
                                        
                                        # Convert position tuple to array for JavaScript
                                        position_array = [position[0], position[1]]
                                        
                                        # Get biome data
                                        biome_color = obstacle_body.userData.get('color', None)
                                        biome_type = obstacle_body.userData.get('biome', None)
                                        
                                        obstacle_data = {
                                            'type': obstacle_type,
                                            'position': position_array,
                                            'size': size,
                                            'height': size,  # Height for 3D rendering
                                            'source': 'dynamic_world',
                                            'danger_level': 0.4,  # Medium difficulty for small obstacles
                                            'active': True,
                                            'biome_color': list(biome_color) if biome_color else None,  # Convert tuple to list for JSON
                                            'biome_type': biome_type,
                                            'is_dynamic_world': True,
                                            'tile_id': obstacle_body.userData.get('tile_id', None)
                                        }
                                        
                                        obstacles_for_ui.append(obstacle_data)
                                        
                                except Exception as obstacle_error:
                                    print(f"⚠️ Error processing dynamic obstacle: {obstacle_error}")
                                    continue
                                    
                except Exception as e:
                    # Don't fail if dynamic world isn't available
                    pass
            
            # Add obstacles that have physics bodies (including any remaining dynamic ones)
            if hasattr(self, 'obstacle_bodies'):
                for obstacle_id, body in self.obstacle_bodies.items():
                    if body and hasattr(body, 'userData') and body.userData:
                        obstacle_type = body.userData.get('obstacle_type', 'unknown')
                        position = (body.position.x, body.position.y)
                        
                        # 🎨 EXTRACT BIOME COLOR from dynamic world obstacles
                        biome_color = body.userData.get('color', None)
                        biome_type = body.userData.get('biome', None)
                        is_dynamic_world = body.userData.get('type') == 'dynamic_obstacle'
                        
                        # Determine size based on fixtures
                        size = 2.0  # Default
                        if body.fixtures:
                            fixture = body.fixtures[0]
                            shape = fixture.shape
                            if hasattr(shape, 'radius'):  # Circle
                                size = shape.radius * 2
                            elif hasattr(shape, 'vertices'):  # Polygon
                                # Approximate size from polygon bounds
                                vertices = shape.vertices
                                if vertices:
                                    x_coords = [v[0] for v in vertices]
                                    y_coords = [v[1] for v in vertices]
                                    size = max(max(x_coords) - min(x_coords), max(y_coords) - min(y_coords))
                        
                        # Convert position tuple to array for JavaScript
                        position_array = [position[0], position[1]]
                        
                        # Add danger level based on obstacle type
                        danger_level = 0.5  # Default
                        if obstacle_type == 'pit':
                            danger_level = 0.8  # High danger
                        elif obstacle_type == 'wall':
                            danger_level = 0.3  # Low danger
                        elif obstacle_type == 'boulder':
                            danger_level = 0.6  # Medium danger
                        
                        obstacle_data = {
                            'id': obstacle_id,
                            'type': obstacle_type,
                            'position': position_array,  # JavaScript expects array [x, y]
                            'size': size,
                            'danger_level': danger_level,  # JavaScript expects this for coloring
                            'active': True
                        }
                        
                        # 🌍 ADD BIOME COLOR INFORMATION for dynamic world obstacles
                        if is_dynamic_world and biome_color:
                            obstacle_data['biome_color'] = list(biome_color)  # Convert tuple to list for JSON
                            obstacle_data['biome_type'] = biome_type
                            obstacle_data['is_dynamic_world'] = True
                        
                        obstacles_for_ui.append(obstacle_data)
            
            return obstacles_for_ui
            
        except Exception as e:
            print(f"⚠️ Error getting obstacle data for UI: {e}")
            return []

    def _cleanup_performance_data(self):
        """Clean up accumulated performance data to prevent memory growth."""
        try:
            # DETAILED LOGGING: Track what's accumulating
            total_action_history = 0
            total_replay_buffer = 0
            total_attention_history = 0
            agents_with_large_buffers = 0
            
            # Clean up Q-learning history data
            for agent in self.agents:
                if getattr(agent, '_destroyed', False):
                    continue
                    
                # Track action history sizes
                if hasattr(agent, 'action_history'):
                    action_size = len(agent.action_history)
                    total_action_history += action_size
                    if action_size > 50:
                        agent.action_history = agent.action_history[-50:]
                
                # Track replay buffer sizes
                if (hasattr(agent, '_learning_system') and agent._learning_system and 
                    hasattr(agent._learning_system, 'memory') and 
                    hasattr(agent._learning_system.memory, 'buffer')):
                    buffer_size = len(agent._learning_system.memory.buffer)
                    total_replay_buffer += buffer_size
                    if buffer_size > 1000:
                        agents_with_large_buffers += 1
                
                # Track attention history sizes
                if (hasattr(agent, '_learning_system') and agent._learning_system and
                    hasattr(agent._learning_system, 'attention_history')):
                    attention_size = len(agent._learning_system.attention_history)
                    total_attention_history += attention_size
                
                
                # Clean up replay buffer if too large (FIXED: correct path)
                if (hasattr(agent, '_learning_system') and agent._learning_system and 
                    hasattr(agent._learning_system, 'memory') and 
                    hasattr(agent._learning_system.memory, 'buffer')):
                    buffer = agent._learning_system.memory.buffer
                    buffer_capacity = getattr(agent._learning_system.memory, 'maxlen', 3000)
                    if len(buffer) > buffer_capacity * 0.5:  # More aggressive - clean at 50%
                        # Remove oldest 50% of experiences more aggressively
                        old_size = len(buffer)
                        remove_count = old_size // 2
                        for _ in range(remove_count):
                            if buffer:
                                buffer.popleft()
            
            # Clean up old robot stats for destroyed agents
            active_agent_ids = {agent.id for agent in self.agents if not getattr(agent, '_destroyed', False)}
            old_stats_keys = [k for k in self.robot_stats.keys() if k not in active_agent_ids]
            for old_key in old_stats_keys[:10]:  # Remove up to 10 old entries at a time
                del self.robot_stats[old_key]
            
            # Clean up ecosystem data
            if hasattr(self.ecosystem_dynamics, 'food_sources'):
                # Remove depleted food sources
                self.ecosystem_dynamics.food_sources = [
                    f for f in self.ecosystem_dynamics.food_sources if f.amount > 0.05
                ]
            
            # Memory pool handles its own cleanup automatically
            
            # DETAILED MEMORY TRACKING LOGS
            print(f"🧹 MEMORY TRACKING - Performance cleanup completed:")
            print(f"   📊 Active agents: {len(self.agents)}")
            print(f"   📊 Robot stats entries: {len(self.robot_stats)}")
            print(f"   📊 Total action history entries: {total_action_history}")
            print(f"   📊 Total replay buffer entries: {total_replay_buffer}")
            print(f"   📊 Total attention history entries: {total_attention_history}")
            print(f"   📊 Agents with large buffers (>1k): {agents_with_large_buffers}")
            print(f"   📊 Avg replay buffer per agent: {total_replay_buffer / max(1, len(self.agents)):.1f}")
            print(f"   📊 Avg attention per agent: {total_attention_history / max(1, len(self.agents)):.1f}")
            
            # AGGRESSIVE: Clean up attention network specific data
            attention_networks_cleaned = 0
            for agent in self.agents:
                if getattr(agent, '_destroyed', False):
                    continue
                
                # Clean up attention networks (FIXED: correct path)
                if hasattr(agent, '_learning_system') and agent._learning_system:
                    learning_system = agent._learning_system
                    
                    # Clean attention history aggressively
                    if hasattr(learning_system, 'attention_history'):
                        old_size = len(learning_system.attention_history)
                        if old_size > 10:  # Keep only 10 most recent (was 25)
                            learning_system.attention_history = deque(
                                list(learning_system.attention_history)[-10:], 
                                maxlen=25  # Reduced from 50
                            )
                            attention_networks_cleaned += 1
                    
                    # ULTRA AGGRESSIVE: Clean experience replay buffer for performance
                    if hasattr(learning_system, 'memory') and hasattr(learning_system.memory, 'buffer'):
                        buffer_size = len(learning_system.memory.buffer)
                        if buffer_size > 500:  # PERFORMANCE FIX: Clean at 500 entries (was 1000)
                            # Keep only most recent 500 experiences
                            learning_system.memory.buffer = deque(
                                list(learning_system.memory.buffer)[-500:],
                                maxlen=1000  # PERFORMANCE FIX: Reduced max from 3000 to 1000
                            )
            
            if attention_networks_cleaned > 0:
                print(f"🧹 Aggressively cleaned {attention_networks_cleaned} attention networks")
            
        except Exception as e:
            print(f"⚠️ Error during performance cleanup: {e}")



    def _cleanup_attention_networks(self):
        """Dedicated cleanup for attention networks to prevent memory accumulation."""
        try:
            cleaned_count = 0
            total_attention_records = 0
            total_buffer_size = 0
            
            for agent in self.agents:
                if getattr(agent, '_destroyed', False):
                    continue
                    
                if hasattr(agent, '_attention_dqn') and agent._attention_dqn:
                    # Track total attention records
                    if hasattr(agent._attention_dqn, 'attention_history'):
                        total_attention_records += len(agent._attention_dqn.attention_history)
                    
                    # Track total buffer size
                    if hasattr(agent._attention_dqn, 'memory') and hasattr(agent._attention_dqn.memory, 'buffer'):
                        total_buffer_size += len(agent._attention_dqn.memory.buffer)
                    
                    # Force cleanup if agent has cleanup method
                    if hasattr(agent._attention_dqn, '_cleanup_attention_data'):
                        agent._attention_dqn._cleanup_attention_data()
                        cleaned_count += 1
            
            # Force GPU memory cleanup
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            
            if cleaned_count > 0:
                print(f"🧹 Attention cleanup: {cleaned_count} networks, {total_attention_records} attention records, {total_buffer_size} buffer entries")
                
        except Exception as e:
            print(f"⚠️ Error in attention network cleanup: {e}")

# Create Flask app and SocketIO instance
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Create training environment instance
env = TrainingEnvironment()

# Add missing main web interface route
@app.route('/')
def index():
    """Serve the main web interface with WebGL rendering."""
    return render_template_string(get_webgl_template())

@app.route('/status')
def status():
    """Get current training status for the web interface."""
    # Get canvas dimensions and culling preference if provided via query parameters
    canvas_width = request.args.get('canvas_width', type=int, default=1200)
    canvas_height = request.args.get('canvas_height', type=int, default=800)
    viewport_culling = request.args.get('viewport_culling', default='true').lower() == 'true'
    # CRITICAL FIX: Get current frontend camera position for accurate viewport culling
    camera_x = request.args.get('camera_x', type=float, default=0.0)
    camera_y = request.args.get('camera_y', type=float, default=0.0)
    return jsonify(env.get_status(canvas_width, canvas_height, viewport_culling, camera_x, camera_y))

# Add missing reward signal endpoints to training system's Flask app
@app.route('/reward_signal_status')
def reward_signal_status():
    """Get reward signal status from training environment's adapter instance."""
    try:
        if hasattr(env, 'reward_signal_adapter') and env.reward_signal_adapter:
            status = env.reward_signal_adapter.get_system_status()
            return jsonify(status)
        else:
            return jsonify({'error': 'Reward signal adapter not available'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reward_signal_metrics')
def reward_signal_metrics():
    """Get reward signal metrics from training environment's adapter instance."""
    try:
        if hasattr(env, 'reward_signal_adapter') and env.reward_signal_adapter:
            metrics = env.reward_signal_adapter.get_all_reward_metrics()
            return jsonify(metrics)
        else:
            return jsonify({'error': 'Reward signal adapter not available'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint for training system data."""
    try:
        if not hasattr(env, 'reward_signal_adapter') or not env.reward_signal_adapter:
            return "# No reward signal adapter available\n", 200, {'Content-Type': 'text/plain'}
        
        # Get system status and metrics
        status = env.reward_signal_adapter.get_system_status()
        reward_metrics = env.reward_signal_adapter.get_all_reward_metrics()
        
        # Build Prometheus metrics format
        metrics_output = []
        
        # System-level metrics
        metrics_output.append(f"# HELP walker_total_agents Total number of agents registered")
        metrics_output.append(f"# TYPE walker_total_agents gauge")
        metrics_output.append(f"walker_total_agents {status.get('total_agents', 0)}")
        
        metrics_output.append(f"# HELP walker_total_rewards Total number of rewards recorded")
        metrics_output.append(f"# TYPE walker_total_rewards counter")
        metrics_output.append(f"walker_total_rewards {status.get('total_rewards_recorded', 0)}")
        
        metrics_output.append(f"# HELP walker_agents_with_metrics Number of agents with quality metrics")
        metrics_output.append(f"# TYPE walker_agents_with_metrics gauge")
        metrics_output.append(f"walker_agents_with_metrics {status.get('agents_with_metrics', 0)}")
        
        # Per-agent reward quality metrics (if any)
        if reward_metrics:
            metrics_output.append(f"# HELP walker_reward_quality_score Reward quality score per agent")
            metrics_output.append(f"# TYPE walker_reward_quality_score gauge")
            
            metrics_output.append(f"# HELP walker_reward_signal_to_noise Signal to noise ratio per agent") 
            metrics_output.append(f"# TYPE walker_reward_signal_to_noise gauge")
            
            for agent_id, metrics in reward_metrics.items():
                agent_id_clean = agent_id.replace('-', '_')  # Prometheus-safe agent ID
                metrics_output.append(f'walker_reward_quality_score{{agent_id="{agent_id_clean}"}} {metrics.quality_score:.4f}')
                metrics_output.append(f'walker_reward_signal_to_noise{{agent_id="{agent_id_clean}"}} {metrics.signal_to_noise_ratio:.4f}')
        
        # Training system health
        metrics_output.append(f"# HELP walker_system_active Training system active status")
        metrics_output.append(f"# TYPE walker_system_active gauge")
        metrics_output.append(f"walker_system_active {1 if status.get('active', False) else 0}")
        
        return "\n".join(metrics_output) + "\n", 200, {'Content-Type': 'text/plain'}
        
    except Exception as e:
        # Return basic error metric
        error_output = [
            "# HELP walker_metrics_error Metrics collection error",
            "# TYPE walker_metrics_error gauge", 
            f"walker_metrics_error 1",
            f"# Error: {str(e)}"
        ]
        return "\n".join(error_output) + "\n", 200, {'Content-Type': 'text/plain'}

# Add missing interactive endpoints that the frontend JavaScript expects
@app.route('/click', methods=['POST'])
def click():
    """Handle click events from the frontend for agent focusing."""
    try:
        data = request.get_json()
        agent_id = data.get('agent_id')
        
        if agent_id:
            # Find the agent by ID - convert both to strings for comparison
            agent_to_focus = next((agent for agent in env.agents if str(agent.id) == str(agent_id)), None)
            if agent_to_focus:
                env.focus_on_agent(agent_to_focus)
                return jsonify({'status': 'success', 'message': f'Focused on agent {agent_id}', 'agent_id': agent_id})
            else:
                env.focus_on_agent(None)  # Clear focus if agent not found
                return jsonify({'status': 'error', 'message': f'Agent {agent_id} not found', 'agent_id': None})
        else:
            # Clear focus if no agent_id provided
            env.focus_on_agent(None)
            return jsonify({'status': 'success', 'message': 'Focus cleared', 'agent_id': None})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_agent_at_position', methods=['POST'])
def get_agent_at_position():
    """Get agent information at a specific world position."""
    try:
        data = request.get_json()
        world_x = data.get('x', 0)  # Frontend sends 'x', not 'world_x'
        world_y = data.get('y', 0)  # Frontend sends 'y', not 'world_y'
        
        agent = env.get_agent_at_position(world_x, world_y)
        if agent:
            return jsonify({
                'status': 'success',
                'agent_id': agent.id,  # Frontend expects 'agent_id' directly, not nested
                'position': [agent.body.position.x, agent.body.position.y] if agent.body else [0, 0]
            })
        else:
            return jsonify({'status': 'error', 'message': 'No agent found at position', 'agent_id': None})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e), 'agent_id': None}), 500

@app.route('/move_agent', methods=['POST'])
def move_agent():
    """Move an agent to a specific world position."""
    try:
        data = request.get_json()
        agent_id = data.get('agent_id')
        x = data.get('x', 0)
        y = data.get('y', 0)
        
        success = env.move_agent(agent_id, x, y)
        if success:
            return jsonify({'status': 'success', 'message': f'Agent {agent_id} moved to ({x}, {y})'})
        else:
            return jsonify({'status': 'error', 'message': f'Failed to move agent {agent_id}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/update_zoom', methods=['POST'])
def update_zoom():
    """Update the user's zoom level preference."""
    try:
        data = request.get_json()
        zoom_level = data.get('zoom', 1.0)
        
        env.update_user_zoom(zoom_level)
        return jsonify({'status': 'success', 'zoom': zoom_level})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/reset_view', methods=['POST'])
def reset_view():
    """Reset camera view to default position and zoom."""
    try:
        env.reset_camera_position()
        env.reset_user_zoom()
        return jsonify({'status': 'success', 'message': 'View reset to default'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/clear_zoom_override', methods=['POST'])
def clear_zoom_override():
    """Clear zoom override flag after frontend receives it."""
    try:
        env.clear_zoom_override()
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/update_agent_params', methods=['POST'])
def update_agent_params():
    """Update agent parameters."""
    try:
        data = request.get_json()
        params = data.get('params', {})
        target_agent_id = data.get('target_agent_id')
        
        result = env.update_agent_params(params, target_agent_id)
        return jsonify({'status': 'success', 'updated_params': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/set_simulation_speed', methods=['POST'])
def set_simulation_speed():
    """Set the simulation speed multiplier."""
    try:
        data = request.get_json()
        speed = data.get('speed', 1.0)
        
        # Validate speed range
        if not isinstance(speed, (int, float)) or speed <= 0:
            return jsonify({'status': 'error', 'message': 'Speed must be a positive number'}), 400
        
        if speed > env.max_speed_multiplier:
            return jsonify({'status': 'error', 'message': f'Speed cannot exceed {env.max_speed_multiplier}x'}), 400
        
        # Set the speed
        env.simulation_speed_multiplier = float(speed)
        
        return jsonify({
            'status': 'success', 
            'message': f'Simulation speed set to {speed}x',
            'speed': speed
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/ai_optimization_settings', methods=['GET', 'POST'])
def ai_optimization_settings():
    """Get or set AI optimization settings."""
    try:
        if request.method == 'GET':
            return jsonify({
                'status': 'success',
                'settings': {
                    'ai_optimization_enabled': env.ai_optimization_enabled,
                    'ai_batch_percentage': env.ai_batch_percentage,
                    'ai_spatial_culling_enabled': env.ai_spatial_culling_enabled,
                    'ai_spatial_culling_distance': env.ai_spatial_culling_distance
                }
            })
        
        elif request.method == 'POST':
            data = request.get_json()
            
            # Update settings if provided
            if 'ai_optimization_enabled' in data:
                env.ai_optimization_enabled = bool(data['ai_optimization_enabled'])
            
            if 'ai_batch_percentage' in data:
                percentage = float(data['ai_batch_percentage'])
                if 0.1 <= percentage <= 1.0:
                    env.ai_batch_percentage = percentage
                else:
                    return jsonify({'status': 'error', 'message': 'ai_batch_percentage must be between 0.1 and 1.0'}), 400
            
            if 'ai_spatial_culling_enabled' in data:
                env.ai_spatial_culling_enabled = bool(data['ai_spatial_culling_enabled'])
            
            if 'ai_spatial_culling_distance' in data:
                distance = float(data['ai_spatial_culling_distance'])
                if distance > 0:
                    env.ai_spatial_culling_distance = distance
                else:
                    return jsonify({'status': 'error', 'message': 'ai_spatial_culling_distance must be positive'}), 400
            
            return jsonify({
                'status': 'success',
                'message': 'AI optimization settings updated',
                'settings': {
                    'ai_optimization_enabled': env.ai_optimization_enabled,
                    'ai_batch_percentage': env.ai_batch_percentage,
                    'ai_spatial_culling_enabled': env.ai_spatial_culling_enabled,
                    'ai_spatial_culling_distance': env.ai_spatial_culling_distance
                }
            })
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/toggle_health_bars', methods=['POST'])
def toggle_health_bars():
    """Toggle health bar rendering on/off for performance optimization."""
    try:
        data = request.get_json() or {}
        enable = data.get('enable', not env.show_health_bars)  # Toggle if not specified
        
        env.show_health_bars = bool(enable)
        
        message = f"Health bars {'enabled' if env.show_health_bars else 'disabled'}"
        return jsonify({
            'status': 'success',
            'message': message,
            'show_health_bars': env.show_health_bars
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# WebGL routes removed - WebGL is now the only rendering mode

@app.route('/elite_robots', methods=['GET', 'POST'])
def elite_robots():
    """Manage elite robots - get stats or load elites into population."""
    try:
        if request.method == 'GET':
            # Get elite robot statistics
            stats = env.elite_manager.get_elite_statistics()
            top_elites = env.elite_manager.get_top_elites(10)
            
            return jsonify({
                'elite_statistics': stats,
                'top_elites': top_elites,
                'auto_save_enabled': env.auto_save_elites,
                'current_generation': env.evolution_engine.generation
            })
        
        elif request.method == 'POST':
            # Load elite robots into current population
            data = request.get_json() or {}
            count = data.get('count', 5)
            min_generation = data.get('min_generation', max(0, env.evolution_engine.generation - 5))
            
            # Load elite robots
            elite_robots = env.elite_manager.restore_elite_robots(
                world=env.world,
                count=count,
                min_generation=min_generation
            )
            
            if elite_robots:
                # Replace random agents with elite robots
                agents_replaced = min(len(elite_robots), len(env.agents) // 4)  # Replace up to 25%
                
                # Remove random agents
                for _ in range(agents_replaced):
                    if env.agents:
                        removed_agent = env.agents.pop(random.randint(0, len(env.agents) - 1))
                        env._safe_destroy_agent(removed_agent)
                
                # Add elite robots
                for elite_robot in elite_robots[:agents_replaced]:
                    env.agents.append(elite_robot)
                    env._initialize_single_agent_ecosystem(elite_robot)
                
                return jsonify({
                    'success': True,
                    'message': f'Loaded {agents_replaced} elite robots into population',
                    'agents_replaced': agents_replaced,
                    'elite_count': len(elite_robots),
                    'population_size': len(env.agents)
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'No elite robots found to load',
                    'agents_replaced': 0
                })
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Ray casting visualization endpoint removed for performance

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
            # Optional: Print periodic status
            if hasattr(env, 'step_count') and env.step_count % 18000 == 0:  # Every 5 minutes
                print(f"🔄 System running: Step {env.step_count}, Generation {env.evolution_engine.generation}")
    except KeyboardInterrupt:
        print("\n🛑 Shutting down training environment...")
        env.stop()
        print("✅ Training stopped.")

if __name__ == "__main__":
    main()

# When the script exits, ensure the environment is stopped
import atexit
atexit.register(lambda: env.stop()) 