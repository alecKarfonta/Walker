#!/usr/bin/env python3
"""
Web-based training visualization with actual physics world rendering.
Shows the real robots, arms, and physics simulation in the browser.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import threading
import time
import json
import logging
import numpy as np
import Box2D as b2
from src.agents.crawling_crate_agent import CrawlingCrateAgent
from src.population.population_controller import PopulationController
from src.population.evolution import EvolutionEngine
import pygame
import io
import base64

# New Dash App Integration
from src.ui.main_window import create_app
from typing import List

logging.getLogger('werkzeug').setLevel(logging.ERROR)

class TrainingEnvironment:
    """
    Manages the physics simulation, agents, and their training.
    This class will be passed to the Dash application.
    """
    def __init__(self, num_agents=50):
        self.num_agents = num_agents
        self.world = b2.b2World(gravity=(0, -10), doSleep=True)
        self.ground = self._create_ground()
        self.agents: List[CrawlingCrateAgent] = []
        self.population = PopulationController(population_size=self.num_agents)
        self.evolution_engine = EvolutionEngine(self.population)
        
        self.simulation_time_s = 0.0
        self.time_step = 1.0 / 60.0
        self.velocity_iterations = 8
        self.position_iterations = 3
        
        self.is_running = False
        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        
        self.camera_center = (0, 15)
        self.camera_zoom = 1.0
        self.target_zoom = 1.0
        self.focused_agent = None
        self.camera_position = (0, 0)
        self.camera_target = (0, 0)

        self._init_robot_stats()
        # Automatically spawn agents at startup
        for _ in range(self.num_agents):
            self.spawn_agent()

    def _create_ground(self):
        """Creates a static ground body."""
        ground_body = self.world.CreateStaticBody(position=(0, -1))
        
        # Calculate ground width based on number of agents
        ground_width = max(500, self.num_agents * 10)  # Ensure enough width for all agents
        
        # The ground's mask is set to collide with the agent category
        ground_fixture = ground_body.CreateFixture(
            shape=b2.b2PolygonShape(box=(ground_width, 1)),
            density=0.0,
            friction=0.9,
            filter=b2.b2Filter(
                categoryBits=0x0001,
                maskBits=0x0002  # Collide with all agents
            )
        )
        print(f"🔧 Ground setup complete with width {ground_width} for {self.num_agents} agents.")

    def _update_statistics(self):
        """Update population statistics."""
        if not self.agents:
            return
        
        # Calculate distances and fitness
        distances = []
        for i, agent in enumerate(self.agents):
            # Update robot statistics
            self.robot_stats[i]['current_position'] = tuple(agent.body.position)
            self.robot_stats[i]['velocity'] = tuple(agent.body.linearVelocity)
            self.robot_stats[i]['arm_angles']['shoulder'] = agent.upper_arm.angle
            self.robot_stats[i]['arm_angles']['elbow'] = agent.lower_arm.angle
            self.robot_stats[i]['steps_alive'] += 1
            self.robot_stats[i]['episode_reward'] = agent.total_reward
            self.robot_stats[i]['q_updates'] = agent.q_table.update_count if hasattr(agent.q_table, 'update_count') else 0
            self.robot_stats[i]['action_history'] = agent.action_history
            
            # Calculate distance traveled
            distance = agent.body.position.x - agent.initial_position[0]
            self.robot_stats[i]['total_distance'] = distance
            self.robot_stats[i]['fitness'] = distance
            distances.append(distance)
        
        # Update population statistics
        self.population_stats = {
            'best_distance': max(distances),
            'average_distance': sum(distances) / len(distances),
            'worst_distance': min(distances),
            'total_agents': len(self.agents),
            'q_learning_stats': {
                'avg_epsilon': sum(agent.epsilon for agent in self.agents) / len(self.agents),
                'total_q_updates': sum(self.robot_stats[i]['q_updates'] for i in range(len(self.agents)))
            }
        }

    def training_loop(self):
        """Main training loop."""
        self.is_running = True
        last_step_time = time.time()
        last_stats_time = time.time()
        last_debug_time = time.time()
        step_count = 0
        
        print("🚀 Training loop started!")
        print(f"🔧 World gravity: {self.world.gravity}")
        print(f"🔧 Number of agents: {len(self.agents)}")
        print(f"🔧 Physics timestep: {self.time_step}")
        
        # Initialize robot statistics
        self._init_robot_stats()
        
        # Test physics world
        print("🔧 Testing physics world...")
        for i in range(5):
            self.world.Step(self.time_step, self.velocity_iterations, self.position_iterations)
            print(f"   Step {i}: World bodies: {len(self.world.bodies)}")
        
        while self.is_running:
            current_time = time.time()
            delta_time = min(current_time - last_step_time, 1.0 / 30.0)  # Cap at 30 FPS
            
            # Update camera
            self.update_camera(delta_time)
            
            # Step the physics world
            self.world.Step(self.time_step, self.velocity_iterations, self.position_iterations)
            step_count += 1
            
            # Update all agents
            for agent in self.agents:
                agent.step(delta_time)
            
            # Check for fallen agents and reset them
            for agent in self.agents:
                if agent.body.position.y < -20.0:
                    agent.reset_position()

            # Update statistics periodically
            if current_time - last_stats_time > 0.1:  # Update every 0.1 seconds
                self._update_statistics()
                last_stats_time = current_time
            
            # Debug output every 2 seconds
            if current_time - last_debug_time > 2.0:
                print(f"🔧 Physics step {step_count}: {len(self.agents)} agents active")
                if self.agents:
                    first_agent = self.agents[0]
                    print(f"   Agent 0: pos=({first_agent.body.position.x:.2f}, {first_agent.body.position.y:.2f}), "
                          f"vel=({first_agent.body.linearVelocity.x:.2f}, {first_agent.body.linearVelocity.y:.2f}), "
                          f"reward={first_agent.total_reward:.2f}")
                    print(f"   Agent 0: action={first_agent.current_action_tuple}, "
                          f"state={first_agent.current_state}, "
                          f"steps={first_agent.steps}")
                    
                    # Check if agent is awake
                    print(f"   Agent 0 awake: {first_agent.body.awake}, "
                          f"upper_arm awake: {first_agent.upper_arm.awake}, "
                          f"lower_arm awake: {first_agent.lower_arm.awake}")
                    
                    # Check arm angles
                    print(f"   Agent 0 arm angles: shoulder={first_agent.upper_arm.angle:.2f}, "
                          f"elbow={first_agent.lower_arm.angle:.2f}")
                last_debug_time = current_time
            
            last_step_time = current_time
            time.sleep(max(0, self.time_step - (time.time() - current_time)))

    def update_agent_params(self, params, target_agent_id=None):
        """Update parameters for specific agent or all agents."""
        if target_agent_id is not None:
            # Update only the focused agent
            target_agent = next((agent for agent in self.agents if agent.id == target_agent_id), None)
            if not target_agent:
                print(f"❌ Agent {target_agent_id} not found")
                return False
            
            agents_to_update = [target_agent]
        else:
            # Update all agents
            agents_to_update = self.agents
        
        for agent in agents_to_update:
            for key, value in params.items():
                # Handle special physical properties
                if key == 'friction':
                    for part in [agent.body, agent.upper_arm, agent.lower_arm] + agent.wheels:
                        for fixture in part.fixtures:
                            fixture.friction = value
                elif key == 'density':
                    for part in [agent.body, agent.upper_arm, agent.lower_arm] + agent.wheels:
                        for fixture in part.fixtures:
                            fixture.density = value
                    # Important: must call ResetMassData after changing density
                    agent.body.ResetMassData()
                    agent.upper_arm.ResetMassData()
                    agent.lower_arm.ResetMassData()
                    for wheel in agent.wheels:
                        wheel.ResetMassData()
                elif key == 'linear_damping':
                     for part in [agent.body, agent.upper_arm, agent.lower_arm] + agent.wheels:
                        part.linearDamping = value
                # Handle generic agent attributes
                elif hasattr(agent, key):
                    setattr(agent, key, value)
        
        target_desc = f"agent {target_agent_id}" if target_agent_id else "all agents"
        print(f"✅ Updated {target_desc} parameters: {params}")
        return True

    def get_status(self):
        """Returns a comprehensive status of the environment for rendering."""
        with self._lock:
            agent_data = []
            for agent_index in range(len(self.agents)):
                agent: CrawlingCrateAgent = self.agents[agent_index]
                body_pos = agent.body.position
                upper_arm_angle = agent.upper_arm.angle
                lower_arm_angle = agent.lower_arm.angle
                
                agent_data.append({
                    'id': agent.id,
                    'type': 'crawling_crate',
                    'chassis_pos': (body_pos.x, body_pos.y),
                    'chassis_angle': agent.body.angle,
                    'wheel_pos': (agent.wheels[0].position.x, agent.wheels[0].position.y),
                    'wheel_angle': agent.wheels[0].angle,
                    'upper_arm_pos': (agent.upper_arm.position.x, agent.upper_arm.position.y),
                    'upper_arm_angle': upper_arm_angle,
                    'lower_arm_pos': (agent.lower_arm.position.x, agent.lower_arm.position.y),
                    'lower_arm_angle': lower_arm_angle,
                    'performance': agent.total_reward,
                    'q_values': agent.q_table.get_best_q_values_for_all_states() if agent.q_table else {},
                    'action_history': list(agent.action_history),
                })
            
            best_agent = self.get_best_agent()
            
            return {
                'simulation_time_s': self.simulation_time_s,
                'num_agents': len(self.agents),
                'agents': agent_data,
                'camera': self.get_camera_state(),
                'best_agent_id': best_agent.id if best_agent else None,
                'population_stats': self.population.get_stats(),
                'evolution_stats': self.evolution_engine.get_stats(),
            }

    def start(self):
        """Starts the training loop in a separate thread."""
        if not self.is_running:
            print("🔄 Starting training loop thread...")
            self._thread = threading.Thread(target=self.training_loop)
            self._thread.daemon = True
            self._thread.start()
            print("✅ Training loop thread started successfully")
        else:
            print("⚠️  Training loop is already running")

    def stop(self):
        """Stops the training loop."""
        print("🛑 Stopping training loop...")
        self.is_running = False
        if self._thread:
            self._thread.join()
            print("✅ Training loop stopped")

    def get_best_agent(self):
        """Utility to get the best agent based on fitness (distance)."""
        if not self.agents:
            return None
        return max(self.agents, key=lambda agent: agent.get_fitness())

    def spawn_agent(self):
        """Adds a new, random agent to the simulation."""
        new_id = len(self.agents)
        spacing = 8 if self.num_agents > 20 else 15
        position = (new_id * spacing, 6)
        
        new_agent = CrawlingCrateAgent(
            self.world,
            agent_id=new_id,
            position=position,
            category_bits=0x0002,
            mask_bits=0x0001
        )
        self.agents.append(new_agent)
        self.population.add_agent(new_agent)
        self.num_agents = len(self.agents)
        print(f"🐣 Spawned new agent {new_id}. Total agents: {self.num_agents}")

    def clone_best_agent(self):
        """Clones the best performing agent."""
        best_agent = self.get_best_agent()
        if not best_agent:
            print("No agents to clone.")
            return

        new_id = len(self.agents)
        spacing = 8 if self.num_agents > 20 else 15
        position = (new_id * spacing, 6)

        # Create a new agent with the same parameters
        cloned_agent = CrawlingCrateAgent(
            self.world,
            agent_id=new_id,
            position=position,
            category_bits=0x0002,
            mask_bits=0x0001
        )
        
        # Copy the learned parameters from the best agent
        if hasattr(best_agent, 'q_table') and hasattr(cloned_agent, 'q_table'):
            # Create a deep copy of the Q-table based on its type
            if hasattr(best_agent.q_table, 'q_values') and hasattr(best_agent.q_table.q_values, 'copy'):
                # Regular QTable with numpy arrays
                cloned_agent.q_table.q_values = best_agent.q_table.q_values.copy()
                if hasattr(best_agent.q_table, 'visit_counts'):
                    cloned_agent.q_table.visit_counts = best_agent.q_table.visit_counts.copy()
            elif hasattr(best_agent.q_table, 'q_values') and isinstance(best_agent.q_table.q_values, dict):
                # SparseQTable with dictionary
                cloned_agent.q_table.q_values = best_agent.q_table.q_values.copy()
                if hasattr(best_agent.q_table, 'visit_counts'):
                    cloned_agent.q_table.visit_counts = best_agent.q_table.visit_counts.copy()
        
        # Copy other learning parameters
        cloned_agent.learning_rate = best_agent.learning_rate
        cloned_agent.epsilon = best_agent.epsilon
        cloned_agent.discount_factor = best_agent.discount_factor
        
        self.agents.append(cloned_agent)
        self.population.add_agent(cloned_agent)
        self.num_agents = len(self.agents)
        print(f"👯 Cloned best agent {best_agent.id} to new agent {new_id}. Total agents: {self.num_agents}")

    def evolve_population(self):
        """Runs the evolution engine to create a new generation."""
        # Update fitness values before evolution
        for agent in self.agents:
            self.population.update_agent_fitness(agent, agent.get_fitness())
        
        new_population = self.evolution_engine.evolve_generation()
        
        # Simple replacement: clear old agents and add new ones
        for agent in self.agents:
            agent.destroy()

        self.agents = new_population
        self.num_agents = len(self.agents)

        # Re-initialize controller and stats
        self.population = PopulationController(len(self.agents))
        self._init_robot_stats() # Helper to re-init stats
        print(f"🧬 Evolved population. New generation has {self.num_agents} agents.")

    def _init_robot_stats(self):
        self.robot_stats = {}
        for i, agent in enumerate(self.agents):
             self.robot_stats[i] = {
                'id': agent.id,
                'initial_position': tuple(agent.initial_position),
                'current_position': tuple(agent.body.position),
                'total_distance': 0,
                'velocity': (0, 0),
                'arm_angles': {'shoulder': 0, 'elbow': 0},
                'fitness': 0,
                'steps_alive': 0,
                'last_position': tuple(agent.body.position),
                'steps_tilted': 0,  # Track how long robot has been tilted
                'episode_reward': 0,
                'q_updates': 0,
                'action_history': []  # Track last actions taken
            }
            
    def update_camera(self, delta_time):
        """Update camera position with smooth following."""
        if self.focused_agent:
            # Get the focused agent's position
            agent_pos = self.focused_agent.body.position
            self.camera_target = (agent_pos.x, agent_pos.y)
        
        # Smooth camera movement using lerp
        self.camera_position = (
            self.camera_position[0] + (self.camera_target[0] - self.camera_position[0]) * 0.05,
            self.camera_position[1] + (self.camera_target[1] - self.camera_position[1]) * 0.05
        )
        
        # Smooth zoom
        if abs(self.target_zoom - self.camera_zoom) > 0.001:
            self.camera_zoom += (self.target_zoom - self.camera_zoom) * 0.05

    def focus_on_agent(self, agent):
        """Focus the camera on a specific agent."""
        self.focused_agent = agent
        if agent:
            print(f"🎯 Camera focused on agent {agent.id}")
        else:
            print("🎯 Camera focus cleared")

    def get_agent_at_position(self, world_x, world_y):
        """Find an agent at the given world coordinates."""
        for agent in self.agents:
            # Check if click is near the agent's body
            agent_pos = agent.body.position
            distance = ((world_x - agent_pos.x) ** 2 + (world_y - agent_pos.y) ** 2) ** 0.5
            if distance < 2.0:  # Click radius
                return agent
        return None

    def move_agent(self, agent_id, x, y):
        """Move an agent to the specified world coordinates."""
        agent = next((a for a in self.agents if a.id == agent_id), None)
        if not agent:
            print(f"❌ Agent {agent_id} not found for moving")
            return False
        
        # Set the agent's position
        agent.body.position = (x, y)
        
        # Reset velocity to prevent physics issues
        agent.body.linearVelocity = (0, 0)
        agent.body.angularVelocity = 0
        
        print(f"🤖 Moved agent {agent_id} to ({x:.2f}, {y:.2f})")
        return True

    def handle_click(self, screen_x, screen_y, canvas_width, canvas_height):
        """Handle mouse click to select an agent."""
        # Convert screen coordinates to world coordinates
        # Assuming the world view is centered and scaled
        world_x = (screen_x - canvas_width / 2) / self.camera_zoom + self.camera_position[0]
        world_y = (canvas_height / 2 - screen_y) / self.camera_zoom + self.camera_position[1]
        
        # Find agent at click position
        clicked_agent = self.get_agent_at_position(world_x, world_y)
        
        if clicked_agent:
            self.focus_on_agent(clicked_agent)
            return clicked_agent.id
        else:
            self.focus_on_agent(None)
            return None

    def get_camera_state(self):
        """Get current camera state for rendering."""
        return {
            'position': self.camera_position,
            'zoom': self.camera_zoom,
            'focused_agent_id': self.focused_agent.id if self.focused_agent else None
        }

    def render_to_base64_image(self, width=800, height=600):
        """Renders the current world state to a base64 encoded PNG image."""
        try:
            pygame.init()
            screen = pygame.Surface((width, height))
            screen.fill((0, 0, 0)) # Black background

            # --- Camera and Coordinate Conversion ---
            PPM = 20.0  # pixels per meter
            TARGET_FPS = 60
            TIME_STEP = 1.0 / TARGET_FPS
            SCREEN_WIDTH, SCREEN_HEIGHT = width, height

            camera_offset = (
                self.camera_center[0] * PPM - SCREEN_WIDTH / 2,
                self.camera_center[1] * PPM - SCREEN_HEIGHT / 2
            )

            def world_to_screen(pos):
                return (
                    pos[0] * PPM - camera_offset[0],
                    SCREEN_HEIGHT - (pos[1] * PPM - camera_offset[1])
                )

            # --- Colors ---
            colors = {
                b2.b2_staticBody: (255, 255, 255, 255),
                b2.b2_dynamicBody: (127, 127, 127, 255),
            }

            # --- Drawing ---
            with self._lock:
                for body in self.world.bodies:
                    for fixture in body.fixtures:
                        shape = fixture.shape
                        
                        # Determine color
                        color = colors.get(body.type, (255, 0, 0, 255))
                        # Find the agent this body belongs to, for special coloring
                        part_of_agent = None
                        for agent in self.agents:
                            if hasattr(agent, 'body') and body == agent.body:
                                part_of_agent = agent
                                break
                        if part_of_agent:
                            color = (224, 60, 51, 255) # Red for agents

                        if isinstance(shape, b2.b2CircleShape):
                            pos = world_to_screen(body.GetWorldPoint(shape.pos))
                            pygame.draw.circle(screen, color, pos, shape.radius * PPM)
                        
                        elif isinstance(shape, b2.b2PolygonShape):
                            vertices = [world_to_screen(body.GetWorldPoint(v)) for v in shape.vertices]
                            pygame.draw.polygon(screen, color, vertices)

            # --- Convert to Base64 ---
            img_data = pygame.image.tostring(screen, 'RGB')
            img_byte_io = io.BytesIO()
            
            # Create a Pygame surface from the raw RGB data
            rgb_surface = pygame.image.fromstring(img_data, (width, height), 'RGB')
            pygame.image.save(rgb_surface, img_byte_io, 'PNG')
            img_byte_io.seek(0)
            
            base64_string = base64.b64encode(img_byte_io.read()).decode('utf-8')
            return "data:image/png;base64," + base64_string
        
        except Exception as e:
            print(f"Error rendering image: {e}")
            return None
        finally:
            pygame.quit()

def main():
    """
    Initializes the training environment and starts the Dash web server.
    """
    port = 8050
    
    # Create the training environment
    env = TrainingEnvironment(num_agents=50)
    
    # Create the Dash app and pass the environment to it
    app = create_app(env)

    # Start the training loop in a background thread
    env.start()

    print(f"✅ Starting web server on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)

if __name__ == '__main__':
    main()

# When the script exits, ensure the environment is stopped
import atexit
atexit.register(lambda: env.stop() if 'env' in locals() else None) 