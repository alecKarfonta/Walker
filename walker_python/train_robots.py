#!/usr/bin/env python3
"""
Main training script for Walker robots.
Run this to see robots learning to crawl in real-time!
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import pygame
import pymunk
import pymunk.pygame_util
import numpy as np
from typing import List
from src.agents.crawling_crate import CrawlingCrate
from src.population.population_controller import PopulationController
from src.population.evolution import EvolutionEngine


class RobotTrainer:
    """Main training system for robots."""
    
    def __init__(self, population_size=8):
        # Initialize Pygame
        pygame.init()
        self.width = 1600
        self.height = 1000
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Walker Robot Training - Watch them learn!")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.GRAY = (128, 128, 128)
        self.DARK_GRAY = (64, 64, 64)
        
        # Fonts
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 32)
        
        # Physics setup
        self.space = pymunk.Space()
        self.space.gravity = (0, -9.8)
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        
        # Create ground
        self._create_ground()
        
        # Training state
        self.population_size = population_size
        self.agents = []
        self.generation = 0
        self.episode = 0
        self.best_fitness = 0.0
        self.avg_fitness = 0.0
        self.fitness_history = []
        
        # UI state
        self.paused = False
        self.show_debug = True
        self.training_speed = 1  # 1 = normal, 2 = fast, 0.5 = slow
        
        # Evolution
        self.evolution_engine = None
        self.population_controller = None
        
    def _create_ground(self):
        """Create the ground terrain."""
        static_body = self.space.static_body
        # Flat ground
        ground = pymunk.Segment(static_body, (-200, 0), (200, 0), 2.0)
        ground.friction = 1.0
        ground.color = (100, 100, 100, 255)
        self.space.add(ground)
        
    def create_population(self):
        """Create initial population of robots."""
        self.agents = []
        spacing = 15
        start_x = 50
        
        for i in range(self.population_size):
            x_pos = start_x + i * spacing
            agent = CrawlingCrate(self.space, position=(x_pos, 20))
            self.agents.append(agent)
            
        # Initialize evolution system with correct parameters
        self.population_controller = PopulationController(population_size=self.population_size)
        
        # Add all agents to the population controller
        for agent in self.agents:
            self.population_controller.add_agent(agent)
            
        self.evolution_engine = EvolutionEngine(self.population_controller, elite_size=2)
        
    def evaluate_agent(self, agent: CrawlingCrate, max_steps=150) -> float:
        """Evaluate a single agent's fitness."""
        agent.reset()
        start_x = agent.body.position.x
        total_reward = 0.0
        
        for step in range(max_steps):
            # Get current state
            state = agent.get_state()
            
            # For now, use simple random actions
            # In the future, this will use the agent's learned Q-table
            action = (
                np.random.uniform(-3, 3),
                np.random.uniform(-3, 3)
            )
            
            # Apply action
            agent.apply_action(action)
            agent.step(1/60.0)
            
            # Calculate reward
            reward = agent.get_reward(start_x)
            total_reward += reward
            start_x = agent.body.position.x
            
            # End if agent falls or flips
            if agent.body.position.y < -2 or abs(agent.body.angle) > np.pi/2:
                break
                
        # Final fitness: forward progress
        final_fitness = agent.body.position.x - agent.position[0]
        return max(0, final_fitness)
        
    def run_training_step(self):
        """Run one step of training for all agents."""
        if self.paused:
            return
            
        # Evaluate all agents
        fitnesses = []
        for agent in self.agents:
            fitness = self.evaluate_agent(agent)
            fitnesses.append(fitness)
            
        # Update statistics
        self.best_fitness = max(fitnesses)
        self.avg_fitness = np.mean(fitnesses)
        self.fitness_history.append(self.avg_fitness)
        
        # Keep only last 100 points for display
        if len(self.fitness_history) > 100:
            self.fitness_history = self.fitness_history[-100:]
            
        # Reset agents for next evaluation
        for agent in self.agents:
            agent.reset()
            
        self.episode += 1
        
        # Evolution every 10 episodes
        if self.episode % 10 == 0:
            self.run_evolution()
            
    def run_evolution(self):
        """Run one generation of evolution."""
        # Update fitness scores for all agents
        for i, agent in enumerate(self.agents):
            # Use the correct method name
            self.population_controller.update_agent_fitness(agent, self.fitness_history[-1])
            
        # Get ranked agents
        ranked_agents = self.population_controller.get_ranked_agents()
        
        # Create new population through evolution
        new_population = self.evolution_engine.evolve_generation()
        
        # Replace old agents with new ones
        for i, new_agent in enumerate(new_population):
            if i < len(self.agents):
                # Remove old agent from physics
                self.agents[i].destroy()
                # Create new agent at same position
                x_pos = 50 + i * 15
                new_agent = CrawlingCrate(self.space, position=(x_pos, 20))
                self.agents[i] = new_agent
                
        self.generation += 1
        
    def draw_agents(self):
        """Draw all agents in the population."""
        for i, agent in enumerate(self.agents):
            # Get agent info
            debug = agent.get_debug_info()
            pos = debug['crate_pos']
            vel = debug['crate_vel']
            
            # Draw agent ID
            text = self.font_small.render(f"Robot {i}", True, self.BLACK)
            self.screen.blit(text, (pos[0] - 20, pos[1] - 30))
            
            # Draw velocity indicator
            if abs(vel[0]) > 0.1:
                color = self.GREEN if vel[0] > 0 else self.RED
                pygame.draw.circle(self.screen, color, (int(pos[0] + 25), int(pos[1] - 20)), 3)
                
            # Draw arm angles
            left_angle = debug['left_arm_angle']
            right_angle = debug['right_arm_angle']
            angle_text = f"L:{left_angle:.1f} R:{right_angle:.1f}"
            angle_surface = self.font_small.render(angle_text, True, self.BLUE)
            self.screen.blit(angle_surface, (pos[0] - 30, pos[1] + 15))
            
    def draw_ui(self):
        """Draw the user interface."""
        # Background for UI
        ui_rect = pygame.Rect(self.width - 350, 0, 350, self.height)
        pygame.draw.rect(self.screen, self.DARK_GRAY, ui_rect)
        
        # Training statistics
        y_offset = 20
        texts = [
            "🤖 WALKER ROBOT TRAINING 🤖",
            "",
            f"Generation: {self.generation}",
            f"Episode: {self.episode}",
            f"Best Fitness: {self.best_fitness:.2f}",
            f"Avg Fitness: {self.avg_fitness:.2f}",
            f"Population: {self.population_size}",
            "",
            "🎮 Controls:",
            "SPACE: Pause/Resume",
            "R: Reset Population", 
            "D: Toggle Debug",
            "1/2/3: Speed (Slow/Normal/Fast)",
            "ESC: Quit",
            "",
            "📊 Learning Progress:",
            "Green dots = Moving forward",
            "Red dots = Moving backward",
            "Watch them learn to crawl!"
        ]
        
        for text in texts:
            if text:
                color = self.YELLOW if text.startswith("🤖") else self.WHITE
                surface = self.font_medium.render(text, True, color)
                self.screen.blit(surface, (self.width - 340, y_offset))
            y_offset += 25
            
        # Fitness history graph
        if len(self.fitness_history) > 1:
            self._draw_fitness_graph()
            
    def _draw_fitness_graph(self):
        """Draw a simple fitness history graph."""
        graph_x = self.width - 340
        graph_y = 500
        graph_width = 320
        graph_height = 200
        
        # Draw graph background
        pygame.draw.rect(self.screen, self.BLACK, 
                        (graph_x, graph_y, graph_width, graph_height))
        
        # Draw fitness line
        if len(self.fitness_history) > 1:
            max_fitness = max(self.fitness_history) if self.fitness_history else 1
            if max_fitness > 0:
                points = []
                for i, fitness in enumerate(self.fitness_history):
                    x = graph_x + (i / len(self.fitness_history)) * graph_width
                    y = graph_y + graph_height - (fitness / max_fitness) * graph_height
                    points.append((x, y))
                    
                if len(points) > 1:
                    pygame.draw.lines(self.screen, self.GREEN, False, points, 2)
                    
        # Draw graph title and labels
        title = self.font_medium.render("📈 Fitness Progress", True, self.WHITE)
        self.screen.blit(title, (graph_x, graph_y - 25))
        
        # Draw some grid lines
        for i in range(5):
            y = graph_y + (i * graph_height / 4)
            pygame.draw.line(self.screen, self.GRAY, 
                           (graph_x, y), (graph_x + graph_width, y), 1)
        
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.create_population()
                elif event.key == pygame.K_d:
                    self.show_debug = not self.show_debug
                elif event.key == pygame.K_1:
                    self.training_speed = 0.5
                elif event.key == pygame.K_2:
                    self.training_speed = 1.0
                elif event.key == pygame.K_3:
                    self.training_speed = 2.0
                elif event.key == pygame.K_ESCAPE:
                    return False
        return True
        
    def run(self):
        """Main training loop."""
        print("🤖 Starting Walker Robot Training!")
        print("🎮 Controls:")
        print("   SPACE: Pause/Resume")
        print("   R: Reset Population")
        print("   D: Toggle Debug")
        print("   1/2/3: Training Speed")
        print("   ESC: Quit")
        print("\n📊 Watch the robots learn to crawl!")
        
        self.create_population()
        clock = pygame.time.Clock()
        
        running = True
        while running:
            # Handle events
            running = self.handle_events()
            
            # Run training step
            for _ in range(int(self.training_speed)):
                self.run_training_step()
            
            # Clear screen
            self.screen.fill(self.WHITE)
            
            # Draw physics world
            self.space.debug_draw(self.draw_options)
            
            # Draw agents
            self.draw_agents()
            
            # Draw UI
            self.draw_ui()
            
            # Update display
            pygame.display.flip()
            clock.tick(30)  # 30 FPS
            
        pygame.quit()
        print("\n👋 Training finished!")


def main():
    """Main function to run the training."""
    trainer = RobotTrainer(population_size=8)
    trainer.run()


if __name__ == "__main__":
    main() 