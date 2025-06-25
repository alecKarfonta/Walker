"""
Training visualizer for watching robots learn to crawl in real-time.
"""

import pygame
import pymunk
import pymunk.pygame_util
import numpy as np
from typing import List, Dict, Any, Tuple
from src.agents.crawling_crate import CrawlingCrate
from src.population.population_controller import PopulationController
from src.population.evolution import EvolutionEngine
from src.agents.basic_agent import BasicAgent


class TrainingVisualizer:
    """Real-time visualization of robot training and evolution."""
    
    def __init__(self, width=1600, height=1000):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Walker Robot Training")
        
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
        self.generation = 0
        self.episode = 0
        self.best_fitness = 0.0
        self.avg_fitness = 0.0
        self.population_size = 8
        self.agents = []
        self.fitness_history = []
        
        # UI state
        self.paused = False
        self.show_debug = True
        self.camera_x = 0
        self.camera_y = 0
        
    def _create_ground(self):
        """Create the ground terrain."""
        static_body = self.space.static_body
        # Flat ground
        ground = pymunk.Segment(static_body, (-200, 0), (200, 0), 2.0)
        ground.friction = 1.0
        ground.color = (100, 100, 100, 255)
        self.space.add(ground)
        
    def create_agent(self, x_pos: float) -> CrawlingCrate:
        """Create a crawling crate agent at the specified position."""
        agent = CrawlingCrate(self.space, position=(x_pos, 20))
        return agent
        
    def create_population(self):
        """Create a population of agents."""
        self.agents = []
        spacing = 15
        start_x = 50
        
        for i in range(self.population_size):
            x_pos = start_x + i * spacing
            agent = self.create_agent(x_pos)
            self.agents.append(agent)
            
    def reset_population(self):
        """Reset all agents in the population."""
        for agent in self.agents:
            agent.reset()
            
    def evaluate_fitness(self, agent: CrawlingCrate, max_steps=200) -> float:
        """Evaluate fitness of a single agent."""
        agent.reset()
        start_x = agent.body.position.x
        
        for step in range(max_steps):
            # Get agent state
            state = agent.get_state()
            
            # For now, use random actions (agents will learn through evolution)
            # In a real implementation, this would use the agent's Q-table
            action = np.random.uniform(-3, 3, 2)
            
            agent.apply_action(action)
            agent.step(1/60.0)
            
            # End if agent falls or flips
            if agent.body.position.y < -2 or abs(agent.body.angle) > np.pi/2:
                break
                
        # Fitness: forward progress
        fitness = agent.body.position.x - start_x
        return max(0, fitness)  # No negative fitness
        
    def run_training_step(self):
        """Run one step of training for all agents."""
        if self.paused:
            return
            
        # Evaluate all agents
        fitnesses = []
        for agent in self.agents:
            fitness = self.evaluate_fitness(agent, max_steps=100)
            fitnesses.append(fitness)
            
        # Update statistics
        self.best_fitness = max(fitnesses)
        self.avg_fitness = np.mean(fitnesses)
        self.fitness_history.append(self.avg_fitness)
        
        # Keep only last 100 points for display
        if len(self.fitness_history) > 100:
            self.fitness_history = self.fitness_history[-100:]
            
        # Reset agents for next evaluation
        self.reset_population()
        self.episode += 1
        
    def draw_agents(self):
        """Draw all agents in the population."""
        for i, agent in enumerate(self.agents):
            # Get agent info
            debug = agent.get_debug_info()
            pos = debug['crate_pos']
            vel = debug['crate_vel']
            
            # Draw agent ID
            text = self.font_small.render(f"Agent {i}", True, self.BLACK)
            self.screen.blit(text, (pos[0] - 20, pos[1] - 30))
            
            # Draw velocity indicator
            if abs(vel[0]) > 0.1:
                color = self.GREEN if vel[0] > 0 else self.RED
                pygame.draw.circle(self.screen, color, (int(pos[0] + 25), int(pos[1] - 20)), 3)
                
    def draw_ui(self):
        """Draw the user interface."""
        # Background for UI
        ui_rect = pygame.Rect(self.width - 300, 0, 300, self.height)
        pygame.draw.rect(self.screen, self.DARK_GRAY, ui_rect)
        
        # Training statistics
        y_offset = 20
        texts = [
            f"Generation: {self.generation}",
            f"Episode: {self.episode}",
            f"Best Fitness: {self.best_fitness:.2f}",
            f"Avg Fitness: {self.avg_fitness:.2f}",
            f"Population: {self.population_size}",
            "",
            "Controls:",
            "SPACE: Pause/Resume",
            "R: Reset Population",
            "D: Toggle Debug",
            "ESC: Quit"
        ]
        
        for text in texts:
            if text:
                surface = self.font_medium.render(text, True, self.WHITE)
                self.screen.blit(surface, (self.width - 290, y_offset))
            y_offset += 25
            
        # Fitness history graph
        if len(self.fitness_history) > 1:
            self._draw_fitness_graph()
            
    def _draw_fitness_graph(self):
        """Draw a simple fitness history graph."""
        graph_x = self.width - 290
        graph_y = 350
        graph_width = 280
        graph_height = 150
        
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
                    
        # Draw graph title
        title = self.font_small.render("Fitness History", True, self.WHITE)
        self.screen.blit(title, (graph_x, graph_y - 20))
        
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.reset_population()
                elif event.key == pygame.K_d:
                    self.show_debug = not self.show_debug
                elif event.key == pygame.K_ESCAPE:
                    return False
        return True
        
    def run(self):
        """Main training loop."""
        self.create_population()
        clock = pygame.time.Clock()
        
        running = True
        while running:
            # Handle events
            running = self.handle_events()
            
            # Run training step
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


def main():
    """Main function to run the training visualizer."""
    visualizer = TrainingVisualizer()
    visualizer.run()


if __name__ == "__main__":
    main() 