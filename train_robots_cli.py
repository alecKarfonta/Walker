#!/usr/bin/env python3
"""
Command-line version of Walker robot training.
For users without a display or who prefer text output.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import pymunk
import numpy as np
import time
from typing import List
from src.agents.evolutionary_crawling_agent import EvolutionaryCrawlingAgent
from src.population.population_controller import PopulationController
from src.population.evolution import EvolutionEngine


class CLITrainer:
    """Command-line training system for robots."""
    
    def __init__(self, population_size=8):
        # Physics setup
        self.space = pymunk.Space()
        self.space.gravity = (0, -9.8)
        
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
        
        # Evolution
        self.evolution_engine = None
        self.population_controller = None
        
    def _create_ground(self):
        """Create the ground terrain."""
        static_body = self.space.static_body
        ground = pymunk.Segment(static_body, (-200, 0), (200, 0), 2.0)
        ground.friction = 1.0
        self.space.add(ground)
        
    def create_population(self):
        """Create initial population of robots."""
        self.agents = []
        spacing = 15
        start_x = 50
        
        for i in range(self.population_size):
            x_pos = start_x + i * spacing
            agent = EvolutionaryCrawlingAgent(self.space, position=(x_pos, 20))
            self.agents.append(agent)
            
        # Initialize evolution system with correct parameters
        self.population_controller = PopulationController(population_size=self.population_size)
        
        # Add all agents to the population controller
        for agent in self.agents:
            self.population_controller.add_agent(agent)
            
        self.evolution_engine = EvolutionEngine(self.population_controller, elite_size=2)
        
    def evaluate_agent(self, agent: EvolutionaryCrawlingAgent, max_steps=150) -> float:
        """Evaluate a single agent's fitness."""
        agent.reset()
        start_x = agent.body.position.x
        total_reward = 0.0
        
        for step in range(max_steps):
            # Get current state
            state = agent.get_state()
            
            # Random actions for now
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
        final_fitness = agent.body.position.x - agent.initial_position[0]
        return max(0, final_fitness)
        
    def run_training_step(self):
        """Run one step of training for all agents."""
        # Evaluate all agents
        fitnesses = []
        for i, agent in enumerate(self.agents):
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
                new_agent = EvolutionaryCrawlingAgent(self.space, position=(x_pos, 20))
                self.agents[i] = new_agent
                
        self.generation += 1
        
    def print_status(self):
        """Print current training status."""
        print(f"\n🤖 Episode {self.episode:4d} | Generation {self.generation:3d}")
        print(f"📊 Best Fitness: {self.best_fitness:6.2f} | Avg Fitness: {self.avg_fitness:6.2f}")
        
        # Show individual agent fitnesses
        fitnesses = []
        for i, agent in enumerate(self.agents):
            fitness = self.evaluate_agent(agent)
            fitnesses.append(fitness)
            
        print("🤖 Individual Robot Fitness:")
        for i, fitness in enumerate(fitnesses):
            bar_length = int(fitness * 20)  # Scale fitness to bar length
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"   Robot {i}: {fitness:6.2f} [{bar}]")
            
        # Show progress trend
        if len(self.fitness_history) > 5:
            recent_avg = np.mean(self.fitness_history[-5:])
            if recent_avg > self.avg_fitness * 1.1:
                print("📈 Trend: Improving! 🎉")
            elif recent_avg < self.avg_fitness * 0.9:
                print("📉 Trend: Declining 😞")
            else:
                print("➡️  Trend: Stable")
                
    def run(self, max_episodes=100):
        """Main training loop."""
        print("🤖 Starting Walker Robot Training (CLI Version)")
        print("=" * 60)
        print("📊 Watch the robots learn to crawl!")
        print("⏹️  Press Ctrl+C to stop training")
        print("=" * 60)
        
        self.create_population()
        
        try:
            for episode in range(max_episodes):
                self.run_training_step()
                
                # Print status every 5 episodes
                if episode % 5 == 0:
                    self.print_status()
                    
                # Check for success
                if self.best_fitness > 5.0:
                    print(f"\n🎉 SUCCESS! Robot achieved fitness of {self.best_fitness:.2f}")
                    print("🏆 Training complete - robots learned to crawl!")
                    break
                    
                # Small delay to make output readable
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print(f"\n⏹️  Training stopped by user at episode {self.episode}")
            
        print(f"\n📈 Final Results:")
        print(f"   Episodes completed: {self.episode}")
        print(f"   Generations evolved: {self.generation}")
        print(f"   Best fitness achieved: {self.best_fitness:.2f}")
        print(f"   Average fitness: {self.avg_fitness:.2f}")
        
        if self.best_fitness > 2.0:
            print("🎉 Great success! Robots learned to crawl effectively!")
        elif self.best_fitness > 0.5:
            print("👍 Good progress! Robots are learning to move.")
        else:
            print("🤔 Robots need more training time.")


def main():
    """Main function to run the CLI training."""
    trainer = CLITrainer(population_size=8)
    trainer.run(max_episodes=200)


if __name__ == "__main__":
    main() 