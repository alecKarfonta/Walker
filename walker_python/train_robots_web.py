#!/usr/bin/env python3
"""
Web-based training visualization for remote server access.
Access via browser at http://your-server:8080
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import pymunk
import numpy as np
import time
import json
import threading
from flask import Flask, render_template_string, jsonify, request
from src.agents.crawling_crate import CrawlingCrate
from src.population.population_controller import PopulationController
from src.population.evolution import EvolutionEngine


# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🤖 Walker Robot Training</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #333; color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .stat-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .stat-value { font-size: 2em; font-weight: bold; color: #007bff; }
        .controls { background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .btn { padding: 10px 20px; margin: 5px; border: none; border-radius: 5px; cursor: pointer; }
        .btn-primary { background: #007bff; color: white; }
        .btn-success { background: #28a745; color: white; }
        .btn-warning { background: #ffc107; color: black; }
        .btn-danger { background: #dc3545; color: white; }
        .robot-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px; }
        .robot-card { background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .fitness-bar { background: #e9ecef; height: 20px; border-radius: 10px; overflow: hidden; margin: 10px 0; }
        .fitness-fill { background: linear-gradient(90deg, #28a745, #20c997); height: 100%; transition: width 0.3s; }
        .chart-container { background: white; padding: 20px; border-radius: 10px; height: 400px; }
        .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
        .status.running { background: #d4edda; color: #155724; }
        .status.paused { background: #fff3cd; color: #856404; }
        .status.stopped { background: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Walker Robot Training</h1>
            <p>Watch robots learn to crawl in real-time!</p>
        </div>
        
        <div class="controls">
            <button class="btn btn-primary" onclick="toggleTraining()" id="toggleBtn">⏸️ Pause</button>
            <button class="btn btn-success" onclick="resetPopulation()">🔄 Reset</button>
            <button class="btn btn-warning" onclick="changeSpeed(0.5)">🐌 Slow</button>
            <button class="btn btn-warning" onclick="changeSpeed(1.0)">⚡ Normal</button>
            <button class="btn btn-warning" onclick="changeSpeed(2.0)">🚀 Fast</button>
            <span class="status" id="status">🟢 Running</span>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <h3>Generation</h3>
                <div class="stat-value" id="generation">0</div>
            </div>
            <div class="stat-card">
                <h3>Episode</h3>
                <div class="stat-value" id="episode">0</div>
            </div>
            <div class="stat-card">
                <h3>Best Fitness</h3>
                <div class="stat-value" id="bestFitness">0.00</div>
            </div>
            <div class="stat-card">
                <h3>Avg Fitness</h3>
                <div class="stat-value" id="avgFitness">0.00</div>
            </div>
        </div>
        
        <div class="robot-grid" id="robotGrid">
            <!-- Robot cards will be populated here -->
        </div>
        
        <div class="chart-container">
            <h3>📈 Fitness Progress</h3>
            <canvas id="fitnessChart" width="800" height="300"></canvas>
        </div>
    </div>
    
    <script>
        let chartCtx;
        let fitnessData = [];
        
        function initChart() {
            const canvas = document.getElementById('fitnessChart');
            chartCtx = canvas.getContext('2d');
            chartCtx.fillStyle = '#f8f9fa';
            chartCtx.fillRect(0, 0, canvas.width, canvas.height);
        }
        
        function updateChart(data) {
            if (!chartCtx) return;
            
            const canvas = document.getElementById('fitnessChart');
            chartCtx.clearRect(0, 0, canvas.width, canvas.height);
            
            if (data.length < 2) return;
            
            const maxFitness = Math.max(...data);
            const scaleY = canvas.height / (maxFitness * 1.2);
            const scaleX = canvas.width / (data.length - 1);
            
            chartCtx.strokeStyle = '#007bff';
            chartCtx.lineWidth = 2;
            chartCtx.beginPath();
            
            data.forEach((fitness, i) => {
                const x = i * scaleX;
                const y = canvas.height - (fitness * scaleY);
                if (i === 0) {
                    chartCtx.moveTo(x, y);
                } else {
                    chartCtx.lineTo(x, y);
                }
            });
            
            chartCtx.stroke();
        }
        
        function updateRobotGrid(robots) {
            const grid = document.getElementById('robotGrid');
            grid.innerHTML = '';
            
            robots.forEach((robot, i) => {
                const card = document.createElement('div');
                card.className = 'robot-card';
                
                const fitnessPercent = Math.min(100, (robot.fitness / 10) * 100);
                const direction = robot.velocity_x > 0 ? '🟢' : robot.velocity_x < 0 ? '🔴' : '⚪';
                
                card.innerHTML = `
                    <h4>Robot ${i}</h4>
                    <p>Position: (${robot.pos_x.toFixed(1)}, ${robot.pos_y.toFixed(1)})</p>
                    <p>Velocity: ${robot.velocity_x.toFixed(2)} m/s ${direction}</p>
                    <p>Fitness: ${robot.fitness.toFixed(2)}</p>
                    <div class="fitness-bar">
                        <div class="fitness-fill" style="width: ${fitnessPercent}%"></div>
                    </div>
                `;
                grid.appendChild(card);
            });
        }
        
        function updateStats(stats) {
            document.getElementById('generation').textContent = stats.generation;
            document.getElementById('episode').textContent = stats.episode;
            document.getElementById('bestFitness').textContent = stats.best_fitness.toFixed(2);
            document.getElementById('avgFitness').textContent = stats.avg_fitness.toFixed(2);
        }
        
        function toggleTraining() {
            fetch('/toggle', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    const btn = document.getElementById('toggleBtn');
                    const status = document.getElementById('status');
                    if (data.paused) {
                        btn.textContent = '▶️ Resume';
                        status.textContent = '🟡 Paused';
                        status.className = 'status paused';
                    } else {
                        btn.textContent = '⏸️ Pause';
                        status.textContent = '🟢 Running';
                        status.className = 'status running';
                    }
                });
        }
        
        function resetPopulation() {
            fetch('/reset', {method: 'POST'});
        }
        
        function changeSpeed(speed) {
            fetch('/speed', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({speed: speed})});
        }
        
        function updateDisplay() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    updateStats(data);
                    updateRobotGrid(data.robots);
                    updateChart(data.fitness_history);
                });
        }
        
        // Initialize and start updates
        initChart();
        setInterval(updateDisplay, 1000); // Update every second
    </script>
</body>
</html>
"""


class WebTrainer:
    """Web-based training system for remote access."""
    
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
        
        # UI state
        self.paused = False
        self.training_speed = 1.0
        
        # Evolution
        self.evolution_engine = None
        self.population_controller = None
        
        # Training thread
        self.training_thread = None
        self.running = False
        
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
            agent = CrawlingCrate(self.space, position=(x_pos, 20))
            self.agents.append(agent)
            
        # Initialize evolution system
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
        
    def get_status(self):
        """Get current training status for web display."""
        robots = []
        for agent in self.agents:
            debug = agent.get_debug_info()
            robots.append({
                'pos_x': debug['crate_pos'][0],
                'pos_y': debug['crate_pos'][1],
                'velocity_x': debug['crate_vel'][0],
                'velocity_y': debug['crate_vel'][1],
                'fitness': self.evaluate_agent(agent, max_steps=50)  # Quick evaluation
            })
            
        return {
            'generation': self.generation,
            'episode': self.episode,
            'best_fitness': self.best_fitness,
            'avg_fitness': self.avg_fitness,
            'robots': robots,
            'fitness_history': self.fitness_history,
            'paused': self.paused
        }
        
    def training_loop(self):
        """Main training loop running in background thread."""
        while self.running:
            for _ in range(int(self.training_speed)):
                self.run_training_step()
            time.sleep(0.1)  # Small delay
            
    def start_training(self):
        """Start the training in a background thread."""
        if not self.running:
            self.running = True
            self.training_thread = threading.Thread(target=self.training_loop)
            self.training_thread.daemon = True
            self.training_thread.start()
            
    def stop_training(self):
        """Stop the training."""
        self.running = False


# Flask app
app = Flask(__name__)
trainer = WebTrainer(population_size=8)

@app.route('/')
def index():
    return HTML_TEMPLATE

@app.route('/status')
def status():
    return jsonify(trainer.get_status())

@app.route('/toggle', methods=['POST'])
def toggle():
    trainer.paused = not trainer.paused
    return jsonify({'paused': trainer.paused})

@app.route('/reset', methods=['POST'])
def reset():
    trainer.create_population()
    trainer.generation = 0
    trainer.episode = 0
    trainer.best_fitness = 0.0
    trainer.avg_fitness = 0.0
    trainer.fitness_history = []
    return jsonify({'status': 'reset'})

@app.route('/speed', methods=['POST'])
def speed():
    data = request.get_json()
    trainer.training_speed = data.get('speed', 1.0)
    return jsonify({'speed': trainer.training_speed})

def main():
    """Start the web server."""
    print("🌐 Starting Web-Based Walker Robot Training")
    print("=" * 50)
    print("📊 Access the training at: http://localhost:8080")
    print("🌍 For remote access: http://your-server-ip:8080")
    print("⏹️  Press Ctrl+C to stop")
    print("=" * 50)
    
    # Create population and start training
    trainer.create_population()
    trainer.start_training()
    
    # Start Flask app
    app.run(host='0.0.0.0', port=7777, debug=False)

if __name__ == "__main__":
    main() 