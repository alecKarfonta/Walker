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
from flask import Flask, render_template_string, jsonify, request
import numpy as np
import Box2D as b2
from src.agents.crawling_crate_agent import CrawlingCrateAgent
from src.population.population_controller import PopulationController
from src.population.evolution import EvolutionEngine
from flask_socketio import SocketIO

# Suppress Flask logging for status endpoint
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# HTML template with Canvas rendering
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Walker Training Visualizer (Box2D)</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body, html {
            width: 100%;
            height: 100%;
            overflow: hidden;
            background: #1a1a2e;
            color: #e8e8e8;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        #app-container {
            display: flex;
            flex-direction: column;
            width: 100%;
            height: 100%;
        }
        
        #canvas-wrapper {
            flex-grow: 1; /* Canvas takes up all available space */
            position: relative;
        }

        canvas { 
            display: block;
            width: 100%;
            height: 100%;
        }
        
        /* The new RTS-style bottom bar */
        #bottom-bar {
            flex-shrink: 0; /* Prevent the bar from shrinking */
            height: 220px; /* Adjust height as needed */
            background: rgba(15, 20, 35, 0.95);
            border-top: 3px solid #e74c3c;
            box-shadow: 0 -5px 25px rgba(0,0,0,0.4);
            display: flex;
            padding: 10px;
            gap: 15px;
            z-index: 100;
            overflow: hidden;
        }

        /* Sections within the bottom bar */
        .bottom-bar-section {
            background: rgba(26, 26, 46, 0.8);
            border-radius: 10px;
            border: 1px solid #3498db;
            padding: 15px;
            display: flex;
            flex-direction: column;
            overflow-y: auto;
        }

        #leaderboard-panel {
            flex: 1; /* Flexible width */
        }
        
        #summary-panel {
            flex: 1;
        }

        #controls-panel {
            flex: 2; /* More space for controls */
            display: flex;
            gap: 10px;
        }
        
        .control-column {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .panel-title {
            color: #3498db;
            font-size: 16px;
            font-weight: 700;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .stat-row, .robot-stat-row { 
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 6px 0;
            border-bottom: 1px solid rgba(52, 152, 219, 0.2);
            font-size: 14px;
        }

        .stat-label, .robot-stat-label { color: #bdc3c7; }
        .stat-value, .robot-stat-value {
            color: #ecf0f1;
            font-weight: 700;
            background: #34495e;
            padding: 3px 8px;
            border-radius: 4px;
        }

        /* Collapsible control panels */
        .control-panel {
            background: rgba(30, 40, 60, 0.9);
            border-radius: 8px;
            border: 1px solid #2980b9;
            padding: 10px;
            flex-grow: 1;
        }

        .control-panel-title {
            color: #3498db;
            font-weight: 600;
            cursor: pointer;
            user-select: none;
        }
        
        .control-panel-title::before {
            content: '▶ ';
            display: inline-block;
            transition: transform 0.2s ease-in-out;
        }
        
        .control-panel.open .control-panel-title::before {
            transform: rotate(90deg);
        }
        
        .control-panel-content {
            padding-top: 10px;
            display: none;
        }
        
        .control-panel.open .control-panel-content {
            display: block;
        }
        
        /* Scrollbar styling for panels */
        .bottom-bar-section::-webkit-scrollbar { width: 6px; }
        .bottom-bar-section::-webkit-scrollbar-track { background: transparent; }
        .bottom-bar-section::-webkit-scrollbar-thumb { background: #3498db; border-radius: 3px; }

    </style>
</head>
<body>
    <div id="app-container">
        <div id="canvas-wrapper">
            <canvas id="simulation-canvas"></canvas>
            <button id="resetView" style="position:absolute; top:10px; left:10px; z-index:50;">Reset View</button>
            <div id="focus-indicator" style="display:none; position:absolute; top:10px; right:10px; z-index:50; background:rgba(231, 76, 60, 0.9); color:white; padding:10px 15px; border-radius:5px;">
                🎯 Focused on Agent: <span id="focused-agent-id">-</span>
            </div>
        </div>

        <div id="bottom-bar">
            <!-- Section 1: Leaderboard -->
            <div id="leaderboard-panel" class="bottom-bar-section">
                <div class="panel-title">🏆 Leaderboard</div>
                <div id="leaderboard-content"></div>
            </div>

            <!-- Section 2: Population Summary -->
            <div id="summary-panel" class="bottom-bar-section">
                <div class="panel-title">📊 Population Summary</div>
                <div id="population-summary-content"></div>
            </div>

            <!-- Section 3: Controls -->
            <div id="controls-panel" class="bottom-bar-section">
                <div class="control-column">
                    <div class="control-panel" id="learning-panel">
                        <div class="control-panel-title">Learning Settings</div>
                        <div class="control-panel-content"></div>
                    </div>
                </div>
                <div class="control-column">
                    <div class="control-panel" id="physical-panel">
                        <div class="control-panel-title">Physical Settings</div>
                        <div class="control-panel-content"></div>
                    </div>
                </div>
                <div class="control-column">
                     <div class="control-panel" id="evolution-panel">
                        <div class="control-panel-title">Evolution Settings</div>
                        <div class="control-panel-content"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('simulation-canvas');
        const ctx = canvas.getContext('2d');
        let scale = 15; // pixels per meter
        let offsetX = 0;
        let offsetY = 0;
        let isDragging = false;
        let isDraggingRobot = false;
        let draggedRobotId = null;
        let lastMouseX, lastMouseY;
        let focusedAgentId = null;
        let cameraPosition = { x: 0, y: 0 };
        let cameraZoom = 1.0;
        let mouseDownTime = 0;
        let mouseDownX = 0;
        let mouseDownY = 0;
        let mouseDownRobotId = null;
        const CLICK_THRESHOLD = 5; // pixels
        const CLICK_TIME_THRESHOLD = 200; // milliseconds

        function resizeCanvas() {
            const wrapper = document.getElementById('canvas-wrapper');
            if (!wrapper) return;
            canvas.width = wrapper.clientWidth;
            canvas.height = wrapper.clientHeight;
        }
        
        // Initialize canvas immediately
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);

        document.body.addEventListener('click', (e) => {
            if (e.target.closest('.robot-stat')) {
                const agentId = e.target.closest('.robot-stat').dataset.agentId;
                console.log(`Leaderboard item clicked for agent: ${agentId}`);
                
                fetch('/click', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ agent_id: parseInt(agentId) })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        focusedAgentId = data.agent_id;
                        updateFocusIndicator();
                        console.log(`✅ Agent ${data.agent_id} selected via leaderboard!`);
                    }
                });
            }
        });

        canvas.addEventListener('mousedown', (e) => {
            mouseDownTime = Date.now();
            mouseDownX = e.clientX;
            mouseDownY = e.clientY;
            isDragging = false;
            isDraggingRobot = false;
            draggedRobotId = null;
            lastMouseX = e.clientX;
            lastMouseY = e.clientY;
            
            // Check if we clicked on a robot
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            // Convert screen coordinates to world coordinates using the new camera system
            const worldX = (x - canvas.width / 2) / cameraZoom + cameraPosition.x;
            const worldY = (canvas.height / 2 - y) / cameraZoom + cameraPosition.y;
            
            console.log(`🎯 Mouse down at screen (${x}, ${y}) -> world (${worldX.toFixed(2)}, ${worldY.toFixed(2)})`);
            console.log(`🎯 Canvas rect: ${rect.left}, ${rect.top}, ${rect.width}, ${rect.height}`);
            console.log(`🎯 Camera: pos(${cameraPosition.x.toFixed(2)}, ${cameraPosition.y.toFixed(2)}), zoom: ${cameraZoom}`);
            
            // Visual debug indicator
            const debugOverlay = document.getElementById('debug-overlay');
            debugOverlay.innerHTML = `<div style="position: absolute; left: ${e.clientX}px; top: ${e.clientY}px; width: 20px; height: 20px; background: red; border-radius: 50%; pointer-events: none; z-index: 9999;"></div>`;
            setTimeout(() => debugOverlay.innerHTML = '', 1000);
            
            // Find robot at click position
            fetch('/get_agent_at_position', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    x: worldX,
                    y: worldY
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success' && data.agent_id !== null) {
                    mouseDownRobotId = data.agent_id;
                    console.log(`🤖 Mouse down on robot ${data.agent_id}`);
                    // Visual feedback for robot click
                    debugOverlay.innerHTML += `<div style="position: absolute; left: ${e.clientX}px; top: ${e.clientY}px; width: 30px; height: 30px; background: green; border-radius: 50%; pointer-events: none; z-index: 9999; border: 3px solid yellow;"></div>`;
                } else {
                    mouseDownRobotId = null;
                    console.log(`🖱️ Mouse down on empty space`);
                }
            })
            .catch(error => {
                console.error('Error checking robot at position:', error);
                mouseDownRobotId = null;
            });
        });
        
        canvas.addEventListener('mousemove', (e) => {
            if (mouseDownTime > 0) {
                const distance = Math.sqrt((e.clientX - mouseDownX) ** 2 + (e.clientY - mouseDownY) ** 2);
                
                if (distance > CLICK_THRESHOLD) {
                    if (mouseDownRobotId !== null) {
                        // Dragging a robot
                        if (!isDraggingRobot) {
                            isDraggingRobot = true;
                            draggedRobotId = mouseDownRobotId;
                            console.log(`🤖 Started dragging robot ${draggedRobotId}`);
                        }
                        
                        // Move robot to cursor position
                        const rect = canvas.getBoundingClientRect();
                        const x = e.clientX - rect.left;
                        const y = e.clientY - rect.top;
                        const worldX = (x - canvas.width / 2) / cameraZoom + cameraPosition.x;
                        const worldY = (canvas.height / 2 - y) / cameraZoom + cameraPosition.y;
                        
                        fetch('/move_agent', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({
                                agent_id: draggedRobotId,
                                x: worldX,
                                y: worldY
                            })
                        })
                        .catch(error => {
                            console.error('Error moving robot:', error);
                        });
                    } else {
                        // Dragging camera
                        isDragging = true;
                        cameraPosition.x -= (e.clientX - lastMouseX) / cameraZoom;
                        cameraPosition.y += (e.clientY - lastMouseY) / cameraZoom;
                        lastMouseX = e.clientX;
                        lastMouseY = e.clientY;
                    }
                }
            }
        });
        
        canvas.addEventListener('mouseup', (e) => {
            const timeDiff = Date.now() - mouseDownTime;
            const distance = Math.sqrt((e.clientX - mouseDownX)**2 + (e.clientY - mouseDownY)**2);

            if (!isDragging && !isDraggingRobot && timeDiff < CLICK_TIME_THRESHOLD && distance < CLICK_THRESHOLD) {
                // This is a click, not a drag
                const rect = canvas.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                
                fetch('/click', {
                     method: 'POST',
                     headers: { 'Content-Type': 'application/json' },
                     body: JSON.stringify({
                         screen_x: x,
                         screen_y: y,
                         canvas_width: canvas.width,
                         canvas_height: canvas.height
                     })
                 })
                 .then(response => response.json())
                 .then(data => {
                     if (data.status === 'success') {
                         focusedAgentId = data.agent_id;
                         updateFocusIndicator();
                         if (data.agent_id !== null) {
                             console.log(`✅ Agent ${data.agent_id} selected!`);
                         } else {
                             console.log(`✅ Camera focus cleared`);
                         }
                     }
                 });
            }

            if (isDraggingRobot) {
                console.log(`🤖 Finished dragging robot ${draggedRobotId}`);
            }
            isDragging = false;
            isDraggingRobot = false;
            draggedRobotId = null;
            mouseDownTime = 0;
            mouseDownRobotId = null;
        });
        
        canvas.addEventListener('mouseleave', () => {
            isDragging = false;
            isDraggingRobot = false;
            draggedRobotId = null;
            mouseDownTime = 0;
            mouseDownRobotId = null;
        });

        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const zoomFactor = 1.1;
            const newScale = e.deltaY < 0 ? cameraZoom * zoomFactor : cameraZoom / zoomFactor;
            cameraZoom = Math.max(0.01, Math.min(20, newScale));
        });

        document.getElementById('resetView').addEventListener('click', () => {
            focusedAgentId = null;
            // Reset camera to default view
            cameraPosition = { x: 0, y: 0 };
            cameraZoom = 1.0;

            // Also reset legacy offset and scale if they are used elsewhere
            scale = 15;
            offsetX = canvas.width / 2;
            offsetY = canvas.height * 0.8;
            
            // Hide focus indicator
            const focusIndicator = document.getElementById('focus-indicator');
            if(focusIndicator) {
                focusIndicator.style.display = 'none';
            }
        });

        function getRewardColor(reward) {
            // Define threshold for "close to zero"
            const threshold = 0.1;
            
            if (Math.abs(reward) < threshold) {
                return '#f39c12'; // Yellow for values close to zero
            } else if (reward > 0) {
                return '#27ae60'; // Green for positive values
            } else {
                return '#e74c3c'; // Red for negative values
            }
        }

        function getActionHistoryString(actionHistory) {
            if (!actionHistory || actionHistory.length === 0) {
                return "No actions yet";
            }
            
            // Map action indices to readable names
            const actionNames = {
                0: "None", 1: "S-Fwd", 2: "E-Fwd", 3: "Both-Fwd", 
                4: "S-Back", 5: "E-Back", 6: "S-Back", 7: "E-Back"
            };
            
            // Get the last 5 actions for display
            const recentActions = actionHistory.slice(-5);
            const actionStrings = recentActions.map(idx => actionNames[idx] || `A${idx}`);
            
            return actionStrings.join(" → ");
        }

        function updateStats(data) {
            if (!data) return;

            // Update leaderboard
            const leaderboardContent = document.getElementById('leaderboard-content');
            if (leaderboardContent && data.leaderboard) {
                leaderboardContent.innerHTML = data.leaderboard.map(robot => `
                    <div class="robot-stat-row" data-agent-id="${robot.id}">
                        <span class="robot-stat-label">${robot.name}</span>
                        <span class="robot-stat-value">${robot.distance.toFixed(2)}m</span>
                    </div>
                `).join('');
            }

            // Update population summary
            const populationSummaryContent = document.getElementById('population-summary-content');
            if (populationSummaryContent && data.statistics) {
                 populationSummaryContent.innerHTML = `
                    <div class="stat-row">
                        <span class="stat-label">Generation:</span>
                        <span class="stat-value">${data.statistics.generation || 1}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Avg Distance:</span>
                        <span class="stat-value">${(data.statistics.average_distance || 0).toFixed(2)}m</span>
                    </div>
                 `;
            }
        }

        function drawWorld(data) {
            if (!canvas || !data) return;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.save();
            
            // Apply camera transform:
            // 1. Move origin to center of canvas
            ctx.translate(canvas.width / 2, canvas.height / 2);
            // 2. Zoom and flip Y axis to match physics coordinates
            ctx.scale(cameraZoom, -cameraZoom);
            // 3. Pan the world so the camera position is at the center
            ctx.translate(-cameraPosition.x, -cameraPosition.y);
            
            // Draw ground from geometry
            if (data.shapes && data.shapes.ground) {
                 const gradient = ctx.createLinearGradient(0, -1, 0, 1);
                gradient.addColorStop(0, '#5e738c');
                gradient.addColorStop(1, '#34495e');
                ctx.fillStyle = gradient;

                data.shapes.ground.forEach(geom => {
                    if (geom.type === 'polygon' && geom.vertices.length > 0) {
                        ctx.beginPath();
                        ctx.moveTo(geom.vertices[0][0], geom.vertices[0][1]);
                        for (let i = 1; i < geom.vertices.length; i++) {
                            ctx.lineTo(geom.vertices[i][0], geom.vertices[i][1]);
                        }
                        ctx.closePath();
                        ctx.fill();
                    } else if (geom.type === 'line') {
                        // Fallback for old line-based ground
                        ctx.strokeStyle = '#34495e';
                        ctx.lineWidth = 0.1;
                        ctx.beginPath();
                        ctx.moveTo(geom.vertices[0][0], geom.vertices[0][1]);
                        ctx.lineTo(geom.vertices[1][0], geom.vertices[1][1]);
                        ctx.stroke();
                    }
                });
            }

            if (data.shapes && data.shapes.robots) {
                data.shapes.robots.forEach(robot => {
                    const isFocused = robot.id === focusedAgentId;
                    
                    robot.body_parts.forEach(part => {
                        if (part.type === 'circle') { // Wheels
                            const wheelGradient = ctx.createRadialGradient(
                                part.center[0], part.center[1], 0,
                                part.center[0], part.center[1], part.radius
                            );
                            wheelGradient.addColorStop(0, '#3498db');
                            wheelGradient.addColorStop(1, '#2980b9');
                            ctx.fillStyle = wheelGradient;
                        } else { // Body parts
                            ctx.fillStyle = isFocused ? '#e74c3c' : '#c0392b'; // Red if focused
                        }
                        
                        ctx.strokeStyle = '#2c3e50';
                        ctx.lineWidth = 0.05;

                        if (part.type === 'polygon') {
                            ctx.beginPath();
                            ctx.moveTo(part.vertices[0][0], part.vertices[0][1]);
                            for (let i = 1; i < part.vertices.length; i++) {
                                ctx.lineTo(part.vertices[i][0], part.vertices[i][1]);
                            }
                            ctx.closePath();
                            ctx.fill();
                            ctx.stroke();
                        } else if (part.type === 'circle') {
                            ctx.beginPath();
                            ctx.arc(part.center[0], part.center[1], part.radius, 0, Math.PI * 2);
                            ctx.fill();
                            ctx.stroke();
                        }
                    });
                });
            }
            ctx.restore();
        }

        function fetchData() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    drawWorld(data);
                    updateStats(data);
                    requestAnimationFrame(fetchData);
                })
                .catch(error => {
                    console.error('Error fetching data:', error);
                    setTimeout(fetchData, 1000); // Try again after a second
                });
        }
        fetchData();

        function updateFocusIndicator() {
            const indicator = document.getElementById('focus-indicator');
            const agentIdSpan = document.getElementById('focused-agent-id');
            if (!indicator || !agentIdSpan) return;

            if (focusedAgentId !== null) {
                agentIdSpan.textContent = focusedAgentId;
                indicator.style.display = 'block';
            } else {
                indicator.style.display = 'none';
            }
        }

        // --- Control Panel Interactivity ---
        function createSlider(id, label, min, max, step, value) {
            const container = document.createElement('div');
            container.className = 'control-row'; // You might need to style this class
            
            const labelEl = document.createElement('span');
            labelEl.className = 'control-label';
            labelEl.textContent = label;
            
            const slider = document.createElement('input');
            slider.type = 'range';
            slider.id = id;
            slider.min = min;
            slider.max = max;
            slider.step = step;
            slider.value = value;
            
            const valueEl = document.createElement('span');
            valueEl.className = 'control-value';
            valueEl.textContent = parseFloat(value).toFixed(3);
            
            slider.addEventListener('input', () => {
                valueEl.textContent = parseFloat(slider.value).toFixed(3);
            });

            slider.addEventListener('change', () => {
                updateAgentParams({ [id]: parseFloat(slider.value) });
            });
            
            container.appendChild(labelEl);
            container.appendChild(slider);
            container.appendChild(valueEl);
            
            return container;
        }

        const learningPanelContent = document.querySelector('#learning-panel .control-panel-content');
        if (learningPanelContent) {
            learningPanelContent.appendChild(createSlider('learning_rate', 'Learning Rate', 0.001, 0.1, 0.001, 0.005));
            learningPanelContent.appendChild(createSlider('epsilon', 'Epsilon (Randomness)', 0.0, 1.0, 0.01, 0.3));
        }

        document.querySelectorAll('.control-panel-title').forEach(title => {
            title.addEventListener('click', () => {
                title.parentElement.classList.toggle('open');
            });
        });

        async function updateAgentParams(params) {
            // Include focused agent ID if available
            if (focusedAgentId !== null) {
                params.target_agent_id = focusedAgentId;
            }
            
            try {
                await fetch('/update_agent_params', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(params)
                });
                console.log('Agent parameters updated:', params);
            } catch (err) {
                console.error('Error updating agent parameters:', err);
            }
        }

    </script>
</body>
</html>
"""


class TrainingEnvironment:
    """
    Manages the physics simulation and training of crawling crate agents using Box2D.
    """
    def __init__(self, num_agents=50):
        self.num_agents = num_agents
        self.world = b2.b2World(gravity=(0, -10), doSleep=True)
        self.dt = 1.0 / 60.0

        # World bounds for resetting fallen agents
        self.world_bounds_y = -20.0 # Reset if agent falls below this y-coordinate
        
        # --- Collision Filtering Setup ---
        # Box2D uses 16-bit collision categories, so we need a different approach for 50 agents
        # Instead of unique categories per agent, we'll use a simpler approach:
        # - Ground: category 0x0001
        # - All agents: category 0x0002 (shared category)
        # - Agents only collide with ground, not with each other
        self.GROUND_CATEGORY = 0x0001
        self.AGENT_CATEGORY = 0x0002

        # 2. Create the ground with its own category.
        #    Its mask is set to collide with ALL agents.
        self._create_ground()

        # 3. Create agents, all with the same category.
        #    Their masks are set to collide ONLY with the ground.
        self.agents = []
        for i in range(self.num_agents):
            # Reduce spacing for 50 agents to fit them all
            spacing = 8 if self.num_agents > 20 else 15
            agent = CrawlingCrateAgent(
                self.world,
                agent_id=i,
                position=(i * spacing, 6),
                category_bits=self.AGENT_CATEGORY,
                mask_bits=self.GROUND_CATEGORY  # Only collide with the ground
            )
            agent.body.awake = True
            self.agents.append(agent)

        # --- Statistics and State ---
        self.step_count = 0
        self.robot_stats = {}
        self.population_stats = {
            'total_distance': 0,
            'best_distance': 0,
            'average_distance': 0,
            'generation': 1,
            'total_steps': 0,
            'q_learning_stats': {
                'avg_epsilon': 0,
                'avg_learning_rate': 0,
                'total_q_updates': 0,
                'avg_q_value': 0
            }
        }
        self.is_running = False
        self.thread = None
        self.episode_length = 1200
        self.episode_step = 0
        
        # Statistics update timing
        self.stats_update_interval = 0.1  # Update stats every 0.1 seconds
        self.steps_per_stats_update = int(self.stats_update_interval / self.dt)  # 6 steps at 60fps
        self.last_stats_update = 0
        
        # Settle the world
        for _ in range(10):
            self.world.Step(self.dt, 8, 3)

        self.population_controller = PopulationController(len(self.agents))
        self.evolution_engine = EvolutionEngine(self.population_controller)
        self.mutation_rate = 0.1 # Default mutation rate

        # Camera and focus system
        self.focused_agent = None
        self.camera_target = (0, 0)
        self.camera_position = (0, 0)
        self.camera_zoom = 1.0
        self.target_zoom = 1.0
        self.follow_speed = 0.05
        self.zoom_speed = 0.05

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
                categoryBits=self.GROUND_CATEGORY,
                maskBits=self.AGENT_CATEGORY  # Collide with all agents
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
        print(f"🔧 Physics timestep: {self.dt}")
        
        # Initialize robot statistics
        self._init_robot_stats()
        
        # Test physics world
        print("🔧 Testing physics world...")
        for i in range(5):
            self.world.Step(self.dt, 8, 3)
            print(f"   Step {i}: World bodies: {len(self.world.bodies)}")
        
        while self.is_running:
            current_time = time.time()
            delta_time = min(current_time - last_step_time, 1.0 / 30.0)  # Cap at 30 FPS
            
            # Update camera
            self.update_camera(delta_time)
            
            # Step the physics world
            self.world.Step(self.dt, 8, 3)
            step_count += 1
            
            # Update all agents
            for agent in self.agents:
                agent.step(delta_time)
            
            # Check for fallen agents and reset them
            for agent in self.agents:
                if agent.body.position.y < self.world_bounds_y:
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
            time.sleep(max(0, self.dt - (time.time() - current_time)))

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
        """Returns the current state of the simulation for rendering."""
        if not self.is_running:
            return {'shapes': {}, 'leaderboard': [], 'robots': [], 'statistics': {}, 'camera': self.get_camera_state()}

        # 1. Get agent shapes for drawing
        robot_shapes = []
        for agent in self.agents:
            body_parts = []
            # Chassis, Arms, Wheels
            for part in [agent.body] + agent.wheels + [agent.upper_arm, agent.lower_arm]:
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
            robot_shapes.append({'id': agent.id, 'body_parts': body_parts})

        # 2. Get ground shapes for drawing
        ground_shapes = []
        for body in self.world.bodies:
            if body.type == b2.b2_staticBody:
                for fixture in body.fixtures:
                    shape = fixture.shape
                    if isinstance(shape, b2.b2PolygonShape):
                        ground_shapes.append({
                            'type': 'polygon',
                            'vertices': [tuple(body.GetWorldPoint(v)) for v in shape.vertices]
                        })
        
        # 3. Get leaderboard data (top 10 robots)
        sorted_robots = sorted(self.robot_stats.values(), key=lambda r: r.get('total_distance', 0), reverse=True)
        leaderboard_data = [
            {'name': f"Robot {r['id']}", 'distance': r.get('total_distance', 0)}
            for r in sorted_robots[:10]
        ]
        
        # 4. Get detailed stats for side panel (top 10)
        robot_details = []
        for i, r_stat in enumerate(sorted_robots[:10]):
            agent = self.agents[r_stat['id']]
            robot_details.append({
                'id': r_stat['id'],
                'name': f"Robot {r_stat['id']}",
                'rank': i + 1,
                'distance': r_stat.get('total_distance', 0),
                'position': r_stat.get('current_position', (0,0)),
                'episode_reward': r_stat.get('episode_reward', 0)
            })

        return {
            'shapes': {'robots': robot_shapes, 'ground': ground_shapes},
            'leaderboard': leaderboard_data,
            'robots': robot_details,
            'statistics': self.population_stats,
            'camera': self.get_camera_state()
        }

    def start(self):
        """Starts the training loop in a separate thread."""
        if not self.is_running:
            print("🔄 Starting training loop thread...")
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
        if self.thread:
            self.thread.join()
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
            category_bits=self.AGENT_CATEGORY,
            mask_bits=self.GROUND_CATEGORY
        )
        self.agents.append(new_agent)
        self.population_controller.add_agent(new_agent)
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
            category_bits=self.AGENT_CATEGORY,
            mask_bits=self.GROUND_CATEGORY
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
        self.population_controller.add_agent(cloned_agent)
        self.num_agents = len(self.agents)
        print(f"👯 Cloned best agent {best_agent.id} to new agent {new_id}. Total agents: {self.num_agents}")

    def evolve_population(self):
        """Runs the evolution engine to create a new generation."""
        # Update fitness values before evolution
        for agent in self.agents:
            self.population_controller.update_agent_fitness(agent, agent.get_fitness())
        
        new_population = self.evolution_engine.evolve_generation()
        
        # Simple replacement: clear old agents and add new ones
        for agent in self.agents:
            agent.destroy()

        self.agents = new_population
        self.num_agents = len(self.agents)

        # Re-initialize controller and stats
        self.population_controller = PopulationController(len(self.agents))
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
            self.camera_position[0] + (self.camera_target[0] - self.camera_position[0]) * self.follow_speed,
            self.camera_position[1] + (self.camera_target[1] - self.camera_position[1]) * self.follow_speed
        )
        
        # Smooth zoom
        if abs(self.target_zoom - self.camera_zoom) > 0.001:
            self.camera_zoom += (self.target_zoom - self.camera_zoom) * self.zoom_speed

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

# --- Main Execution ---
app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')
env = TrainingEnvironment(num_agents=50)

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/status')
def status():
    # Suppress logging for status endpoint to reduce noise
    return jsonify(env.get_status())

@app.route('/start', methods=['POST'])
def start_training():
    print("🚀 Starting training via web endpoint")
    env.start()
    return jsonify({'status': 'Training started'})

@app.route('/stop', methods=['POST'])
def stop_training():
    print("🛑 Stopping training via web endpoint")
    env.stop()
    return jsonify({'status': 'Training stopped'})

@app.route('/click', methods=['POST'])
def handle_click():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No data provided'}), 400

    agent_id = None
    if 'agent_id' in data:
        # Click from the leaderboard
        clicked_agent = next((agent for agent in env.agents if agent.id == data['agent_id']), None)
        env.focus_on_agent(clicked_agent)
        agent_id = clicked_agent.id if clicked_agent else None
    elif 'screen_x' in data and 'screen_y' in data:
        # Click from the canvas
        agent_id = env.handle_click(
            data['screen_x'], data['screen_y'],
            data.get('canvas_width', 800),
            data.get('canvas_height', 600)
        )
    
    return jsonify({
        'status': 'success',
        'agent_id': agent_id,
        'focused': agent_id is not None
    })

@app.route('/get_agent_at_position', methods=['POST'])
def get_agent_at_position():
    data = request.get_json()
    if not data or 'x' not in data or 'y' not in data:
        return jsonify({'status': 'error', 'message': 'Missing coordinates'}), 400
    
    world_x = data['x']
    world_y = data['y']
    
    agent = env.get_agent_at_position(world_x, world_y)
    agent_id = agent.id if agent else None
    
    return jsonify({
        'status': 'success',
        'agent_id': agent_id
    })

@app.route('/move_agent', methods=['POST'])
def move_agent():
    data = request.get_json()
    if not data or 'agent_id' not in data or 'x' not in data or 'y' not in data:
        return jsonify({'status': 'error', 'message': 'Missing agent_id or coordinates'}), 400
    
    agent_id = data['agent_id']
    x = data['x']
    y = data['y']
    
    success = env.move_agent(agent_id, x, y)
    
    if success:
        return jsonify({'status': 'success', 'agent_id': agent_id})
    else:
        return jsonify({'status': 'error', 'message': f'Failed to move agent {agent_id}'}), 500

@app.route('/update_agent_params', methods=['POST'])
def update_agent_params():
    params = request.get_json()
    if not params:
        return jsonify({'status': 'error', 'message': 'No parameters provided'}), 400
    
    # Check if we should target a specific agent
    target_agent_id = None
    if 'target_agent_id' in params:
        target_agent_id = params.pop('target_agent_id')
    
    success = env.update_agent_params(params, target_agent_id)
    
    if success:
        return jsonify({'status': 'success', 'updated_params': params})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to update parameters'}), 500

@app.route('/evolution_event', methods=['POST'])
def evolution_event():
    data = request.get_json()
    if not data or 'event' not in data:
        return jsonify({'status': 'error', 'message': 'No event specified'}), 400
    
    event = data['event']
    
    try:
        if event == 'spawn':
            env.spawn_agent()
        elif event == 'clone':
            env.clone_best_agent()
        elif event == 'evolve':
            env.evolve_population()
        else:
            return jsonify({'status': 'error', 'message': f'Unknown event: {event}'}), 400
        
        return jsonify({'status': 'success', 'event': event})
    except Exception as e:
        print(f"❌ Evolution event '{event}' failed: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

def main():
    # Set a different port for the web server to avoid conflicts
    web_port = 7777
    
    # Start the training loop
    env.start()
    
    # Start the web server in a separate thread
    server_thread = threading.Thread(
        target=lambda: socketio.run(app, host='0.0.0.0', port=web_port, allow_unsafe_werkzeug=True),
        daemon=True
    )
    server_thread.start()
    
    print(f"✅ Web server started on http://localhost:{web_port}")
    
    # Keep the main thread alive to allow background threads to run
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("🛑 Shutting down training environment...")
        env.stop()
        print("✅ Training stopped.")

if __name__ == "__main__":
    main()

# When the script exits, ensure the environment is stopped
import atexit
atexit.register(lambda: env.stop()) 