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
from typing import Dict, Any, List, Optional
from flask import Flask, render_template_string, jsonify, request
import numpy as np
import Box2D as b2
from src.agents.evolutionary_crawling_agent import EvolutionaryCrawlingAgent
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
from src.agents.ecosystem_interface import EcosystemInterface
from src.agents.survival_q_integration_patch import upgrade_agent_to_survival_learning
from src.agents.learning_manager import LearningManager, LearningApproach

# Import elite robot management
from src.persistence import EliteManager

# Import realistic terrain generation
from src.terrain_generation import generate_robot_scale_terrain

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
            height: 280px; /* Increased height for better visibility */
            background: rgba(15, 20, 35, 0.95);
            border-top: 2px solid #e74c3c;
            box-shadow: 0 -5px 20px rgba(0,0,0,0.3);
            display: flex;
            padding: 12px; /* Increased padding */
            gap: 12px;   /* Increased gap */
            z-index: 100;
            overflow: hidden;
        }

        /* Sections within the bottom bar */
        .bottom-bar-section {
            background: rgba(26, 26, 46, 0.8);
            border-radius: 8px;
            border: 1px solid #3498db;
            padding: 10px; /* Reduced padding */
            display: flex;
            flex-direction: column;
            overflow-y: auto;
        }

        #leaderboard-panel {
            flex: 2; /* More space for leaderboard */
        }
        
        #robot-details-panel {
            flex: 1.5;
            min-width: 200px;
        }

        #summary-and-controls-panel {
            flex: 3;
            display: flex;
            flex-direction: column;
            gap: 8px;
            background: transparent;
            border: none;
            padding: 0;
        }

        #summary-panel {
            flex-shrink: 0;
        }

        #controls-panel {
            flex-grow: 1; 
            display: flex;
            gap: 8px;
            padding: 0;
            border: none;
            background: transparent;
        }
        
        .control-column {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .panel-title {
            color: #3498db;
            font-size: 14px; /* Smaller font */
            font-weight: 600;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px; /* Tighter spacing */
        }

        .stat-row { 
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 4px 0; /* Tighter padding */
            border-bottom: 1px solid rgba(52, 152, 219, 0.15);
            font-size: 13px; /* Smaller font */
        }

        .robot-stat-row { 
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 12px; /* More padding for button-like feel */
            border-bottom: 1px solid rgba(52, 152, 219, 0.15);
            font-size: 13px;
            cursor: pointer; /* Show it's clickable */
            transition: all 0.2s ease;
            border-radius: 6px;
            margin: 2px 0;
            background: rgba(52, 152, 219, 0.1);
            border-left: 3px solid transparent;
            user-select: none; /* Prevent text selection */
            position: relative;
        }

        .robot-stat-row:hover {
            background: rgba(52, 152, 219, 0.3);
            transform: translateX(3px);
            border-left: 3px solid #3498db;
            box-shadow: 0 3px 8px rgba(0,0,0,0.3);
        }

        .robot-stat-row:active {
            background: rgba(52, 152, 219, 0.4);
            transform: translateX(1px);
            box-shadow: 0 1px 3px rgba(0,0,0,0.4);
        }

        .robot-stat-row.focused {
            background: rgba(231, 76, 60, 0.8);
            border-left: 3px solid #c0392b;
            color: white;
        }

        .robot-stat-row.focused:hover {
            background: rgba(192, 57, 43, 0.9);
            border-left: 3px solid #a93226;
        }

        .stat-label, .robot-stat-label { color: #bdc3c7; }
        .stat-value, .robot-stat-value {
            color: #ecf0f1;
            font-weight: 700;
            background: #34495e;
            padding: 3px 8px;
            border-radius: 4px;
        }

        .robot-stat-row.focused .robot-stat-label,
        .robot-stat-row.focused .robot-stat-value {
            color: #fff;
        }

        .robot-stat-row.focused .robot-stat-value {
            background: rgba(255, 255, 255, 0.2);
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

        .robot-details-title {
            font-size: 12px;
            font-weight: bold;
            color: #3498db;
            margin-bottom: 8px;
            border-bottom: 1px solid #3498db;
            padding-bottom: 4px;
        }

        .robot-details-content {
            font-size: 10px;
            line-height: 1.3;
        }

        .detail-section {
            margin-bottom: 8px;
            padding: 6px;
            background: rgba(52, 152, 219, 0.1);
            border-radius: 4px;
            border-left: 2px solid #3498db;
        }

        .detail-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 3px;
        }

        .detail-row:last-child {
            margin-bottom: 0;
        }

        .detail-label {
            color: #bdc3c7;
            font-weight: 500;
        }

        .detail-value {
            color: #ecf0f1;
            font-weight: bold;
        }

        /* Control panel styling */

    </style>
</head>
<body>
    <div id="app-container">
        <div id="canvas-wrapper">
            <canvas id="simulation-canvas"></canvas>
            <button id="resetView" style="position:absolute; top:10px; left:10px; z-index:50;">Reset View</button>
            <button id="toggleFoodLines" onclick="toggleFoodLines()" style="position:absolute; top:10px; left:120px; z-index:50; background:#4CAF50; color:white; border:none; padding:5px 10px; border-radius:3px; cursor:pointer;">Show Food Lines</button>
            <button id="testCarnivoreFeeding" onclick="testCarnivoreFeeding()" style="position:absolute; top:10px; left:250px; z-index:50; background:#FF4444; color:white; border:none; padding:5px 10px; border-radius:3px; cursor:pointer;">Test Carnivore</button>
            <div id="focus-indicator" style="display:none; position:absolute; top:1%; left:50%; transform:translate(-50%, -50%); z-index:50; background:rgba(231, 76, 60, 0.95); color:white; padding:15px 20px; border-radius:8px; box-shadow:0 4px 20px rgba(0,0,0,0.3); border:2px solid rgba(255,255,255,0.2);">
                🎯 Focused on Agent: <span id="focused-agent-id">-</span>
            </div>
        </div>

        <div id="bottom-bar">
            <!-- Section 1: Leaderboard -->
            <div id="leaderboard-panel" class="bottom-bar-section">
                <div class="panel-title">🏆 Leaderboard (Food)</div>
                <div id="leaderboard-content"></div>
            </div>

            <!-- Section 2: Focused Robot Details -->
            <div id="robot-details-panel" class="bottom-bar-section">
                <div class="panel-title">🤖 Robot Details</div>
                <div id="robot-details-content">
                    <div class="placeholder">Select a robot to see details.</div>
                </div>
            </div>

            <!-- Section 3: Summary and Controls -->
            <div id="summary-and-controls-panel">
                <div id="summary-panel" class="bottom-bar-section">
                    <div class="panel-title">📊 Population Summary</div>
                    <div id="population-summary-content"></div>
                </div>

                <div id="controls-panel">
                    <div class="control-column">
                        <div class="control-panel" id="learning-panel">
                            <div class="control-panel-title">Learning</div>
                            <div class="control-panel-content"></div>
                        </div>
                    </div>
                    <div class="control-column">
                        <div class="control-panel" id="physical-panel">
                            <div class="control-panel-title">Physical</div>
                            <div class="control-panel-content"></div>
                        </div>
                    </div>
                    <div class="control-column">
                         <div class="control-panel" id="evolution-panel">
                            <div class="control-panel-title">Evolution</div>
                            <div class="control-panel-content"></div>
                        </div>
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
        let userHasManuallyPanned = false; // track manual camera pan
        let mouseDownTime = 0;
        let mouseDownX = 0;
        let mouseDownY = 0;
        let mouseDownRobotId = null;
        let lastLeaderboardHtml = ''; // Variable to store the last state of the leaderboard HTML
        let showFoodLines = false; // Food lines disabled by default for performance
        let showFpsCounters = true; // FPS counters enabled by default
        
        // FPS Tracking
        let uiFpsCounter = 0;
        let uiFpsStartTime = Date.now();
        let lastUiFpsUpdate = Date.now();
        let currentUiFps = 0;
        let physicsStepsCounter = 0;
        let physicsStepsStartTime = Date.now();
        let lastPhysicsFpsUpdate = Date.now();
        let currentPhysicsFps = 0;
        
        // Missing constants that were causing errors
        const CLICK_THRESHOLD = 5; // pixels
        const CLICK_TIME_THRESHOLD = 200; // milliseconds
        
        // Enhanced visualization constants
        const ECOSYSTEM_COLORS = {
            'carnivore': '#FF4444',    // Red for predators
            'herbivore': '#44AA44',    // Green for prey
            'omnivore': '#FF8844',     // Orange for omnivores
            'scavenger': '#8844AA',    // Purple for scavengers
            'symbiont': '#4488FF'      // Blue for symbionts
        };
        
        const STATUS_COLORS = {
            'hunting': '#FF0000',      // Bright red
            'feeding': '#00FF00',      // Bright green
            'fleeing': '#FFFF00',      // Yellow
            'territorial': '#FF8800',  // Orange
            'idle': '#CCCCCC',         // Gray
            'moving': '#FFFFFF',       // White
            'active': '#88DDFF'        // Light blue
        };
        
        // Visualization state
        let ecosystemData = null;
        let environmentData = null;
        let predationEvents = [];
        // Animation trails disabled for performance optimization

        function resizeCanvas() {
            const wrapper = document.getElementById('canvas-wrapper');
            if (!wrapper) return;
            canvas.width = wrapper.clientWidth;
            canvas.height = wrapper.clientHeight;
        }
        
        // Initialize canvas immediately
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);

        const leaderboardPanel = document.getElementById('leaderboard-panel');
        leaderboardPanel.addEventListener('click', function(e) {
            const robotRow = e.target.closest('.robot-stat-row');
            if (robotRow && robotRow.dataset.agentId) {
                e.preventDefault();
                e.stopPropagation();

                const agentId = robotRow.dataset.agentId;  // Keep as string, don't parseInt
                console.log(`🎯 CLIENT: Leaderboard button clicked for agent: ${agentId}`);

                // Send the click to the server to select the agent
                fetch('./click', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ agent_id: agentId })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        focusedAgentId = data.agent_id;
                        if (data.agent_id !== null) {
                            // Re-enable camera tracking when selecting from leaderboard
                            userHasManuallyPanned = false;
                            console.log(`✅ Agent ${data.agent_id} selected from leaderboard! Camera tracking enabled.`);
                        }
                    }
                })
                .catch(error => {
                    console.error('❌ Error during leaderboard click fetch:', error);
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
            
            // Find robot at click position
            fetch('./get_agent_at_position', {
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
                        
                        fetch('./move_agent', {
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
                        userHasManuallyPanned = true; // flag manual pan
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
                // This is a click, not a drag. We use the robot ID found during mousedown.
                fetch('./click', {
                     method: 'POST',
                     headers: { 'Content-Type': 'application/json' },
                     body: JSON.stringify({ agent_id: mouseDownRobotId }) // Send ID directly
                 })
                 .then(response => response.json())
                 .then(data => {
                     if (data.status === 'success') {
                         focusedAgentId = data.agent_id;
                         if (data.agent_id !== null) {
                             // Re-enable camera tracking when focusing on a robot
                             userHasManuallyPanned = false;
                             console.log(`✅ Agent ${data.agent_id} selected! Camera tracking enabled.`);
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
            
            // Send zoom update to backend (backend will track that user manually zoomed)
            fetch('./update_zoom', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ zoom: cameraZoom })
            })
            .catch(error => {
                console.error('Error updating zoom:', error);
            });
        });

        document.getElementById('resetView').addEventListener('click', () => {
            focusedAgentId = null;
            userHasManuallyPanned = false; // Reset manual pan flag
            
            // Reset camera to default view
            cameraPosition = { x: 0, y: 0 };
            cameraZoom = 1.0;

            // Tell backend to reset zoom preferences and focus
            fetch('./reset_view', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            })
            .catch(error => {
                console.error('Error resetting view:', error);
            });

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

            // Update global focused agent ID from backend
            if (data.focused_agent_id !== undefined) {
                focusedAgentId = data.focused_agent_id;
            }

            // Update camera from backend if available and user isn't manually controlling camera
            if (data.camera && !isDragging && !userHasManuallyPanned) {
                // Always update camera position when not dragging
                if (data.camera.position && Array.isArray(data.camera.position) && data.camera.position.length === 2) {
                    cameraPosition.x = data.camera.position[0];
                    cameraPosition.y = data.camera.position[1];
                }
            }
            
            // Handle zoom override separately (can happen even during manual pan)
            if (data.camera && data.camera.zoom_override !== undefined && data.camera.zoom_override !== null) {
                cameraZoom = data.camera.zoom_override;
                
                // Tell backend we've applied the zoom override
                fetch('./clear_zoom_override', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                })
                .catch(error => {
                    console.error('Error clearing zoom override:', error);
                });
            }

            // Update leaderboard only if it has changed to prevent re-rendering
            const leaderboardContent = document.getElementById('leaderboard-content');
            if (leaderboardContent && data.leaderboard) {
                const newLeaderboardHtml = data.leaderboard.map((robot, index) => {
                    const isFocused = robot.id === data.focused_agent_id;
                    const focusedClass = isFocused ? ' focused' : '';
                    return `
                        <div class="robot-stat-row${focusedClass}" data-agent-id="${robot.id}" title="Click to focus on Robot ${robot.id}">
                            <span class="robot-stat-label">${robot.name}${isFocused ? ' 🎯' : ''}</span>
                            <span class="robot-stat-value">🍽️ ${robot.food_consumed.toFixed(2)}</span>
                        </div>
                    `;
                }).join('');
                
                // Only update the DOM if the content has actually changed.
                if (newLeaderboardHtml !== lastLeaderboardHtml) {
                    leaderboardContent.innerHTML = newLeaderboardHtml;
                    lastLeaderboardHtml = newLeaderboardHtml;
                }
            }

            // Update population summary
            const populationSummaryContent = document.getElementById('population-summary-content');
            if (populationSummaryContent && data.statistics) {
                // Calculate role distribution
                const roleDistribution = {};
                let totalAgents = 0;
                
                if (data.agents) {
                    data.agents.forEach(agent => {
                        const role = agent.ecosystem?.role || 'omnivore';
                        roleDistribution[role] = (roleDistribution[role] || 0) + 1;
                        totalAgents++;
                    });
                }
                
                // Role icons
                const roleIcons = {
                    'carnivore': '🦁',
                    'herbivore': '🐰', 
                    'omnivore': '🐻',
                    'scavenger': '🦅',
                    'symbiont': '🐠'
                };
                
                // Create role distribution display
                let roleHtml = '';
                Object.entries(roleDistribution).forEach(([role, count]) => {
                    const icon = roleIcons[role] || '🤖';
                    const percentage = totalAgents > 0 ? ((count / totalAgents) * 100).toFixed(0) : 0;
                    roleHtml += `
                        <div class="stat-row">
                            <span class="stat-label">${icon} ${role.charAt(0).toUpperCase() + role.slice(1)}:</span>
                            <span class="stat-value">${count} (${percentage}%)</span>
                        </div>
                    `;
                });
                
                 populationSummaryContent.innerHTML = `
                    <div class="stat-row">
                        <span class="stat-label">Generation:</span>
                        <span class="stat-value">${data.statistics.generation || 1}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Population:</span>
                        <span class="stat-value">${totalAgents} agents</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Avg Distance:</span>
                        <span class="stat-value">${(data.statistics.average_distance || 0).toFixed(2)}m</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Total Food Consumed:</span>
                        <span class="stat-value">🍽️ ${(data.statistics.total_food_consumed || 0).toFixed(1)}</span>
                    </div>
                    ${roleHtml}
                 `;
            }

            // Update robot details panel
            updateRobotDetails(data);
            
            // Update focus indicator
            updateFocusIndicator();
        }

        function updateRobotDetails(data) {
            const robotDetailsPanel = document.getElementById('robot-details-panel');
            const focusedAgentId = data.focused_agent_id;
            
            if (focusedAgentId === null) {
                robotDetailsPanel.innerHTML = '<div class="robot-details-title">🤖 Robot Details</div><div class="robot-details-content">Click on a robot to view details</div>';
                return;
            }
            
            // Find the focused agent
            const agent = data.agents.find(a => a.id === focusedAgentId);
            if (!agent) {
                robotDetailsPanel.innerHTML = '<div class="robot-details-title">🤖 Robot Details</div><div class="robot-details-content">Robot not found</div>';
                return;
            }
            
            // Calculate arm angles from positions
            const shoulderAngle = Math.atan2(agent.upper_arm.y - agent.body.y, agent.upper_arm.x - agent.body.x);
            const elbowAngle = Math.atan2(agent.lower_arm.y - agent.upper_arm.y, agent.lower_arm.x - agent.upper_arm.x);
            
            // Get ecosystem data
            const ecosystem = agent.ecosystem || {};
            const role = ecosystem.role || 'omnivore';
            const status = ecosystem.status || 'idle';
            const health = ecosystem.health || 1.0;
            const energy = ecosystem.energy || 1.0;
            const speed = ecosystem.speed || 0.0;
            const alliances = ecosystem.alliances || [];
            const territories = ecosystem.territories || [];
            
            // Role symbols and colors
            const roleSymbols = {
                'carnivore': '🦁',
                'herbivore': '🐰',
                'omnivore': '🐻',
                'scavenger': '🦅',
                'symbiont': '🐠'
            };
            
            const statusSymbols = {
                'hunting': '🎯',
                'feeding': '🍃',
                'fleeing': '💨',
                'territorial': '🛡️',
                'idle': '😴',
                'moving': '➡️',
                'active': '⚡'
            };
            
            // Format the details
            const details = `
                <div class="robot-details-title">🤖 Robot ${agent.id}</div>
                <div class="robot-details-content">
                    <div class="detail-section">
                        <div class="detail-row">
                            <span class="detail-label">Ecosystem Role:</span>
                            <span class="detail-value" style="color: ${ECOSYSTEM_COLORS[role] || '#888888'};">${roleSymbols[role] || '🤖'} ${role.charAt(0).toUpperCase() + role.slice(1)}</span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">Status:</span>
                            <span class="detail-value" style="color: ${STATUS_COLORS[status] || '#FFFFFF'};">${statusSymbols[status] || '●'} ${status.charAt(0).toUpperCase() + status.slice(1)}</span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">Health:</span>
                            <span class="detail-value" style="color: ${health > 0.5 ? '#4CAF50' : health > 0.25 ? '#FF9800' : '#F44336'};">${(health * 100).toFixed(1)}%</span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">Energy:</span>
                            <span class="detail-value" style="color: ${energy > 0.5 ? '#2196F3' : energy > 0.25 ? '#FF9800' : '#F44336'};">${(energy * 100).toFixed(1)}%</span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">Speed:</span>
                            <span class="detail-value">${speed.toFixed(2)} m/s</span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">Alliances:</span>
                            <span class="detail-value">${alliances.length > 0 ? alliances.length + ' allies' : 'None'}</span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">Territories:</span>
                            <span class="detail-value">${territories.length > 0 ? territories.length + ' claimed' : 'None'}</span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">Closest Food:</span>
                            <span class="detail-value" style="color: ${ecosystem.closest_food_distance >= 999999 || ecosystem.closest_food_distance > 50 ? '#FF8844' : ecosystem.closest_food_distance < 5 ? '#4CAF50' : '#FFF'};">
                                ${ecosystem.closest_food_distance >= 999999 ? 'None available' : ecosystem.closest_food_distance.toFixed(1) + 'm'}
                            </span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">X-Axis Distance:</span>
                            <span class="detail-value" style="color: ${ecosystem.closest_food_signed_x_distance === undefined ? '#888' : ecosystem.closest_food_signed_x_distance > 0 ? '#4CAF50' : '#FF9800'};">
                                ${ecosystem.closest_food_signed_x_distance === undefined ? 'N/A' : (ecosystem.closest_food_signed_x_distance > 0 ? '+' : '') + ecosystem.closest_food_signed_x_distance.toFixed(1) + 'm ' + (ecosystem.closest_food_signed_x_distance > 0 ? '→' : '←')}
                            </span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">Food Type:</span>
                            <span class="detail-value" style="color: ${ecosystem.closest_food_source === 'prey' ? '#FF6B6B' : '#4CAF50'};">
                                ${ecosystem.closest_food_type || 'Unknown'}
                                ${ecosystem.closest_food_source === 'prey' ? ' 🎯' : ecosystem.closest_food_source === 'environment' ? ' 🌿' : ''}
                            </span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">Food Source:</span>
                            <span class="detail-value" style="color: ${ecosystem.closest_food_source === 'prey' ? '#FF6B6B' : '#4CAF50'};">
                                ${ecosystem.closest_food_source || 'Unknown'}
                            </span>
                        </div>
                        ${energy < 0.4 ? `
                        <div class="detail-row" style="color: #FF8844; font-weight: bold; background: rgba(255, 136, 68, 0.1); padding: 4px; border-radius: 3px;">
                            <span class="detail-label">🍽️ Hungry:</span>
                            <span class="detail-value">Looking for food sources</span>
                        </div>` : ''}
                        ${energy < 0.2 ? `
                        <div class="detail-row" style="color: #FF4444; font-weight: bold; background: rgba(255, 68, 68, 0.1); padding: 4px; border-radius: 3px;">
                            <span class="detail-label">⚠️ Warning:</span>
                            <span class="detail-value">Low Energy - Risk of Starvation!</span>
                        </div>` : ''}
                        ${energy < 0.05 ? `
                        <div class="detail-row" style="color: #FF0000; font-weight: bold; background: rgba(255, 0, 0, 0.2); padding: 4px; border-radius: 3px; animation: blink 1s infinite;">
                            <span class="detail-label">💀 CRITICAL:</span>
                            <span class="detail-value">DYING - Find food immediately!</span>
                        </div>` : ''}
                        ${status === 'dying' ? `
                        <div class="detail-row" style="color: #8B0000; font-weight: bold; background: rgba(139, 0, 0, 0.3); padding: 4px; border-radius: 3px; animation: pulse 0.5s infinite;">
                            <span class="detail-label">⚰️ STATUS:</span>
                            <span class="detail-value">STARVING TO DEATH</span>
                        </div>` : ''}
                    </div>
                    
                    <div class="detail-section">
                        <div class="detail-row">
                            <span class="detail-label">Position:</span>
                            <span class="detail-value">(${agent.body.x.toFixed(2)}, ${agent.body.y.toFixed(2)})</span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">Velocity:</span>
                            <span class="detail-value">(${agent.body.velocity.x.toFixed(2)}, ${agent.body.velocity.y.toFixed(2)})</span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">Episode Reward:</span>
                            <span class="detail-value">${agent.total_reward.toFixed(2)}</span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">Best Reward:</span>
                            <span class="detail-value" style="color: #27ae60;">${(agent.best_reward || 0).toFixed(4)}</span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">Worst Reward:</span>
                            <span class="detail-value" style="color: #e74c3c;">${(agent.worst_reward || 0).toFixed(4)}</span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">Steps:</span>
                            <span class="detail-value">${agent.steps || 0}</span>
                        </div>
                    </div>
                    
                    <div class="detail-section">
                        <div class="detail-row">
                            <span class="detail-label">Shoulder Angle:</span>
                            <span class="detail-value">${(shoulderAngle * 180 / Math.PI).toFixed(1)}°</span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">Elbow Angle:</span>
                            <span class="detail-value">${(elbowAngle * 180 / Math.PI).toFixed(1)}°</span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">Current Action:</span>
                            <span class="detail-value">(${agent.current_action[0]}, ${agent.current_action[1]})</span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">State (Discretized):</span>
                            <span class="detail-value">S-bin:${agent.state[0] || 'N/A'}, E-bin:${agent.state[1] || 'N/A'}</span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">State (Angles):</span>
                            <span class="detail-value">S:${agent.state[0] ? ((agent.state[0] * 10) - 180).toFixed(0) + '°' : 'N/A'}, E:${agent.state[1] ? ((agent.state[1] * 10) - 180).toFixed(0) + '°' : 'N/A'}</span>
                        </div>
                    </div>
                    
                    <div class="detail-section">
                        <div class="detail-row">
                            <span class="detail-label">Q-Table Size:</span>
                            <span class="detail-value">${Object.keys(agent.q_table || {}).length}</span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">Action History:</span>
                            <span class="detail-value">${getActionHistoryString(agent.action_history)}</span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">Total Actions:</span>
                            <span class="detail-value">${(agent.action_history || []).length}</span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">Awake:</span>
                            <span class="detail-value">${agent.awake ? 'Yes' : 'No'}</span>
                        </div>
                    </div>
                    
                    <div class="detail-section">
                        <div style="margin-bottom: 6px; font-weight: bold; color: #3498db;">Learning Approach Controls</div>
                        <div style="display: flex; flex-direction: column; gap: 3px;">
                            <button onclick="switchLearningApproach('${agent.id}', 'basic_q_learning')" 
                                    style="background: #27ae60; color: white; border: none; padding: 3px 6px; border-radius: 3px; font-size: 10px; cursor: pointer;">
                                Basic Q-Learning
                            </button>
                            <button onclick="switchLearningApproach('${agent.id}', 'enhanced_survival_q')" 
                                    style="background: #e74c3c; color: white; border: none; padding: 3px 6px; border-radius: 3px; font-size: 10px; cursor: pointer;">
                                Enhanced Survival Q
                            </button>
                            <button onclick="switchLearningApproach('${agent.id}', 'deep_survival_q')" 
                                    style="background: #8e44ad; color: white; border: none; padding: 3px 6px; border-radius: 3px; font-size: 10px; cursor: pointer;">
                                Deep Survival Q (GPU)
                            </button>
                            <button onclick="switchLearningApproach('${agent.id}', 'auto_advanced')" 
                                    style="background: #f39c12; color: white; border: none; padding: 3px 6px; border-radius: 3px; font-size: 10px; cursor: pointer;">
                                Auto Advanced Learning
                            </button>
                        </div>
                        <div style="font-size: 9px; color: #95a5a6; margin-top: 4px;">
                            Current: ${agent.learning_approach || 'basic_q_learning'}
                        </div>
                    </div>
                </div>
            `;
            
            robotDetailsPanel.innerHTML = details;
        }

        function drawWorld(data) {
            if (!ctx) return;

            // Store ecosystem and environment data for enhanced rendering
            if (data.ecosystem) ecosystemData = data.ecosystem;
            if (data.environment) environmentData = data.environment;
            if (data.ecosystem && data.ecosystem.predation_events) {
                predationEvents = data.ecosystem.predation_events;
            }

            // Update UI FPS counter
            uiFpsCounter++;
            const now = Date.now();
            if (now - lastUiFpsUpdate >= 1000) { // Update every second
                currentUiFps = Math.round(uiFpsCounter * 1000 / (now - uiFpsStartTime));
                uiFpsCounter = 0;
                uiFpsStartTime = now;
                lastUiFpsUpdate = now;
            }
            
            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Apply camera transformations
            ctx.save();
            ctx.translate(canvas.width / 2, canvas.height / 2); // Center of canvas
            ctx.scale(cameraZoom, -cameraZoom); // Zoom and flip Y-axis
            ctx.translate(-cameraPosition.x, -cameraPosition.y); // Pan

            // Draw environmental elements first (background layer)
            drawEnvironmentalElements(data);
            
            // Draw ecosystem elements
            drawEcosystemElements(data);

            // Draw ground
            if (data.shapes && data.shapes.ground) {
                ctx.strokeStyle = '#8e8e8e';
                ctx.lineWidth = 0.1;
                data.shapes.ground.forEach(shape => {
                    if (shape.type === 'polygon' && shape.vertices.length > 1) {
                        ctx.beginPath();
                        ctx.moveTo(shape.vertices[0][0], shape.vertices[0][1]);
                        for (let i = 1; i < shape.vertices.length; i++) {
                            ctx.lineTo(shape.vertices[i][0], shape.vertices[i][1]);
                        }
                        ctx.closePath();
                        ctx.stroke();
                    }
                });
            }

            // Draw enhanced robots with ecosystem roles
            drawEnhancedRobots(data);
            
            // Draw food lines if enabled
            if (showFoodLines) {
                drawFoodLines(data);
            }
            
            // All animation events disabled for maximum performance

            ctx.restore(); // Restore to pre-camera transform state
            
            // Draw FPS counters (overlay, not affected by camera transform)
            if (showFpsCounters) {
                drawFpsCounters(data);
            }
        }
        
        function drawEnvironmentalElements(data) {
            if (!environmentData || !environmentData.obstacles) return;
            
            // Draw environmental obstacles
            environmentData.obstacles.forEach(obstacle => {
                const dangerLevel = obstacle.danger_level || 0;
                const [x, y] = obstacle.position;
                const size = obstacle.size;
                
                // Color based on danger level
                const red = Math.floor(100 + (dangerLevel * 155));
                const green = Math.floor(150 - (dangerLevel * 100));
                const blue = Math.floor(100 - (dangerLevel * 50));
                
                ctx.fillStyle = `rgba(${red}, ${green}, ${blue}, 0.6)`;
                ctx.strokeStyle = `rgb(${red}, ${green}, ${blue})`;
                ctx.lineWidth = 0.1;
                
                // Draw obstacle based on type
                ctx.beginPath();
                if (obstacle.type === 'boulder' || obstacle.type === 'wall') {
                    ctx.rect(x - size/2, y - size/2, size, size);
                } else {
                    ctx.arc(x, y, size/2, 0, 2 * Math.PI);
                }
                ctx.fill();
                ctx.stroke();
                
                // Add danger indicator for high-danger obstacles - fix text flipping
                if (dangerLevel > 0.5) {
                    ctx.save();
                    ctx.scale(1, -1); // Counter the Y-axis flip for text
                    ctx.fillStyle = '#FF0000';
                    ctx.font = `${size/3}px Arial`;
                    ctx.textAlign = 'center';
                    ctx.fillText('⚠', x, -(y + size/6));
                    ctx.restore();
                }
            });
        }
        
        function drawEcosystemElements(data) {
            if (!ecosystemData) return;
            
            // Draw territories
            if (ecosystemData.territories) {
                ecosystemData.territories.forEach(territory => {
                    const [x, y] = territory.position;
                    const size = territory.size;
                    const contested = territory.contested;
                    
                    // Territory color based on type and resource value
                    const alpha = Math.min(0.3, territory.resource_value * 0.2);
                    let territoryColor = '#4CAF50'; // Default green
                    
                    switch (territory.type) {
                        case 'feeding_ground': territoryColor = '#8BC34A'; break;
                        case 'nesting_area': territoryColor = '#FF9800'; break;
                        case 'water_source': territoryColor = '#2196F3'; break;
                        case 'shelter': territoryColor = '#9C27B0'; break;
                    }
                    
                    ctx.strokeStyle = contested ? '#FF0000' : territoryColor;
                    ctx.lineWidth = contested ? 0.3 : 0.15;
                    // Simplified territory boundary - no dashed lines for performance
                    ctx.beginPath();
                    ctx.arc(x, y, size/2, 0, 2 * Math.PI);
                    ctx.stroke();
                });
            }
            
            // Draw food sources
            if (ecosystemData.food_sources) {
                ecosystemData.food_sources.forEach(food => {
                    const [x, y] = food.position;
                    const amount = food.amount;
                    const maxCapacity = food.max_capacity;
                    const ratio = amount / maxCapacity;
                    
                    // Food color based on type
                    let foodColor = '#4CAF50'; // Default green for plants
                    switch (food.type) {
                        case 'plants': foodColor = '#4CAF50'; break;
                        // meat removed - carnivores hunt robots instead
                        case 'insects': foodColor = '#795548'; break;
                        case 'seeds': foodColor = '#FF9800'; break;
                    }
                    
                    const baseRadius = 0.8;
                    const radius = baseRadius + (ratio * 1.2); // Size based on remaining amount
                    const alpha = 0.5 + (ratio * 0.5); // Transparency based on amount
                    
                    // Draw resource base
                    ctx.fillStyle = foodColor + Math.floor(alpha * 255).toString(16).padStart(2, '0');
                    ctx.strokeStyle = foodColor;
                    ctx.lineWidth = 0.1;
                    
                    ctx.beginPath();
                    ctx.arc(x, y, radius, 0, 2 * Math.PI);
                    ctx.fill();
                    ctx.stroke();
                    
                    // Draw resource type icon - fix text flipping
                    ctx.save();
                    ctx.scale(1, -1); // Counter the Y-axis flip for text
                    ctx.fillStyle = '#FFFFFF';
                    ctx.font = '0.8px Arial';
                    ctx.textAlign = 'center';
                    const typeIcons = {
                        'plants': '🌿',
                        // meat removed - carnivores hunt robots instead
                        'insects': '🐛',
                        'seeds': '🌰'
                    };
                    ctx.fillText(typeIcons[food.type] || '🍃', x, -(y + 0.3));
                    ctx.restore();
                    
                    // Depletion warning animations disabled for performance
                    
                    // Draw resource amount bar
                    const barWidth = radius * 2;
                    const barHeight = 0.2;
                    const barY = y - radius - 0.5;
                    
                    // Background bar
                    ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
                    ctx.fillRect(x - barWidth/2, barY, barWidth, barHeight);
                    
                    // Amount bar
                    ctx.fillStyle = ratio > 0.5 ? '#4CAF50' : ratio > 0.2 ? '#FF9800' : '#F44336';
                    ctx.fillRect(x - barWidth/2, barY, barWidth * ratio, barHeight);
                    
                    // Amount text - fix text flipping
                    ctx.save();
                    ctx.scale(1, -1); // Counter the Y-axis flip for text
                    ctx.fillStyle = '#FFFFFF';
                    ctx.font = '0.4px Arial';
                    ctx.textAlign = 'center';
                    ctx.fillText(`${amount.toFixed(0)}/${maxCapacity.toFixed(0)}`, x, -(barY - 0.2));
                    ctx.restore();
                });
            }
        }
        
        function drawEnhancedRobots(data) {
            if (!data.shapes || !data.shapes.robots || !data.agents) return;
            
            data.shapes.robots.forEach(robot => {
                // Find corresponding agent data for ecosystem info
                const agent = data.agents.find(a => a.id === robot.id);
                if (!agent) return;
                
                const isFocused = (robot.id === focusedAgentId);
                const ecosystem = agent.ecosystem || {};
                const role = ecosystem.role || 'omnivore';
                const status = ecosystem.status || 'idle';
                const health = ecosystem.health || 1.0;
                const energy = ecosystem.energy || 1.0;
                const speed = ecosystem.speed || 0.0;
                
                // Get role-based color
                const baseColor = ECOSYSTEM_COLORS[role] || '#888888';
                const statusColor = STATUS_COLORS[status] || baseColor;
                
                // Apply focus highlighting
                const strokeColor = isFocused ? '#FFD700' : baseColor; // Gold for focused
                const fillColor = isFocused ? `${baseColor}88` : `${baseColor}66`; // More transparent
                
                ctx.strokeStyle = strokeColor;
                ctx.fillStyle = fillColor;
                ctx.lineWidth = isFocused ? 0.25 : 0.15;
                
                // Draw agent body parts
                robot.body_parts.forEach(part => {
                    ctx.beginPath();
                    if (part.type === 'polygon' && part.vertices.length > 1) {
                        ctx.moveTo(part.vertices[0][0], part.vertices[0][1]);
                        for (let i = 1; i < part.vertices.length; i++) {
                            ctx.lineTo(part.vertices[i][0], part.vertices[i][1]);
                        }
                        ctx.closePath();
                    } else if (part.type === 'circle') {
                        ctx.arc(part.center[0], part.center[1], part.radius, 0, 2 * Math.PI);
                    }
                    ctx.fill();
                    ctx.stroke();
                });
                
                // Draw status indicators above agent
                const agentPos = [agent.body.x, agent.body.y];
                drawAgentStatusIndicators(agentPos, role, status, health, energy, speed, isFocused);
                
                // Movement trails disabled for performance optimization
                
                // Alliance connections disabled for performance
            });
            
            // FEEDING ANIMATION: Draw consumption lines from robots to food with role-based colors
            if (data.ecosystem && data.ecosystem.consumption_events) {
                data.ecosystem.consumption_events.forEach(consumption => {
                    const agentPos = [consumption.agent_position[0], consumption.agent_position[1]];
                    const foodPos = [consumption.food_position[0], consumption.food_position[1]];
                    
                    // CLEAR LINE COLORS based on food type and role
                    let lineColor = '#4CAF50'; // Default green
                    let consumptionLineWidth = 2;
                    
                    // Environmental food colors
                    if (consumption.food_type.includes('plants')) lineColor = '#4CAF50'; // Green for plants
                    if (consumption.food_type.includes('insects')) lineColor = '#795548'; // Brown for insects  
                    if (consumption.food_type.includes('seeds')) lineColor = '#FF9800'; // Orange for seeds
                    
                    // Robot consumption colors (thicker lines)
                    if (consumption.food_type.includes('robot')) {
                        consumptionLineWidth = 4; // Thicker line for robot consumption
                        if (consumption.food_type.includes('carnivore')) lineColor = '#D32F2F'; // Dark red for carnivore
                        if (consumption.food_type.includes('scavenger')) lineColor = '#7B1FA2'; // Purple for scavenger  
                        if (consumption.food_type.includes('omnivore')) lineColor = '#F57C00'; // Dark orange for omnivore
                    }
                    
                    // Animated line with pulsing effect
                    const alpha = 0.8 - (consumption.progress * 0.5); // Fade out as animation progresses
                    const animationLineWidth = 0.2 + (consumption.progress * 0.1); // Get slightly thicker as animation progresses
                    
                    ctx.strokeStyle = lineColor;
                    ctx.globalAlpha = alpha;
                    ctx.lineWidth = animationLineWidth;
                    ctx.setLineDash([0.5, 0.3]); // Small dashed line
                    
                    ctx.beginPath();
                    ctx.moveTo(agentPos[0], agentPos[1]);
                    ctx.lineTo(foodPos[0], foodPos[1]);
                    ctx.stroke();
                    
                    // Reset line style
                    ctx.setLineDash([]);
                    ctx.globalAlpha = 1.0;
                    
                    // Energy gain indicator at robot position - fix text flipping
                    ctx.save();
                    ctx.scale(1, -1); // Counter the Y-axis flip for text
                    ctx.fillStyle = lineColor;
                    ctx.font = '0.6px Arial';
                    ctx.textAlign = 'center';
                    ctx.fillText(`+${consumption.energy_gained.toFixed(1)}`, agentPos[0], -(agentPos[1] - 1.5));
                    ctx.restore();
                 });
             }
        }
        
        function drawAgentStatusIndicators(position, role, status, health, energy, speed, isFocused) {
            const [x, y] = position;
            const barWidth = 2.0;
            const barHeight = 0.3;
            const barSpacing = 0.4;
            const baseY = y + 3.0; // Position above agent
            
            // Health bar
            ctx.fillStyle = '#333333';
            ctx.fillRect(x - barWidth/2, baseY, barWidth, barHeight);
            ctx.fillStyle = health > 0.5 ? '#4CAF50' : health > 0.25 ? '#FF9800' : '#F44336';
            ctx.fillRect(x - barWidth/2, baseY, barWidth * health, barHeight);
            
            // Energy bar
            ctx.fillStyle = '#333333';
            ctx.fillRect(x - barWidth/2, baseY + barSpacing, barWidth, barHeight);
            ctx.fillStyle = energy > 0.5 ? '#2196F3' : energy > 0.25 ? '#FF9800' : '#F44336';
            ctx.fillRect(x - barWidth/2, baseY + barSpacing, barWidth * energy, barHeight);
            
            // Role indicator
            const roleSymbols = {
                'carnivore': '🦁',
                'herbivore': '🐰',
                'omnivore': '🐻',
                'scavenger': '🦅',
                'symbiont': '🐠'
            };
            
            if (isFocused || speed > 0.5) {
                // Fix text flipping by temporarily countering the Y-axis flip
                ctx.save();
                ctx.scale(1, -1); // Counter the Y-axis flip
                ctx.fillStyle = '#FFFFFF';
                ctx.font = '1px Arial';
                ctx.textAlign = 'center';
                ctx.fillText(roleSymbols[role] || '🤖', x, -(baseY + barSpacing * 2 + 0.8));
                ctx.restore();
            }
            
            // Status indicator for active agents
            if (status !== 'idle' && speed > 0.1) {
                const statusSymbols = {
                    'hunting': '🎯',
                    'feeding': '🍃',
                    'fleeing': '💨',
                    'territorial': '🛡️',
                    'moving': '➡️',
                    'active': '⚡'
                };
                
                // Fix text flipping by temporarily countering the Y-axis flip
                ctx.save();
                ctx.scale(1, -1); // Counter the Y-axis flip
                ctx.fillStyle = STATUS_COLORS[status] || '#FFFFFF';
                ctx.font = '0.8px Arial';
                ctx.textAlign = 'center';
                ctx.fillText(statusSymbols[status] || '●', x + 1.5, -(y + 2.0));
                ctx.restore();
            }
        }
        
        // Movement trail function removed for performance optimization
        
        // Alliance connections function removed for performance
        
        function drawFoodLines(data) {
            if (!data.agents || !focusedAgentId) {
                console.log("🎯 No agents or no focused agent, skipping food lines");
                return;
            }
            
            // Only draw food line for the focused robot
            const focusedAgent = data.agents.find(agent => agent.id === focusedAgentId);
            if (!focusedAgent) {
                console.log(`🎯 Focused agent ${focusedAgentId} not found in agent list`);
                return;
            }
            
            const ecosystem = focusedAgent.ecosystem || {};
            const foodPosition = ecosystem.closest_food_position;
            const signedXDistance = ecosystem.closest_food_signed_x_distance;
            
            console.log(`🎯 Food data for agent ${focusedAgentId}:`, {
                foodPosition,
                signedXDistance,
                ecosystemKeys: Object.keys(ecosystem)
            });
            
            // Only draw line if food position is available
            if (foodPosition && Array.isArray(foodPosition) && foodPosition.length >= 2) {
                const robotPos = [focusedAgent.body.x, focusedAgent.body.y];
                const [foodX, foodY] = foodPosition;
                
                console.log(`🎯 Drawing food line from robot ${robotPos} to food ${foodPosition}`);
                
                // Calculate distance for coloring
                const distance = signedXDistance !== undefined ? Math.abs(signedXDistance) : 
                                Math.sqrt((foodX - robotPos[0]) ** 2 + (foodY - robotPos[1]) ** 2);
                
                // Bright visible colors for debugging
                const lineColor = signedXDistance !== undefined && signedXDistance > 0 ? 
                    '#00FFFF' : '#FF6600'; // Cyan for right, orange for left
                
                // Thicker line for visibility
                ctx.strokeStyle = lineColor;
                ctx.lineWidth = 0.3;
                
                ctx.beginPath();
                ctx.moveTo(robotPos[0], robotPos[1]);
                ctx.lineTo(foodX, foodY);
                ctx.stroke();
                
                // Draw arrow at food position
                const angle = Math.atan2(foodY - robotPos[1], foodX - robotPos[0]);
                const arrowLength = 1.0;
                
                ctx.strokeStyle = lineColor;
                ctx.lineWidth = 0.2;
                
                ctx.beginPath();
                ctx.moveTo(foodX, foodY);
                ctx.lineTo(
                    foodX - arrowLength * Math.cos(angle - 0.5),
                    foodY - arrowLength * Math.sin(angle - 0.5)
                );
                ctx.lineTo(
                    foodX - arrowLength * Math.cos(angle + 0.5),
                    foodY - arrowLength * Math.sin(angle + 0.5)
                );
                ctx.lineTo(foodX, foodY);
                ctx.stroke();
                
                console.log(`🎯 Food line drawn successfully`);
            } else {
                console.log(`🎯 No valid food position for agent ${focusedAgentId}`);
            }
        }
        
        // Predation events function removed for performance
        
        // Death events function removed for performance

        function drawFpsCounters(data) {
            // Get physics FPS from server data (more accurate than client-side calculation)
            if (data && data.physics_fps !== undefined) {
                currentPhysicsFps = data.physics_fps;
            }
            
            // Save current context
            ctx.save();
            
            // Reset transform for UI overlay
            ctx.setTransform(1, 0, 0, 1, 0, 0);
            
            // FPS counter background
            const fpsBoxWidth = 180;
            const fpsBoxHeight = 70;
            const fpsBoxX = canvas.width - fpsBoxWidth - 10;
            const fpsBoxY = 10;
            
            ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
            ctx.fillRect(fpsBoxX, fpsBoxY, fpsBoxWidth, fpsBoxHeight);
            
            ctx.strokeStyle = '#444444';
            ctx.lineWidth = 1;
            ctx.strokeRect(fpsBoxX, fpsBoxY, fpsBoxWidth, fpsBoxHeight);
            
            // FPS text
            ctx.fillStyle = '#FFFFFF';
            ctx.font = '14px monospace';
            ctx.textAlign = 'left';
            
            // UI FPS
            const uiFpsColor = currentUiFps >= 20 ? '#4CAF50' : currentUiFps >= 15 ? '#FF9800' : '#F44336';
            ctx.fillStyle = uiFpsColor;
            ctx.fillText(`UI FPS: ${currentUiFps}`, fpsBoxX + 10, fpsBoxY + 20);
            
            // Physics FPS
            const physicsFpsColor = currentPhysicsFps >= 50 ? '#4CAF50' : currentPhysicsFps >= 30 ? '#FF9800' : '#F44336';
            ctx.fillStyle = physicsFpsColor;
            ctx.fillText(`Physics: ${currentPhysicsFps}`, fpsBoxX + 10, fpsBoxY + 40);
            
            // Performance indicator
            ctx.fillStyle = '#BBBBBB';
            ctx.font = '10px monospace';
            const perfStatus = (currentUiFps >= 20 && currentPhysicsFps >= 50) ? 'OPTIMAL' : 
                              (currentUiFps >= 15 && currentPhysicsFps >= 30) ? 'GOOD' : 'SLOW';
            ctx.fillText(`Status: ${perfStatus}`, fpsBoxX + 10, fpsBoxY + 60);
            
            // Restore context
            ctx.restore();
        }

        let lastFetchTime = 0;
        const fetchInterval = 33; // ~30 FPS instead of 60 FPS for performance
        
        function fetchData() {
            const now = Date.now();
            if (now - lastFetchTime < fetchInterval) {
                requestAnimationFrame(fetchData);
                return;
            }
            lastFetchTime = now;
            
            fetch('./status')  // Use relative path
                .then(response => response.json())
                .then(data => {
                    window.lastData = data; // Store latest data globally
                    drawWorld(data);
                    updateStats(data);
                    requestAnimationFrame(fetchData);
                })
                .catch(error => {
                    console.error('Error fetching data:', error);
                    setTimeout(fetchData, 1000); // Try again after a second
                });
        }
        
        // Start the main loop
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

        function updateLeaderboardVisualFocus() {
            // Update the visual state of leaderboard items
            const leaderboardItems = document.querySelectorAll('.robot-stat-row[data-agent-id]');
            leaderboardItems.forEach(item => {
                const itemAgentId = item.dataset.agentId;  // Keep as string, don't parseInt
                if (itemAgentId === focusedAgentId) {
                    item.classList.add('focused');
                } else {
                    item.classList.remove('focused');
                }
            });
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
                await fetch('./update_agent_params', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(params)
                });
                console.log('Agent parameters updated:', params);
            } catch (err) {
                console.error('Error updating agent parameters:', err);
            }
        }

        // Learning approach switching function
        async function switchLearningApproach(agentId, approach) {
            try {
                const response = await fetch('./switch_learning_approach', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        agent_id: agentId, 
                        approach: approach 
                    })
                });
                const result = await response.json();
                
                if (result.status === 'success') {
                    console.log(`✅ Switched agent ${agentId} to ${approach} learning approach`);
                } else {
                    console.error(`❌ Failed to switch learning approach: ${result.message}`);
                }
            } catch (err) {
                console.error('Error switching learning approach:', err);
            }
        }

        // Toggle food lines display
        function toggleFoodLines() {
            showFoodLines = !showFoodLines;
            const button = document.getElementById('toggleFoodLines');
            if (showFoodLines) {
                button.textContent = 'Hide Food Lines';
                button.style.background = '#FF5722';
            } else {
                button.textContent = 'Show Food Lines';
                button.style.background = '#4CAF50';
            }
            console.log(`🎯 Food lines ${showFoodLines ? 'enabled' : 'disabled'}`);
        }

        // Test carnivore feeding mechanics
        function testCarnivoreFeeding() {
            console.log('🧪 Testing carnivore feeding mechanics...');
            fetch('./test_carnivore_feeding', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    console.log('✅ Carnivore feeding test completed');
                    alert('Carnivore feeding test completed - check console logs for results');
                } else {
                    console.error('❌ Carnivore feeding test failed:', data.message);
                    alert('Carnivore feeding test failed: ' + data.message);
                }
            })
            .catch(error => {
                console.error('❌ Error during carnivore feeding test:', error);
                alert('Error running carnivore feeding test');
            });
        }

    </script>
</body>
</html>
"""

def safe_convert_numeric(value):
    """Convert numpy numeric types to JSON-serializable types without recursion."""
    if isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, (int, float, str, bool)) or value is None:
        return value
    else:
        # For other types, try to convert to float if possible, otherwise return as-is
        try:
            return float(value)
        except (ValueError, TypeError):
            return value

def safe_convert_list(lst):
    """Convert a list of potentially numpy values efficiently."""
    if not lst:
        return lst
    return [safe_convert_numeric(item) for item in lst]

def safe_convert_position(pos):
    """Convert a position tuple/list safely."""
    if hasattr(pos, 'x') and hasattr(pos, 'y'):
        # Box2D vector
        return (float(pos.x), float(pos.y))
    elif isinstance(pos, (tuple, list)) and len(pos) >= 2:
        return (safe_convert_numeric(pos[0]), safe_convert_numeric(pos[1]))
    return pos

class TrainingEnvironment:
    """
    Enhanced training environment with evolutionary physical parameters.
    Manages physics simulation and evolution of diverse crawling robots.
    Enhanced with comprehensive evaluation framework.
    """
    def __init__(self, num_agents=30, enable_evaluation=True):  # Reduced from 50 to 30 to save memory
        self.num_agents = num_agents
        # CRITICAL FIX: Disable sleeping at the world level to prevent static bodies from going to sleep
        self.world = b2.b2World(gravity=(0, -9.8), doSleep=False)
        self.dt = 1.0 / 60.0  # 60 FPS

        # World bounds for resetting fallen agents
        self.world_bounds_y = -20.0
        
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

        # Create initial diverse population
        self.agents = self.evolution_engine.initialize_population()

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
        self.episode_length = 12000  # 200 seconds at 60 Hz - much longer to prevent constant resets
        
        # Enhanced thread safety for Box2D operations
        import threading
        self._physics_lock = threading.RLock()  # Use RLock for re-entrant locking
        self._evolution_lock = threading.Lock()  # Separate lock for evolution state
        self._is_evolving = False  # Flag to prevent concurrent evolution
        self._agents_pending_destruction = []  # Safe destruction queue
        
        # Statistics update timing
        self.stats_update_interval = 1.0
        self.last_stats_update = 0
        
        # Evolution timing with safety
        self.evolution_interval = 180.0  # 3 minutes between generations
        self.last_evolution_time = time.time()
        self.auto_evolution_enabled = False
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
        self.agent_health = {}  # Track agent health/energy for visualization
        self.agent_statuses = {}  # Track agent statuses (hunting, feeding, etc.)
        self.predation_events = []  # Track recent predation events for visualization
        self.last_ecosystem_update = time.time()
        self.ecosystem_update_interval = 20.0  # Update ecosystem every 20 seconds for better responsiveness
        
        # Resource generation system - REDUCED FREQUENCY for stability
        self.last_resource_generation = time.time()
        self.resource_generation_interval = 120.0  # Generate strategic resources every 2 minutes (was 45s) for stable rewards
        self.agent_energy_levels = {}  # Track agent energy levels for resource consumption
        
        # Death and survival system
        self.death_events = []  # Track recent death events for visualization
        self.agents_pending_replacement = []  # Queue for dead agents needing replacement
        
        # Food consumption animation system
        self.consumption_events = []  # Track active food consumption for animation
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
        
        # Generate realistic terrain at startup
        self._generate_realistic_terrain()

        # Initialize elite robot management system
        self.elite_manager = EliteManager(
            storage_directory="robot_storage",
            elite_per_generation=3,  # Preserve top 3 robots each generation
            max_elite_storage=150,   # Maximum 150 elite robots in storage
            min_fitness_threshold=1.0  # Only preserve robots with fitness > 1.0
        )
        
        # Flag to track if we should restore elites on startup
        self.restore_elites_on_start = True

        # Initialize Learning Manager for advanced learning approaches
        try:
            self.learning_manager = LearningManager(
                ecosystem_interface=EcosystemInterface(self)
            )
            print("🧠 Learning Manager initialized successfully")
        except Exception as e:
            print(f"⚠️ Learning Manager initialization failed: {e}")
            self.learning_manager = None

        # ✨ INITIALIZE RANDOM LEARNING APPROACHES FOR ALL AGENTS (after learning_manager is initialized)
        self._initialize_random_learning_approaches()

        # Performance optimization tracking
        self.last_performance_cleanup = time.time()
        self.performance_cleanup_interval = 120.0  # Clean up every 2 minutes
        
        # Web interface throttling
        self.last_web_interface_update = time.time()
        self.web_interface_update_interval = 0.05  # 20 FPS instead of 60 FPS
        self.web_data_cache = {}
        self.web_cache_valid = False
        
        # Physics FPS tracking
        self.physics_fps_counter = 0
        self.physics_fps_start_time = time.time()
        self.last_physics_fps_update = time.time()
        self.current_physics_fps = 0
        
        # Background processing
        import threading
        self._background_processing_active = False
        self._background_lock = threading.Lock()

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
            
            # Connect learning manager to memory pool for knowledge transfer
            if self.learning_manager:
                self.robot_memory_pool.set_learning_manager(self.learning_manager)
            
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

        # Reward signal evaluation system
        self.reward_signal_evaluator = None
        self.reward_signal_adapter = None
        try:
            from src.evaluation.reward_signal_integration import reward_signal_adapter
            self.reward_signal_adapter = reward_signal_adapter
            
            # Register all existing agents with the reward signal evaluator
            for agent in self.agents:
                if not getattr(agent, '_destroyed', False):
                    # Determine agent type from learning approach or fallback to default
                    agent_type = getattr(agent, 'learning_approach', 'evolutionary')
                    self.reward_signal_adapter.register_agent(
                        agent.id,
                        agent_type,
                        metadata={
                            'physical_params': str(agent.physical_params) if hasattr(agent, 'physical_params') else None,
                            'created_at': time.time()
                        }
                    )
            
            print(f"📊 Reward signal evaluation system initialized - tracking {len(self.agents)} agents")
        except ImportError as e:
            print(f"⚠️ Reward signal evaluation not available: {e}")
        except Exception as e:
            print(f"⚠️ Reward signal evaluation initialization failed: {e}")

        print(f"🧬 Enhanced Training Environment initialized:")
        print(f"   Population: {len(self.agents)} diverse agents")
        print(f"   Evolution: {self.evolution_config.population_size} agents, {self.evolution_config.elite_size} elite")
        print(f"   Diversity target: {self.evolution_config.target_diversity}")
        print(f"   Auto-evolution every {self.evolution_interval}s")
        print(f"🌿 Ecosystem dynamics and visualization systems active")
        print(f"🏆 Elite preservation: {self.elite_manager.elite_per_generation} per generation, max {self.elite_manager.max_elite_storage} stored")
        print(f"🏞️ Realistic terrain generated: {len(self.terrain_collision_bodies)} terrain bodies using '{self.terrain_style}' style")

    def _create_ground(self):
        """Creates a static ground body."""
        ground_body = self.world.CreateStaticBody(position=(0, -1))
        
        # Calculate ground width to accommodate evolution engine spawn area
        # Evolution engine uses: max(800, population_size * min_spacing * 1.5)
        # With min_spacing = 12, this ensures ground covers the full spawn area
        min_spacing = 12  # Must match evolution engine's min_spacing
        calculated_spawn_width = max(800, self.num_agents * min_spacing * 1.5)
        
        # Add extra margin for robot movement beyond spawn area
        ground_width = calculated_spawn_width + 200  # Extra 200 units for exploration
        
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
        print(f"🔧 Ground setup complete with width {ground_width} (spawn area: {calculated_spawn_width}) for {self.num_agents} agents.")

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
        
        # Update style and regenerate
        self.terrain_style = new_style
        self._generate_realistic_terrain()
        
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
                    'alliances': [],
                    'territories': []
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
        """Initialize random learning approaches for all agents to ensure diversity."""
        if not self.learning_manager:
            print("⚠️ Learning Manager not available - agents will use default learning approach")
            return
        
        # Import learning approaches
        from src.agents.learning_manager import LearningApproach
        import random
        
        # Available learning approaches with weights (HEAVILY favor Deep Q-Learning)
        learning_approaches = [
            (LearningApproach.BASIC_Q_LEARNING, 0.10),      # 10% - Simple baseline
            (LearningApproach.ENHANCED_Q_LEARNING, 0.20),   # 20% - Advanced tabular
            (LearningApproach.SURVIVAL_Q_LEARNING, 0.30),   # 30% - Survival-focused
            (LearningApproach.DEEP_Q_LEARNING, 0.40),       # 40% - Neural networks (INCREASED!)
        ]
        
        # Create weighted list for random selection
        weighted_approaches = []
        for approach, weight in learning_approaches:
            weighted_approaches.extend([approach] * int(weight * 100))
        
        # Track assignments for reporting
        approach_counts = {approach: 0 for approach, _ in learning_approaches}
        successful_assignments = 0
        failed_assignments = 0
        
        print(f"🎯 Assigning random learning approaches to {len(self.agents)} agents...")
        
        for i, agent in enumerate(self.agents):
            if getattr(agent, '_destroyed', False):
                continue
                
            try:
                # Select random learning approach
                selected_approach = random.choice(weighted_approaches)
                
                # Apply the learning approach
                success = self.learning_manager.set_agent_approach(agent, selected_approach)
                
                if success:
                    # Store the approach name on the agent for web interface
                    setattr(agent, 'learning_approach', selected_approach.value)
                    approach_counts[selected_approach] += 1
                    successful_assignments += 1
                    
                    # Log first few assignments for verification
                    if i < 5:
                        print(f"   Agent {agent.id[:8]}: {selected_approach.value}")
                else:
                    failed_assignments += 1
                    print(f"   ❌ Failed to assign {selected_approach.value} to agent {agent.id[:8]}")
                    
            except Exception as e:
                failed_assignments += 1
                print(f"   ❌ Error assigning learning approach to agent {agent.id}: {e}")
        
        # Report final distribution
        print(f"🧠 Learning Approach Distribution:")
        total_assigned = successful_assignments
        for approach, count in approach_counts.items():
            if count > 0:
                percentage = (count / total_assigned * 100) if total_assigned > 0 else 0
                approach_info = self.learning_manager.approach_info[approach]
                icon = approach_info['icon']
                name = approach_info['name']
                print(f"   {icon} {name}: {count} agents ({percentage:.1f}%)")
        
        if failed_assignments > 0:
            print(f"   ⚠️ Failed assignments: {failed_assignments}")
        
        print(f"✅ Successfully assigned learning approaches to {successful_assignments}/{len(self.agents)} agents")

    def _assign_random_learning_approach_single(self, agent):
        """Assign a random learning approach to a single agent (for replacement agents)."""
        if not self.learning_manager:
            return
        
        # Import learning approaches
        from src.agents.learning_manager import LearningApproach
        import random
        
        # Available learning approaches with weights (HEAVILY favor Deep Q-Learning for replacements)
        learning_approaches = [
            (LearningApproach.BASIC_Q_LEARNING, 0.10),      # 10% - Simple baseline
            (LearningApproach.ENHANCED_Q_LEARNING, 0.20),   # 20% - Advanced tabular
            (LearningApproach.SURVIVAL_Q_LEARNING, 0.30),   # 30% - Survival-focused
            (LearningApproach.DEEP_Q_LEARNING, 0.40),       # 40% - Neural networks (INCREASED!)
        ]
        
        # Create weighted list for random selection
        weighted_approaches = []
        for approach, weight in learning_approaches:
            weighted_approaches.extend([approach] * int(weight * 100))
        
        try:
            # Select random learning approach
            selected_approach = random.choice(weighted_approaches)
            
            # Apply the learning approach
            success = self.learning_manager.set_agent_approach(agent, selected_approach)
            
            if success:
                # Store the approach name on the agent for web interface
                setattr(agent, 'learning_approach', selected_approach.value)
                approach_info = self.learning_manager.approach_info[selected_approach]
                icon = approach_info['icon']
                name = approach_info['name']
                print(f"   🎯 Assigned {icon} {name} to replacement agent {agent.id[:8]}")
                
                # Special logging for Deep Q-Learning to track creation
                if selected_approach == LearningApproach.DEEP_Q_LEARNING:
                    print(f"   🧠 DEEP Q-LEARNING AGENT CREATED: {agent.id[:8]} - Neural network active!")
            else:
                print(f"   ❌ Failed to assign {selected_approach.value} to replacement agent {agent.id[:8]}")
                
        except Exception as e:
            print(f"   ❌ Error assigning learning approach to replacement agent {agent.id}: {e}")

    def _update_ecosystem_dynamics(self):
        """Update ecosystem dynamics including agent interactions, territories, and predation."""
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
                    
                    # Energy decreases over time, affected by environmental factors
                    energy_drain = 0.01 + environmental_effects.get('energy_cost', 0.0)
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
            
            # Simulate predation events (visual only)
            self._simulate_predation_events()
            
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
            
            # Update alliances and territories from ecosystem
            status_data['alliances'] = list(self.ecosystem_dynamics.alliances.get(agent_id, set()))
            status_data['territories'] = [
                {'type': t.territory_type.value, 'position': t.position, 'size': t.size}
                for t in self.ecosystem_dynamics.territories if t.owner_id == agent_id
            ]
    
    def _simulate_predation_events(self):
        """Simulate predation events for visualization purposes."""
        current_time = time.time()
        
        # Find carnivores and herbivores
        carnivores = []
        herbivores = []
        
        for agent in self.agents:
            if getattr(agent, '_destroyed', False) or not agent.body:
                continue
                
            agent_id = agent.id
            if agent_id in self.agent_statuses:
                role = self.agent_statuses[agent_id]['role']
                if role == 'carnivore':
                    carnivores.append(agent)
                elif role == 'herbivore':
                    herbivores.append(agent)
        
        # Simulate predation attempts
        for predator in carnivores:
            if self.agent_statuses[predator.id]['status'] != 'hunting':
                continue
                
            # Find nearby prey
            for prey in herbivores:
                distance = ((predator.body.position.x - prey.body.position.x) ** 2 + 
                           (predator.body.position.y - prey.body.position.y) ** 2) ** 0.5
                
                if distance < 5.0:  # Within hunting range
                    # Predation attempt based on relative speeds/fitness
                    predator_fitness = predator.get_evolutionary_fitness()
                    prey_fitness = prey.get_evolutionary_fitness()
                    
                    success_chance = min(0.3, predator_fitness / (prey_fitness + 1.0))
                    
                    if random.random() < success_chance:
                        # Successful predation event
                        self.predation_events.append({
                            'predator_id': predator.id,
                            'prey_id': prey.id,
                            'position': (predator.body.position.x, predator.body.position.y),
                            'timestamp': current_time,
                            'success': True
                        })
                        
                        # Update prey status
                        if prey.id in self.agent_statuses:
                            self.agent_statuses[prey.id]['status'] = 'fleeing'
                        
                        # Update predator energy
                        if predator.id in self.agent_health:
                            self.agent_health[predator.id]['energy'] = min(1.0, 
                                self.agent_health[predator.id]['energy'] + 0.2)
                        
                        break  # One predation per predator per update

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
        """Validate that resources maintain proper distance from agents - NO MORE MOVING FOOD CLOSE TO AGENTS."""
        try:
            minimum_safe_distance = 6.0  # Must be at least 6m from any agent (matches ecosystem_dynamics)
            resources_to_remove = []
            
            for food_source in self.ecosystem_dynamics.food_sources:
                food_pos = food_source.position
                
                # Check if this resource is too close to ANY agent
                too_close_to_agent = False
                for agent_id, agent_pos in agent_positions:
                    distance = ((food_pos[0] - agent_pos[0])**2 + (food_pos[1] - agent_pos[1])**2)**0.5
                    if distance < minimum_safe_distance:
                        too_close_to_agent = True
                        break
                
                # CRITICAL: Remove resources that are too close instead of moving them
                # This prevents random rewards from food appearing right next to robots
                if too_close_to_agent:
                    resources_to_remove.append(food_source)
            
            # Remove resources that are too close to agents
            if resources_to_remove:
                for food_source in resources_to_remove:
                    self.ecosystem_dynamics.food_sources.remove(food_source)
                print(f"🚫 Removed {len(resources_to_remove)} food sources that were too close to agents (<{minimum_safe_distance}m)")
                print(f"   📍 This prevents random rewards and ensures fair competition")
                
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
                
                # Apply consistent energy decay (not paused during eating)
                base_decay = 0.0001  
                
                # Minimal additional decay based on movement 
                velocity = agent.body.linearVelocity
                speed = (velocity.x ** 2 + velocity.y ** 2) ** 0.5
                movement_cost = speed * 0.0001  # Very minimal movement cost
                
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
                            print(f"💚 {agent_id[:8]} is recovering health (energy: {current_energy:.2f})")
                    
                    elif current_energy < 0.1:
                        # LOW ENERGY: Health degrades from starvation
                        health_degradation = 0.002  # Health loss when starving
                        self.agent_health[agent_id]['health'] = max(0.0, current_health - health_degradation)
                        if current_health - health_degradation <= 0.0:
                            print(f"💀 {agent_id[:8]} is starving to death (health: {current_health:.3f})")
                
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
                    
                    print(f"🍴 {agent_id[:8]} is {food_description} (energy: +{energy_gain:.2f}) [decay paused]")
                    
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
                
                # Log significant consumption events (keep minimal logging)
                #if energy_gain > 0.1:  # Only log substantial energy gains
                #    print(f"🍽️ Agent {agent_id[:8]} consumed energy: +{boosted_energy_gain:.2f}")
                
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
            
            # Replace dead agents with new ones
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
                events_with_lifespan = [event for event in self.death_events[-100:] if 'lifespan' in event]
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
        """Replace dead agents with new randomly generated agents."""
        with self._physics_lock:
            try:
                for dead_agent in dead_agents:
                    # Mark the dead agent for destruction
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
                        
                        # ✨ ASSIGN RANDOM LEARNING APPROACH TO REPLACEMENT AGENT
                        self._assign_random_learning_approach_single(replacement_agent)
                        
                        print(f"🐣 Spawned replacement agent {replacement_agent.id} for dead agent {dead_agent.id}")
                
            except Exception as e:
                print(f"❌ Error replacing dead agents: {e}")
    
    def _create_replacement_agent(self):
        """Create a new agent to replace a dead one using memory pool if available."""
        try:
            # Find a good spawn position (spread them out)
            existing_positions = []
            for agent in self.agents:
                if not getattr(agent, '_destroyed', False) and agent.body:
                    existing_positions.append(agent.body.position.x)
            
            # Find gaps in the population or place at edges
            if existing_positions:
                min_x = min(existing_positions) - 20
                max_x = max(existing_positions) + 20
                spawn_x = random.uniform(min_x, max_x)
            else:
                spawn_x = random.uniform(-50, 50)
            
            spawn_position = (spawn_x, 5.0)  # Spawn slightly above ground
            
            # Create random physical parameters (fresh genetics)
            from src.agents.physical_parameters import PhysicalParameters
            random_params = PhysicalParameters.random_parameters()
            
            # Use memory pool if available for efficient reuse with learning preservation
            if self.robot_memory_pool:
                new_agent = self.robot_memory_pool.acquire_robot(
                    position=spawn_position,
                    physical_params=random_params,
                    parent_lineage=[],  # Fresh agent, no lineage
                    restore_learning=True  # Try to restore previous learning if available
                )
                logger.debug(f"♻️ Acquired replacement agent {new_agent.id} from memory pool")
            else:
                # Fallback: Create new agent directly
                from src.agents.evolutionary_crawling_agent import EvolutionaryCrawlingAgent
                new_agent = EvolutionaryCrawlingAgent(
                    world=self.world,
                    agent_id=None,  # Generate new UUID automatically
                    position=spawn_position,
                    category_bits=self.AGENT_CATEGORY,
                    mask_bits=self.GROUND_CATEGORY | self.OBSTACLE_CATEGORY,  # Collide with ground AND obstacles
                    physical_params=random_params
                )
                logger.warning(f"🆕 Created new replacement agent {new_agent.id} (no memory pool)")
            
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
                            'source': 'replacement'
                        }
                    )
                except Exception as e:
                    print(f"⚠️ Failed to register replacement agent {new_agent.id} with reward signal adapter: {e}")
            
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
                'alliances': [],
                'territories': []
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

    def _update_statistics(self):
        """Update population statistics with enhanced safety checks."""
        if not self.agents:
            return
        
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
                self.robot_stats[agent_id]['episode_reward'] = getattr(agent, 'total_reward', 0.0)
                self.robot_stats[agent_id]['q_updates'] = agent.q_table.update_count if hasattr(agent.q_table, 'update_count') else 0
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
                    self.robot_memory_pool.return_robot(agent, preserve_learning=True)
                    print(f"♻️ Returned agent {agent.id} to memory pool with learning preserved")
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

        while self.is_running:
            current_time = time.time()
            frame_time = current_time - last_time
            last_time = current_time
            
            # Add frame time to the accumulator
            accumulator += frame_time
            
            # Fixed-step physics updates with enhanced thread safety
            while accumulator >= self.dt:
                with self._physics_lock:  # Protect ALL Box2D operations
                    try:
                        # Process any pending destructions first
                        self._process_destruction_queue()
                        
                        # Step the physics world only if not evolving
                        if not self._is_evolving:
                            self.world.Step(self.dt, 8, 3)
                            
                            # Update all agents (copy list to avoid iteration issues)
                            current_agents = self.agents.copy()
                            agents_to_reset = []
                            
                            for agent in current_agents:
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
                                    
                                    agent.step(self.dt)

                                    # Check for reset conditions but don't reset immediately
                                    if agent.body and agent.body.position.y < self.world_bounds_y:
                                        agents_to_reset.append(('world_bounds', agent))
                                    elif agent.steps >= self.episode_length:
                                        agents_to_reset.append(('episode_end', agent))
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
                                        agent.reset_position()
                                    elif reset_type == 'episode_end':
                                        agent.reset()  # preserves Q-table
                                        agent.reset_position()
                                except Exception as e:
                                    print(f"⚠️  Error resetting agent {agent.id}: {e}")
                                    
                    except Exception as e:
                        print(f"❌ Critical error in physics loop: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Decrement accumulator
                accumulator -= self.dt
                self.step_count += 1
                
                # Update physics FPS tracking
                self.physics_fps_counter += 1
                if current_time - self.last_physics_fps_update >= 1.0:  # Update every second
                    self.current_physics_fps = round(self.physics_fps_counter / (current_time - self.physics_fps_start_time))
                    self.physics_fps_counter = 0
                    self.physics_fps_start_time = current_time
                    self.last_physics_fps_update = current_time

            # Update camera and statistics (can be done once per frame)
            self.update_camera(frame_time)
            
            # Update ecosystem dynamics periodically
            if current_time - self.last_ecosystem_update > self.ecosystem_update_interval:
                self._update_ecosystem_dynamics()
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
            
            # Health check logging every 30 seconds
            if current_time - last_health_check > 30.0:
                try:
                    import psutil
                    import os
                    process = psutil.Process(os.getpid())
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    cpu_percent = process.cpu_percent()
                    active_agents = len([a for a in self.agents if not getattr(a, '_destroyed', False)])
                    print(f"💚 HEALTH CHECK: Memory={memory_mb:.1f}MB, CPU={cpu_percent:.1f}%, Agents={active_agents}, Step={self.step_count}")
                except Exception as e:
                    print(f"💚 HEALTH CHECK: Step={self.step_count}, Agents={len(self.agents)} (Error getting system stats: {e})")
                last_health_check = current_time
            
            # Performance cleanup every 2 minutes
            if current_time - self.last_performance_cleanup > self.performance_cleanup_interval:
                self._cleanup_performance_data()
                self.last_performance_cleanup = current_time
            
            # Update enhanced learning systems (after current_agents is defined)
            if hasattr(self, 'learning_manager') and self.learning_manager:
                try:
                    # Create a safe copy of current agents for this section
                    current_agents = [agent for agent in self.agents if not getattr(agent, '_destroyed', False)]
                    
                    # Update agent performance for elite identification
                    for agent in current_agents:
                        if not getattr(agent, '_destroyed', False) and agent.body:
                            try:
                                self.learning_manager.update_agent_performance(agent, self.step_count)
                            except Exception as e:
                                if self.step_count % 1000 == 0:  # Log occasionally to avoid spam
                                    print(f"⚠️ Error updating agent performance: {e}")
                    
                    # Update elite agents periodically
                    self.learning_manager.update_elites(current_agents, self.step_count)
                except Exception as e:
                    if self.step_count % 1000 == 0:  # Log occasionally to avoid spam
                        print(f"⚠️ Error updating enhanced learning systems: {e}")
                
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
                self._update_statistics()
                
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
                                    'position_x': agent.body.position.x if agent.body else 0,
                                    'q_table_size': len(agent.q_table.q_values) if hasattr(agent.q_table, 'q_values') else 0
                                }
                                self.mlflow_integration.log_individual_robot_metrics(
                                    f"top_{i+1}_{agent.id[:8]}", individual_metrics, self.step_count
                                )
                        
                    except Exception as e:
                        print(f"⚠️  Error logging to MLflow: {e}")
                
                last_mlflow_log = current_time
            
            # Check for periodic learning
            if current_time - self.last_learning_time >= self.learning_interval:
                self.perform_periodic_learning()
                self.last_learning_time = current_time
            
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

    def get_status(self):
        """Returns the current state of the simulation for rendering with enhanced safety."""
        if not self.is_running:
            return {'shapes': {}, 'leaderboard': [], 'robots': [], 'agents': [], 'statistics': {}, 'camera': self.get_camera_state(), 'focused_agent_id': None}

        # Use a read lock to safely access agents
        with self._physics_lock:
            try:
                current_time = time.time()  # For death event age calculations
                
                # Create a safe copy of agents list
                current_agents = [agent for agent in self.agents if not getattr(agent, '_destroyed', False)]
                
                # 1. Get agent shapes for drawing
                robot_shapes = []
                for agent in current_agents:
                    try:
                        if not agent.body:  # Skip agents without bodies
                            continue
                            
                        body_parts = []
                        # Chassis, Arms, Wheels
                        body_list = [agent.body] + (agent.wheels or [])
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
                
                # 3. Get leaderboard data (top 10 robots) - safely sorted by food consumption
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
                agents_data = []
                try:
                    for agent in current_agents:
                        if not agent.body:  # Skip agents without bodies
                            continue
                            
                        agent_id = agent.id
                        is_focused = (self.focused_agent and not getattr(self.focused_agent, '_destroyed', False) and self.focused_agent.id == agent_id)
                        
                        # Basic data for all agents (needed for rendering)
                        basic_agent_data = {
                            'id': agent.id,
                            'body': {
                                'x': safe_convert_numeric(agent.body.position.x),
                                'y': safe_convert_numeric(agent.body.position.y),
                                'velocity': {
                                    'x': safe_convert_numeric(agent.body.linearVelocity.x),
                                    'y': safe_convert_numeric(agent.body.linearVelocity.y)
                                }
                            },
                            'upper_arm': {
                                'x': safe_convert_numeric(agent.upper_arm.position.x) if agent.upper_arm else 0,
                                'y': safe_convert_numeric(agent.upper_arm.position.y) if agent.upper_arm else 0
                            },
                            'lower_arm': {
                                'x': safe_convert_numeric(agent.lower_arm.position.x) if agent.lower_arm else 0,
                                'y': safe_convert_numeric(agent.lower_arm.position.y) if agent.lower_arm else 0
                            },
                            'total_reward': safe_convert_numeric(agent.total_reward),
                            # Basic ecosystem data for rendering
                            'ecosystem': {
                                'role': self.agent_statuses.get(agent_id, {}).get('role', 'omnivore'),
                                'status': self.agent_statuses.get(agent_id, {}).get('status', 'idle'),
                                'health': safe_convert_numeric(self.agent_health.get(agent_id, {'health': 1.0})['health']),
                                'energy': safe_convert_numeric(self.agent_health.get(agent_id, {'energy': 1.0})['energy']),
                                'speed': safe_convert_numeric((agent.body.linearVelocity.x ** 2 + agent.body.linearVelocity.y ** 2) ** 0.5)
                            }
                        }
                        agents_data.append(basic_agent_data)
                        
                        # Add detailed data for focused agent
                        if is_focused:
                            agent_status = self.agent_statuses.get(agent_id, {})
                            agent_health = self.agent_health.get(agent_id, {'health': 1.0, 'energy': 1.0})
                            closest_food_info = self._get_closest_food_distance_for_agent(agent)
                            
                            # Add detailed data to the basic agent data
                            basic_agent_data.update({
                                'steps': safe_convert_numeric(agent.steps),
                                'current_action': safe_convert_list(agent.current_action_tuple),
                                'state': safe_convert_list(agent.current_state),
                                'q_table': len(agent.q_table.q_values) if hasattr(agent.q_table, 'q_values') else 0,
                                'action_history': safe_convert_list(agent.action_history[-10:]) if agent.action_history else [],
                                'best_reward': safe_convert_numeric(getattr(agent, 'best_reward_received', 0.0)),
                                'worst_reward': safe_convert_numeric(getattr(agent, 'worst_reward_received', 0.0)),
                                'awake': agent.body.awake if agent.body else False,
                                'learning_approach': getattr(agent, 'learning_approach', 'basic_q_learning'),
                            })
                            
                            # Add detailed ecosystem data
                            basic_agent_data['ecosystem'].update({
                                'speed_factor': safe_convert_numeric(agent_status.get('speed_factor', 1.0)),
                                'alliances': agent_status.get('alliances', []),
                                'territories': agent_status.get('territories', []),
                                'closest_food_distance': safe_convert_numeric(closest_food_info['distance']),
                                'closest_food_signed_x_distance': safe_convert_numeric(closest_food_info.get('signed_x_distance', closest_food_info['distance'])),
                                'closest_food_type': closest_food_info['food_type'],
                                'closest_food_source': closest_food_info.get('source_type', 'environment'),
                                'closest_food_position': safe_convert_position(closest_food_info.get('food_position', None))
                            })
                            
                except Exception as e:
                    print(f"⚠️  Error creating agents data: {e}")
                    agents_data = []

                # 6. Get focused agent ID safely
                focused_agent_id = None
                if self.focused_agent and not getattr(self.focused_agent, '_destroyed', False):
                    focused_agent_id = self.focused_agent.id

                # 7. Get ecosystem and environmental data
                ecosystem_status = self.ecosystem_dynamics.get_ecosystem_status()
                environmental_status = self.environmental_system.get_status()
                
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
                    'agents': agents_data,
                    'statistics': self.population_stats,
                    'camera': self.get_camera_state(),
                    'focused_agent_id': focused_agent_id,
                    # Enhanced visualization data
                    'ecosystem': {
                        'status': ecosystem_status,
                        'territories': [
                            {
                                'type': t.territory_type.value,
                                'position': t.position,
                                'size': t.size,
                                'resource_value': t.resource_value,
                                'owner_id': t.owner_id,
                                'contested': t.contested
                            }
                            for t in self.ecosystem_dynamics.territories
                        ],
                        'food_sources': [
                            {
                                'position': f.position,
                                'type': f.food_type,
                                'amount': f.amount,
                                'max_capacity': f.max_capacity
                            }
                            for f in self.ecosystem_dynamics.food_sources
                        ],
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
                        'obstacles': self._get_obstacle_data_for_ui()  # Use new physics-body-based obstacle data
                    },
                    'physics_fps': getattr(self, 'current_physics_fps', 0)
                }
                
            except Exception as e:
                print(f"❌ Critical error in get_status: {e}")
                import traceback
                traceback.print_exc()
                return {'shapes': {}, 'leaderboard': [], 'robots': [], 'agents': [], 'statistics': {}, 'camera': self.get_camera_state(), 'focused_agent_id': None}

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
    
    def perform_periodic_learning(self):
        """Make all robots learn from the best performing robot with safety checks."""
        leader = self.find_leader()
        if not leader:
            print("⚠️ No leader found for periodic learning")
            return
        
        # Double-check leader has valid body before proceeding
        if not leader.body:
            print("⚠️ Leader has no body, skipping periodic learning")
            return
        
        # Count how many agents actually learned
        learning_count = 0
        
        print(f"🎓 === PERIODIC LEARNING EVENT ===")
        
        try:
            leader_distance = leader.body.position.x - leader.initial_position[0]
            print(f"🏆 Leader: Robot {leader.id} (Distance: {leader_distance:.2f})")
        except Exception as e:
            print(f"🏆 Leader: Robot {leader.id} (Distance calculation failed: {e})")
        
        # Only include valid agents in learning
        valid_agents = [agent for agent in self.agents if not getattr(agent, '_destroyed', False)]
        
        for agent in valid_agents:
            if agent == leader:
                continue  # Leader doesn't learn from itself
            
            # Make agent learn from leader's Q-table
            if hasattr(agent, 'q_table') and hasattr(leader, 'q_table'):
                try:
                    # Use the existing learn_from_other_table method
                    agent.q_table.learn_from_other_table(leader.q_table, self.learning_rate)
                    learning_count += 1
                except Exception as e:
                    print(f"❌ Error during learning for Agent {agent.id}: {e}")
        
        print(f"📚 {learning_count} robots learned from Robot {leader.id}")
        print(f"🔄 Learning rate: {self.learning_rate:.1%}")
        print(f"⏰ Next learning session in {self.learning_interval} seconds")
        print(f"🎓 === LEARNING EVENT COMPLETE ===")
        print()  # Add spacing for readability

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
                    self.agents:List[EvolutionaryCrawlingAgent] = new_population
                    
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
        
        new_agent = EvolutionaryCrawlingAgent(
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
        cloned_agent.reset_with_new_position(position)
        
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
                agent = next((a for a in self.agents if a.id == agent_id and not getattr(a, '_destroyed', False)), None)
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
        print(f"🖱️ SERVER: Received click for agent_id: {agent_id}")

        if agent_id is not None:
            # Find the agent by ID
            agent_to_focus = next((agent for agent in self.agents if agent.id == agent_id), None)
            if agent_to_focus:
                self.focus_on_agent(agent_to_focus)
                return jsonify({'status': 'success', 'message': f'Focused on agent {agent_id}', 'agent_id': agent_id})
            else:
                self.focus_on_agent(None) # Clear focus if agent not found
                return jsonify({'status': 'error', 'message': f'Agent {agent_id} not found', 'agent_id': None})
        else:
            # If no agent_id is provided, it's a click on empty space, so clear focus
            self.focus_on_agent(None)
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
    
    def update_user_zoom(self, zoom_level):
        """Update the user's preferred zoom level."""
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

    def switch_agent_learning_approach(self, agent_id: str, approach: str) -> bool:
        """
        Switch a specific agent to a new learning approach.
        
        Args:
            agent_id: ID of the agent to switch
            approach: Learning approach name ('basic_q_learning', 'enhanced_survival_q', etc.)
            
        Returns:
            bool: True if switch was successful, False otherwise
        """
        if not self.learning_manager:
            print(f"❌ Learning Manager not available for agent {agent_id}")
            return False
        
        # Find the agent by ID
        agent = next((a for a in self.agents if a.id == agent_id and not getattr(a, '_destroyed', False)), None)
        if not agent:
            print(f"❌ Agent {agent_id} not found or destroyed")
            return False
        
        # Map string approach to enum
        from src.agents.learning_manager import LearningApproach
        approach_mapping = {
            'basic_q_learning': LearningApproach.BASIC_Q_LEARNING,
            'enhanced_survival_q': LearningApproach.SURVIVAL_Q_LEARNING,
            'deep_survival_q': LearningApproach.DEEP_Q_LEARNING,
            'auto_advanced': LearningApproach.ENHANCED_Q_LEARNING  # Fallback for auto-advanced
        }
        
        learning_approach = approach_mapping.get(approach)
        if not learning_approach:
            print(f"❌ Unknown learning approach: {approach}")
            return False
        
        # Perform the switch
        success = self.learning_manager.set_agent_approach(agent, learning_approach)
        
        if success:
            # Update agent data to include learning approach for frontend
            setattr(agent, 'learning_approach', approach)
            
            print(f"✅ Agent {agent_id} switched to {approach}")
        else:
            print(f"❌ Failed to switch agent {agent_id} to {approach}")
        
        return success

    def _get_agent_learning_approach_name(self, agent_id: str) -> str:
        """Get the learning approach name for an agent."""
        agent = next((a for a in self.agents if a.id == agent_id and not getattr(a, '_destroyed', False)), None)
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
                food_type_desc = f"robot prey {prey_id[:8]} (energy: {prey_energy:.2f})"
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
            
            # Add obstacles that have physics bodies (including any remaining dynamic ones)
            if hasattr(self, 'obstacle_bodies'):
                for obstacle_id, body in self.obstacle_bodies.items():
                    if body and hasattr(body, 'userData') and body.userData:
                        obstacle_type = body.userData.get('obstacle_type', 'unknown')
                        position = (body.position.x, body.position.y)
                        
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
                        
                        obstacles_for_ui.append({
                            'id': obstacle_id,
                            'type': obstacle_type,
                            'position': position_array,  # JavaScript expects array [x, y]
                            'size': size,
                            'danger_level': danger_level,  # JavaScript expects this for coloring
                            'active': True
                        })
            
            return obstacles_for_ui
            
        except Exception as e:
            print(f"⚠️ Error getting obstacle data for UI: {e}")
            return []

    def _cleanup_performance_data(self):
        """Clean up accumulated performance data to prevent memory growth."""
        try:
            # Clean up Q-learning history data
            for agent in self.agents:
                if getattr(agent, '_destroyed', False):
                    continue
                    
                # Limit action history
                if hasattr(agent, 'action_history') and len(agent.action_history) > 50:
                    agent.action_history = agent.action_history[-50:]
                
                # Clean up Q-table data if it exists and has history
                if hasattr(agent, 'q_table'):
                    if hasattr(agent.q_table, 'q_value_history') and len(agent.q_table.q_value_history) > 100:
                        agent.q_table.q_value_history = agent.q_table.q_value_history[-100:]
                    
                    # Clean up visit counts for very large Q-tables
                    if hasattr(agent.q_table, 'visit_counts') and hasattr(agent.q_table.visit_counts, '__len__'):
                        try:
                            if len(agent.q_table.visit_counts) > 5000:
                                # Reset visit counts for least visited state-actions
                                pass  # Skip complex cleanup for now
                        except:
                            pass
                
                # Clean up replay buffer if too large
                if hasattr(agent, 'replay_buffer') and hasattr(agent.replay_buffer, 'buffer'):
                    buffer_capacity = getattr(agent.replay_buffer, 'capacity', 3000)
                    if len(agent.replay_buffer.buffer) > buffer_capacity * 0.9:
                        # Remove oldest 25% of experiences
                        old_size = len(agent.replay_buffer.buffer)
                        remove_count = old_size // 4
                        for _ in range(remove_count):
                            if agent.replay_buffer.buffer:
                                agent.replay_buffer.buffer.popleft()
            
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
            print(f"🧹 Performance cleanup completed (agents: {len(self.agents)}, stats: {len(self.robot_stats)})")
            
        except Exception as e:
            print(f"⚠️ Error during performance cleanup: {e}")

    def test_carnivore_feeding(self):
        """Test method to place a carnivore next to prey to verify feeding mechanics"""
        print("🧪 Running carnivore feeding test...")
        
        # Find a carnivore and a herbivore
        carnivore = None
        herbivore = None
        
        for agent in self.agents:
            if getattr(agent, '_destroyed', False) or not agent.body:
                continue
            role = self.agent_statuses.get(agent.id, {}).get('role', 'omnivore')
            if role == 'carnivore' and carnivore is None:
                carnivore = agent
            elif role == 'herbivore' and herbivore is None:
                herbivore = agent
            
            if carnivore and herbivore:
                break
        
        if not carnivore or not herbivore:
            print("❌ Test failed: Could not find both carnivore and herbivore")
            return
        
        # Record initial states
        carnivore_initial_energy = self.agent_energy_levels.get(carnivore.id, 1.0)
        herbivore_initial_energy = self.agent_energy_levels.get(herbivore.id, 1.0)
        herbivore_initial_health = self.agent_health.get(herbivore.id, {'health': 1.0})['health']
        
        # Position carnivore next to herbivore (within consumption distance)
        herbivore_pos = (herbivore.body.position.x, herbivore.body.position.y)
        test_position = (herbivore_pos[0] + 2.0, herbivore_pos[1])  # 2 meters away (within 4m consumption distance)
        
        # Move carnivore to test position
        carnivore.body.position = test_position
        carnivore.body.linearVelocity = (0, 0)  # Stop movement
        
        # Lower carnivore's energy to trigger hunting behavior
        self.agent_energy_levels[carnivore.id] = 0.5  # Below 0.6 threshold
        
        print(f"🔬 Test setup:")
        print(f"   Carnivore {carnivore.id[:8]} - Energy: {carnivore_initial_energy:.2f} -> 0.5")
        print(f"   Herbivore {herbivore.id[:8]} - Energy: {herbivore_initial_energy:.2f}, Health: {herbivore_initial_health:.2f}")
        print(f"   Distance: {math.sqrt((test_position[0] - herbivore_pos[0])**2 + (test_position[1] - herbivore_pos[1])**2):.1f}m")
        
        # Run consumption updates for several frames to see if feeding occurs
        print("🔄 Running consumption updates...")
        for frame in range(10):  # Run for 10 frames
            # Manually trigger resource consumption update
            self._update_resource_consumption()
            
            # Check current states
            carnivore_energy = self.agent_energy_levels.get(carnivore.id, 1.0)
            herbivore_energy = self.agent_energy_levels.get(herbivore.id, 1.0)
            herbivore_health = self.agent_health.get(herbivore.id, {'health': 1.0})['health']
            carnivore_status = self.agent_statuses.get(carnivore.id, {}).get('status', 'unknown')
            
            print(f"   Frame {frame+1}: Carnivore energy {carnivore_energy:.3f}, status: {carnivore_status}")
            print(f"   Frame {frame+1}: Herbivore energy {herbivore_energy:.3f}, health: {herbivore_health:.3f}")
            
            # Check if carnivore is consuming herbivore
            if carnivore_energy > 0.5 or carnivore_status == 'eating':
                print(f"✅ SUCCESS: Carnivore is consuming herbivore!")
                print(f"   Energy gained: {carnivore_energy - 0.5:.3f}")
                print(f"   Herbivore health lost: {herbivore_initial_health - herbivore_health:.3f}")
                return
        
        print("❌ FAILED: Carnivore did not consume herbivore after 10 frames")
        print("   This indicates the robot consumption system is not working properly")
        
        # Reset positions to avoid disrupting normal simulation
        carnivore.body.position = (random.uniform(-30, 30), 5.0)
        self.agent_energy_levels[carnivore.id] = carnivore_initial_energy

# --- Main Execution ---
app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')
env = TrainingEnvironment(num_agents=30)  # Reduced from 50 to 30 for memory stability

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
    env.stop()
    return jsonify({'status': 'success'})

@app.route('/click', methods=['POST'])
def handle_click():
    """Handles a click event from the frontend."""
    data = request.get_json()
    agent_id = data.get('agent_id')
    print(f"🖱️ SERVER: Received click for agent_id: {agent_id}")

    if agent_id is not None:
        # Find the agent by ID
        agent_to_focus = next((agent for agent in env.agents if agent.id == agent_id), None)
        if agent_to_focus:
            env.focus_on_agent(agent_to_focus)
            return jsonify({'status': 'success', 'message': f'Focused on agent {agent_id}', 'agent_id': agent_id})
        else:
            env.focus_on_agent(None) # Clear focus if agent not found
            return jsonify({'status': 'error', 'message': f'Agent {agent_id} not found', 'agent_id': None})
    else:
        # If no agent_id is provided, it's a click on empty space, so clear focus
        env.focus_on_agent(None)
        return jsonify({'status': 'success', 'message': 'Focus cleared', 'agent_id': None})

@app.route('/get_agent_at_position', methods=['POST'])
def get_agent_at_position():
    """Gets the agent at a specific mouse position."""
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

@app.route('/update_zoom', methods=['POST'])
def update_zoom():
    data = request.get_json()
    if not data or 'zoom' not in data:
        return jsonify({'status': 'error', 'message': 'No zoom level provided'}), 400
    
    zoom_level = data['zoom']
    env.update_user_zoom(zoom_level)
    
    return jsonify({'status': 'success', 'zoom': env.user_zoom_level})

@app.route('/reset_view', methods=['POST'])
def reset_view():
    # Clear focus, reset zoom preferences, and reset camera position
    env.focus_on_agent(None)
    env.reset_user_zoom()
    env.reset_camera_position()
    
    return jsonify({'status': 'success', 'message': 'View reset'})

@app.route('/clear_zoom_override', methods=['POST'])
def clear_zoom_override():
    env.clear_zoom_override()
    return jsonify({'status': 'success'})

@app.route('/switch_learning_approach', methods=['POST'])
def switch_learning_approach():
    """Switch an agent's learning approach."""
    data = request.get_json()
    if not data or 'agent_id' not in data or 'approach' not in data:
        return jsonify({'status': 'error', 'message': 'Missing agent_id or approach'}), 400
    
    agent_id = data['agent_id']
    approach = data['approach']
    
    success = env.switch_agent_learning_approach(agent_id, approach)
    
    if success:
        return jsonify({'status': 'success', 'agent_id': agent_id, 'approach': approach})
    else:
        return jsonify({'status': 'error', 'message': f'Failed to switch agent {agent_id} to {approach}'}), 500

@app.route('/change_terrain_style', methods=['POST'])
def change_terrain_style():
    """Change the terrain generation style and regenerate the terrain."""
    try:
        data = request.get_json()
        if not data or 'style' not in data:
            return jsonify({'status': 'error', 'message': 'Missing style parameter'}), 400
        
        new_style = data['style']
        
        # Get available robot-scale terrain styles
        robot_terrain_styles = [
            'flat', 'gentle_hills', 'obstacle_course', 'slopes_and_ramps', 
            'rough_terrain', 'varied', 'mixed'
        ]
        
        if new_style not in robot_terrain_styles:
            return jsonify({
                'status': 'error', 
                'message': f'Unknown style. Available: {robot_terrain_styles}'
            }), 400
        
        # Change the terrain style
        success = env.change_terrain_style(new_style)
        
        if success:
            # Robot-scale terrain style descriptions
            style_descriptions = {
                'flat': 'Mostly flat terrain with occasional small features',
                'gentle_hills': 'Small, navigable hills and gentle slopes',
                'obstacle_course': 'Various sized obstacles and challenges',
                'slopes_and_ramps': 'Terrain focused on slopes and ramps',
                'rough_terrain': 'Rough, uneven terrain',
                'varied': 'Varied terrain with all feature types',
                'mixed': 'Balanced mixed terrain good for robot training'
            }
            
            return jsonify({
                'status': 'success', 
                'message': f'Terrain changed to {new_style}',
                'style': new_style,
                'description': style_descriptions.get(new_style, 'Robot-scale terrain'),
                'terrain_bodies_created': len(env.terrain_collision_bodies)
            })
        else:
            return jsonify({'status': 'error', 'message': 'Failed to change terrain style'}), 500
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_terrain_styles', methods=['GET'])
def get_terrain_styles():
    """Get available robot-scale terrain styles."""
    try:
        # Robot-scale terrain styles with descriptions
        robot_terrain_styles = {
            'flat': {
                'name': 'flat',
                'description': 'Mostly flat terrain with occasional small features',
                'style': 'flat'
            },
            'gentle_hills': {
                'name': 'gentle_hills',
                'description': 'Small, navigable hills and gentle slopes',
                'style': 'gentle_hills'
            },
            'obstacle_course': {
                'name': 'obstacle_course',
                'description': 'Various sized obstacles and challenges',
                'style': 'obstacle_course'
            },
            'slopes_and_ramps': {
                'name': 'slopes_and_ramps',
                'description': 'Terrain focused on slopes and ramps',
                'style': 'slopes_and_ramps'
            },
            'rough_terrain': {
                'name': 'rough_terrain',
                'description': 'Rough, uneven terrain',
                'style': 'rough_terrain'
            },
            'varied': {
                'name': 'varied',
                'description': 'Varied terrain with all feature types',
                'style': 'varied'
            },
            'mixed': {
                'name': 'mixed',
                'description': 'Balanced mixed terrain good for robot training',
                'style': 'mixed'
            }
        }
        
        return jsonify({
            'status': 'success',
            'styles': robot_terrain_styles,
            'current_style': env.terrain_style if hasattr(env, 'terrain_style') else 'mixed'
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

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
            env.trigger_evolution()  # Use new evolution method
        elif event == 'learn_from_leader':
            env.perform_periodic_learning()
        elif event == 'toggle_auto_evolution':
            enabled = env.toggle_auto_evolution()
            return jsonify({'status': 'success', 'event': event, 'auto_evolution_enabled': enabled})
        else:
            return jsonify({'status': 'error', 'message': f'Unknown event: {event}'}), 400
        
        return jsonify({'status': 'success', 'event': event})
    except Exception as e:
        print(f"❌ Evolution event '{event}' failed: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/evolution_status', methods=['GET'])
def evolution_status():
    """Get current evolution status."""
    try:
        status = env.get_evolution_status()
        return jsonify({'status': 'success', 'evolution_status': status})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_diverse_agents', methods=['GET'])
def get_diverse_agents():
    """Get diverse representatives from the population."""
    try:
        count = request.args.get('count', 5, type=int)
        diverse_agents = env.evolution_engine.get_diverse_representatives(count)
        
        agent_info = []
        for agent in diverse_agents:
            info = {
                'id': agent.id,
                'fitness': agent.get_evolutionary_fitness(),
                'generation': agent.generation,
                'diversity_metrics': agent.get_diversity_metrics(),
                'physical_summary': {
                    'body_size': f"{agent.physical_params.body_width:.2f}x{agent.physical_params.body_height:.2f}",
                    'wheel_radius': agent.physical_params.wheel_radius,
                    'motor_torque': agent.physical_params.motor_torque,
                    'learning_rate': agent.physical_params.learning_rate,
                    'epsilon': agent.physical_params.epsilon
                }
            }
            agent_info.append(info)
        
        return jsonify({'status': 'success', 'diverse_agents': agent_info})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/evaluation_metrics', methods=['GET'])
def get_evaluation_metrics():
    """Get comprehensive evaluation metrics."""
    try:
        if not env.enable_evaluation or not env.metrics_collector:
            return jsonify({'status': 'error', 'message': 'Evaluation framework not enabled'}), 400
        
        metrics_summary = env.metrics_collector.get_current_metrics_summary()
        return jsonify({'status': 'success', 'metrics': metrics_summary})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/evaluation_diagnostics', methods=['GET'])
def get_evaluation_diagnostics():
    """Get training diagnostics and recommendations."""
    try:
        if not env.enable_evaluation or not env.metrics_collector:
            return jsonify({'status': 'error', 'message': 'Evaluation framework not enabled'}), 400
        
        diagnostics = env.metrics_collector.get_training_diagnostics()
        return jsonify({'status': 'success', 'diagnostics': diagnostics})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/test_carnivore_feeding', methods=['POST'])
def test_carnivore_feeding():
    """Test endpoint to verify carnivore feeding mechanics"""
    try:
        if env:
            env.test_carnivore_feeding()
            return jsonify({'status': 'success', 'message': 'Carnivore feeding test completed - check console for results'})
        else:
            return jsonify({'status': 'error', 'message': 'Training environment not available'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Test failed: {str(e)}'})

@app.route('/q_learning_status', methods=['GET'])
def get_q_learning_status():
    """Get comprehensive Q-learning evaluation status."""
    try:
        from src.evaluation.q_learning_integration import get_q_learning_status_for_api
        status = get_q_learning_status_for_api(env)
        return jsonify(status)
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Q-learning status error: {str(e)}'})

@app.route('/q_learning_agent/<agent_id>', methods=['GET'])
def get_agent_q_learning_metrics(agent_id):
    """Get detailed Q-learning metrics for a specific agent."""
    try:
        if not hasattr(env, 'q_learning_evaluator') or env.q_learning_evaluator is None:
            return jsonify({'status': 'error', 'message': 'Q-learning evaluator not initialized'}), 400
        
        metrics = env.q_learning_evaluator.get_agent_metrics(agent_id)
        if metrics:
            return jsonify({'status': 'success', 'metrics': metrics.to_dict()})
        else:
            return jsonify({'status': 'error', 'message': f'No metrics found for agent {agent_id}'}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/q_learning_agent/<agent_id>/diagnostics', methods=['GET'])
def get_agent_q_learning_diagnostics(agent_id):
    """Get Q-learning diagnostics and recommendations for a specific agent."""
    try:
        if not hasattr(env, 'q_learning_evaluator') or env.q_learning_evaluator is None:
            return jsonify({'status': 'error', 'message': 'Q-learning evaluator not initialized'}), 400
        
        diagnostics = env.q_learning_evaluator.get_learning_diagnostics(agent_id)
        return jsonify({'status': 'success', 'diagnostics': diagnostics})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/q_learning_comparison', methods=['GET'])
def get_q_learning_type_comparison():
    """Get comparative analysis of Q-learning performance across agent types."""
    try:
        if not hasattr(env, 'q_learning_evaluator') or env.q_learning_evaluator is None:
            return jsonify({'status': 'error', 'message': 'Q-learning evaluator not initialized'}), 400
        
        comparison = env.q_learning_evaluator.get_type_comparison()
        return jsonify({'status': 'success', 'comparison': comparison})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/q_learning_summary', methods=['GET'])
def get_q_learning_summary():
    """Get comprehensive Q-learning summary report."""
    try:
        if not hasattr(env, 'q_learning_evaluator') or env.q_learning_evaluator is None:
            return jsonify({'status': 'error', 'message': 'Q-learning evaluator not initialized'}), 400
        
        summary = env.q_learning_evaluator.generate_summary_report()
        return jsonify({'status': 'success', 'summary': summary})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/reward_signal_status', methods=['GET'])
def get_reward_signal_status():
    """Get overall reward signal evaluation status."""
    try:
        from src.evaluation.reward_signal_integration import reward_signal_adapter
        status = reward_signal_adapter.get_system_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting reward signal status: {e}")
        return jsonify({'error': str(e), 'timestamp': time.time()})

@app.route('/reward_signal_summary', methods=['GET'])
def get_reward_signal_summary():
    """Get comprehensive reward signal quality summary."""
    try:
        from src.evaluation.reward_signal_integration import reward_signal_adapter
        summary = reward_signal_adapter.get_reward_comparative_report()
        return jsonify(summary)
    except Exception as e:
        logger.error(f"Error getting reward signal summary: {e}")
        return jsonify({'error': str(e), 'timestamp': time.time()})

@app.route('/reward_signal_agent/<agent_id>', methods=['GET'])
def get_agent_reward_signal_metrics(agent_id):
    """Get reward signal metrics for a specific agent."""
    try:
        from src.evaluation.reward_signal_integration import reward_signal_adapter
        metrics = reward_signal_adapter.get_agent_reward_metrics(agent_id)
        
        if metrics:
            return jsonify({
                'agent_id': agent_id,
                'metrics': metrics.to_dict(),
                'timestamp': time.time()
            })
        else:
            return jsonify({
                'agent_id': agent_id,
                'status': 'no_data',
                'message': 'No reward signal data available for this agent',
                'timestamp': time.time()
            })
    except Exception as e:
        logger.error(f"Error getting reward signal metrics for agent {agent_id}: {e}")
        return jsonify({'error': str(e), 'agent_id': agent_id, 'timestamp': time.time()})

@app.route('/reward_signal_agent/<agent_id>/diagnostics', methods=['GET'])
def get_agent_reward_signal_diagnostics(agent_id):
    """Get detailed reward signal diagnostics for a specific agent."""
    try:
        from src.evaluation.reward_signal_integration import reward_signal_adapter
        diagnostics = reward_signal_adapter.get_agent_diagnostics(agent_id)
        return jsonify(diagnostics)
    except Exception as e:
        logger.error(f"Error getting reward signal diagnostics for agent {agent_id}: {e}")
        return jsonify({'error': str(e), 'agent_id': agent_id, 'timestamp': time.time()})

@app.route('/reward_signal_comparison', methods=['GET'])
def get_reward_signal_comparison():
    """Get comparative analysis of reward signal quality across agents."""
    try:
        from src.evaluation.reward_signal_integration import reward_signal_adapter
        all_metrics = reward_signal_adapter.get_all_reward_metrics()
        
        if not all_metrics:
            return jsonify({
                'status': 'no_data',
                'message': 'No reward signal data available',
                'timestamp': time.time()
            })
        
        # Organize by quality tiers
        quality_tiers = {
            'excellent': [],
            'good': [],
            'fair': [],
            'poor': [],
            'very_poor': []
        }
        
        for agent_id, metrics in all_metrics.items():
            if metrics.quality_score >= 0.8:
                tier = 'excellent'
            elif metrics.quality_score >= 0.6:
                tier = 'good'
            elif metrics.quality_score >= 0.4:
                tier = 'fair'
            elif metrics.quality_score >= 0.2:
                tier = 'poor'
            else:
                tier = 'very_poor'
            
            quality_tiers[tier].append({
                'agent_id': agent_id,
                'quality_score': metrics.quality_score,
                'signal_to_noise_ratio': metrics.signal_to_noise_ratio,
                'reward_consistency': metrics.reward_consistency,
                'exploration_incentive': metrics.exploration_incentive,
                'main_issues': [issue.value for issue in metrics.quality_issues[:3]]
            })
        
        # Sort each tier by quality score
        for tier in quality_tiers.values():
            tier.sort(key=lambda x: x['quality_score'], reverse=True)
        
        return jsonify({
            'quality_tiers': quality_tiers,
            'tier_counts': {tier: len(agents) for tier, agents in quality_tiers.items()},
            'total_agents': len(all_metrics),
            'timestamp': time.time()
        })
        
    except Exception as e:
        logger.error(f"Error getting reward signal comparison: {e}")
        return jsonify({'error': str(e), 'timestamp': time.time()})

@app.route('/performance_status')
def get_performance_status():
    """Enhanced performance status with Q-learning metrics."""
    try:
        # Get basic performance status
        performance_data = {
            'entities': {
                'world_bodies': len(env.world.bodies) if env.world else 0,
                'food_sources': len(getattr(env.ecosystem_dynamics, 'food_sources', [])),
                'static_bodies': len(getattr(env, 'terrain_collision_bodies', [])),
                'dynamic_bodies': len([a for a in env.agents if not getattr(a, '_destroyed', False)]),
                'total_agents': len(env.agents),
                'active_agents': len([a for a in env.agents if not getattr(a, '_destroyed', False)])
            },
            'performance': {
                'step_count': getattr(env, 'step_count', 0),
                'memory_mb': 0.0,  # Will be filled by system monitoring
                'cpu_percent': 0.0,  # Will be filled by system monitoring
                'fps': getattr(env, 'fps', 0.0)
            }
        }
        
        # Add system resource information
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            performance_data['performance']['memory_mb'] = process.memory_info().rss / 1024 / 1024
            performance_data['performance']['cpu_percent'] = process.cpu_percent()
        except:
            pass
        
        # Add Q-learning evaluation data if available
        if hasattr(env, 'q_learning_evaluator') and env.q_learning_evaluator is not None:
            try:
                q_metrics = env.q_learning_evaluator.get_all_agent_metrics()
                q_comparison = env.q_learning_evaluator.get_type_comparison()
                
                performance_data['q_learning'] = {
                    'agents_monitored': len(q_metrics),
                    'agent_types': list(q_comparison.keys()),
                    'overall_stats': {
                        'avg_prediction_mae': float(sum(m.value_prediction_mae for m in q_metrics.values()) / len(q_metrics)) if q_metrics else 0.0,
                        'avg_convergence_score': float(sum(m.convergence_score for m in q_metrics.values()) / len(q_metrics)) if q_metrics else 0.0,
                        'agents_with_issues': len([m for m in q_metrics.values() if m.learning_issues]),
                        'agents_learning_well': len([m for m in q_metrics.values() if m.learning_efficiency_score > 0.6])
                    },
                    'type_performance': q_comparison
                }
            except Exception as e:
                performance_data['q_learning'] = {'error': str(e)}
        else:
            performance_data['q_learning'] = {'status': 'not_initialized'}
        
        # Add reward signal evaluation data if available
        try:
            from src.evaluation.reward_signal_integration import reward_signal_adapter
            reward_status = reward_signal_adapter.get_system_status()
            reward_metrics = reward_signal_adapter.get_all_reward_metrics()
            
            if reward_metrics:
                # Calculate aggregate statistics
                avg_quality = sum(m.quality_score for m in reward_metrics.values()) / len(reward_metrics)
                avg_snr = sum(m.signal_to_noise_ratio for m in reward_metrics.values()) / len(reward_metrics)
                avg_consistency = sum(m.reward_consistency for m in reward_metrics.values()) / len(reward_metrics)
                
                # Count quality issues
                all_issues = []
                for metrics in reward_metrics.values():
                    all_issues.extend([issue.value for issue in metrics.quality_issues])
                
                issue_counts = {}
                for issue in all_issues:
                    issue_counts[issue] = issue_counts.get(issue, 0) + 1
                
                performance_data['reward_signals'] = {
                    'agents_monitored': len(reward_metrics),
                    'total_rewards_recorded': reward_status['total_rewards_recorded'],
                    'overall_stats': {
                        'avg_quality_score': float(avg_quality),
                        'avg_signal_to_noise_ratio': float(avg_snr),
                        'avg_consistency': float(avg_consistency),
                        'agents_with_good_rewards': len([m for m in reward_metrics.values() if m.quality_score > 0.6]),
                        'agents_with_issues': len([m for m in reward_metrics.values() if m.quality_issues]),
                        'sparse_reward_agents': len([m for m in reward_metrics.values() if m.reward_sparsity > 0.8])
                    },
                    'common_issues': issue_counts,
                    'status': 'active' if reward_status['active'] else 'inactive'
                }
            else:
                performance_data['reward_signals'] = {
                    'status': 'no_data',
                    'agents_monitored': 0,
                    'total_rewards_recorded': reward_status.get('total_rewards_recorded', 0)
                }
        except Exception as e:
            performance_data['reward_signals'] = {'error': str(e)}
        
        return jsonify(performance_data)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

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