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
            <div id="focus-indicator" style="display:none; position:absolute; top:10px; right:10px; z-index:50; background:rgba(231, 76, 60, 0.9); color:white; padding:10px 15px; border-radius:5px;">
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
                    <div class="stat-section">
                        <div class="stat-row">
                            <span class="stat-label">🌿 Herbivores:</span>
                            <span class="stat-value" style="color: #44AA44;">${roleDistribution.herbivore || 0}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">🦁 Carnivores:</span>
                            <span class="stat-value" style="color: #FF4444;">${roleDistribution.carnivore || 0}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">🐻 Omnivores:</span>
                            <span class="stat-value" style="color: #FF8844;">${roleDistribution.omnivore || 0}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">🦅 Scavengers:</span>
                            <span class="stat-value" style="color: #8B4513;">${roleDistribution.scavenger || 0}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">🐠 Symbionts:</span>
                            <span class="stat-value" style="color: #20B2AA;">${roleDistribution.symbiont || 0}</span>
                        </div>
                    </div>
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
                        <div class="detail-row" style="color: #FF0000; font-weight: bold; background: rgba(255, 0, 0, 0.2); padding: 4px; border-radius: 3px;">
                            <span class="detail-label">💀 CRITICAL:</span>
                            <span class="detail-value">DYING - Find food immediately!</span>
                        </div>` : ''}
                        ${status === 'dying' ? `
                        <div class="detail-row" style="color: #8B0000; font-weight: bold; background: rgba(139, 0, 0, 0.3); padding: 4px; border-radius: 3px;">
                            <span class="detail-label">⚰️ STATUS:</span>
                            <span class="detail-value">STARVING TO DEATH</span>
                        </div>` : ''}
                    </div>
                    
                    <div class="detail-section">
                        <div class="detail-row">
                            <span class="detail-label">Learning Approach:</span>
                            <span class="detail-value" id="learning-approach-${agent.id}">${agent.learning_approach || 'Enhanced Q-Learning'}</span>
                        </div>
                        <div class="learning-controls" style="margin-top: 8px;">
                            <button class="learning-btn" onclick="switchLearningApproach('${agent.id}', 'basic_q_learning')" style="background: #808080; color: white; border: none; padding: 4px 8px; margin: 2px; border-radius: 3px; font-size: 10px; cursor: pointer;">🔤 Basic</button>
                            <button class="learning-btn" onclick="switchLearningApproach('${agent.id}', 'enhanced_q_learning')" style="background: #3498db; color: white; border: none; padding: 4px 8px; margin: 2px; border-radius: 3px; font-size: 10px; cursor: pointer;">⚡ Enhanced</button>
                            <button class="learning-btn" onclick="switchLearningApproach('${agent.id}', 'survival_q_learning')" style="background: #27ae60; color: white; border: none; padding: 4px 8px; margin: 2px; border-radius: 3px; font-size: 10px; cursor: pointer;">🍃 Survival</button>
                            <button class="learning-btn" onclick="switchLearningApproach('${agent.id}', 'deep_q_learning')" style="background: #9b59b6; color: white; border: none; padding: 4px 8px; margin: 2px; border-radius: 3px; font-size: 10px; cursor: pointer;" disabled>🧠 Deep</button>
                        </div>
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
            
            // Animation effects disabled for performance optimization

            ctx.restore(); // Restore to pre-camera transform state
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
                
                // Add danger indicator for high-danger obstacles
                if (dangerLevel > 0.5) {
                    ctx.fillStyle = '#FF0000';
                    ctx.font = `${size/3}px Arial`;
                    ctx.textAlign = 'center';
                    ctx.fillText('⚠', x, y + size/6);
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
                    ctx.setLineDash(contested ? [0.5, 0.5] : []);
                    
                    // Draw territory boundary
                    ctx.beginPath();
                    ctx.arc(x, y, size/2, 0, 2 * Math.PI);
                    ctx.stroke();
                    
                    ctx.setLineDash([]); // Reset line dash
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
                        case 'meat': foodColor = '#F44336'; break;
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
                    
                    // Draw resource type icon
                    ctx.fillStyle = '#FFFFFF';
                    ctx.font = '0.8px Arial';
                    ctx.textAlign = 'center';
                    const typeIcons = {
                        'plants': '🌿',
                        'meat': '🥩',
                        'insects': '🐛',
                        'seeds': '🌰'
                    };
                    ctx.fillText(typeIcons[food.type] || '🍃', x, y + 0.3);
                    
                    // Add depletion warning animation
                    if (ratio < 0.4) {
                        const time = Date.now() / 1000;
                        const pulse = 0.3 + 0.7 * Math.sin(time * 6);
                        const warningColor = ratio < 0.2 ? 'rgba(255, 0, 0, ' : 'rgba(255, 165, 0, ';
                        
                        ctx.strokeStyle = warningColor + pulse + ')';
                        ctx.lineWidth = 0.2;
                        ctx.beginPath();
                        ctx.arc(x, y, radius + 0.4, 0, 2 * Math.PI);
                        ctx.stroke();
                        
                        // Sparkle effects disabled for performance optimization
                    }
                    
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
                    
                    // Amount text
                    ctx.fillStyle = '#FFFFFF';
                    ctx.font = '0.4px Arial';
                    ctx.textAlign = 'center';
                    ctx.fillText(`${amount.toFixed(0)}/${maxCapacity.toFixed(0)}`, x, barY - 0.2);
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
                
                // Alliance connections disabled for performance optimization
            });
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
                ctx.fillStyle = '#FFFFFF';
                ctx.font = '1px Arial';
                ctx.textAlign = 'center';
                ctx.fillText(roleSymbols[role] || '🤖', x, baseY + barSpacing * 2 + 0.8);
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
                    
                    ctx.fillStyle = STATUS_COLORS[status] || '#FFFFFF';
                    ctx.font = '0.8px Arial';
                    ctx.textAlign = 'center';
                    ctx.fillText(statusSymbols[status] || '●', x + 1.5, y + 2.0);
                }
                
                // Simple consumption indicator for feeding agents (no expensive animations)
                if (status === 'feeding' && energy < 0.9) {
                    // Simple green circle around feeding agents
                    ctx.strokeStyle = 'rgba(50, 200, 50, 0.8)';
                    ctx.lineWidth = 0.2;
                    ctx.beginPath();
                    ctx.arc(x, y, 1.5, 0, 2 * Math.PI);
                    ctx.stroke();
                }
        }
        
        function drawMovementTrail(agentId, position, speed) {
            // Movement trails disabled for performance - too expensive with many agents
            // Simple approach: no trails, just current position rendering
            return;
        }
        
        function drawAllianceConnections(agentId, position, alliances, allAgents) {
            // Alliance connections disabled for performance - expensive distance calculations and line drawing
            // The alliance system still works in the backend, just not visually displayed
            return;
        }
        
        function drawPredationEvents() {
            // Predation effects disabled for performance - too many particles with trigonometry
            // The predation events are still logged and tracked, just not visually animated
            return;
        }
        
        function drawDeathEvents() {
            if (!window.lastData || !window.lastData.ecosystem || !window.lastData.ecosystem.death_events) return;
            
            const deathEvents = window.lastData.ecosystem.death_events;
            
            deathEvents.forEach(event => {
                if (event.age > 3.0) return; // Only show for 3 seconds
                
                const [x, y] = event.position;
                const alpha = Math.max(0, 1.0 - (event.age / 3.0)); // Fade over 3 seconds
                
                // Simple role-based death icon
                const deathSymbols = {
                    'carnivore': '💀',
                    'herbivore': '🥀', 
                    'omnivore': '⚰️',
                    'scavenger': '👻',
                    'symbiont': '💫'
                };
                
                // Simple icon display - no complex animations
                ctx.fillStyle = `rgba(255, 255, 255, ${alpha})`;
                ctx.font = '1.0px Arial';
                ctx.textAlign = 'center';
                ctx.fillText(deathSymbols[event.role] || '💀', x, y);
            });
        }

        // Throttle frame rate for better performance - 30 FPS instead of 60 FPS
        let lastFrameTime = 0;
        const targetFPS = 30;
        const frameInterval = 1000 / targetFPS; // ~33ms between frames
        
        function fetchData() {
            const now = Date.now();
            if (now - lastFrameTime < frameInterval) {
                requestAnimationFrame(fetchData);
                return;
            }
            lastFrameTime = now;
            
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
            
            // Add learning approach controls
            const approachControls = document.createElement('div');
            approachControls.innerHTML = `
                <div style="margin-top: 10px; padding: 8px; background: rgba(30, 40, 60, 0.7); border-radius: 5px;">
                    <div style="font-weight: bold; margin-bottom: 5px; color: #3498db;">🎛️ Bulk Learning Control</div>
                    <div style="display: flex; flex-wrap: wrap; gap: 4px;">
                        <button onclick="bulkSwitchLearning('basic_q_learning')" style="background: #808080; color: white; border: none; padding: 4px 6px; border-radius: 3px; font-size: 9px; cursor: pointer;">🔤 All Basic</button>
                        <button onclick="bulkSwitchLearning('enhanced_q_learning')" style="background: #3498db; color: white; border: none; padding: 4px 6px; border-radius: 3px; font-size: 9px; cursor: pointer;">⚡ All Enhanced</button>
                        <button onclick="bulkSwitchLearning('survival_q_learning')" style="background: #27ae60; color: white; border: none; padding: 4px 6px; border-radius: 3px; font-size: 9px; cursor: pointer;">🍃 All Survival</button>
                        <button onclick="randomizeLearningApproaches()" style="background: #e67e22; color: white; border: none; padding: 4px 6px; border-radius: 3px; font-size: 9px; cursor: pointer;">🎲 Randomize</button>
                    </div>
                    <div id="learning-stats" style="margin-top: 8px; font-size: 9px; color: #bdc3c7;">
                        Loading learning statistics...
                    </div>
                </div>
            `;
            learningPanelContent.appendChild(approachControls);
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

        // Learning approach switching functions
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
                    console.log(`✅ Switched agent ${agentId} to ${approach}`);
                    
                    // Update the UI to show new approach
                    const approachElement = document.getElementById(`learning-approach-${agentId}`);
                    if (approachElement) {
                        const approachNames = {
                            'basic_q_learning': '🔤 Basic Q-Learning',
                            'enhanced_q_learning': '⚡ Enhanced Q-Learning',
                            'survival_q_learning': '🍃 Survival Q-Learning',
                            'deep_q_learning': '🧠 Deep Q-Learning'
                        };
                        approachElement.textContent = approachNames[approach] || approach;
                    }
                } else {
                    console.error(`❌ Failed to switch agent ${agentId}:`, result.message);
                }
            } catch (error) {
                console.error('Error switching learning approach:', error);
            }
        }

        async function bulkSwitchLearning(approach) {
            try {
                const response = await fetch('./bulk_switch_learning', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        approach: approach
                    })
                });
                
                const result = await response.json();
                if (result.status === 'success') {
                    console.log(`✅ Bulk switched ${result.success_count} agents to ${approach}`);
                } else {
                    console.error('❌ Bulk switch failed:', result.message);
                }
            } catch (error) {
                console.error('Error bulk switching learning approaches:', error);
            }
        }

        async function randomizeLearningApproaches() {
            try {
                const response = await fetch('./randomize_learning_approaches', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });
                
                const result = await response.json();
                if (result.status === 'success') {
                    console.log(`✅ Randomized learning approaches for ${result.agents_updated} agents`);
                } else {
                    console.error('❌ Randomization failed:', result.message);
                }
            } catch (error) {
                console.error('Error randomizing learning approaches:', error);
            }
        }

        async function updateLearningStatistics() {
            try {
                const response = await fetch('./learning_statistics');
                const result = await response.json();
                
                if (result.status === 'success' && result.statistics) {
                    const stats = result.statistics;
                    const statsElement = document.getElementById('learning-stats');
                    if (statsElement) {
                        const distribution = stats.approach_distribution;
                        const total = stats.total_agents;
                        
                        let statsHtml = `Total: ${total} agents<br/>`;
                        if (distribution.basic_q_learning > 0) statsHtml += `🔤 Basic: ${distribution.basic_q_learning} `;
                        if (distribution.enhanced_q_learning > 0) statsHtml += `⚡ Enhanced: ${distribution.enhanced_q_learning} `;
                        if (distribution.survival_q_learning > 0) statsHtml += `🍃 Survival: ${distribution.survival_q_learning} `;
                        if (distribution.deep_q_learning > 0) statsHtml += `🧠 Deep: ${distribution.deep_q_learning} `;
                        
                        statsElement.innerHTML = statsHtml;
                    }
                }
            } catch (error) {
                console.error('Error updating learning statistics:', error);
            }
        }

        // Update learning statistics every 5 seconds
        setInterval(updateLearningStatistics, 5000);
        updateLearningStatistics(); // Initial update

    </script>
</body>
</html>
"""

def convert_numpy_types(obj):
    """Convert numpy types to JSON-serializable types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        # Create a copy of the dictionary to avoid "dictionary changed size during iteration" error
        # This happens when Q-learning agents are actively updating the Q-table while we read it
        try:
            dict_copy = dict(obj)  # Make a shallow copy
            return {key: convert_numpy_types(value) for key, value in dict_copy.items()}
        except RuntimeError:
            # If we still get a RuntimeError, return an empty dict as fallback
            return {}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

class TrainingEnvironment:
    """
    Enhanced training environment with evolutionary physical parameters.
    Manages physics simulation and evolution of diverse crawling robots.
    Enhanced with comprehensive evaluation framework.
    """
    def __init__(self, num_agents=30, enable_evaluation=True):  # Reduced from 50 to 30 to save memory
        self.num_agents = num_agents
        self.world = b2.b2World(gravity=(0, -10), doSleep=True)
        self.dt = 1.0 / 60.0

        # World bounds for resetting fallen agents
        self.world_bounds_y = -20.0
        
        # Collision filtering setup
        self.GROUND_CATEGORY = 0x0001
        self.AGENT_CATEGORY = 0x0002

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

        # Initialize robot memory pool for efficient object reuse
        try:
            from src.agents.robot_memory_pool import RobotMemoryPool
            self.robot_memory_pool = RobotMemoryPool(
                world=self.world,
                min_pool_size=10,
                max_pool_size=50,
                category_bits=self.AGENT_CATEGORY,
                mask_bits=self.GROUND_CATEGORY
            )
            print("🏊 Robot memory pool initialized for efficient object reuse")
        except Exception as e:
            print(f"⚠️ Could not initialize robot memory pool: {e}")
            self.robot_memory_pool = None

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
        self.auto_evolution_enabled = True
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
        self.eating_status_buffers = {}  # Track eating status duration
        self.predation_events = []  # Track recent predation events for visualization
        self.last_ecosystem_update = time.time()
        self.ecosystem_update_interval = 6.0  # Update ecosystem every 2 seconds to quickly remove depleted resources
        
        # Resource generation system  
        self.last_resource_generation = time.time()
        self.resource_generation_interval = 5.0  # Generate resources very frequently to ensure abundance
        self.agent_energy_levels = {}  # Track agent energy levels for resource consumption
        
        # Death and survival system
        self.death_events = []  # Track recent death events for visualization
        self.agents_pending_replacement = []  # Queue for dead agents needing replacement
        self.survival_stats = {
            'total_deaths': 0,
            'deaths_by_starvation': 0,
            'average_lifespan': 0,
            'agent_birth_times': {}  # Track when each agent was born/created
        }
        
        # Initialize ecosystem roles for existing agents
        self._initialize_ecosystem_roles()

        # FLEXIBLE LEARNING SYSTEM INTEGRATION
        print(f"\n🎛️ === FLEXIBLE LEARNING SYSTEM INTEGRATION ===")
        self._initialize_learning_manager()
        
        # Connect learning manager to memory pool for comprehensive transfer
        if hasattr(self, 'robot_memory_pool') and hasattr(self, 'learning_manager') and self.robot_memory_pool:
            self.robot_memory_pool.set_learning_manager(self.learning_manager)
            print("🔗 Learning manager connected to memory pool for comprehensive knowledge transfer")

        print(f"\n🧬 Enhanced Training Environment initialized:")
        print(f"   Population: {len(self.agents)} diverse agents")
        print(f"   Evolution: {self.evolution_config.population_size} agents, {self.evolution_config.elite_size} elite")
        print(f"   Diversity target: {self.evolution_config.target_diversity}")
        print(f"   Auto-evolution every {self.evolution_interval}s")
        print(f"🌿 Ecosystem dynamics and visualization systems active")
        print(f"🎛️ Flexible learning system: Multiple approaches available via UI")
        print(f"🧠 Survival learning: {len(getattr(self, 'survival_adapters', []))} agents upgraded")
        print(f"")
        print(f"🎮 WEB INTERFACE FEATURES:")
        print(f"   • Click robots to focus and see detailed stats")
        print(f"   • Individual learning approach switching (🔤⚡🍃🧠)")
        print(f"   • Bulk learning controls in the Learning panel")
        print(f"   • Real-time learning performance comparison")
        print(f"   • Dynamic ecosystem visualization with food/energy")

    def _create_ground(self):
        """Creates a static ground body."""
        ground_body = self.world.CreateStaticBody(position=(0, -1))
        
        # Calculate ground width based on number of agents (reasonable size)
        ground_width = max(500, self.num_agents * 15)  # Ensure enough width for all agents
        
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

    def _initialize_ecosystem_roles(self):
        """Initialize ecosystem roles for all agents based on their characteristics."""
        for agent in self.agents:
            if not getattr(agent, '_destroyed', False):
                # Extract fitness traits from agent's physical parameters
                fitness_traits = {
                    'speed': getattr(agent.physical_params, 'motor_speed', 5.0) / 10.0,  # Normalize to 0-1
                    'strength': getattr(agent.physical_params, 'motor_torque', 50.0) / 100.0,  # Normalize to 0-1
                    'cooperation': min(1.0, getattr(agent.physical_params, 'learning_rate', 0.1) * 10.0)  # Higher learning rate = more cooperative
                }
                
                # Assign ecosystem role
                role = self.ecosystem_dynamics.assign_ecosystem_role(agent.id, fitness_traits)
                
                # Initialize agent status tracking
                self.agent_statuses[agent.id] = {
                    'role': role.value,
                    'status': 'idle',  # idle, hunting, feeding, fleeing, territorial
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

    def _initialize_learning_manager(self):
        """Initialize the flexible learning manager system."""
        try:
            # Create ecosystem interface
            self.ecosystem_interface = EcosystemInterface(self)
            
            # Create learning manager
            self.learning_manager = LearningManager(ecosystem_interface=self.ecosystem_interface)
            
            # Initialize all agents with random learning approaches for diversity
            import random
            
            # All available learning approaches
            all_approaches = [
                LearningApproach.BASIC_Q_LEARNING,
                LearningApproach.ENHANCED_Q_LEARNING,
                LearningApproach.SURVIVAL_Q_LEARNING,
                LearningApproach.DEEP_Q_LEARNING
            ]
            
            initialized_count = 0
            approach_counts = {approach: 0 for approach in all_approaches}
            
            for agent in self.agents:
                if not getattr(agent, '_destroyed', False) and agent.body:
                    try:
                        # Random approach assignment for maximum diversity
                        random_approach = random.choice(all_approaches)
                        success = self.learning_manager.set_agent_approach(agent, random_approach)
                        
                        if success:
                            initialized_count += 1
                            approach_counts[random_approach] += 1
                            print(f"🎲 Agent {agent.id[:8]} assigned {random_approach.value} learning")
                        
                    except Exception as e:
                        print(f"⚠️ Failed to initialize learning for agent {agent.id}: {e}")
                        continue
            
            # Legacy survival adapters for backward compatibility
            self.survival_adapters = []
            for agent_id, adapter in self.learning_manager.agent_adapters.items():
                if self.learning_manager.get_agent_approach(agent_id) == LearningApproach.SURVIVAL_Q_LEARNING:
                    self.survival_adapters.append(adapter)
            
            # Add survival statistics tracking
            survival_count = approach_counts.get(LearningApproach.SURVIVAL_Q_LEARNING, 0)
            self.survival_learning_stats = {
                'agents_upgraded': survival_count,
                'total_food_consumed': 0,
                'average_learning_stage': 'basic_movement',
                'stage_distribution': {
                    'basic_movement': survival_count,
                    'food_seeking': 0,
                    'survival_mastery': 0
                }
            }
            
            print(f"✅ Flexible learning system initialized with RANDOM approaches:")
            print(f"   • {initialized_count} agents initialized with diverse learning approaches")
            print(f"   • {approach_counts[LearningApproach.BASIC_Q_LEARNING]} agents using Basic Q-Learning")
            print(f"   • {approach_counts[LearningApproach.ENHANCED_Q_LEARNING]} agents using Enhanced Q-Learning")
            print(f"   • {approach_counts[LearningApproach.SURVIVAL_Q_LEARNING]} agents using Survival Q-Learning")
            print(f"   • {approach_counts[LearningApproach.DEEP_Q_LEARNING]} agents using Deep Q-Learning (GPU)")
            print(f"🎲 Each robot started with a RANDOM learning policy for maximum diversity!")
            print(f"🎛️ Learning approaches available:")
            print(f"   • Basic Q-Learning: Simple tabular approach")
            print(f"   • Enhanced Q-Learning: Advanced tabular with experience replay")
            print(f"   • Survival Q-Learning: Ecosystem-aware learning")
            print(f"   • Deep Q-Learning: Neural network-based (coming soon)")
            
        except Exception as e:
            print(f"❌ Critical error initializing learning manager: {e}")
            self.learning_manager = None
            self.ecosystem_interface = None
            import traceback
            traceback.print_exc()

    def _update_survival_learning_stats(self):
        """Update survival learning statistics and progress tracking."""
        try:
            if not hasattr(self, 'survival_adapters') or not self.survival_adapters:
                return
            
            # Collect stats from all survival adapters
            total_food_consumed = 0
            stage_counts = {'basic_movement': 0, 'food_seeking': 0, 'survival_mastery': 0}
            total_energy_gained = 0.0
            
            for adapter in self.survival_adapters:
                try:
                    stats = adapter.get_learning_stats()
                    
                    # Accumulate food consumption
                    total_food_consumed += stats.get('food_consumed', 0)
                    total_energy_gained += stats.get('energy_gained', 0.0)
                    
                    # Count learning stages
                    stage = stats.get('learning_stage', 'basic_movement')
                    if stage in stage_counts:
                        stage_counts[stage] += 1
                    
                except Exception as e:
                    continue  # Skip problematic adapters
            
            # Update global survival stats
            self.survival_learning_stats.update({
                'total_food_consumed': total_food_consumed,
                'total_energy_gained': total_energy_gained,
                'stage_distribution': stage_counts
            })
            
            # Determine average learning stage
            if stage_counts['survival_mastery'] > 0:
                self.survival_learning_stats['average_learning_stage'] = 'survival_mastery'
            elif stage_counts['food_seeking'] > stage_counts['basic_movement']:
                self.survival_learning_stats['average_learning_stage'] = 'food_seeking'
            else:
                self.survival_learning_stats['average_learning_stage'] = 'basic_movement'
            
            # Add to population stats for monitoring
            if hasattr(self, 'population_stats'):
                self.population_stats['survival_learning'] = self.survival_learning_stats
            
        except Exception as e:
            print(f"⚠️ Error updating survival learning stats: {e}")

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
            
            # Clean up old death events (simple icons only)
            self.death_events = [
                event for event in self.death_events 
                if current_time - event['timestamp'] < 3.0  # Only keep 3 seconds for simple icon display
            ]
            
        except Exception as e:
            print(f"⚠️ Error updating ecosystem dynamics: {e}")
    
    def _update_agent_statuses(self):
        """Update agent statuses based on their current behaviors and ecosystem role."""
        current_time = time.time()
        
        for agent in self.agents:
            if getattr(agent, '_destroyed', False) or not agent.body:
                continue
                
            agent_id = agent.id
            if agent_id not in self.agent_statuses:
                continue
                
            status_data = self.agent_statuses[agent_id]
            role = status_data['role']
            
            # Update speed factor based on velocity
            velocity = agent.body.linearVelocity
            speed = (velocity.x ** 2 + velocity.y ** 2) ** 0.5
            status_data['speed_factor'] = min(2.0, speed / 2.0)  # Normalize and cap at 2x
            
            # Update status based on role and behavior
            if role == 'carnivore':
                # Carnivores hunt when they have energy
                if self.agent_health[agent_id]['energy'] > 0.3 and speed > 1.0:
                    status_data['status'] = 'hunting'
                elif speed < 0.5:
                    status_data['status'] = 'idle'
                else:
                    status_data['status'] = 'moving'
            elif role == 'herbivore':
                # Herbivores feed or flee
                if speed > 2.0:
                    status_data['status'] = 'fleeing'
                elif speed < 0.5:
                    status_data['status'] = 'feeding'
                else:
                    status_data['status'] = 'moving'
            else:
                # Other roles have simpler status updates
                if speed > 1.5:
                    status_data['status'] = 'moving'
                elif speed < 0.5:
                    status_data['status'] = 'idle'
                else:
                    status_data['status'] = 'active'
            
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
                
                # Generate resources between agents using ecosystem dynamics
                self.ecosystem_dynamics.generate_resources_between_agents(agent_positions)
                
                print(f"🌱 Resource generation cycle completed. Total resources: {len(self.ecosystem_dynamics.food_sources)}")
        
        except Exception as e:
            print(f"⚠️ Error generating resources: {e}")
    
    def _get_energy_based_status(self, agent_id: str, current_energy: float) -> str:
        """Get appropriate status based on energy level and agent role"""
        if current_energy < 0.05:
            return 'dying'
        elif current_energy < 0.4:
            return 'feeding'
        else:
            role = self.agent_statuses.get(agent_id, {}).get('role', 'omnivore')
            if role == 'carnivore':
                return 'hunting'
            else:
                return 'idle'
    
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
                
                # Proper energy decay over time - realistic survival pressure
                base_decay = 0.0005  # Lose 0.05% energy per frame (about 1.8% per second at 60fps)
                
                # Movement-based energy cost
                velocity = agent.body.linearVelocity
                speed = (velocity.x ** 2 + velocity.y ** 2) ** 0.5
                movement_cost = speed * 0.0002  # Additional cost for movement
                
                # Role-based energy costs (minimal differences)
                role = self.agent_statuses.get(agent_id, {}).get('role', 'omnivore')
                role_multipliers = {
                    'carnivore': 1.05,  # Very slightly more energy needed
                    'herbivore': 0.95,  # Very slightly more efficient
                    'omnivore': 1.0,    # Balanced
                    'scavenger': 0.98,  # Very slightly efficient
                    'symbiont': 0.92    # More efficient but not extreme
                }
                role_multiplier = role_multipliers.get(role, 1.0)
                
                total_decay = (base_decay + movement_cost) * role_multiplier
                current_energy = max(0.0, current_energy - total_decay)
                
                # Try to consume nearby resources (balanced energy gain)
                consumption_result = self.ecosystem_dynamics.consume_resource(agent_id, agent_position)
                energy_gain = consumption_result['energy_gained']
                
                # Further reduced boost to energy gain for proper balance with new consumption rate
                boosted_energy_gain = energy_gain * 1.5  # 1.5x energy gain from resources (was 2x)
                current_energy = min(1.0, current_energy + boosted_energy_gain)
                
                # Try predation for carnivores and scavengers  
                from src.ecosystem_dynamics import EcosystemRole
                agent_role = self.ecosystem_dynamics.agent_roles.get(agent_id, None)
                if agent_role in [EcosystemRole.CARNIVORE, EcosystemRole.SCAVENGER]:
                    # Gather potential prey information
                    potential_prey = []
                    for other_agent in self.agents:  
                        if (other_agent.id != agent_id and 
                            not getattr(other_agent, '_destroyed', False) and 
                            other_agent.body):
                            other_position = (other_agent.body.position.x, other_agent.body.position.y)
                            other_energy = self.agent_energy_levels.get(other_agent.id, 1.0)
                            potential_prey.append((other_agent.id, other_position, other_energy))
                    
                    # Attempt predation
                    predation_result = self.ecosystem_dynamics.attempt_predation(agent_id, agent_position, potential_prey)
                    
                    if predation_result['consumed']:
                        # Successful hunt! Add energy
                        hunt_energy_gain = predation_result['energy_gained']
                        current_energy = min(1.0, current_energy + hunt_energy_gain)
                        
                        # Track predation for leaderboard (counts as food consumption)
                        if agent_id in self.robot_stats:
                            self.robot_stats[agent_id]['food_consumed'] += hunt_energy_gain
                        
                        # Set hunting status
                        if agent_id in self.agent_statuses:
                            prey_id = predation_result['prey_id']
                            self.agent_statuses[agent_id]['status'] = f'eating 🥩'
                            
                            # Set eating buffer for predation
                            current_time = time.time()
                            self.eating_status_buffers[agent_id] = {
                                'eating_status': 'eating 🥩',
                                'start_time': current_time,
                                'duration': 5.0,  # Hold hunting status longer (5 seconds)
                                'previous_status': self._get_energy_based_status(agent_id, current_energy)
                            }
                        
                        # Damage the prey (reduce their energy significantly)
                        prey_id = predation_result['prey_id']
                        if prey_id in self.agent_energy_levels:
                            damage = min(0.6, self.agent_energy_levels[prey_id] * 0.8)  # Up to 80% damage
                            self.agent_energy_levels[prey_id] = max(0.0, self.agent_energy_levels[prey_id] - damage)
                
                # Track food consumption for leaderboard
                if energy_gain > 0 and agent_id in self.robot_stats:
                    self.robot_stats[agent_id]['food_consumed'] += energy_gain
                
                # Update agent status to "eating" if actively consuming
                if agent_id in self.agent_statuses:
                    current_time = time.time()
                    
                    if consumption_result['consumed']:
                        # Set status to eating with food type and start buffer timer
                        food_type = consumption_result['food_type']
                        food_emoji = {
                            'plants': '🌿',
                            'meat': '🥩', 
                            'insects': '🐛',
                            'seeds': '🌰'
                        }.get(food_type, '🍽️')
                        
                        eating_status = f'eating {food_emoji}'
                        self.agent_statuses[agent_id]['status'] = eating_status
                        self.agent_statuses[agent_id]['last_status_change'] = current_time
                        
                        # Set eating buffer timer (hold eating status for 3 seconds)
                        self.eating_status_buffers[agent_id] = {
                            'eating_status': eating_status,
                            'start_time': current_time,
                            'duration': 3.0,  # Hold eating status for 3 seconds
                            'previous_status': self._get_energy_based_status(agent_id, current_energy)
                        }
                        #print(f"🍽️ Status: {agent_id[:8]} set to '{eating_status}' (buffered for 3s)")
                    
                    # Check if agent should still be in eating status due to buffer
                    if agent_id in self.eating_status_buffers:
                        buffer_info = self.eating_status_buffers[agent_id]
                        time_since_eating = current_time - buffer_info['start_time']
                        
                        if time_since_eating < buffer_info['duration']:
                            # Still within buffer time, maintain eating status
                            self.agent_statuses[agent_id]['status'] = buffer_info['eating_status']
                        else:
                            # Buffer expired, revert to appropriate status
                            new_status = self._get_energy_based_status(agent_id, current_energy)
                            self.agent_statuses[agent_id]['status'] = new_status
                            
                            # Remove buffer entry
                            del self.eating_status_buffers[agent_id]
                            #print(f"🔄 Status: {agent_id[:8]} buffer expired, reverted to '{new_status}'")
                
                # Update energy level
                self.agent_energy_levels[agent_id] = current_energy
                
                # Check for death by starvation
                if current_energy <= 0.0:
                    print(f"💀 Agent {agent_id} died from starvation! (Role: {role})")
                    agents_to_replace.append(agent)
                    self._record_death_event(agent, 'starvation')
                    continue
                
                # Update agent health data for visualization
                if agent_id in self.agent_health:
                    self.agent_health[agent_id]['energy'] = current_energy
                
                # Update agent status data for visualization
                if agent_id in self.agent_statuses:
                    self.agent_statuses[agent_id]['energy'] = current_energy
                    
                    # Update status based on energy level (more balanced thresholds)
                    if current_energy < 0.05:  # Only truly critical energy triggers dying
                        self.agent_statuses[agent_id]['status'] = 'dying'  # Critical energy
                    elif current_energy < 0.4:  # More reasonable threshold for seeking food
                        self.agent_statuses[agent_id]['status'] = 'feeding'  # Looking for food
                    elif current_energy > 0.7:  # Lower threshold for high energy activities
                        role = self.agent_statuses[agent_id]['role']
                        if role == 'carnivore':
                            self.agent_statuses[agent_id]['status'] = 'hunting'  # High energy, ready to hunt
            
            # Replace dead agents with new ones
            if agents_to_replace:
                self._replace_dead_agents(agents_to_replace)
        
        except Exception as e:
            print(f"⚠️ Error updating resource consumption: {e}")
    
    def _record_death_event(self, agent, cause='starvation'):
        """Record a death event for visualization and statistics."""
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
        
        # Calculate average lifespan
        if self.survival_stats['total_deaths'] > 0:
            total_lifespan = sum(event['lifespan'] for event in self.death_events[-100:])  # Last 100 deaths
            recent_deaths = min(len(self.death_events), 100)
            self.survival_stats['average_lifespan'] = total_lifespan / recent_deaths
        
        # Clean up old tracking data
        if agent.id in self.survival_stats['agent_birth_times']:
            del self.survival_stats['agent_birth_times'][agent.id]
        if agent.id in self.agent_energy_levels:
            del self.agent_energy_levels[agent.id]
        if agent.id in self.agent_health:
            del self.agent_health[agent.id]
        if agent.id in self.agent_statuses:
            del self.agent_statuses[agent.id]
    
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
                        
                        print(f"🐣 Spawned replacement agent {replacement_agent.id} for dead agent {dead_agent.id}")
                
            except Exception as e:
                print(f"❌ Error replacing dead agents: {e}")
    
    def _find_safe_spawn_position(self, min_distance=15.0, max_attempts=50):
        """Find a safe spawn position that doesn't collide with existing agents."""
        # Get all existing agent positions
        existing_positions = []
        for agent in self.agents:
            if not getattr(agent, '_destroyed', False) and agent.body:
                existing_positions.append((agent.body.position.x, agent.body.position.y))
        
        # Try to find a safe position
        for attempt in range(max_attempts):
            # Generate candidate position with increasing search radius
            search_radius = 50 + (attempt * 10)  # Start at 50, expand by 10 each attempt
            
            spawn_x = random.uniform(-search_radius, search_radius)
            spawn_y = random.uniform(3.0, 10.0)  # Slightly above ground with some variation
            candidate_position = (spawn_x, spawn_y)
            
            # Check distance from all existing agents
            is_safe = True
            for existing_pos in existing_positions:
                distance = ((candidate_position[0] - existing_pos[0])**2 + 
                           (candidate_position[1] - existing_pos[1])**2)**0.5
                if distance < min_distance:
                    is_safe = False
                    break
            
            if is_safe:
                print(f"✅ Found safe spawn position: ({spawn_x:.1f}, {spawn_y:.1f}) after {attempt + 1} attempts")
                return candidate_position
        
        # If we couldn't find a safe position, return a far edge position
        edge_x = random.choice([-100, 100]) + random.uniform(-20, 20)
        edge_y = random.uniform(5.0, 12.0)
        print(f"⚠️ Using edge spawn position: ({edge_x:.1f}, {edge_y:.1f}) after {max_attempts} failed attempts")
        return (edge_x, edge_y)

    def _create_replacement_agent(self):
        """Create a new agent to replace a dead one."""
        try:
            # Find a safe spawn position with collision detection
            spawn_position = self._find_safe_spawn_position()
            if spawn_position is None:
                print("⚠️ Could not find safe spawn position for replacement agent")
                return None
            
            # Create random physical parameters (fresh genetics)
            from src.agents.physical_parameters import PhysicalParameters
            random_params = PhysicalParameters.random_parameters()
            
            # Create new agent
            from src.agents.evolutionary_crawling_agent import EvolutionaryCrawlingAgent
            new_agent = EvolutionaryCrawlingAgent(
                world=self.world,
                agent_id=None,  # Generate new UUID automatically
                position=spawn_position,
                category_bits=self.AGENT_CATEGORY,
                mask_bits=self.GROUND_CATEGORY,
                physical_params=random_params
            )
            
            return new_agent
            
        except Exception as e:
            print(f"❌ Error creating replacement agent: {e}")
            return None
    
    def _initialize_single_agent_ecosystem(self, agent):
        """Initialize ecosystem data for a single new agent."""
        try:
            agent_id = agent.id
            
            # Extract fitness traits from agent's physical parameters
            fitness_traits = {
                'speed': getattr(agent.physical_params, 'motor_speed', 5.0) / 10.0,
                'strength': getattr(agent.physical_params, 'motor_torque', 50.0) / 100.0,
                'cooperation': min(1.0, getattr(agent.physical_params, 'learning_rate', 0.1) * 10.0)
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
                        'action_history': [],
                        'food_consumed': 0.0  # Track total food consumed for leaderboard
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
        
                    # Update survival learning statistics
            self._update_survival_learning_stats()
        
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
                
                self.population_stats = {
                    'generation': evolution_summary['generation'],
                    'best_distance': best_distance,
                    'average_distance': avg_distance,
                    'worst_distance': worst_distance,
                    'best_fitness': best_fitness,
                    'average_fitness': avg_fitness,
                    'diversity': evolution_summary['diversity'],
                    'total_agents': len(self.robot_stats),
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
        """Safely destroy an agent with proper error handling."""
        if not agent or getattr(agent, '_destroyed', False):
            return  # Already destroyed
            
        try:
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
            
            print(f"✅ Successfully destroyed agent {agent.id}")
            
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
                                    agent.step(self.dt)

                                    # Check for reset conditions but don't reset immediately
                                    if agent.body and agent.body.position.y < self.world_bounds_y:
                                        agents_to_reset.append(('world_bounds', agent))
                                    elif agent.steps >= self.episode_length:
                                        agents_to_reset.append(('episode_end', agent))
                                except Exception as e:
                                    print(f"⚠️  Error updating agent {agent.id}: {e}")
                                    # Mark problematic agent as destroyed to prevent further issues
                                    if not getattr(agent, '_destroyed', False):
                                        agent._destroyed = True
                            
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

            # Update camera and statistics (can be done once per frame)
            self.update_camera(frame_time)
            
            # Update ecosystem dynamics periodically
            if current_time - self.last_ecosystem_update > self.ecosystem_update_interval:
                self._update_ecosystem_dynamics()
                self.last_ecosystem_update = current_time
            
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
                
                # 3. Get leaderboard data (top 10 robots) - safely
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
                        if '🔤' in approach_name:
                            approach_icon = '🔤'
                        elif '⚡' in approach_name:
                            approach_icon = '⚡'
                        elif '🍃' in approach_name:
                            approach_icon = '🍃'
                        elif '🧠' in approach_name:
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
                            # Get agent ID for use throughout this loop iteration
                            agent_id = r_stat['id']
                            
                            # Calculate distance to nearest food that this robot type can actually eat
                            nearest_food_distance = float('inf')
                            nearest_food_type = "none"
                            
                            if hasattr(self, 'ecosystem_dynamics') and self.ecosystem_dynamics.food_sources:
                                agent_pos = r_stat.get('current_position', (0, 0))
                                
                                # Get robot's ecosystem role to determine what foods it can eat
                                agent_role = self.ecosystem_dynamics.agent_roles.get(agent_id, None)
                                
                                if agent_role:
                                    # Define consumption efficiency matrix (what each role can eat)
                                    from src.ecosystem_dynamics import EcosystemRole
                                    consumption_efficiency = {
                                        EcosystemRole.HERBIVORE: {"plants": 1.0, "seeds": 0.8, "insects": 0.0, "meat": 0.0},
                                        EcosystemRole.CARNIVORE: {"plants": 0.1, "seeds": 0.05, "insects": 0.6, "meat": 1.0},
                                        EcosystemRole.OMNIVORE: {"plants": 0.8, "insects": 0.8, "seeds": 0.7, "meat": 0.8},
                                        EcosystemRole.SCAVENGER: {"plants": 0.2, "seeds": 0.1, "insects": 0.7, "meat": 1.0},
                                        EcosystemRole.SYMBIONT: {"plants": 0.9, "seeds": 0.8, "insects": 0.5, "meat": 0.0}
                                    }
                                    
                                    role_efficiency = consumption_efficiency.get(agent_role, {})
                                    
                                    # Check environmental food sources that this robot can eat
                                    for food_source in self.ecosystem_dynamics.food_sources:
                                        if hasattr(food_source, 'amount') and food_source.amount > 0.1:  # Only consider non-depleted food
                                            food_type = food_source.food_type
                                            efficiency = role_efficiency.get(food_type, 0.0)
                                            
                                            # Only consider foods this robot can actually consume (efficiency > 0)
                                            if efficiency > 0.0:
                                                food_pos = food_source.position
                                                distance = ((agent_pos[0] - food_pos[0])**2 + (agent_pos[1] - food_pos[1])**2)**0.5
                                                if distance < nearest_food_distance:
                                                    nearest_food_distance = distance
                                                    nearest_food_type = food_type
                                    
                                    # For carnivores and scavengers, also consider other agents as potential prey
                                    if agent_role in [EcosystemRole.CARNIVORE, EcosystemRole.SCAVENGER]:
                                        for other_agent in current_agents:
                                            if (other_agent.id != agent_id and 
                                                not getattr(other_agent, '_destroyed', False) and 
                                                other_agent.body):
                                                
                                                other_pos = (other_agent.body.position.x, other_agent.body.position.y)
                                                distance = ((agent_pos[0] - other_pos[0])**2 + (agent_pos[1] - other_pos[1])**2)**0.5
                                                
                                                # For scavengers, only consider weak prey (more likely targets)
                                                if agent_role == EcosystemRole.SCAVENGER:
                                                    other_energy = self.agent_energy_levels.get(other_agent.id, 1.0)
                                                    if other_energy < 0.5:  # Only weak prey for scavengers
                                                        if distance < nearest_food_distance:
                                                            nearest_food_distance = distance
                                                            nearest_food_type = "prey (weak)"
                                                else:  # Carnivore - considers any prey
                                                    if distance < nearest_food_distance:
                                                        nearest_food_distance = distance
                                                        nearest_food_type = "prey"
                                else:
                                    # Fallback: if no role assigned, check all food sources
                                    for food_source in self.ecosystem_dynamics.food_sources:
                                        if hasattr(food_source, 'amount') and food_source.amount > 0.1:
                                            food_pos = food_source.position
                                            distance = ((agent_pos[0] - food_pos[0])**2 + (agent_pos[1] - food_pos[1])**2)**0.5
                                            if distance < nearest_food_distance:
                                                nearest_food_distance = distance
                                                nearest_food_type = food_source.food_type
                            
                            # If no consumable food found, set to a large but finite value
                            if nearest_food_distance == float('inf'):
                                nearest_food_distance = 999.9
                                nearest_food_type = "none"
                            
                            # Get robot's ecosystem role for display
                            agent_role_str = "unknown"
                            if hasattr(self, 'ecosystem_dynamics') and agent_id in self.ecosystem_dynamics.agent_roles:
                                agent_role_str = self.ecosystem_dynamics.agent_roles[agent_id].value
                            
                            robot_details.append({
                                'id': r_stat['id'],
                                'name': f"Robot {r_stat['id']}",
                                'rank': i + 1,
                                'distance': r_stat.get('total_distance', 0),
                                'position': r_stat.get('current_position', (0,0)),
                                'episode_reward': r_stat.get('episode_reward', 0),
                                'nearest_food_distance': round(nearest_food_distance, 1),
                                'nearest_food_type': nearest_food_type,
                                'ecosystem_role': agent_role_str
                            })
                except Exception as e:
                    print(f"⚠️  Error creating robot details: {e}")
                    robot_details = []

                # 5. Get full agent data for robot details panel with enhanced visualization data
                agents_data = []
                try:
                    for agent in current_agents:
                        if not agent.body:  # Skip agents without bodies
                            continue
                            
                        agent_id = agent.id
                        
                        # Get ecosystem and health data
                        agent_status = self.agent_statuses.get(agent_id, {})
                        agent_health = self.agent_health.get(agent_id, {'health': 1.0, 'energy': 1.0})
                        
                        # Calculate speed for visualization
                        velocity = agent.body.linearVelocity
                        speed = (velocity.x ** 2 + velocity.y ** 2) ** 0.5
                        
                        agent_data = {
                            'id': agent.id,
                            'body': {
                                'x': convert_numpy_types(agent.body.position.x),
                                'y': convert_numpy_types(agent.body.position.y),
                                'velocity': {
                                    'x': convert_numpy_types(agent.body.linearVelocity.x),
                                    'y': convert_numpy_types(agent.body.linearVelocity.y)
                                }
                            },
                            'upper_arm': {
                                'x': convert_numpy_types(agent.upper_arm.position.x) if agent.upper_arm else 0,
                                'y': convert_numpy_types(agent.upper_arm.position.y) if agent.upper_arm else 0
                            },
                            'lower_arm': {
                                'x': convert_numpy_types(agent.lower_arm.position.x) if agent.lower_arm else 0,
                                'y': convert_numpy_types(agent.lower_arm.position.y) if agent.lower_arm else 0
                            },
                            'total_reward': convert_numpy_types(agent.total_reward),
                            'steps': convert_numpy_types(agent.steps),
                            'current_action': convert_numpy_types(agent.current_action_tuple),
                            'state': convert_numpy_types(agent.current_state),
                            'q_table': convert_numpy_types(agent.q_table.q_values if hasattr(agent.q_table, 'q_values') else {}),
                            'action_history': convert_numpy_types(agent.action_history),
                            'best_reward': convert_numpy_types(getattr(agent, 'best_reward_received', 0.0)),
                            'worst_reward': convert_numpy_types(getattr(agent, 'worst_reward_received', 0.0)),
                            'awake': agent.body.awake if agent.body else False,
                            'learning_approach': self._get_agent_learning_approach_name(agent.id),
                            # Enhanced visualization data
                            'ecosystem': {
                                'role': agent_status.get('role', 'omnivore'),
                                'status': agent_status.get('status', 'idle'),
                                'health': convert_numpy_types(agent_health['health']),
                                'energy': convert_numpy_types(agent_health['energy']),
                                'speed': convert_numpy_types(speed),
                                'speed_factor': convert_numpy_types(agent_status.get('speed_factor', 1.0)),
                                'alliances': agent_status.get('alliances', []),
                                'territories': agent_status.get('territories', [])
                            }
                        }
                        agents_data.append(agent_data)
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
                                'role': d['role'],
                                'age': current_time - d['timestamp']
                            }
                            for d in self.death_events if current_time - d['timestamp'] < 3.0  # Simple icons for 3 seconds only
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
                        'obstacles': [
                            {
                                'type': obs.get('type', 'unknown'),
                                'position': obs.get('position', [0, 0]),
                                'size': obs.get('size', 2.0),  # Default size for simple obstacles
                                'danger_level': obs.get('danger_level', 0.3)  # Default danger level
                            }
                            for obs in self.environmental_system.obstacles if obs and isinstance(obs, dict)
                        ]
                    }
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
        
        best_distance = -float('inf')
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
            
            # PRESERVE LEARNING APPROACH DIVERSITY - Store current distribution AND set inheritance
            old_approach_distribution = {}
            if hasattr(self, 'learning_manager') and self.learning_manager:
                try:
                    stats = self.learning_manager.get_approach_statistics()
                    old_approach_distribution = stats['approach_distribution'].copy()
                    print(f"🎯 Preserving learning approach diversity:")
                    print(f"   🔤 Basic: {old_approach_distribution.get('basic_q_learning', 0)}")
                    print(f"   ⚡ Enhanced: {old_approach_distribution.get('enhanced_q_learning', 0)}")
                    print(f"   🍃 Survival: {old_approach_distribution.get('survival_q_learning', 0)}")
                    print(f"   🧠 Deep: {old_approach_distribution.get('deep_q_learning', 0)}")
                    
                    # CRITICAL: Set inherited learning approach on all agents before evolution
                    for agent in self.agents:
                        if not getattr(agent, '_destroyed', False) and agent.body:
                            current_approach = self.learning_manager.get_agent_approach(agent.id)
                            agent._inherited_learning_approach = current_approach
                            print(f"🧬 Agent {agent.id[:8]}: Tagged for {current_approach.value} inheritance")
                    
                except Exception as e:
                    print(f"⚠️ Error capturing learning approaches: {e}")
                    old_approach_distribution = {}
            
            # PRESERVE ECOSYSTEM ROLE DIVERSITY - Store current role distribution
            old_role_distribution = {}
            if hasattr(self, 'ecosystem_dynamics') and self.ecosystem_dynamics:
                try:
                    old_role_distribution = {}
                    for role in self.ecosystem_dynamics.agent_roles.values():
                        old_role_distribution[role.value] = old_role_distribution.get(role.value, 0) + 1
                    
                    print(f"🦎 Preserving ecosystem role diversity:")
                    print(f"   🌿 Herbivore: {old_role_distribution.get('herbivore', 0)}")
                    print(f"   🥩 Carnivore: {old_role_distribution.get('carnivore', 0)}")
                    print(f"   🍽️ Omnivore: {old_role_distribution.get('omnivore', 0)}")
                    print(f"   🦴 Scavenger: {old_role_distribution.get('scavenger', 0)}")
                    print(f"   🤝 Symbiont: {old_role_distribution.get('symbiont', 0)}")
                    
                    # Tag agents with their current ecosystem role for inheritance
                    for agent in self.agents:
                        if not getattr(agent, '_destroyed', False) and agent.body:
                            current_role = self.ecosystem_dynamics.agent_roles.get(agent.id)
                            if current_role:
                                agent._inherited_ecosystem_role = current_role
                                print(f"🦎 Agent {agent.id[:8]}: Tagged for {current_role.value} role inheritance")
                    
                except Exception as e:
                    print(f"⚠️ Error capturing ecosystem roles: {e}")
                    old_role_distribution = {}
            
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
            
            # RESTORE LEARNING APPROACH DIVERSITY for new agents
            if hasattr(self, 'learning_manager') and self.learning_manager and old_approach_distribution:
                try:
                    import random
                    
                    # Create a weighted list of approaches based on old distribution
                    approaches_pool = []
                    for approach_str, count in old_approach_distribution.items():
                        if count > 0:
                            try:
                                from src.agents.learning_manager import LearningApproach
                                approach = LearningApproach(approach_str)
                                approaches_pool.extend([approach] * count)
                            except ValueError:
                                continue
                    
                    # If we have approaches to assign, randomly distribute them
                    if approaches_pool and self.agents:
                        # Shuffle the pool to ensure random assignment
                        random.shuffle(approaches_pool)
                        
                        # CRITICAL FIX: Use inherited learning approaches first, then assign randomly
                        # This ensures Q-table data matches the learning approach
                        assigned_count = 0
                        inherited_count = 0
                        switched_count = 0
                        
                        for i, agent in enumerate(self.agents):
                            if not getattr(agent, '_destroyed', False) and agent.body:
                                # First priority: Use inherited learning approach if available
                                inherited_approach = getattr(agent, '_inherited_learning_approach', None)
                                current_approach = self.learning_manager.get_agent_approach(agent.id)
                                
                                if inherited_approach and inherited_approach != current_approach:
                                    # Agent has inherited approach - use it to match Q-table data
                                    print(f"🧬 Agent {agent.id[:8]}: Inheriting {inherited_approach.value} (Q-table preserved)")
                                    success = self.learning_manager.set_agent_approach(agent, inherited_approach)
                                    if success:
                                        inherited_count += 1
                                        assigned_count += 1
                                elif inherited_approach and inherited_approach == current_approach:
                                    # Agent already has correct inherited approach
                                    inherited_count += 1
                                    print(f"💾 Agent {agent.id[:8]}: Already has inherited {inherited_approach.value}")
                                else:
                                    # No inheritance - fall back to random assignment for diversity
                                    target_approach = approaches_pool[i % len(approaches_pool)]
                                    if current_approach != target_approach:
                                        print(f"🎲 Agent {agent.id[:8]}: Random assignment {current_approach.value} → {target_approach.value}")
                                        success = self.learning_manager.set_agent_approach(agent, target_approach)
                                        if success:
                                            switched_count += 1
                                            assigned_count += 1
                        
                        print(f"🎲 Learning approach restoration complete:")
                        print(f"   🧬 {inherited_count} agents used inherited approaches (Q-table preserved)")
                        print(f"   🎲 {switched_count} agents randomly assigned for diversity")
                        print(f"   📊 {assigned_count} total assignments made")
                        
                        # Verify the new distribution
                        new_stats = self.learning_manager.get_approach_statistics()
                        new_distribution = new_stats['approach_distribution']
                        print(f"🎯 New distribution:")
                        print(f"   🔤 Basic: {new_distribution.get('basic_q_learning', 0)}")
                        print(f"   ⚡ Enhanced: {new_distribution.get('enhanced_q_learning', 0)}")
                        print(f"   🍃 Survival: {new_distribution.get('survival_q_learning', 0)}")
                        print(f"   🧠 Deep: {new_distribution.get('deep_q_learning', 0)}")
                        
                except Exception as e:
                    print(f"⚠️ Error restoring learning approaches: {e}")
                    import traceback
                    traceback.print_exc()
            
            # RESTORE ECOSYSTEM ROLE DIVERSITY for new agents
            if hasattr(self, 'ecosystem_dynamics') and self.ecosystem_dynamics and old_role_distribution:
                try:
                    import random
                    from src.ecosystem_dynamics import EcosystemRole
                    
                    # Clear old role assignments
                    self.ecosystem_dynamics.agent_roles.clear()
                    
                    # Create a weighted list of roles based on old distribution
                    roles_pool = []
                    for role_str, count in old_role_distribution.items():
                        if count > 0:
                            try:
                                role = EcosystemRole(role_str)
                                roles_pool.extend([role] * count)
                            except ValueError:
                                continue
                    
                    # Restore roles with inheritance priority
                    if roles_pool and self.agents:
                        random.shuffle(roles_pool)
                        
                        inherited_roles = 0
                        assigned_roles = 0
                        
                        for i, agent in enumerate(self.agents):
                            if not getattr(agent, '_destroyed', False) and agent.body:
                                # First priority: Use inherited role if available
                                inherited_role = getattr(agent, '_inherited_ecosystem_role', None)
                                
                                if inherited_role:
                                    # Agent has inherited role - use it
                                    self.ecosystem_dynamics.agent_roles[agent.id] = inherited_role
                                    inherited_roles += 1
                                    print(f"🦎 Agent {agent.id[:8]}: Inherited {inherited_role.value} role")
                                else:
                                    # No inheritance - use distribution-based assignment
                                    target_role = roles_pool[i % len(roles_pool)]
                                    self.ecosystem_dynamics.agent_roles[agent.id] = target_role
                                    assigned_roles += 1
                                    print(f"🎲 Agent {agent.id[:8]}: Assigned {target_role.value} role for diversity")
                        
                        print(f"🦎 Ecosystem role restoration complete:")
                        print(f"   🧬 {inherited_roles} agents used inherited roles")
                        print(f"   🎲 {assigned_roles} agents assigned for diversity")
                        
                        # Verify the new role distribution
                        new_role_distribution = {}
                        for role in self.ecosystem_dynamics.agent_roles.values():
                            new_role_distribution[role.value] = new_role_distribution.get(role.value, 0) + 1
                        
                        print(f"🦎 New role distribution:")
                        print(f"   🌿 Herbivore: {new_role_distribution.get('herbivore', 0)}")
                        print(f"   🥩 Carnivore: {new_role_distribution.get('carnivore', 0)}")
                        print(f"   🍽️ Omnivore: {new_role_distribution.get('omnivore', 0)}")
                        print(f"   🦴 Scavenger: {new_role_distribution.get('scavenger', 0)}")
                        print(f"   🤝 Symbiont: {new_role_distribution.get('symbiont', 0)}")
                        
                except Exception as e:
                    print(f"⚠️ Error restoring ecosystem roles: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Fallback: let normal ecosystem initialization handle role assignment
                    print("🔄 Falling back to normal ecosystem role assignment")
                    for agent in self.agents:
                        if not getattr(agent, '_destroyed', False) and agent.body:
                            if agent.id not in self.ecosystem_dynamics.agent_roles:
                                # Use normal assignment as fallback
                                self._initialize_single_agent_ecosystem(agent)
            
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
        # Find a safe spawn position with collision detection
        position = self._find_safe_spawn_position()
        if position is None:
            print("⚠️ Could not find safe spawn position for new agent")
            return
        
        # Create random physical parameters for diversity
        random_params = PhysicalParameters.random_parameters()
        
        new_agent = EvolutionaryCrawlingAgent(
            world=self.world,
            agent_id=None,  # Generate new UUID automatically
            position=position,
            category_bits=self.AGENT_CATEGORY,
            mask_bits=self.GROUND_CATEGORY,
            physical_params=random_params
        )
        self.agents.append(new_agent)
        
        # Initialize ecosystem data for the new agent
        self._initialize_single_agent_ecosystem(new_agent)
        
        print(f"🐣 Spawned new agent {new_agent.id} at safe position ({position[0]:.1f}, {position[1]:.1f}). Total agents: {len(self.agents)}")

    def clone_best_agent(self):
        """Clones the best performing agent using evolutionary methods."""
        best_agent = self.evolution_engine.get_best_agent()
        if not best_agent:
            print("No agents to clone.")
            return

        # Find a safe spawn position with collision detection
        position = self._find_safe_spawn_position()
        if position is None:
            print("⚠️ Could not find safe spawn position for cloned agent")
            return

        # Use the evolutionary clone method with slight mutation
        cloned_agent = best_agent.clone_with_mutation(mutation_rate=0.05)
        
        # Update position for the clone (ID is already generated)
        cloned_agent.initial_position = position
        cloned_agent.reset_with_new_position(position)
        
        self.agents.append(cloned_agent)
        
        # Initialize ecosystem data for the cloned agent
        self._initialize_single_agent_ecosystem(cloned_agent)
        
        print(f"👯 Cloned best agent {best_agent.id} to new agent {cloned_agent.id} (fitness: {best_agent.get_evolutionary_fitness():.3f}) at safe position ({position[0]:.1f}, {position[1]:.1f}). Total agents: {len(self.agents)}")

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
                    'action_history': []  # Track last actions taken
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
                self.camera_target = (0, 0)
        else:
            # If no agent is focused or focused agent is invalid, smoothly return to the origin
            if self.focused_agent and (getattr(self.focused_agent, '_destroyed', False) or not self.focused_agent.body):
                self.focused_agent = None  # Clear invalid focus
            self.camera_target = (0, 0)
            
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
    
    def clear_zoom_override(self):
        """Clear the zoom override flag after it's been sent."""
        if hasattr(self, '_zoom_override'):
            delattr(self, '_zoom_override')
    
    def _get_agent_learning_approach_name(self, agent_id: str) -> str:
        """Get the display name of the learning approach for an agent."""
        try:
            if hasattr(self, 'learning_manager') and self.learning_manager:
                approach = self.learning_manager.get_agent_approach(agent_id)
                approach_names = {
                    LearningApproach.BASIC_Q_LEARNING: '🔤 Basic Q-Learning',
                    LearningApproach.ENHANCED_Q_LEARNING: '⚡ Enhanced Q-Learning',
                    LearningApproach.SURVIVAL_Q_LEARNING: '🍃 Survival Q-Learning',
                    LearningApproach.DEEP_Q_LEARNING: '🧠 Deep Q-Learning'
                }
                return approach_names.get(approach, '⚡ Enhanced Q-Learning')
            else:
                return '⚡ Enhanced Q-Learning'  # Default fallback
        except Exception as e:
            print(f"⚠️ Error getting learning approach for agent {agent_id}: {e}")
            return '⚡ Enhanced Q-Learning'

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
    # Clear focus and reset zoom preferences
    env.focus_on_agent(None)
    env.reset_user_zoom()
    
    return jsonify({'status': 'success', 'message': 'View reset'})

@app.route('/clear_zoom_override', methods=['POST'])
def clear_zoom_override():
    env.clear_zoom_override()
    return jsonify({'status': 'success'})

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

@app.route('/switch_learning_approach', methods=['POST'])
def switch_learning_approach():
    """Switch a specific agent's learning approach."""
    try:
        data = request.get_json()
        if not data or 'agent_id' not in data or 'approach' not in data:
            return jsonify({'status': 'error', 'message': 'Missing agent_id or approach'}), 400
        
        agent_id = data['agent_id']
        approach_str = data['approach']
        
        # Convert string to enum
        try:
            approach = LearningApproach(approach_str)
        except ValueError:
            return jsonify({'status': 'error', 'message': f'Invalid learning approach: {approach_str}'}), 400
        
        # Find the agent
        agent = next((a for a in env.agents if a.id == agent_id and not getattr(a, '_destroyed', False)), None)
        if not agent:
            return jsonify({'status': 'error', 'message': f'Agent {agent_id} not found'}), 404
        
        # Switch approach using learning manager
        if hasattr(env, 'learning_manager') and env.learning_manager:
            success = env.learning_manager.set_agent_approach(agent, approach)
            if success:
                return jsonify({
                    'status': 'success', 
                    'agent_id': agent_id,
                    'new_approach': approach_str,
                    'message': f'Successfully switched agent {agent_id} to {approach_str}'
                })
            else:
                return jsonify({'status': 'error', 'message': f'Failed to switch agent {agent_id} to {approach_str}'}), 500
        else:
            return jsonify({'status': 'error', 'message': 'Learning manager not available'}), 500
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/bulk_switch_learning', methods=['POST'])
def bulk_switch_learning():
    """Switch all agents to the specified learning approach."""
    try:
        data = request.get_json()
        if not data or 'approach' not in data:
            return jsonify({'status': 'error', 'message': 'Missing approach'}), 400
        
        approach_str = data['approach']
        
        # Convert string to enum
        try:
            approach = LearningApproach(approach_str)
        except ValueError:
            return jsonify({'status': 'error', 'message': f'Invalid learning approach: {approach_str}'}), 400
        
        # Get all valid agents
        valid_agents = [a for a in env.agents if not getattr(a, '_destroyed', False) and a.body]
        agent_ids = [a.id for a in valid_agents]
        
        if hasattr(env, 'learning_manager') and env.learning_manager:
            results = env.learning_manager.bulk_switch_approach(agent_ids, valid_agents, approach)
            success_count = sum(1 for success in results.values() if success)
            
            return jsonify({
                'status': 'success',
                'approach': approach_str,
                'success_count': success_count,
                'total_agents': len(valid_agents),
                'results': results
            })
        else:
            return jsonify({'status': 'error', 'message': 'Learning manager not available'}), 500
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/randomize_learning_approaches', methods=['POST'])
def randomize_learning_approaches():
    """Randomly assign learning approaches to all agents."""
    try:
        import random
        
        # Get all valid agents
        valid_agents = [a for a in env.agents if not getattr(a, '_destroyed', False) and a.body]
        
        if hasattr(env, 'learning_manager') and env.learning_manager:
            approaches = [
                LearningApproach.BASIC_Q_LEARNING,
                LearningApproach.ENHANCED_Q_LEARNING,
                LearningApproach.SURVIVAL_Q_LEARNING,
                LearningApproach.DEEP_Q_LEARNING  # GPU-accelerated deep learning is now fully working
            ]
            
            updated_count = 0
            for agent in valid_agents:
                random_approach = random.choice(approaches)
                success = env.learning_manager.set_agent_approach(agent, random_approach)
                if success:
                    updated_count += 1
            
            return jsonify({
                'status': 'success',
                'agents_updated': updated_count,
                'total_agents': len(valid_agents)
            })
        else:
            return jsonify({'status': 'error', 'message': 'Learning manager not available'}), 500
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/learning_statistics', methods=['GET'])
def get_learning_statistics():
    """Get learning approach statistics."""
    try:
        if hasattr(env, 'learning_manager') and env.learning_manager:
            stats = env.learning_manager.get_approach_statistics()
            return jsonify({'status': 'success', 'statistics': stats})
        else:
            return jsonify({'status': 'error', 'message': 'Learning manager not available'}), 500
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/agent_learning_info/<agent_id>', methods=['GET'])
def get_agent_learning_info(agent_id):
    """Get detailed learning information for a specific agent."""
    try:
        if hasattr(env, 'learning_manager') and env.learning_manager:
            info = env.learning_manager.get_agent_learning_info(agent_id)
            return jsonify({'status': 'success', 'learning_info': info})
        else:
            return jsonify({'status': 'error', 'message': 'Learning manager not available'}), 500
            
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