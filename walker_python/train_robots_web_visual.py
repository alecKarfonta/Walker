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
                <div class="panel-title">🏆 Leaderboard</div>
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
        let agentTrails = new Map(); // Store movement trails for agents

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
                            <span class="robot-stat-value">${robot.distance.toFixed(2)}m</span>
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
            
            // Draw predation events and effects
            drawPredationEvents();

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
                    
                    const radius = 0.5 + (ratio * 1.5); // Size based on remaining amount
                    const alpha = 0.4 + (ratio * 0.4); // Transparency based on amount
                    
                    ctx.fillStyle = foodColor + Math.floor(alpha * 255).toString(16).padStart(2, '0');
                    ctx.strokeStyle = foodColor;
                    ctx.lineWidth = 0.1;
                    
                    ctx.beginPath();
                    ctx.arc(x, y, radius, 0, 2 * Math.PI);
                    ctx.fill();
                    ctx.stroke();
                    
                    // Add depletion animation if amount is low
                    if (ratio < 0.3) {
                        const time = Date.now() / 1000;
                        const pulse = 0.5 + 0.5 * Math.sin(time * 4);
                        ctx.strokeStyle = `rgba(255, 255, 0, ${pulse})`;
                        ctx.lineWidth = 0.2;
                        ctx.beginPath();
                        ctx.arc(x, y, radius + 0.3, 0, 2 * Math.PI);
                        ctx.stroke();
                    }
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
                
                // Draw movement trail for fast-moving agents
                if (speed > 1.0) {
                    drawMovementTrail(robot.id, agentPos, speed);
                }
                
                // Draw alliance connections
                if (ecosystem.alliances && ecosystem.alliances.length > 0) {
                    drawAllianceConnections(robot.id, agentPos, ecosystem.alliances, data.agents);
                }
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
        }
        
        function drawMovementTrail(agentId, position, speed) {
            // Update agent trail
            if (!agentTrails.has(agentId)) {
                agentTrails.set(agentId, []);
            }
            
            const trail = agentTrails.get(agentId);
            trail.push({ x: position[0], y: position[1], time: Date.now() });
            
            // Keep only recent trail points (last 3 seconds)
            const now = Date.now();
            const filteredTrail = trail.filter(point => now - point.time < 3000);
            agentTrails.set(agentId, filteredTrail);
            
            // Draw trail
            if (filteredTrail.length > 1) {
                ctx.strokeStyle = `rgba(255, 255, 255, ${Math.min(0.8, speed / 3.0)})`;
                ctx.lineWidth = 0.1;
                ctx.beginPath();
                ctx.moveTo(filteredTrail[0].x, filteredTrail[0].y);
                
                for (let i = 1; i < filteredTrail.length; i++) {
                    const alpha = i / filteredTrail.length; // Fade over time
                    ctx.lineTo(filteredTrail[i].x, filteredTrail[i].y);
                }
                ctx.stroke();
            }
        }
        
        function drawAllianceConnections(agentId, position, alliances, allAgents) {
            alliances.forEach(allyId => {
                const ally = allAgents.find(a => a.id === allyId);
                if (ally && ally.body) {
                    const allyPos = [ally.body.x, ally.body.y];
                    const distance = Math.sqrt((position[0] - allyPos[0])**2 + (position[1] - allyPos[1])**2);
                    
                    // Only draw connection if agents are close
                    if (distance < 10.0) {
                        ctx.strokeStyle = 'rgba(100, 200, 255, 0.3)';
                        ctx.lineWidth = 0.1;
                        ctx.setLineDash([0.3, 0.3]);
                        
                        ctx.beginPath();
                        ctx.moveTo(position[0], position[1]);
                        ctx.lineTo(allyPos[0], allyPos[1]);
                        ctx.stroke();
                        
                        ctx.setLineDash([]); // Reset line dash
                    }
                }
            });
        }
        
        function drawPredationEvents() {
            if (!predationEvents || predationEvents.length === 0) return;
            
            const now = Date.now() / 1000;
            
            predationEvents.forEach(event => {
                if (event.age > 5.0) return; // Don't draw old events
                
                const [x, y] = event.position;
                const alpha = Math.max(0, 1.0 - (event.age / 5.0)); // Fade over 5 seconds
                
                if (event.success) {
                    // Successful predation - red burst
                    ctx.fillStyle = `rgba(255, 0, 0, ${alpha * 0.6})`;
                    ctx.strokeStyle = `rgba(255, 100, 100, ${alpha})`;
                    ctx.lineWidth = 0.2;
                    
                    const radius = 1.0 + (event.age * 0.5); // Expanding circle
                    ctx.beginPath();
                    ctx.arc(x, y, radius, 0, 2 * Math.PI);
                    ctx.fill();
                    ctx.stroke();
                    
                    // Add particles effect
                    const particleCount = 8;
                    for (let i = 0; i < particleCount; i++) {
                        const angle = (i / particleCount) * 2 * Math.PI;
                        const particleRadius = radius + (event.age * 0.8);
                        const px = x + Math.cos(angle) * particleRadius;
                        const py = y + Math.sin(angle) * particleRadius;
                        
                        ctx.fillStyle = `rgba(255, 50, 50, ${alpha * 0.8})`;
                        ctx.beginPath();
                        ctx.arc(px, py, 0.1, 0, 2 * Math.PI);
                        ctx.fill();
                    }
                }
            });
        }

        function fetchData() {
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
        self.predation_events = []  # Track recent predation events for visualization
        self.last_ecosystem_update = time.time()
        self.ecosystem_update_interval = 30.0  # Update ecosystem every 30 seconds
        
        # Initialize ecosystem roles for existing agents
        self._initialize_ecosystem_roles()

        print(f"🧬 Enhanced Training Environment initialized:")
        print(f"   Population: {len(self.agents)} diverse agents")
        print(f"   Evolution: {self.evolution_config.population_size} agents, {self.evolution_config.elite_size} elite")
        print(f"   Diversity target: {self.evolution_config.target_diversity}")
        print(f"   Auto-evolution every {self.evolution_interval}s")
        print(f"🌿 Ecosystem dynamics and visualization systems active")

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
        
        print(f"🦎 Initialized ecosystem roles for {len(self.agents)} agents")

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
                                         key=lambda r: r.get('total_distance', 0), reverse=True)
                    leaderboard_data = [
                        {'id': r['id'], 'name': f"Robot {r['id']}", 'distance': r.get('total_distance', 0)}
                        for r in sorted_robots[:10]
                    ]
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
                        'predation_events': recent_predation_events
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
            mask_bits=self.GROUND_CATEGORY,
            physical_params=random_params
        )
        self.agents.append(new_agent)
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