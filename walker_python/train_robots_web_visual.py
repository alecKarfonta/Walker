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
from flask import Flask, render_template_string, jsonify, request
import numpy as np
import Box2D as b2
from src.agents.crawling_crate_agent import CrawlingCrateAgent
from src.population.population_controller import PopulationController
from src.population.evolution import EvolutionEngine
from flask_socketio import SocketIO


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
        
        body { 
            margin: 0; 
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: #e8e8e8; 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            overflow: hidden;
        }
        
        canvas { 
            display: block; 
            background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
            box-shadow: inset 0 0 50px rgba(0,0,0,0.3);
        }
        
        #controls { 
            position: absolute; 
            top: 20px; 
            left: 400px; /* Initial position, will be updated by JS */
            z-index: 100;
            transition: left 0.3s ease;
        }
        
        button { 
            background: linear-gradient(145deg, #2c3e50, #34495e);
            color: #ecf0f1; 
            border: 2px solid #3498db; 
            padding: 12px 20px; 
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        
        button:hover {
            background: linear-gradient(145deg, #3498db, #2980b9);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(52, 152, 219, 0.3);
        }
        
        .stats-container {
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(26, 26, 46, 0.95);
            backdrop-filter: blur(10px);
            padding: 20px;
            border-radius: 15px;
            border: 2px solid #3498db;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            max-width: 350px;
            z-index: 100;
            text-align: center;
        }
        
        .stats-title {
            color: #3498db;
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 15px;
            text-align: center;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .stat-row { 
            margin: 8px 0; 
            font-size: 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid rgba(52, 152, 219, 0.2);
        }
        
        .stat-row:last-child {
            border-bottom: none;
        }
        
        .stat-label { 
            color: #bdc3c7; 
            font-weight: 500;
        }
        
        .stat-value { 
            color: #ecf0f1; 
            font-weight: 700;
            background: linear-gradient(45deg, #3498db, #2980b9);
            padding: 4px 8px;
            border-radius: 6px;
            min-width: 60px;
            text-align: center;
        }
        
        .q-learning-section {
            margin-top: 15px;
            padding-top: 15px;
            border-top: 2px solid #e74c3c;
        }
        
        .q-learning-title {
            color: #e74c3c;
            font-size: 16px;
            font-weight: 700;
            margin-bottom: 10px;
            text-align: center;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .q-stat-row {
            margin: 6px 0;
            font-size: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 6px 0;
            border-bottom: 1px solid rgba(231, 76, 60, 0.2);
        }
        
        .q-stat-row:last-child {
            border-bottom: none;
        }
        
        .q-stat-label {
            color: #bdc3c7;
            font-weight: 500;
        }
        
        .q-stat-value {
            color: #ecf0f1;
            font-weight: 600;
            background: rgba(231, 76, 60, 0.3);
            padding: 3px 6px;
            border-radius: 4px;
            min-width: 50px;
            text-align: center;
        }
        
        #robot-stats { 
            position: absolute; 
            top: 0;
            left: 0;
            height: 100vh;
            width: 380px; /* Default width */
            min-width: 250px; /* Min resize width */
            max-width: 800px; /* Max resize width */
            background: rgba(15, 20, 35, 0.9);
            backdrop-filter: blur(12px);
            padding: 20px;
            border-right: 2px solid #e74c3c;
            box-shadow: 5px 0 25px rgba(0,0,0,0.3);
            overflow-y: auto;
            z-index: 100;
            transition: width 0.3s ease, transform 0.3s ease, padding 0.3s ease;
        }

        #robot-stats.collapsed {
            width: 0 !important; /* Use important to override inline style from JS */
            min-width: 0 !important;
            transform: translateX(-100%);
            padding-left: 0;
            padding-right: 0;
        }
        
        #robot-stats.collapsed > * {
            display: none;
        }
        
        #resizer {
            position: absolute;
            top: 0;
            right: 0;
            width: 8px;
            height: 100%;
            cursor: col-resize;
            z-index: 102;
        }

        #collapse-toggle {
            position: absolute;
            top: 50%;
            left: 380px; /* Initial position, will be updated by JS */
            transform: translateY(-50%);
            width: 25px;
            height: 50px;
            background: #e74c3c;
            border: none;
            color: white;
            font-size: 20px;
            cursor: pointer;
            border-radius: 0 5px 5px 0;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 99;
            transition: left 0.3s ease;
        }
        
        #collapse-toggle .arrow {
            display: inline-block;
            transition: transform 0.3s ease;
        }

        #robot-stats.collapsed + #collapse-toggle .arrow {
            transform: rotate(180deg);
        }
        
        .robot-stats-title {
            color: #e74c3c;
            font-size: 16px;
            font-weight: 700;
            margin-bottom: 15px;
            text-align: center;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .robot-stat { 
            margin: 12px 0; 
            padding: 15px; 
            border-left: 4px solid #e74c3c;
            background: rgba(231, 76, 60, 0.1);
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        
        .robot-stat:hover {
            background: rgba(231, 76, 60, 0.25);
            transform: none; /* Removed jarring hover movement */
        }
        
        .robot-name {
            color: #e74c3c;
            font-weight: 700;
            font-size: 14px;
            margin-bottom: 8px;
        }
        
        /* Leaderboard specific styling */
        .robot-stat:nth-child(1) .robot-name {
            color: #f39c12; /* Gold for 1st place */
            font-size: 16px;
        }
        
        .robot-stat:nth-child(2) .robot-name {
            color: #95a5a6; /* Silver for 2nd place */
            font-size: 15px;
        }
        
        .robot-stat:nth-child(3) .robot-name {
            color: #d35400; /* Bronze for 3rd place */
            font-size: 15px;
        }
        
        .robot-stat-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 4px 0;
            font-size: 12px;
        }
        
        .robot-stat-label {
            color: #bdc3c7;
            font-weight: 500;
        }
        
        .robot-stat-value {
            color: #ecf0f1;
            font-weight: 600;
            background: rgba(231, 76, 60, 0.3);
            padding: 2px 6px;
            border-radius: 4px;
            min-width: 50px;
            text-align: center;
        }
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(52, 152, 219, 0.1);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(45deg, #3498db, #2980b9);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(45deg, #2980b9, #1f5f8b);
        }
        
        /* Animation for stats updates */
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        .stat-value.updated {
            animation: pulse 0.3s ease-in-out;
        }
    </style>
</head>
<body>
    <canvas id="world"></canvas>
    
    <div id="robot-stats">
        <div class="robot-stats-title">🏆 Robot Leaderboard</div>
        <div id="robotDetails"></div>
        <div id="resizer"></div>
    </div>
    
    <button id="collapse-toggle"><span class="arrow">‹</span></button>

    <div id="controls">
        <button id="resetView">Reset View</button>
    </div>
    
    <div class="stats-container">
        <div class="stats-title">Population Statistics</div>
        <div class="stat-row">
            <span class="stat-label">Generation:</span>
            <span class="stat-value" id="generation">1</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Total Steps:</span>
            <span class="stat-value" id="totalSteps">0</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Best Distance:</span>
            <span class="stat-value" id="bestDistance">0.00</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Average Distance:</span>
            <span class="stat-value" id="avgDistance">0.00</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Total Distance:</span>
            <span class="stat-value" id="totalDistance">0.00</span>
        </div>
        
        <div class="q-learning-section">
            <div class="q-learning-title">Q-Learning Stats</div>
            <div class="q-stat-row">
                <span class="q-stat-label">Epsilon:</span>
                <span class="q-stat-value" id="epsilon">0.300</span>
            </div>
            <div class="q-stat-row">
                <span class="q-stat-label">Learning Rate:</span>
                <span class="q-stat-value" id="learningRate">0.150</span>
            </div>
            <div class="q-stat-row">
                <span class="q-stat-label">Q Updates:</span>
                <span class="q-stat-value" id="qUpdates">0</span>
            </div>
            <div class="q-stat-row">
                <span class="q-stat-label">Avg Q-Value:</span>
                <span class="q-stat-value" id="avgQValue">0.000</span>
            </div>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('world');
        const ctx = canvas.getContext('2d');
        let scale = 15; // pixels per meter
        let offsetX = 0;
        let offsetY = 0;
        let isDragging = false;
        let lastMouseX, lastMouseY;

        function resizeCanvas() {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
            if (offsetX === 0 && offsetY === 0) {
                offsetX = canvas.width / 2;
                offsetY = canvas.height * 0.8;
            }
        }
        window.addEventListener('resize', resizeCanvas);

        canvas.addEventListener('mousedown', (e) => {
            isDragging = true;
            lastMouseX = e.clientX;
            lastMouseY = e.clientY;
        });
        canvas.addEventListener('mousemove', (e) => {
            if (isDragging) {
                offsetX += e.clientX - lastMouseX;
                offsetY += e.clientY - lastMouseY;
                lastMouseX = e.clientX;
                lastMouseY = e.clientY;
            }
        });
        canvas.addEventListener('mouseup', () => isDragging = false);
        canvas.addEventListener('mouseleave', () => isDragging = false);
        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const zoomFactor = 1.1;
            const newScale = e.deltaY < 0 ? scale * zoomFactor : scale / zoomFactor;
            scale = Math.max(5, Math.min(100, newScale));
        });

        document.getElementById('resetView').addEventListener('click', () => {
            scale = 15;
            offsetX = canvas.width / 2;
            offsetY = canvas.height * 0.8;
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
            if (data.statistics) {
                const elements = {
                    'generation': data.statistics.generation || 1,
                    'totalSteps': data.statistics.total_steps || 0,
                    'bestDistance': (data.statistics.best_distance || 0).toFixed(2),
                    'avgDistance': (data.statistics.average_distance || 0).toFixed(2),
                    'totalDistance': (data.statistics.total_distance || 0).toFixed(2)
                };
                
                Object.entries(elements).forEach(([id, value]) => {
                    const element = document.getElementById(id);
                    if (element && element.textContent !== value.toString()) {
                        element.textContent = value;
                        element.classList.add('updated');
                        setTimeout(() => element.classList.remove('updated'), 300);
                    }
                });
                
                // Update Q-learning statistics
                if (data.statistics.q_learning_stats) {
                    const qStats = data.statistics.q_learning_stats;
                    const qElements = {
                        'epsilon': (qStats.avg_epsilon || 0).toFixed(3),
                        'learningRate': (qStats.avg_learning_rate || 0).toFixed(3),
                        'qUpdates': qStats.total_q_updates || 0,
                        'avgQValue': (qStats.avg_q_value || 0).toFixed(3)
                    };
                    
                    Object.entries(qElements).forEach(([id, value]) => {
                        const element = document.getElementById(id);
                        if (element && element.textContent !== value.toString()) {
                            element.textContent = value;
                            element.classList.add('updated');
                            setTimeout(() => element.classList.remove('updated'), 300);
                        }
                    });
                }
            }
            
            // Update robot details
            const robotDetails = document.getElementById('robotDetails');
            robotDetails.innerHTML = '';
            
            if (data.agents) {
                // Create leaderboard based on distance moved
                const leaderboard = data.agents
                    .map((agent, index) => ({
                        index: index,
                        agent: agent,
                        distance: agent.statistics?.total_distance || 0
                    }))
                    .sort((a, b) => b.distance - a.distance) // Sort by distance descending
                    .slice(0, 10); // Show top 10 performers
                
                // Add leaderboard title
                const leaderboardTitle = document.createElement('div');
                leaderboardTitle.className = 'robot-stats-title';
                leaderboardTitle.style.marginBottom = '10px';
                leaderboardTitle.innerHTML = `🏆 Top 10 Performers (${data.agents.length} total)`;
                robotDetails.appendChild(leaderboardTitle);
                
                leaderboard.forEach((entry, leaderboardIndex) => {
                    const stats = entry.agent.statistics || {};
                    const robotDiv = document.createElement('div');
                    robotDiv.className = 'robot-stat';
                    
                    // Add rank indicator
                    const rank = leaderboardIndex + 1;
                    const rankEmoji = rank === 1 ? '🥇' : rank === 2 ? '🥈' : rank === 3 ? '🥉' : `#${rank}`;
                    
                    const position = stats.current_position || [0, 0];
                    const velocity = stats.velocity || [0, 0];
                    const armAngles = stats.arm_angles || {shoulder: 0, elbow: 0};
                    
                    robotDiv.innerHTML = `
                        <div class="robot-name">${rankEmoji} Robot ${entry.index + 1} (Rank #${rank})</div>
                        <div class="robot-stat-row">
                            <span class="robot-stat-label">Distance:</span>
                            <span class="robot-stat-value" style="color: #27ae60; font-weight: bold;">${(stats.total_distance || 0).toFixed(2)}</span>
                        </div>
                        <div class="robot-stat-row">
                            <span class="robot-stat-label">Position:</span>
                            <span class="robot-stat-value">(${position[0].toFixed(2)}, ${position[1].toFixed(2)})</span>
                        </div>
                        <div class="robot-stat-row">
                            <span class="robot-stat-label">Velocity:</span>
                            <span class="robot-stat-value">(${velocity[0].toFixed(2)}, ${velocity[1].toFixed(2)})</span>
                        </div>
                        <div class="robot-stat-row">
                            <span class="robot-stat-label">Shoulder:</span>
                            <span class="robot-stat-value">${((armAngles.shoulder || 0) * 180 / Math.PI).toFixed(1)}°</span>
                        </div>
                        <div class="robot-stat-row">
                            <span class="robot-stat-label">Elbow:</span>
                            <span class="robot-stat-value">${((armAngles.elbow || 0) * 180 / Math.PI).toFixed(1)}°</span>
                        </div>
                        <div class="robot-stat-row">
                            <span class="robot-stat-label">Episode Reward:</span>
                            <span class="robot-stat-value" style="color: ${getRewardColor(stats.episode_reward || 0)}">${(stats.episode_reward || 0).toFixed(1)}</span>
                        </div>
                        <div class="robot-stat-row">
                            <span class="robot-stat-label">Q Updates:</span>
                            <span class="robot-stat-value">${stats.q_updates || 0}</span>
                        </div>
                        <div class="robot-stat-row">
                            <span class="robot-stat-label">Actions:</span>
                            <span class="robot-stat-value">${getActionHistoryString(stats.action_history || [])}</span>
                        </div>
                    `;
                    robotDetails.appendChild(robotDiv);
                });
                
                // Add summary stats
                if (data.agents.length > 10) {
                    const summaryDiv = document.createElement('div');
                    summaryDiv.className = 'robot-stat';
                    summaryDiv.style.marginTop = '15px';
                    summaryDiv.style.borderTop = '2px solid #3498db';
                    summaryDiv.style.paddingTop = '10px';
                    
                    const allDistances = data.agents.map(agent => agent.statistics?.total_distance || 0);
                    const avgDistance = allDistances.reduce((a, b) => a + b, 0) / allDistances.length;
                    const minDistance = Math.min(...allDistances);
                    const maxDistance = Math.max(...allDistances);
                    
                    summaryDiv.innerHTML = `
                        <div class="robot-name">📊 Population Summary</div>
                        <div class="robot-stat-row">
                            <span class="robot-stat-label">Best Distance:</span>
                            <span class="robot-stat-value" style="color: #27ae60;">${maxDistance.toFixed(2)}</span>
                        </div>
                        <div class="robot-stat-row">
                            <span class="robot-stat-label">Average Distance:</span>
                            <span class="robot-stat-value">${avgDistance.toFixed(2)}</span>
                        </div>
                        <div class="robot-stat-row">
                            <span class="robot-stat-label">Worst Distance:</span>
                            <span class="robot-stat-value" style="color: #e74c3c;">${minDistance.toFixed(2)}</span>
                        </div>
                        <div class="robot-stat-row">
                            <span class="robot-stat-label">Total Robots:</span>
                            <span class="robot-stat-value">${data.agents.length}</span>
                        </div>
                    `;
                    robotDetails.appendChild(summaryDiv);
                }
            }
        }

        function draw(data) {
            console.log("Draw function called with data:", data); // DEBUG
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.save();
            ctx.translate(offsetX, offsetY);
            ctx.scale(scale, -scale); // Flip Y-axis for physics coords

            // Draw ground from geometry
            if (data.ground_geometry && data.ground_geometry.length > 0) {
                const gradient = ctx.createLinearGradient(0, -1, 0, 1);
                gradient.addColorStop(0, '#5e738c');
                gradient.addColorStop(1, '#34495e');
                
                ctx.fillStyle = gradient;
                
                data.ground_geometry.forEach(geom => {
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

            if (data.agents) {
                data.agents.forEach(agent => {
                    agent.body_parts.forEach(part => {
                        // Enhanced colors with gradients
                        if (part.type === 'circle') {
                            // Wheels
                            const wheelGradient = ctx.createRadialGradient(
                                part.center[0], part.center[1], 0,
                                part.center[0], part.center[1], part.radius
                            );
                            wheelGradient.addColorStop(0, '#3498db');
                            wheelGradient.addColorStop(1, '#2980b9');
                            ctx.fillStyle = wheelGradient;
                        } else {
                            // Body parts
                            const bodyGradient = ctx.createLinearGradient(
                                part.vertices[0][0], part.vertices[0][1],
                                part.vertices[2][0], part.vertices[2][1]
                            );
                            bodyGradient.addColorStop(0, '#e74c3c');
                            bodyGradient.addColorStop(1, '#c0392b');
                            ctx.fillStyle = bodyGradient;
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
            console.log("Fetching data..."); // DEBUG
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    draw(data);
                    updateStats(data);
                    requestAnimationFrame(fetchData);
                })
                .catch(err => {
                    console.error('Error fetching data:', err);
                    setTimeout(fetchData, 1000); // Retry after 1s on error
                });
        }

        // --- Sidebar Interactivity ---
        const sidebar = document.getElementById('robot-stats');
        const resizer = document.getElementById('resizer');
        const toggleBtn = document.getElementById('collapse-toggle');
        const controls = document.getElementById('controls');

        // Toggle Logic
        toggleBtn.addEventListener('click', function() {
            sidebar.classList.toggle('collapsed');
            updateLayout();
        });

        // Resizing Logic
        let isResizing = false;
        resizer.addEventListener('mousedown', function(e) {
            isResizing = true;
            document.body.style.cursor = 'col-resize';
            window.addEventListener('mousemove', onMouseMove);
            window.addEventListener('mouseup', onMouseUp);
        });

        function onMouseMove(e) {
            if (!isResizing) return;
            const minWidth = parseInt(getComputedStyle(sidebar).minWidth);
            const maxWidth = parseInt(getComputedStyle(sidebar).maxWidth);
            let newWidth = e.clientX;

            if (newWidth < minWidth) newWidth = minWidth;
            if (newWidth > maxWidth) newWidth = maxWidth;
            
            sidebar.style.width = newWidth + 'px';
            updateLayout();
        }

        function onMouseUp() {
            isResizing = false;
            document.body.style.cursor = 'default';
            window.removeEventListener('mousemove', onMouseMove);
            window.removeEventListener('mouseup', onMouseUp);
        }

        // Central function to update layout based on sidebar state
        function updateLayout() {
            const sidebarWidth = sidebar.offsetWidth;
            if (sidebar.classList.contains('collapsed')) {
                toggleBtn.style.left = '0px';
                controls.style.left = '40px';
            } else {
                toggleBtn.style.left = sidebarWidth + 'px';
                controls.style.left = (sidebarWidth + 20) + 'px';
            }
        }

        updateLayout(); // Set initial positions
        resizeCanvas();
        fetchData();
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
        """Update population and Q-learning statistics."""
        # Update population statistics
        distances = [stats['total_distance'] for stats in self.robot_stats.values()]
        self.population_stats['total_distance'] = sum(distances)
        self.population_stats['best_distance'] = max(distances)
        self.population_stats['average_distance'] = sum(distances) / len(distances)
        self.population_stats['total_steps'] = self.step_count
        
        # Update Q-learning statistics
        q_agents = [agent for agent in self.agents if hasattr(agent, 'q_table') and agent.q_table is not None]
        if q_agents:
            self.population_stats['q_learning_stats']['avg_epsilon'] = float(sum(agent.epsilon for agent in q_agents) / len(q_agents))
            self.population_stats['q_learning_stats']['avg_learning_rate'] = float(sum(agent.learning_rate for agent in q_agents) / len(q_agents))
            self.population_stats['q_learning_stats']['total_q_updates'] = int(sum(stats['q_updates'] for stats in self.robot_stats.values()))
            
            # Calculate average Q-value
            all_q_values = []
            for agent in q_agents:
                if hasattr(agent.q_table, 'q_values'):
                    q_values = agent.q_table.q_values
                    if hasattr(q_values, 'flatten'):
                        # Handle numpy array (QTable)
                        all_q_values.extend(q_values.flatten())
                    elif isinstance(q_values, dict):
                        # Handle dictionary (SparseQTable)
                        for action_values in q_values.values():
                            all_q_values.extend(action_values)
            if all_q_values:
                self.population_stats['q_learning_stats']['avg_q_value'] = float(sum(all_q_values) / len(all_q_values))

    def training_loop(self):
        """The main training loop that steps the physics world and updates agents."""
        self.is_running = True
        step_count = 0
        start_time = time.time()
        last_step_time = start_time
        print("🚀 Training loop started!")
        
        # Initialize robot statistics
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
        
        while self.is_running:
            current_time = time.time()
            step_duration = current_time - last_step_time
            
            for i, agent in enumerate(self.agents):
                # Check if robot is tilted too much (on side or upside down)
                body_angle = abs(agent.body.angle)
                max_tilt_angle = np.pi / 3  # 60 degrees
                max_tilt_steps = 600  # 10 seconds at 60fps
                
                if body_angle > max_tilt_angle:
                    self.robot_stats[i]['steps_tilted'] += 1
                else:
                    self.robot_stats[i]['steps_tilted'] = 0
                
                # Reset robot if it's been tilted too long or episode ended
                should_reset = (
                    self.robot_stats[i]['steps_tilted'] > max_tilt_steps or
                    self.episode_step >= self.episode_length or
                    agent.body.position.y < -5  # Fallen too far
                )
                
                if should_reset:
                    print(f"🤖 Resetting robot {i} - tilt_steps: {self.robot_stats[i]['steps_tilted']}, episode_step: {self.episode_step}")
                    agent.reset()
                    self.robot_stats[i]['steps_tilted'] = 0
                    self.robot_stats[i]['total_distance'] = 0
                    self.robot_stats[i]['last_position'] = tuple(agent.body.position)
                    self.robot_stats[i]['episode_reward'] = 0
                    self.robot_stats[i]['q_updates'] = 0
                    self.robot_stats[i]['action_history'] = []  # Reset action history
                    continue

                # Agent handles its own logic for actions and learning
                agent.step(self.dt)
                
                # Update robot statistics from agent's state
                self.robot_stats[i]['current_position'] = tuple(agent.body.position)
                self.robot_stats[i]['velocity'] = tuple(agent.body.linearVelocity)
                self.robot_stats[i]['arm_angles']['shoulder'] = agent.upper_arm.angle
                self.robot_stats[i]['arm_angles']['elbow'] = agent.lower_arm.angle
                self.robot_stats[i]['steps_alive'] += 1
                self.robot_stats[i]['episode_reward'] = agent.total_reward
                self.robot_stats[i]['q_updates'] = agent.q_table.update_count if hasattr(agent.q_table, 'update_count') else 0
                self.robot_stats[i]['action_history'] = agent.action_history
                
                # Update total distance for fitness
                self.robot_stats[i]['total_distance'] = agent.body.position.x - agent.initial_position[0]
                self.robot_stats[i]['fitness'] = self.robot_stats[i]['total_distance']
            
            # Physics step
            self.world.Step(self.dt, 8, 3)

            step_count += 1
            self.step_count = step_count
            self.episode_step += 1
            
            # Update statistics only every 0.1 seconds (every 6 steps at 60fps)
            if step_count % self.steps_per_stats_update == 0:
                self._update_statistics()
            
            # Print step count every 60 steps (1 second at 60fps)
            if step_count % 60 == 0:
                print(f"⏱️  Step {step_count} - Loop running at {(step_count / (time.time() - start_time)):.1f} steps/sec")
            
            # Reset episode counter when all robots are reset
            if self.episode_step >= self.episode_length:
                self.episode_step = 0
                print(f"🔄 Episode completed at step {step_count}")
            
            # Debug: print robot positions every 60 steps (1 second)
            if step_count % 60 == 0:
                print(f"Step {step_count}: Robot 0 at {self.agents[0].body.position}")
                print(f"Population stats: Best={self.population_stats['best_distance']:.2f}, Avg={self.population_stats['average_distance']:.2f}")
                print(f"⏱️  Step duration: {step_duration*1000:.1f}ms (target: {self.dt*1000:.1f}ms)")
                q_agents = [agent for agent in self.agents if hasattr(agent, 'q_table') and agent.q_table is not None]
                if q_agents:
                    print(f"Q-Learning: ε={self.population_stats['q_learning_stats']['avg_epsilon']:.3f}, Q_updates={self.population_stats['q_learning_stats']['total_q_updates']}")
            
            time.sleep(self.dt)
            last_step_time = time.time()

    def get_status(self):
        """Returns the current state of the simulation for rendering."""
        if not self.is_running:
            return {'agents': [], 'statistics': {}, 'ground_geometry': []}

        agent_states = []
        for i, agent in enumerate(self.agents):
            body_parts = []
            
            # Chassis
            body_parts.append({
                'type': 'polygon',
                'vertices': [tuple(agent.body.GetWorldPoint(v)) for v in agent.body.fixtures[0].shape.vertices],
            })
            
            # Wheels
            for wheel in agent.wheels:
                body_parts.append({
                    'type': 'circle',
                    'center': tuple(wheel.position),
                    'radius': wheel.fixtures[0].shape.radius,
                    'angle': wheel.angle
                })

            # Arms
            body_parts.append({
                'type': 'polygon',
                'vertices': [tuple(agent.upper_arm.GetWorldPoint(v)) for v in agent.upper_arm.fixtures[0].shape.vertices],
            })
            body_parts.append({
                'type': 'polygon',
                'vertices': [tuple(agent.lower_arm.GetWorldPoint(v)) for v in agent.lower_arm.fixtures[0].shape.vertices],
            })

            # Add robot statistics
            robot_data = {
                'id': agent.id, 
                'body_parts': body_parts,
                'statistics': self.robot_stats.get(i, {})
            }
            agent_states.append(robot_data)
        
        # Extract ground geometry for rendering
        ground_geometry = []
        for body in self.world.bodies:
            if body.type == b2.b2_staticBody:
                for fixture in body.fixtures:
                    shape = fixture.shape
                    if isinstance(shape, b2.b2PolygonShape):
                        # Handle polygon shapes for the ground
                        ground_geometry.append({
                            'type': 'polygon',
                            'vertices': [tuple(body.GetWorldPoint(v)) for v in shape.vertices]
                        })
                    elif isinstance(shape, b2.b2EdgeShape):
                        # Handle edge shapes if they are still used
                        ground_geometry.append({
                            'type': 'line',
                            'vertices': [tuple(v) for v in shape.vertices]
                        })

        status_data = {
            'agents': agent_states,
            'statistics': self.population_stats,
            'ground_geometry': ground_geometry
        }
        return status_data

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

# --- Main Execution ---
app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')
env = TrainingEnvironment(num_agents=50)

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/status')
def status():
    return jsonify(env.get_status())

@app.route('/start', methods=['POST'])
def start_training():
    env.start()
    return jsonify({'status': 'Training started'})

@app.route('/stop', methods=['POST'])
def stop_training():
    env.stop()
    return jsonify({'status': 'Training stopped'})

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