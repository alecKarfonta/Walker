"""
WebGL-based rendering system for Walker robot training visualization.
High-performance GPU-accelerated rendering with modern WebGL features.
"""

WEBGL_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Walker Training Visualizer (WebGL)</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body, html {
            width: 100%;
            height: 100vh; /* Fallback for older browsers */
            height: 100dvh; /* Dynamic Viewport Height - accounts for mobile browser chrome */
            overflow: hidden;
            background: #1a1a2e;
            color: #e8e8e8;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
        }
        
        html {
            height: -webkit-fill-available; /* WebKit-specific fix */
        }

        #app-container {
            display: flex;
            flex-direction: column;
            width: 100vw;
            height: 100vh; /* Fallback */
            height: 100dvh; /* Dynamic Viewport Height */
            min-height: 100vh; /* Fallback */
            min-height: 100dvh;
            max-height: 100vh; /* Prevent overflow */
            max-height: 100dvh;
        }
        
        #canvas-wrapper {
            flex: 1;
            position: relative;
            overflow: hidden;
            min-height: 0; /* Important for flexbox to work properly */
        }

        canvas { 
            display: block;
            width: 100%;
            height: 100%;
        }
        
        /* Bottom bar styling - same as Canvas 2D version */
        #bottom-bar {
            flex-shrink: 0;
            height: 280px;
            max-height: 280px;
            background: rgba(15, 20, 35, 0.95);
            border-top: 2px solid #e74c3c;
            box-shadow: 0 -5px 20px rgba(0,0,0,0.3);
            display: flex;
            padding: 12px;
            gap: 12px;
            z-index: 100;
            overflow: hidden;
            position: relative; /* Ensure it's positioned correctly */
        }

        .bottom-bar-section {
            background: rgba(26, 26, 46, 0.8);
            border-radius: 8px;
            border: 1px solid #3498db;
            padding: 10px;
            display: flex;
            flex-direction: column;
            overflow-y: auto;
            max-height: 100%;
        }

        #leaderboard-panel { flex: 2; }
        #robot-details-panel { flex: 1.5; min-width: 200px; }
        #summary-and-controls-panel { flex: 3; display: flex; flex-direction: column; gap: 8px; background: transparent; border: none; padding: 0; }
        #summary-panel { flex-shrink: 0; }
        #controls-panel { flex-grow: 1; display: flex; gap: 8px; padding: 0; border: none; background: transparent; }
        .control-column { flex: 1; display: flex; flex-direction: column; gap: 10px; }
        .panel-title { color: #3498db; font-size: 14px; font-weight: 600; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }
        .stat-row { display: flex; justify-content: space-between; align-items: center; padding: 4px 0; border-bottom: 1px solid rgba(52, 152, 219, 0.15); font-size: 13px; }
        .robot-stat-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; border-bottom: 1px solid rgba(52, 152, 219, 0.15); font-size: 13px; cursor: pointer; transition: all 0.2s ease; border-radius: 6px; margin: 2px 0; background: rgba(52, 152, 219, 0.1); border-left: 3px solid transparent; user-select: none; position: relative; }
        .robot-stat-row:hover { background: rgba(52, 152, 219, 0.3); transform: translateX(3px); border-left: 3px solid #3498db; box-shadow: 0 3px 8px rgba(0,0,0,0.3); }
        .robot-stat-row.focused { background: rgba(231, 76, 60, 0.8); border-left: 3px solid #c0392b; color: white; }
        .stat-label, .robot-stat-label { color: #bdc3c7; }
        .stat-value, .robot-stat-value { color: #ecf0f1; font-weight: 700; background: #34495e; padding: 3px 8px; border-radius: 4px; }
        .control-panel { background: rgba(30, 40, 60, 0.9); border-radius: 8px; border: 1px solid #2980b9; padding: 10px; flex-grow: 1; }
        .control-panel-title { color: #3498db; font-weight: 600; cursor: pointer; user-select: none; }
        .control-panel-title::before { content: '▶ '; display: inline-block; transition: transform 0.2s ease-in-out; }
        .control-panel.open .control-panel-title::before { transform: rotate(90deg); }
        .control-panel-content { padding-top: 10px; display: none; }
        .control-panel.open .control-panel-content { display: block; }
        
        /* Scrollbar styling for panels */
        .bottom-bar-section::-webkit-scrollbar { width: 6px; }
        .bottom-bar-section::-webkit-scrollbar-track { background: transparent; }
        .bottom-bar-section::-webkit-scrollbar-thumb { background: #3498db; border-radius: 3px; }

        /* Population summary specific styling */
        #population-summary-content {
            overflow-y: auto;
            max-height: calc(100% - 30px);
            flex-grow: 1;
        }

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

        .robot-stat-row:active {
            background: rgba(52, 152, 219, 0.4);
            transform: translateX(1px);
            box-shadow: 0 1px 3px rgba(0,0,0,0.4);
        }

        .robot-stat-row.focused:hover {
            background: rgba(192, 57, 43, 0.9);
            border-left: 3px solid #a93226;
        }

        .robot-stat-row.focused .robot-stat-label,
        .robot-stat-row.focused .robot-stat-value {
            color: #fff;
        }

        .robot-stat-row.focused .robot-stat-value {
            background: rgba(255, 255, 255, 0.2);
        }
    </style>
</head>
<body>
    <div id="app-container">
        <div id="canvas-wrapper">
            <canvas id="simulation-canvas"></canvas>
            <button id="toggleFoodLines" onclick="toggleFoodLines()" style="position:absolute; top:10px; left:120px; z-index:50; background:#4CAF50; color:white; border:none; padding:5px 10px; border-radius:3px; cursor:pointer;">Show Food Lines</button>
            <button id="toggleViewportCulling" onclick="toggleViewportCulling()" style="position:absolute; top:10px; left:250px; z-index:50; background:#00BCD4; color:white; border:none; padding:5px 10px; border-radius:3px; cursor:pointer;">🔍 Viewport Culling: ON</button>
            <div id="renderer-status" style="position:absolute; top:45px; left:120px; z-index:50; background:#4CAF50; color:white; border:none; padding:5px 10px; border-radius:3px; font-size:11px; pointer-events:none;">⚡ WebGL Enabled</div>
            
            <div id="speed-controls" style="position:absolute; top:90px; right:10px; z-index:50; background:rgba(0,0,0,0.8); color:white; padding:8px 12px; border-radius:6px; border:1px solid #3498db;">
                <div style="margin-bottom: 6px; font-size: 11px; color: #bdc3c7;">
                    ⚡ Speed: <span id="speed-display" style="color: #3498db; font-weight: bold;">1.0x</span>
                </div>
                <div style="display: flex; gap: 4px;">
                    <button onclick="setSimulationSpeed(0.5)" style="min-width: 28px; background: #95a5a6; color: white; border: none; padding: 3px 6px; border-radius: 3px; font-size: 9px; cursor: pointer;">0.5x</button>
                    <button onclick="setSimulationSpeed(1.0)" style="min-width: 28px; background: #3498db; color: white; border: none; padding: 3px 6px; border-radius: 3px; font-size: 9px; cursor: pointer;">1x</button>
                    <button onclick="setSimulationSpeed(2.0)" style="min-width: 28px; background: #e67e22; color: white; border: none; padding: 3px 6px; border-radius: 3px; font-size: 9px; cursor: pointer;">2x</button>
                    <button onclick="setSimulationSpeed(5.0)" style="min-width: 28px; background: #e74c3c; color: white; border: none; padding: 3px 6px; border-radius: 3px; font-size: 9px; cursor: pointer;">5x</button>
                </div>
            </div>
            
            <div id="focus-indicator" style="display:none; position:absolute; top:1%; left:50%; transform:translate(-50%, -50%); z-index:50; background:rgba(231, 76, 60, 0.95); color:white; padding:15px 20px; border-radius:8px; box-shadow:0 4px 20px rgba(0,0,0,0.3); border:2px solid rgba(255,255,255,0.2);">
                🎯 Focused on Agent: <span id="focused-agent-id">-</span>
            </div>
        </div>

        <div id="bottom-bar">
            <div id="leaderboard-panel" class="bottom-bar-section">
                <div class="panel-title">🏆 Leaderboard (Food)</div>
                <div id="leaderboard-content"></div>
            </div>
            <div id="robot-details-panel" class="bottom-bar-section">
                <div class="panel-title">🤖 Robot Details</div>
                <div id="robot-details-content">
                    <div class="placeholder">Select a robot to see details.</div>
                </div>
            </div>
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
        // WebGL Rendering System
        class WebGLRenderer {
            constructor(canvas) {
                this.canvas = canvas;
                this.gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
                
                if (!this.gl) {
                    console.error('❌ WebGL not supported, falling back to Canvas 2D');
                    this.fallbackToCanvas2D();
                    this.updateRendererStatus('Canvas 2D Fallback', '#FF9800');
                    return;
                }
                
                console.log('🚀 WebGL context initialized successfully');
                this.updateRendererStatus('WebGL Enabled', '#4CAF50');
                this.initialize();
            }
            
            initialize() {
                // Initialize WebGL state
                this.gl.enable(this.gl.DEPTH_TEST);
                this.gl.enable(this.gl.BLEND);
                this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);
                
                // Create shader programs
                this.createShaderPrograms();
                
                // Create buffers
                this.createBuffers();
                
                // Initialize uniforms
                this.initializeUniforms();
                
                console.log('✅ WebGL renderer initialized');
            }
            
            createShaderPrograms() {
                // Vertex shader for basic shapes
                const vertexShaderSource = `
                    attribute vec2 a_position;
                    attribute vec4 a_color;
                    
                    uniform mat3 u_matrix;
                    uniform vec2 u_resolution;
                    
                    varying vec4 v_color;
                    
                    void main() {
                        vec2 position = (u_matrix * vec3(a_position, 1)).xy;
                        
                        // Convert from pixels to clip space
                        vec2 clipSpace = ((position / u_resolution) * 2.0) - 1.0;
                        
                        // Standard WebGL coordinate transformation
                        gl_Position = vec4(clipSpace.x, clipSpace.y, 0, 1);
                        
                        // Fix upside-down rendering by flipping Y coordinate
                        //gl_Position.y = -gl_Position.y;
                        
                        v_color = a_color;
                    }
                `;
                
                // Fragment shader for basic shapes
                const fragmentShaderSource = `
                    precision mediump float;
                    
                    varying vec4 v_color;
                    
                    void main() {
                        gl_FragColor = v_color;
                    }
                `;
                
                this.shapeProgram = this.createProgram(vertexShaderSource, fragmentShaderSource);
                
                // Get attribute and uniform locations
                this.shapeAttribs = {
                    position: this.gl.getAttribLocation(this.shapeProgram, 'a_position'),
                    color: this.gl.getAttribLocation(this.shapeProgram, 'a_color')
                };
                
                this.shapeUniforms = {
                    matrix: this.gl.getUniformLocation(this.shapeProgram, 'u_matrix'),
                    resolution: this.gl.getUniformLocation(this.shapeProgram, 'u_resolution')
                };
            }
            
            createProgram(vertexSource, fragmentSource) {
                const vertexShader = this.createShader(this.gl.VERTEX_SHADER, vertexSource);
                const fragmentShader = this.createShader(this.gl.FRAGMENT_SHADER, fragmentSource);
                
                const program = this.gl.createProgram();
                this.gl.attachShader(program, vertexShader);
                this.gl.attachShader(program, fragmentShader);
                this.gl.linkProgram(program);
                
                if (!this.gl.getProgramParameter(program, this.gl.LINK_STATUS)) {
                    console.error('❌ Error linking shader program:', this.gl.getProgramInfoLog(program));
                    return null;
                }
                
                return program;
            }
            
            createShader(type, source) {
                const shader = this.gl.createShader(type);
                this.gl.shaderSource(shader, source);
                this.gl.compileShader(shader);
                
                if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
                    console.error('❌ Error compiling shader:', this.gl.getShaderInfoLog(shader));
                    this.gl.deleteShader(shader);
                    return null;
                }
                
                return shader;
            }
            
            createBuffers() {
                // Position buffer
                this.positionBuffer = this.gl.createBuffer();
                
                // Color buffer  
                this.colorBuffer = this.gl.createBuffer();
                
                // Index buffer for efficient rendering
                this.indexBuffer = this.gl.createBuffer();
            }
            
            initializeUniforms() {
                this.viewMatrix = this.createMatrix();
                this.projectionMatrix = this.createMatrix();
            }
            
            createMatrix() {
                return new Float32Array([
                    1, 0, 0,
                    0, 1, 0,
                    0, 0, 1
                ]);
            }
            
            updateCamera(x, y, zoom) {
                // Create camera transformation matrix
                const centerX = this.canvas.width / 2;
                const centerY = this.canvas.height / 2;
                
                // Transform matrix: scale, translate to center, then translate by camera position
                this.viewMatrix[0] = zoom;  this.viewMatrix[1] = 0;     this.viewMatrix[2] = 0;
                this.viewMatrix[3] = 0;     this.viewMatrix[4] = zoom;  this.viewMatrix[5] = 0;
                this.viewMatrix[6] = centerX - x * zoom; 
                this.viewMatrix[7] = centerY + y * zoom; // Use minus to flip Y-axis for physics world
                this.viewMatrix[8] = 1;
            }
            
            clear() {
                this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
                this.gl.clearColor(0.1, 0.1, 0.18, 1.0); // Match background color
                this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);
            }
            
            drawPolygon(vertices, color) {
                if (vertices.length < 3) return;
                
                // Convert vertices to flat array
                const positions = [];
                const colors = [];
                const indices = [];
                
                // Create triangulated polygon (fan triangulation)
                for (let i = 0; i < vertices.length; i++) {
                    positions.push(vertices[i][0], vertices[i][1]);
                    colors.push(color[0], color[1], color[2], color[3]);
                }
                
                // Create triangle indices (fan triangulation)
                for (let i = 1; i < vertices.length - 1; i++) {
                    indices.push(0, i, i + 1);
                }
                
                this.renderTriangles(positions, colors, indices);
            }
            
            drawCircle(x, y, radius, color, segments = 32) {
                const positions = [];
                const colors = [];
                const indices = [];
                
                // Center vertex
                positions.push(x, y);
                colors.push(color[0], color[1], color[2], color[3]);
                
                // Circle vertices
                for (let i = 0; i <= segments; i++) {
                    const angle = (i / segments) * 2 * Math.PI;
                    positions.push(x + Math.cos(angle) * radius, y + Math.sin(angle) * radius);
                    colors.push(color[0], color[1], color[2], color[3]);
                }
                
                // Triangle indices (fan from center)
                for (let i = 1; i <= segments; i++) {
                    indices.push(0, i, i + 1);
                }
                
                this.renderTriangles(positions, colors, indices);
            }
            
            drawLine(x1, y1, x2, y2, color, width = 1.0) {
                // Simple line implementation - could be enhanced with proper line rendering
                const dx = x2 - x1;
                const dy = y2 - y1;
                const length = Math.sqrt(dx * dx + dy * dy);
                const angle = Math.atan2(dy, dx);
                
                // Create a thin rectangle for the line
                const halfWidth = width / 2;
                const cos = Math.cos(angle);
                const sin = Math.sin(angle);
                
                const positions = [
                    x1 - sin * halfWidth, y1 + cos * halfWidth,
                    x1 + sin * halfWidth, y1 - cos * halfWidth,
                    x2 + sin * halfWidth, y2 - cos * halfWidth,
                    x2 - sin * halfWidth, y2 + cos * halfWidth
                ];
                
                const colors = [];
                for (let i = 0; i < 4; i++) {
                    colors.push(color[0], color[1], color[2], color[3]);
                }
                
                const indices = [0, 1, 2, 0, 2, 3];
                
                this.renderTriangles(positions, colors, indices);
            }
            
            drawRectangle(x, y, width, height, color) {
                // Draw a rectangle using two triangles
                const positions = [
                    x, y,                    // Bottom-left
                    x + width, y,            // Bottom-right
                    x + width, y + height,   // Top-right
                    x, y + height            // Top-left
                ];
                
                const colors = [];
                for (let i = 0; i < 4; i++) {
                    colors.push(color[0], color[1], color[2], color[3]);
                }
                
                const indices = [0, 1, 2, 0, 2, 3];
                
                this.renderTriangles(positions, colors, indices);
            }
            
            drawHealthEnergyBars(x, y, health, energy) {
                // Health and energy bars positioned above the robot (matching Canvas 2D version)
                const barWidth = 2.0;
                const barHeight = 0.3;
                const barSpacing = 0.4;
                const baseY = y + 3.0; // Position above robot
                
                // Health bar background
                this.drawRectangle(x - barWidth/2, baseY, barWidth, barHeight, [0.2, 0.2, 0.2, 0.8]);
                
                // Health bar foreground
                const healthColor = health > 0.5 ? [0.29, 0.69, 0.31, 1.0] : // Green
                                   health > 0.25 ? [1.0, 0.6, 0.0, 1.0] :     // Orange  
                                   [0.96, 0.26, 0.21, 1.0];                   // Red
                this.drawRectangle(x - barWidth/2, baseY, barWidth * health, barHeight, healthColor);
                
                // Energy bar background
                this.drawRectangle(x - barWidth/2, baseY + barSpacing, barWidth, barHeight, [0.2, 0.2, 0.2, 0.8]);
                
                // Energy bar foreground
                const energyColor = energy > 0.5 ? [0.13, 0.59, 0.95, 1.0] : // Blue
                                   energy > 0.25 ? [1.0, 0.6, 0.0, 1.0] :     // Orange
                                   [0.96, 0.26, 0.21, 1.0];                   // Red
                this.drawRectangle(x - barWidth/2, baseY + barSpacing, barWidth * energy, barHeight, energyColor);
            }
            
            drawEnhancedFoodSource(x, y, radius, amount, maxCapacity, foodType) {
                // Enhanced food source rendering with visual improvements
                const ratio = amount / maxCapacity;
                
                // Base colors for different food types
                let baseColor, glowColor;
                switch (foodType) {
                    case 'plants':
                        baseColor = [0.27, 0.76, 0.31, 0.9];    // Green
                        glowColor = [0.4, 0.9, 0.4, 0.4];      // Light green glow
                        break;
                    case 'insects':
                        baseColor = [0.47, 0.33, 0.28, 0.9];   // Brown
                        glowColor = [0.6, 0.45, 0.35, 0.4];    // Light brown glow
                        break;
                    case 'seeds':
                        baseColor = [1.0, 0.6, 0.0, 0.9];      // Orange
                        glowColor = [1.0, 0.8, 0.2, 0.4];      // Light orange glow
                        break;
                    default:
                        baseColor = [0.5, 0.5, 0.5, 0.9];      // Gray
                        glowColor = [0.7, 0.7, 0.7, 0.4];      // Light gray glow
                }
                
                // Outer glow (larger, transparent)
                const glowRadius = radius * 1.4;
                this.drawCircle(x, y, glowRadius, glowColor);
                
                // Main food circle (size based on amount)
                const mainRadius = 0.6 + (ratio * 1.4); // Varying size based on food amount
                this.drawCircle(x, y, mainRadius, baseColor);
                
                // Inner highlight (smaller, brighter) - only if there's substantial food
                if (ratio > 0.3) {
                    const highlightColor = [
                        Math.min(1.0, baseColor[0] + 0.3),
                        Math.min(1.0, baseColor[1] + 0.3), 
                        Math.min(1.0, baseColor[2] + 0.3),
                        0.6
                    ];
                    const highlightRadius = mainRadius * 0.4;
                    this.drawCircle(x + highlightRadius * 0.3, y + highlightRadius * 0.3, highlightRadius, highlightColor);
                }
                
                // Depletion indicator (darker overlay when low)
                if (ratio < 0.3) {
                    const depletionAlpha = (0.3 - ratio) * 2.0; // 0 to 0.6 alpha
                    const depletionColor = [0.1, 0.1, 0.1, depletionAlpha];
                    this.drawCircle(x, y, mainRadius, depletionColor);
                }
            }
            
            renderTriangles(positions, colors, indices) {
                this.gl.useProgram(this.shapeProgram);
                
                // Upload vertex data
                this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.positionBuffer);
                this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(positions), this.gl.DYNAMIC_DRAW);
                this.gl.enableVertexAttribArray(this.shapeAttribs.position);
                this.gl.vertexAttribPointer(this.shapeAttribs.position, 2, this.gl.FLOAT, false, 0, 0);
                
                // Upload color data
                this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.colorBuffer);
                this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(colors), this.gl.DYNAMIC_DRAW);
                this.gl.enableVertexAttribArray(this.shapeAttribs.color);
                this.gl.vertexAttribPointer(this.shapeAttribs.color, 4, this.gl.FLOAT, false, 0, 0);
                
                // Upload index data
                this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer);
                this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), this.gl.DYNAMIC_DRAW);
                
                // Set uniforms
                this.gl.uniformMatrix3fv(this.shapeUniforms.matrix, false, this.viewMatrix);
                this.gl.uniform2f(this.shapeUniforms.resolution, this.canvas.width, this.canvas.height);
                
                // Draw
                this.gl.drawElements(this.gl.TRIANGLES, indices.length, this.gl.UNSIGNED_SHORT, 0);
            }
            
            updateRendererStatus(message, color) {
                const statusElement = document.getElementById('renderer-status');
                if (statusElement) {
                    statusElement.textContent = `⚡ ${message}`;
                    statusElement.style.background = color;
                }
            }
            
            fallbackToCanvas2D() {
                // If WebGL fails, create a 2D context fallback
                this.ctx = this.canvas.getContext('2d');
                this.isWebGL = false;
                webglSupported = false;
                console.warn('⚠️ Using Canvas 2D fallback');
            }
        }

        // Initialize WebGL renderer (default mode)
        const canvas = document.getElementById('simulation-canvas');
        let renderer;
        let useWebGL = true;
        let webglSupported = true;
        
        // Global rendering state
        let scale = 15;
        let cameraPosition = { x: 0, y: 0 };
        let cameraZoom = 1.0;
        let focusedAgentId = null;
        let showFoodLines = false;
        let viewportCullingEnabled = true;
        
        // FPS tracking
        let uiFpsCounter = 0;
        let uiFpsStartTime = Date.now();
        let currentUiFps = 0;
        
        // Ecosystem colors (WebGL format - RGBA arrays)
        const ECOSYSTEM_COLORS_WEBGL = {
            'carnivore': [1.0, 0.27, 0.27, 0.8],     // Red
            'herbivore': [0.27, 0.67, 0.27, 0.8],    // Green  
            'omnivore': [1.0, 0.53, 0.27, 0.8],      // Orange
            'scavenger': [0.53, 0.27, 0.67, 0.8],    // Purple
            'symbiont': [0.27, 0.53, 1.0, 0.8]       // Blue
        };
        
        // Ecosystem colors (CSS format for UI)
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
        
        function initializeRenderer() {
            renderer = new WebGLRenderer(canvas);
            resizeCanvas();
            window.addEventListener('resize', resizeCanvas);
        }
        
        function resizeCanvas() {
            const wrapper = document.getElementById('canvas-wrapper');
            if (!wrapper) return;
            
            const width = wrapper.clientWidth;
            const height = wrapper.clientHeight;
            
            canvas.width = width;
            canvas.height = height;
            
            if (renderer && renderer.gl) {
                renderer.gl.viewport(0, 0, width, height);
            }
        }
        
        function drawWorld(data) {
            if (!renderer) return;
            
            // Update UI FPS counter
            uiFpsCounter++;
            const now = Date.now();
            if (now - uiFpsStartTime >= 1000) {
                currentUiFps = Math.round(uiFpsCounter * 1000 / (now - uiFpsStartTime));
                uiFpsCounter = 0;
                uiFpsStartTime = now;
            }
            
            // Use WebGL if available, otherwise fall back to Canvas 2D
            if (webglSupported && renderer.gl) {
                drawWorldWebGL(data);
            } else {
                drawWorldCanvas2D(data);
            }
        }
        
        function drawWorldWebGL(data) {
            // Clear screen
            renderer.clear();
            
            // Update camera
            renderer.updateCamera(cameraPosition.x, cameraPosition.y, cameraZoom * scale);
            
            // Draw ground
            if (data.shapes && data.shapes.ground) {
                data.shapes.ground.forEach(shape => {
                    if (shape.type === 'polygon' && shape.vertices.length > 2) {
                        renderer.drawPolygon(shape.vertices, [0.56, 0.56, 0.56, 1.0]); // Gray
                    }
                });
            }
            
            // Draw ecosystem elements (food sources) with enhanced rendering
            if (data.ecosystem && data.ecosystem.food_sources) {
                data.ecosystem.food_sources.forEach(food => {
                    const [x, y] = food.position;
                    const amount = food.amount;
                    const maxCapacity = food.max_capacity;
                    const baseRadius = 0.8;
                    
                    // Use enhanced food source rendering
                    renderer.drawEnhancedFoodSource(x, y, baseRadius, amount, maxCapacity, food.type);
                });
            }
            
            // Draw robots with health and energy bars
            if (data.shapes && data.shapes.robots && data.agents) {
                data.shapes.robots.forEach(robot => {
                    const agent = data.agents.find(a => a.id === robot.id);
                    if (!agent) return;
                    
                    const isFocused = (robot.id === focusedAgentId);
                    const ecosystem = agent.ecosystem || {};
                    const role = ecosystem.role || 'omnivore';
                    const health = ecosystem.health || 1.0;
                    const energy = ecosystem.energy || 1.0;
                    
                    let robotColor = ECOSYSTEM_COLORS_WEBGL[role] || [0.53, 0.53, 0.53, 0.8];
                    
                    // Highlight focused robot
                    if (isFocused) {
                        robotColor = [1.0, 0.84, 0.0, 1.0]; // Gold
                    }
                    
                    // Draw robot body parts
                    robot.body_parts.forEach(part => {
                        if (part.type === 'polygon' && part.vertices.length > 2) {
                            renderer.drawPolygon(part.vertices, robotColor);
                        } else if (part.type === 'circle') {
                            renderer.drawCircle(part.center[0], part.center[1], part.radius, robotColor);
                        }
                    });
                    
                    // Draw health and energy bars above the robot
                    if (agent.body && agent.body.x !== undefined && agent.body.y !== undefined) {
                        const robotX = agent.body.x;
                        const robotY = agent.body.y;
                        renderer.drawHealthEnergyBars(robotX, robotY, health, energy);
                    }
                });
            }
            
            // Draw food lines for focused robot
            if (showFoodLines && focusedAgentId && data.all_agents) {
                const focusedAgent = data.all_agents.find(agent => agent.id === focusedAgentId);
                if (focusedAgent && focusedAgent.ecosystem && focusedAgent.ecosystem.closest_food_position) {
                    const robotPos = [focusedAgent.body.x, focusedAgent.body.y];
                    const foodPos = focusedAgent.ecosystem.closest_food_position;
                    
                    const lineColor = focusedAgent.ecosystem.closest_food_signed_x_distance > 0 ? 
                        [0.0, 1.0, 1.0, 0.8] : [1.0, 0.4, 0.0, 0.8]; // Cyan or orange
                    
                    renderer.drawLine(robotPos[0], robotPos[1], foodPos[0], foodPos[1], lineColor, 0.3);
                }
            }
        }
        
        function drawWorldCanvas2D(data) {
            // Canvas 2D fallback rendering (simplified version)
            if (!renderer.ctx) return;
            
            const ctx = renderer.ctx;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Simple 2D rendering for fallback
            ctx.save();
            ctx.translate(canvas.width / 2, canvas.height / 2);
            ctx.scale(cameraZoom * scale, -cameraZoom * scale);
            ctx.translate(-cameraPosition.x, -cameraPosition.y);
            
            // Draw ground (simplified)
            if (data.shapes && data.shapes.ground) {
                ctx.strokeStyle = '#8e8e8e';
                ctx.lineWidth = 0.1;
                data.shapes.ground.forEach(shape => {
                    if (shape.type === 'polygon' && shape.vertices.length > 2) {
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
            
            // Draw robots (simplified) with health and energy bars
            if (data.shapes && data.shapes.robots) {
                data.shapes.robots.forEach(robot => {
                    const agent = data.agents?.find(a => a.id === robot.id);
                    if (!agent) return;
                    
                    const isFocused = (robot.id === focusedAgentId);
                    const ecosystem = agent.ecosystem || {};
                    const role = ecosystem.role || 'omnivore';
                    const health = ecosystem.health || 1.0;
                    const energy = ecosystem.energy || 1.0;
                    
                    let color = '#888888';
                    switch (role) {
                        case 'carnivore': color = '#FF4444'; break;
                        case 'herbivore': color = '#44AA44'; break;
                        case 'omnivore': color = '#FF8844'; break;
                        case 'scavenger': color = '#8844AA'; break;
                        case 'symbiont': color = '#4488FF'; break;
                    }
                    
                    if (isFocused) color = '#FFD700';
                    
                    ctx.fillStyle = color;
                    ctx.strokeStyle = color;
                    ctx.lineWidth = 0.1;
                    
                    robot.body_parts.forEach(part => {
                        ctx.beginPath();
                        if (part.type === 'polygon' && part.vertices.length > 2) {
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
                    
                    // Draw health and energy bars (Canvas 2D fallback version)
                    if (agent.body && agent.body.x !== undefined && agent.body.y !== undefined) {
                        const robotX = agent.body.x;
                        const robotY = agent.body.y;
                        const barWidth = 2.0;
                        const barHeight = 0.3;
                        const barSpacing = 0.4;
                        const baseY = robotY + 3.0;
                        
                        // Health bar background
                        ctx.fillStyle = '#333333';
                        ctx.fillRect(robotX - barWidth/2, baseY, barWidth, barHeight);
                        
                        // Health bar foreground
                        ctx.fillStyle = health > 0.5 ? '#4CAF50' : health > 0.25 ? '#FF9800' : '#F44336';
                        ctx.fillRect(robotX - barWidth/2, baseY, barWidth * health, barHeight);
                        
                        // Energy bar background
                        ctx.fillStyle = '#333333';
                        ctx.fillRect(robotX - barWidth/2, baseY + barSpacing, barWidth, barHeight);
                        
                        // Energy bar foreground
                        ctx.fillStyle = energy > 0.5 ? '#2196F3' : energy > 0.25 ? '#FF9800' : '#F44336';
                        ctx.fillRect(robotX - barWidth/2, baseY + barSpacing, barWidth * energy, barHeight);
                    }
                });
            }
            
            ctx.restore();
        }
        
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
        }
        
        function toggleViewportCulling() {
            viewportCullingEnabled = !viewportCullingEnabled;
            const button = document.getElementById('toggleViewportCulling');
            if (viewportCullingEnabled) {
                button.textContent = '🔍 Viewport Culling: ON';
                button.style.background = '#00BCD4';
            } else {
                button.textContent = '🔍 Viewport Culling: OFF';
                button.style.background = '#FF5722';
            }
        }
        
        function setSimulationSpeed(speed) {
            fetch('./set_simulation_speed', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ speed: speed })
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    document.getElementById('speed-display').textContent = speed.toFixed(1) + 'x';
                    console.log(`⚡ Simulation speed set to ${speed}x`);
                }
            })
            .catch(error => console.error('Error setting speed:', error));
        }
        
        // Mouse interaction handling
        let isDragging = false;
        let isDraggingRobot = false;
        let draggedRobotId = null;
        let lastMouseX = 0, lastMouseY = 0;
        let userHasManuallyPanned = false;
        let mouseDownTime = 0;
        let mouseDownX = 0, mouseDownY = 0;
        let mouseDownRobotId = null;
        
        // Constants for click detection
        const CLICK_THRESHOLD = 5; // pixels
        const CLICK_TIME_THRESHOLD = 200; // milliseconds
        
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
            
            // Convert screen coordinates to world coordinates (accounting for scale factor)
            const totalZoom = cameraZoom * scale;
            const worldX = (x - canvas.width / 2) / totalZoom + cameraPosition.x;
            const worldY = (canvas.height / 2 - y) / totalZoom + cameraPosition.y;
            
            // Find robot at click position
            fetch('./get_agent_at_position', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ x: worldX, y: worldY })
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success' && data.agent_id !== null) {
                    mouseDownRobotId = data.agent_id;
                    console.log(`🤖 Mouse down on robot ${data.agent_id}`);
                } else {
                    mouseDownRobotId = null;
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
                        const totalZoom = cameraZoom * scale;
                        const worldX = (x - canvas.width / 2) / totalZoom + cameraPosition.x;
                        const worldY = (canvas.height / 2 - y) / totalZoom + cameraPosition.y;
                        
                        fetch('./move_agent', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
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
                        userHasManuallyPanned = true;
                        const totalZoom = cameraZoom * scale;
                        cameraPosition.x -= (e.clientX - lastMouseX) / totalZoom;
                        cameraPosition.y += (e.clientY - lastMouseY) / totalZoom;
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
                fetch('./click', {
                     method: 'POST',
                     headers: { 'Content-Type': 'application/json' },
                     body: JSON.stringify({ agent_id: mouseDownRobotId })
                 })
                 .then(response => response.json())
                 .then(data => {
                     if (data.status === 'success') {
                         focusedAgentId = data.agent_id;
                         if (data.agent_id !== null) {
                             userHasManuallyPanned = false;
                             console.log(`✅ Agent ${data.agent_id} selected!`);
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
            const newZoom = e.deltaY < 0 ? cameraZoom * zoomFactor : cameraZoom / zoomFactor;
            cameraZoom = Math.max(0.01, Math.min(20, newZoom));
            
            // Send zoom update to backend
            fetch('./update_zoom', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ zoom: cameraZoom })
            })
            .catch(error => {
                console.error('Error updating zoom:', error);
            });
        });
        
        // Main rendering loop
        let lastFetchTime = 0;
        const fetchInterval = 33; // ~30 FPS
        
        function fetchData() {
            const now = Date.now();
            if (now - lastFetchTime < fetchInterval) {
                requestAnimationFrame(fetchData);
                return;
            }
            lastFetchTime = now;
            
            const canvasWidth = canvas.width;
            const canvasHeight = canvas.height;
            const cullingParam = viewportCullingEnabled ? '&viewport_culling=true' : '&viewport_culling=false';
            const cameraParam = `&camera_x=${cameraPosition.x}&camera_y=${cameraPosition.y}`;
            
            fetch(`./status?canvas_width=${canvasWidth}&canvas_height=${canvasHeight}${cullingParam}${cameraParam}`)
                .then(response => response.json())
                .then(data => {
                    // Update camera from backend if not manually panning
                    if (data.camera && !isDragging && !userHasManuallyPanned) {
                        if (data.camera.position && Array.isArray(data.camera.position)) {
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
                    
                    // Update focused agent
                    if (data.focused_agent_id !== undefined) {
                        focusedAgentId = data.focused_agent_id;
                    }
                    
                    // Render the world
                    drawWorld(data);
                    
                    // Update UI with full functionality
                    updateStats(data);
                    
                    requestAnimationFrame(fetchData);
                })
                .catch(error => {
                    console.error('Error fetching data:', error);
                    setTimeout(fetchData, 1000);
                });
        }
        
        // UI update functions (from original template)
        let lastLeaderboardHtml = '';
        
        function updateStats(data) {
            if (!data) return;

            // Update global focused agent ID from backend
            if (data.focused_agent_id !== undefined) {
                focusedAgentId = data.focused_agent_id;
            }

            // Update simulation speed display
            if (data.simulation_speed !== undefined) {
                const speedDisplay = document.getElementById('speed-display');
                if (speedDisplay) {
                    speedDisplay.textContent = data.simulation_speed.toFixed(1) + 'x';
                }
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
                
                if (newLeaderboardHtml !== lastLeaderboardHtml) {
                    leaderboardContent.innerHTML = newLeaderboardHtml;
                    lastLeaderboardHtml = newLeaderboardHtml;
                }
            }

            // Update population summary
            const populationSummaryContent = document.getElementById('population-summary-content');
            if (populationSummaryContent && data.statistics) {
                const roleDistribution = {};
                let totalAgents = 0;
                
                if (data.all_agents) {
                    data.all_agents.forEach(agent => {
                        const role = agent.ecosystem?.role || 'omnivore';
                        roleDistribution[role] = (roleDistribution[role] || 0) + 1;
                        totalAgents++;
                    });
                }
                
                const roleIcons = {
                    'carnivore': '🦁',
                    'herbivore': '🐰', 
                    'omnivore': '🐻',
                    'scavenger': '🦅',
                    'symbiont': '🐠'
                };
                
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
                        <span class="stat-label">WebGL FPS:</span>
                        <span class="stat-value" style="color: ${currentUiFps >= 30 ? '#4CAF50' : '#FF5722'}">${currentUiFps}</span>
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
                robotDetailsPanel.innerHTML = '<div class="panel-title">🤖 Robot Details</div><div class="robot-details-content">Click on a robot to view details</div>';
                return;
            }
            
            const agent = (data.all_agents || data.agents)?.find(a => a.id === focusedAgentId);
            if (!agent) {
                robotDetailsPanel.innerHTML = '<div class="panel-title">🤖 Robot Details</div><div class="robot-details-content">Robot not found</div>';
                return;
            }
            
            // Check if agent is outside viewport (limited data available)
            const hasDetailedData = agent.upper_arm && agent.lower_arm && 
                                   agent.upper_arm.x !== undefined && agent.upper_arm.y !== undefined;
            const isOutsideViewport = data.viewport_culling && data.viewport_culling.enabled && !hasDetailedData;
            const viewportWarning = isOutsideViewport ? `
                <div style="background: rgba(255, 152, 0, 0.1); border: 1px solid #FF9800; border-radius: 4px; padding: 6px; margin-bottom: 8px; font-size: 11px;">
                    <span style="color: #FF9800;">⚠️ Robot outside viewport - Limited data available. Pan camera to robot for full details.</span>
                </div>
            ` : '';
            
            const ecosystem = agent.ecosystem || {};
            const role = ecosystem.role || 'omnivore';
            const status = ecosystem.status || 'idle';
            const health = ecosystem.health || 1.0;
            const energy = ecosystem.energy || 1.0;
            const speed = ecosystem.speed || 0.0;
            
            const roleSymbols = {
                'carnivore': '🦁', 'herbivore': '🐰', 'omnivore': '🐻', 'scavenger': '🦅', 'symbiont': '🐠'
            };
            
            const statusSymbols = {
                'hunting': '🎯', 'feeding': '🍃', 'fleeing': '💨', 'territorial': '🛡️', 'idle': '😴', 'moving': '➡️', 'active': '⚡'
            };
            
            const details = `
                <div class="robot-details-title">🤖 Robot ${agent.id}</div>
                <div class="robot-details-content">
                    ${viewportWarning}
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
                            <span class="detail-label">Closest Food:</span>
                            <span class="detail-value" style="color: ${(ecosystem.closest_food_distance === undefined || ecosystem.closest_food_distance >= 999999 || ecosystem.closest_food_distance > 50) ? '#FF8844' : ecosystem.closest_food_distance < 5 ? '#4CAF50' : '#FFF'};">
                                ${ecosystem.closest_food_distance === undefined ? 'N/A (outside viewport)' : ecosystem.closest_food_distance >= 999999 ? 'None available' : ecosystem.closest_food_distance.toFixed(1) + 'm'}
                            </span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">X-Axis Distance:</span>
                            <span class="detail-value" style="color: ${ecosystem.closest_food_signed_x_distance === undefined ? '#888' : ecosystem.closest_food_signed_x_distance > 0 ? '#4CAF50' : '#FF9800'};">
                                ${ecosystem.closest_food_signed_x_distance === undefined ? 'N/A (outside viewport)' : (ecosystem.closest_food_signed_x_distance > 0 ? '+' : '') + ecosystem.closest_food_signed_x_distance.toFixed(1) + 'm ' + (ecosystem.closest_food_signed_x_distance > 0 ? '→' : '←')}
                            </span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">Food Type:</span>
                            <span class="detail-value" style="color: ${ecosystem.closest_food_source === 'prey' ? '#FF6B6B' : '#4CAF50'};">
                                ${ecosystem.closest_food_type || 'Unknown'}
                                ${ecosystem.closest_food_source === 'prey' ? ' 🎯' : ecosystem.closest_food_source === 'environment' ? ' 🌿' : ''}
                            </span>
                        </div>
                    </div>
                    
                    <div class="detail-section">
                        <div class="detail-row">
                            <span class="detail-label">Episode Reward:</span>
                            <span class="detail-value">${(agent.total_reward || 0).toFixed(2)}</span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">Recent Reward:</span>
                            <span class="detail-value" style="color: ${(agent.recent_reward || 0) > 0 ? '#27ae60' : (agent.recent_reward || 0) < 0 ? '#e74c3c' : '#f39c12'};">${(agent.recent_reward || 0).toFixed(4)}</span>
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
                    
                    ${agent.reward_components && typeof agent.reward_components === 'object' && Object.keys(agent.reward_components).length > 0 ? `
                    <div class="detail-section">
                        <div style="margin-bottom: 6px; font-weight: bold; color: #3498db;">Reward Components</div>
                        ${Object.entries(agent.reward_components).map(([key, value]) => `
                        <div class="detail-row">
                            <span class="detail-label">${key.charAt(0).toUpperCase() + key.slice(1)}:</span>
                            <span class="detail-value" style="color: ${value > 0 ? '#27ae60' : value < 0 ? '#e74c3c' : '#f39c12'};">${value.toFixed(4)}</span>
                        </div>
                        `).join('')}
                    </div>` : ''}
                </div>
            `;
            
            robotDetailsPanel.innerHTML = details;
        }
        
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
        
        // Leaderboard click handler
        function setupLeaderboardClickHandler() {
            const leaderboardPanel = document.getElementById('leaderboard-panel');
            if (leaderboardPanel) {
                leaderboardPanel.addEventListener('click', function(e) {
                    const robotRow = e.target.closest('.robot-stat-row');
                    if (robotRow && robotRow.dataset.agentId) {
                        e.preventDefault();
                        e.stopPropagation();

                        const agentId = robotRow.dataset.agentId;
                        console.log(`🎯 Leaderboard button clicked for agent: ${agentId}`);

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
                                    userHasManuallyPanned = false;
                                    console.log(`✅ Agent ${data.agent_id} selected from leaderboard!`);
                                }
                            }
                        })
                        .catch(error => {
                            console.error('❌ Error during leaderboard click fetch:', error);
                        });
                    }
                });
            }
        }
        
        // Control panel setup
        function createSlider(id, label, min, max, step, value) {
            const container = document.createElement('div');
            container.style.marginBottom = '8px';
            
            const labelEl = document.createElement('div');
            labelEl.style.color = '#bdc3c7';
            labelEl.style.fontSize = '11px';
            labelEl.style.marginBottom = '4px';
            labelEl.textContent = label;
            
            const sliderContainer = document.createElement('div');
            sliderContainer.style.display = 'flex';
            sliderContainer.style.alignItems = 'center';
            sliderContainer.style.gap = '8px';
            
            const slider = document.createElement('input');
            slider.type = 'range';
            slider.id = id;
            slider.min = min;
            slider.max = max;
            slider.step = step;
            slider.value = value;
            slider.style.flex = '1';
            
            const valueEl = document.createElement('span');
            valueEl.style.color = '#ecf0f1';
            valueEl.style.fontSize = '10px';
            valueEl.style.minWidth = '40px';
            valueEl.textContent = parseFloat(value).toFixed(3);
            
            slider.addEventListener('input', () => {
                valueEl.textContent = parseFloat(slider.value).toFixed(3);
            });

            slider.addEventListener('change', () => {
                updateAgentParams({ [id]: parseFloat(slider.value) });
            });
            
            sliderContainer.appendChild(slider);
            sliderContainer.appendChild(valueEl);
            container.appendChild(labelEl);
            container.appendChild(sliderContainer);
            
            return container;
        }
        
        function setupControlPanels() {
            // Learning panel
            const learningPanelContent = document.querySelector('#learning-panel .control-panel-content');
            if (learningPanelContent) {
                learningPanelContent.appendChild(createSlider('learning_rate', 'Learning Rate', 0.001, 0.1, 0.001, 0.005));
                learningPanelContent.appendChild(createSlider('epsilon', 'Epsilon (Randomness)', 0.0, 1.0, 0.01, 0.3));
            }

            // Control panel toggle functionality
            document.querySelectorAll('.control-panel-title').forEach(title => {
                title.addEventListener('click', () => {
                    title.parentElement.classList.toggle('open');
                });
            });
        }

        async function updateAgentParams(params) {
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

        // Initialize everything
        document.addEventListener('DOMContentLoaded', () => {
            initializeRenderer();
            setupLeaderboardClickHandler();
            setupControlPanels();
            fetchData();
        });
        
    </script>
</body>
</html>
"""

def get_webgl_template():
    """Get the WebGL HTML template for high-performance rendering."""
    return WEBGL_HTML_TEMPLATE 