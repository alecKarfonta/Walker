"""
World Observation System for Robot Navigation
Provides environmental awareness and obstacle detection for crawling robots.
"""

import math
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

@dataclass
class ObstacleInfo:
    """Information about a detected obstacle."""
    distance: float
    position: Tuple[float, float]
    angle: float  # Relative to robot
    object_type: str
    traversable: bool
    confidence: float = 1.0
    timestamp: float = 0.0

class WorldObservation:
    """
    Provides environmental awareness for robots using multiple sensing strategies.
    Integrates with existing Box2D physics for efficient obstacle detection.
    """
    
    def __init__(self, sensor_range: float = 8.0, resolution: int = 16):
        """
        Initialize world observation system.
        
        Args:
            sensor_range: Maximum detection range in meters
            resolution: Number of sensing directions (like LiDAR rays)
        """
        self.sensor_range = sensor_range
        self.resolution = resolution
        self.last_observation = None
        self.observation_history = []
        
        # Pre-calculate sensing directions for efficiency
        self.sensing_angles = []
        for i in range(resolution):
            angle = (2 * math.pi * i) / resolution
            self.sensing_angles.append(angle)
    
    def observe_environment(self, robot, world, other_agents=None) -> Dict[str, Any]:
        """
        Perform environmental observation from robot's perspective.
        
        Args:
            robot: The robot performing observation
            world: Box2D world containing physics bodies
            other_agents: List of other robots/agents in the environment
            
        Returns:
            Dictionary containing obstacle information and clearances
        """
        if not robot.body:
            return self._empty_observation()
        
        robot_pos = (robot.body.position.x, robot.body.position.y)
        robot_angle = robot.body.angle
        
        # Detect obstacles using multiple strategies
        obstacles = []
        
        # 1. Physics-based obstacle detection
        physics_obstacles = self._detect_physics_obstacles(world, robot_pos, robot_angle)
        obstacles.extend(physics_obstacles)
        
        # 2. Agent-based obstacle detection
        if other_agents:
            agent_obstacles = self._detect_other_agents(robot_pos, robot_angle, other_agents)
            obstacles.extend(agent_obstacles)
        
        # 3. Predictive obstacle detection (based on robot behavior patterns)
        predictive_obstacles = self._predict_obstacles(robot_pos, robot_angle, obstacles)
        obstacles.extend(predictive_obstacles)
        
        # Calculate directional clearances
        clearances = self._calculate_clearances(obstacles)
        
        # Create observation result
        observation = {
            'timestamp': time.time(),
            'robot_position': robot_pos,
            'robot_angle': robot_angle,
            'obstacles': obstacles,
            'clearances': clearances,
            'sensor_range': self.sensor_range,
            'safe_directions': self._find_safe_directions(obstacles),
            'threat_level': self._assess_threat_level(obstacles)
        }
        
        # Store observation history
        self.last_observation = observation
        self.observation_history.append(observation)
        if len(self.observation_history) > 20:  # Keep last 20 observations
            self.observation_history.pop(0)
        
        return observation
    
    def _detect_physics_obstacles(self, world, robot_pos, robot_angle) -> List[ObstacleInfo]:
        """Detect obstacles using Box2D physics bodies."""
        obstacles = []
        
        # Query nearby physics bodies
        for body in world.bodies:
            if not body.userData or body.userData.get('type') == 'robot':
                continue  # Skip robots and non-obstacle bodies
            
            body_pos = (body.position.x, body.position.y)
            distance = math.sqrt((body_pos[0] - robot_pos[0])**2 + (body_pos[1] - robot_pos[1])**2)
            
            if distance <= self.sensor_range:
                # Calculate relative angle
                dx = body_pos[0] - robot_pos[0]
                dy = body_pos[1] - robot_pos[1]
                world_angle = math.atan2(dy, dx)
                relative_angle = world_angle - robot_angle
                
                # Normalize angle to [-π, π]
                relative_angle = math.atan2(math.sin(relative_angle), math.cos(relative_angle))
                
                # Determine object type and traversability
                object_type = "obstacle"
                traversable = False
                
                if body.userData:
                    if body.userData.get('type') == 'terrain':
                        object_type = "terrain"
                        properties = body.userData.get('properties', {})
                        traversable = properties.get('navigable', True)
                    elif body.userData.get('type') == 'obstacle':
                        object_type = body.userData.get('obstacle_type', 'obstacle')
                        traversable = False
                    elif body.userData.get('type') == 'food':
                        object_type = "food"
                        traversable = True
                
                obstacle = ObstacleInfo(
                    distance=distance,
                    position=body_pos,
                    angle=relative_angle,
                    object_type=object_type,
                    traversable=traversable,
                    confidence=1.0,
                    timestamp=time.time()
                )
                obstacles.append(obstacle)
        
        return obstacles
    
    def _detect_other_agents(self, robot_pos, robot_angle, other_agents) -> List[ObstacleInfo]:
        """Detect other robots/agents as dynamic obstacles."""
        obstacles = []
        
        for agent in other_agents:
            if not agent.body or getattr(agent, '_destroyed', False):
                continue
            
            agent_pos = (agent.body.position.x, agent.body.position.y)
            distance = math.sqrt((agent_pos[0] - robot_pos[0])**2 + (agent_pos[1] - robot_pos[1])**2)
            
            if distance <= self.sensor_range and distance > 0.1:  # Avoid self-detection
                # Calculate relative angle
                dx = agent_pos[0] - robot_pos[0]
                dy = agent_pos[1] - robot_pos[1]
                world_angle = math.atan2(dy, dx)
                relative_angle = world_angle - robot_angle
                
                # Normalize angle
                relative_angle = math.atan2(math.sin(relative_angle), math.cos(relative_angle))
                
                # Predict agent movement for better avoidance
                velocity = (0, 0)
                if hasattr(agent.body, 'linearVelocity'):
                    velocity = (agent.body.linearVelocity.x, agent.body.linearVelocity.y)
                
                agent_speed = math.sqrt(velocity[0]**2 + velocity[1]**2)
                
                obstacle = ObstacleInfo(
                    distance=distance,
                    position=agent_pos,
                    angle=relative_angle,
                    object_type="robot",
                    traversable=False,  # Robots are not traversable
                    confidence=0.9,  # Slightly lower confidence for moving objects
                    timestamp=time.time()
                )
                obstacles.append(obstacle)
        
        return obstacles
    
    def _predict_obstacles(self, robot_pos, robot_angle, current_obstacles) -> List[ObstacleInfo]:
        """
        Predict potential obstacles based on movement patterns.
        This implements a simplified version of "people as sensors" concept.
        """
        predicted = []
        
        # If we see robots suddenly changing direction, predict obstacles in their original path
        if len(self.observation_history) >= 2:
            prev_obs = self.observation_history[-1]
            
            for prev_obstacle in prev_obs.get('obstacles', []):
                if prev_obstacle.object_type == "robot":
                    # Look for sudden direction changes that might indicate obstacle avoidance
                    for current_obstacle in current_obstacles:
                        if (current_obstacle.object_type == "robot" and 
                            abs(current_obstacle.distance - prev_obstacle.distance) < 1.0):
                            
                            # Calculate if robot changed direction significantly
                            angle_change = abs(current_obstacle.angle - prev_obstacle.angle)
                            if angle_change > 0.5:  # Significant direction change
                                # Predict obstacle in the robot's original path
                                predicted_pos = (
                                    prev_obstacle.position[0] + math.cos(prev_obstacle.angle + robot_angle) * 2.0,
                                    prev_obstacle.position[1] + math.sin(prev_obstacle.angle + robot_angle) * 2.0
                                )
                                
                                pred_distance = math.sqrt((predicted_pos[0] - robot_pos[0])**2 + 
                                                        (predicted_pos[1] - robot_pos[1])**2)
                                
                                if pred_distance <= self.sensor_range:
                                    dx = predicted_pos[0] - robot_pos[0]
                                    dy = predicted_pos[1] - robot_pos[1]
                                    pred_angle = math.atan2(dy, dx) - robot_angle
                                    pred_angle = math.atan2(math.sin(pred_angle), math.cos(pred_angle))
                                    
                                    predicted_obstacle = ObstacleInfo(
                                        distance=pred_distance,
                                        position=predicted_pos,
                                        angle=pred_angle,
                                        object_type="predicted_obstacle",
                                        traversable=False,
                                        confidence=0.3,  # Lower confidence for predictions
                                        timestamp=time.time()
                                    )
                                    predicted.append(predicted_obstacle)
        
        return predicted
    
    def _calculate_clearances(self, obstacles) -> Dict[str, float]:
        """Calculate clearance distances in different directions."""
        clearances = {
            'front': self.sensor_range,
            'back': self.sensor_range, 
            'left': self.sensor_range,
            'right': self.sensor_range
        }
        
        # Find minimum distance to obstacles in each direction
        for obstacle in obstacles:
            if not obstacle.traversable:
                angle = obstacle.angle
                
                # Classify direction based on angle
                if -math.pi/4 <= angle <= math.pi/4:
                    clearances['front'] = min(clearances['front'], obstacle.distance)
                elif 3*math.pi/4 <= angle or angle <= -3*math.pi/4:
                    clearances['back'] = min(clearances['back'], obstacle.distance)
                elif math.pi/4 < angle < 3*math.pi/4:
                    clearances['left'] = min(clearances['left'], obstacle.distance)
                else:
                    clearances['right'] = min(clearances['right'], obstacle.distance)
        
        return clearances
    
    def _find_safe_directions(self, obstacles) -> List[Tuple[float, float]]:
        """Find directions with sufficient clearance for safe movement."""
        safe_directions = []
        min_clearance = 2.0  # Minimum safe distance
        
        for angle in self.sensing_angles:
            is_safe = True
            
            for obstacle in obstacles:
                if not obstacle.traversable:
                    # Check if obstacle blocks this direction
                    angle_diff = abs(obstacle.angle - angle)
                    if angle_diff < 0.3 and obstacle.distance < min_clearance:  # Within 17 degrees and too close
                        is_safe = False
                        break
            
            if is_safe:
                safe_directions.append((math.cos(angle), math.sin(angle)))
        
        return safe_directions
    
    def _assess_threat_level(self, obstacles) -> float:
        """Assess overall threat level based on nearby obstacles."""
        if not obstacles:
            return 0.0
        
        threat = 0.0
        danger_threshold = 3.0  # Distance considered dangerous
        
        for obstacle in obstacles:
            if not obstacle.traversable:
                if obstacle.distance < danger_threshold:
                    # Closer obstacles are more threatening
                    obstacle_threat = (danger_threshold - obstacle.distance) / danger_threshold
                    threat += obstacle_threat * obstacle.confidence
        
        return min(threat, 1.0)  # Cap at 1.0
    
    def _empty_observation(self) -> Dict[str, Any]:
        """Return empty observation when robot has no body."""
        return {
            'timestamp': time.time(),
            'robot_position': (0, 0),
            'robot_angle': 0,
            'obstacles': [],
            'clearances': {'front': 0, 'back': 0, 'left': 0, 'right': 0},
            'sensor_range': self.sensor_range,
            'safe_directions': [],
            'threat_level': 1.0  # Maximum threat when no observation possible
        }
    
    def get_navigation_suggestions(self) -> Dict[str, Any]:
        """Get navigation suggestions based on current observations."""
        if not self.last_observation:
            return {'action': 'stop', 'reason': 'no_observation'}
        
        obs = self.last_observation
        clearances = obs['clearances']
        threat_level = obs['threat_level']
        
        # Simple navigation logic
        if threat_level > 0.8:
            return {'action': 'retreat', 'reason': 'high_threat', 'direction': 'back'}
        
        # Find best direction to move
        best_direction = max(clearances.keys(), key=lambda k: clearances[k])
        
        if clearances[best_direction] > 3.0:
            return {
                'action': 'advance', 
                'reason': 'clear_path', 
                'direction': best_direction,
                'clearance': clearances[best_direction]
            }
        elif clearances[best_direction] > 1.5:
            return {
                'action': 'proceed_cautiously', 
                'reason': 'limited_clearance', 
                'direction': best_direction,
                'clearance': clearances[best_direction]
            }
        else:
            return {'action': 'explore_alternatives', 'reason': 'all_directions_blocked'} 