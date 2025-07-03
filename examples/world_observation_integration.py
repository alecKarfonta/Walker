"""
Example: Integrating World Observation with Existing Robots

This example shows how to add environmental awareness to your crawling robots
using the new WorldObservation system.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.agents.world_observation import WorldObservation, ObstacleInfo
from src.agents.crawling_agent import CrawlingAgent

class ObservationAwareCrawlingAgent(CrawlingAgent):
    """
    Enhanced crawling agent with world observation capabilities.
    Extends the existing CrawlingAgent with environmental awareness.
    """
    
    def __init__(self, world, agent_id=None, position=(0, 10), 
                 category_bits=0x0002, mask_bits=0x0001, 
                 physical_params=None, parent_lineage=None):
        """Initialize agent with world observation capabilities."""
        
        # Initialize base agent
        super().__init__(world, agent_id, position, category_bits, mask_bits, 
                        physical_params, parent_lineage)
        
        # Add world observation system
        self.world_observer = WorldObservation(
            sensor_range=8.0,  # 8 meter sensing range
            resolution=16      # 16 sensing directions (22.5° apart)
        )
        
        # Observation data
        self.current_observation = None
        self.observation_enabled = True
        
        # Navigation state
        self.stuck_timer = 0
        self.last_position = position
        self.movement_threshold = 0.1  # Minimum movement to not be considered stuck
        
    def step(self, dt: float, other_agents=None):
        """Enhanced step function with environmental observation."""
        
        # Perform world observation before taking action
        if self.observation_enabled and not getattr(self, '_destroyed', False):
            self.current_observation = self.world_observer.observe_environment(
                robot=self,
                world=self.world,
                other_agents=other_agents
            )
            
            # Use observation to modify movement behavior
            self._apply_observation_guided_behavior()
        
        # Execute the normal step using parent class
        result = super().step(dt)
        
        # Check if robot is stuck and needs alternative behavior
        self._check_stuck_state()
        
        return result
    
    def _apply_observation_guided_behavior(self):
        """Apply behavioral modifications based on environmental observation."""
        if not self.current_observation:
            return
        
        # Get navigation suggestions from observation system
        nav_suggestions = self.world_observer.get_navigation_suggestions()
        
        # Extract observation data
        clearances = self.current_observation['clearances']
        threat_level = self.current_observation['threat_level']
        
        # Modify robot's behavior based on observation
        # This is a simple example - you can integrate this with your RL training
        if threat_level > 0.8:
            print(f"🚨 Robot {self.id}: High threat detected ({threat_level:.2f}), taking evasive action")
            # In a real implementation, you would modify the robot's neural network input
            # or adjust joint targets here. For demonstration, we just log the behavior.
        elif threat_level > 0.4:
            print(f"⚠️ Robot {self.id}: Moderate threat detected, proceeding cautiously")
        # The actual movement will be handled by the parent class step() method
    
    def _observation_guided_action(self, original_action):
        """
        Modify robot's action based on environmental observation.
        
        Args:
            original_action: The original action from RL or other controller
            
        Returns:
            Modified action that considers environmental obstacles
        """
        if not self.current_observation:
            return original_action
        
        # Get navigation suggestions from observation system
        nav_suggestions = self.world_observer.get_navigation_suggestions()
        
        # Extract observation data
        clearances = self.current_observation['clearances']
        threat_level = self.current_observation['threat_level']
        safe_directions = self.current_observation['safe_directions']
        
        # High threat - override original action with safety behavior
        if threat_level > 0.8:
            print(f"🚨 Robot {self.id}: High threat detected ({threat_level:.2f}), taking evasive action")
            return self._emergency_avoidance_action(clearances)
        
        # Moderate threat - modify original action
        elif threat_level > 0.4:
            return self._cautious_action(original_action, clearances, nav_suggestions)
        
        # Low threat - proceed with original action but with awareness
        else:
            return self._aware_action(original_action, clearances)
    
    def _emergency_avoidance_action(self, clearances):
        """Generate emergency avoidance action when threat level is high."""
        
        # Find direction with most clearance
        best_direction = max(clearances.keys(), key=lambda k: clearances[k])
        
        if best_direction == 'back':
            # Move backward
            return [-0.5, 0.5]  # Reverse both arms
        elif best_direction == 'left':
            # Turn left
            return [0.8, -0.3]
        elif best_direction == 'right':
            # Turn right  
            return [-0.3, 0.8]
        else:
            # Move forward if front is clear
            return [0.5, 0.5]
    
    def _cautious_action(self, original_action, clearances, nav_suggestions):
        """Modify action to be more cautious when moderate threat detected."""
        
        if original_action is None:
            original_action = [0.0, 0.0]
        
        # Reduce action magnitude for caution
        cautious_action = [a * 0.7 for a in original_action]
        
        # If moving forward but front clearance is limited, prefer turning
        front_clearance = clearances.get('front', 0)
        if front_clearance < 2.0:
            # Bias toward turning instead of moving forward
            if clearances.get('left', 0) > clearances.get('right', 0):
                cautious_action[0] *= 1.2  # Favor left arm
                cautious_action[1] *= 0.5
            else:
                cautious_action[0] *= 0.5
                cautious_action[1] *= 1.2  # Favor right arm
        
        return cautious_action
    
    def _aware_action(self, original_action, clearances):
        """Apply minor adjustments to action based on environmental awareness."""
        
        if original_action is None:
            original_action = [0.0, 0.0]
        
        # Minor bias toward directions with more clearance
        left_clearance = clearances.get('left', 0)
        right_clearance = clearances.get('right', 0)
        
        # Subtle bias toward side with more clearance
        if left_clearance > right_clearance + 1.0:
            original_action[0] *= 1.1  # Slight boost to left arm
        elif right_clearance > left_clearance + 1.0:
            original_action[1] *= 1.1  # Slight boost to right arm
        
        return original_action
    
    def _check_stuck_state(self):
        """Check if robot is stuck and needs help."""
        if not self.body:
            return
        
        current_pos = (self.body.position.x, self.body.position.y)
        
        # Calculate movement since last check
        if hasattr(self, 'last_position'):
            movement = ((current_pos[0] - self.last_position[0])**2 + 
                       (current_pos[1] - self.last_position[1])**2)**0.5
            
            if movement < self.movement_threshold:
                self.stuck_timer += 1
            else:
                self.stuck_timer = 0
        
        self.last_position = current_pos
        
        # If stuck for too long, try unsticking behavior
        if self.stuck_timer > 50:  # Stuck for 50 steps
            print(f"🔄 Robot {self.id}: Detected stuck state, trying unstick behavior")
            self._unstick_behavior()
    
    def _unstick_behavior(self):
        """Behavior to help robot get unstuck."""
        # This could be implemented as a special action sequence
        # For now, just reset the stuck timer
        self.stuck_timer = 0
    
    def get_observation_data(self):
        """Get current observation data for debugging/visualization."""
        if not self.current_observation:
            return None
        
        return {
            'obstacles_count': len(self.current_observation['obstacles']),
            'clearances': self.current_observation['clearances'],
            'threat_level': self.current_observation['threat_level'],
            'safe_directions_count': len(self.current_observation['safe_directions']),
            'sensor_range': self.current_observation['sensor_range']
        }
    
    def print_observation_summary(self):
        """Print a summary of current observations for debugging."""
        if not self.current_observation:
            print(f"Robot {self.id}: No observation data")
            return
        
        obs = self.current_observation
        print(f"🤖 Robot {self.id} Observation Summary:")
        print(f"   📍 Position: ({obs['robot_position'][0]:.1f}, {obs['robot_position'][1]:.1f})")
        print(f"   🚨 Threat Level: {obs['threat_level']:.2f}")
        print(f"   🔍 Obstacles Detected: {len(obs['obstacles'])}")
        print(f"   🚦 Clearances: Front={obs['clearances']['front']:.1f}m, "
              f"Back={obs['clearances']['back']:.1f}m, "
              f"Left={obs['clearances']['left']:.1f}m, "
              f"Right={obs['clearances']['right']:.1f}m")
        print(f"   ✅ Safe Directions: {len(obs['safe_directions'])}")


def create_observation_aware_robot(world, position=(0, 10)):
    """
    Factory function to create an observation-aware robot.
    
    Args:
        world: Box2D world
        position: Starting position
        
    Returns:
        ObservationAwareCrawlingAgent instance
    """
    from src.agents.physical_parameters import PhysicalParameters
    
    # Create robot with random physical parameters
    physical_params = PhysicalParameters.random_parameters()
    
    robot = ObservationAwareCrawlingAgent(
        world=world,
        position=position,
        physical_params=physical_params
    )
    
    return robot


# Example usage in your training environment:
"""
# In train_robots_web_visual.py or similar training script:

def create_agents_with_observation(self):
    '''Create agents with world observation capabilities.'''
    
    # Replace regular CrawlingAgent creation with observation-aware version
    for i in range(self.num_agents):
        position = self._calculate_spawn_position(i)
        
        # Create observation-aware robot
        robot = create_observation_aware_robot(self.world, position)
        
        # Set robot userData for identification by observation system
        if robot.body:
            robot.body.userData = {
                'type': 'robot',
                'robot_id': robot.id
            }
        
        self.agents.append(robot)
        
        # Print initial observation capability
        print(f"🤖 Created observation-aware robot {robot.id} at {position}")


def training_step_with_observation(self):
    '''Modified training step that uses observation data.'''
    
    for agent in self.agents:
        if getattr(agent, '_destroyed', False):
            continue
        
        # Get current observation
        if hasattr(agent, 'current_observation') and agent.current_observation:
            obs_data = agent.get_observation_data()
            
            # Log high-threat situations
            if obs_data and obs_data['threat_level'] > 0.5:
                print(f"⚠️ Robot {agent.id}: High threat environment detected")
                agent.print_observation_summary()
        
        # Perform step with all other agents as context
        other_agents = [a for a in self.agents if a != agent and not getattr(a, '_destroyed', False)]
        result = agent.step(other_agents=other_agents)
""" 