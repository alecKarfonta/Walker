"""
Ecosystem Interface for Survival Q-Learning Integration
Provides a clean interface between survival Q-learning and ecosystem dynamics.
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import math


class EcosystemInterface:
    """
    Interface between survival Q-learning and ecosystem dynamics.
    Provides survival data for agents to make informed decisions.
    """
    
    def __init__(self, training_environment):
        self.env = training_environment
        self.ecosystem_dynamics = training_environment.ecosystem_dynamics
        self.agent_health = training_environment.agent_health
        self.agent_statuses = training_environment.agent_statuses
        self.agent_energy_levels = training_environment.agent_energy_levels
        
        print("🌿 EcosystemInterface initialized for survival Q-learning")
    
    def get_agent_survival_data(self, agent_id: str, agent_position: Tuple[float, float]) -> Dict[str, Any]:
        """
        Get comprehensive survival data for an agent.
        
        Args:
            agent_id: Agent identifier
            agent_position: Agent's current position (x, y)
            
        Returns:
            Dictionary containing survival-relevant data
        """
        try:
            survival_data = {}
            
            # Basic energy and health
            survival_data['energy_level'] = self.agent_energy_levels.get(agent_id, 1.0)
            health_data = self.agent_health.get(agent_id, {'health': 1.0, 'energy': 1.0})
            survival_data['health_level'] = health_data['health']
            
            # Food awareness - consider agent's dietary restrictions
            food_info = self._get_nearest_food_info(agent_id, agent_position)
            survival_data.update(food_info)
            
            # Social context
            social_info = self._get_social_context(agent_id, agent_position)
            survival_data.update(social_info)
            
            # Environmental threats
            threat_info = self._get_threat_assessment(agent_position)
            survival_data.update(threat_info)
            
            # Agent status
            status_data = self.agent_statuses.get(agent_id, {})
            survival_data['role'] = status_data.get('role', 'omnivore')
            survival_data['status'] = status_data.get('status', 'idle')
            
            # Enhanced data for predator Q-learning state calculation
            survival_data['agent_roles'] = self.ecosystem_dynamics.agent_roles
            survival_data['agent_energy'] = self.agent_energy_levels
            survival_data['nearby_agents_data'] = self._get_nearby_agents_data(agent_id, agent_position)
            survival_data['food_sources'] = self._get_food_sources_data()
            
            return survival_data
            
        except Exception as e:
            print(f"⚠️ Error getting survival data for agent {agent_id}: {e}")
            # Return safe defaults
            return {
                'energy_level': 1.0,
                'health_level': 1.0,
                'nearest_food_direction': 0.0,
                'nearest_food_distance': 10.0,
                'food_type': 'plants',
                'nearby_agents': 0,
                'competition_pressure': 0.0,
                'threat_level': 0.0,
                'role': 'omnivore',
                'status': 'idle'
            }
    
    def _get_nearest_food_info(self, agent_id: str, agent_position: Tuple[float, float]) -> Dict[str, Any]:
        """Get information about the nearest food source that the agent can actually eat."""
        try:
            if not self.ecosystem_dynamics.food_sources:
                return {
                    'nearest_food_direction': 0.0,
                    'nearest_food_distance': 10.0,
                    'food_type': 'plants',
                    'food_abundance': 0.0
                }
            
            agent_x, agent_y = agent_position
            nearest_food = None
            nearest_distance = float('inf')
            
            # Get agent's ecosystem role to determine dietary restrictions
            agent_status = self.agent_statuses.get(agent_id, {})
            agent_role_str = agent_status.get('role', 'omnivore')
            
            # Convert role string to EcosystemRole enum
            from src.ecosystem_dynamics import EcosystemRole
            role_map = {
                'herbivore': EcosystemRole.HERBIVORE,
                'carnivore': EcosystemRole.CARNIVORE,
                'omnivore': EcosystemRole.OMNIVORE,
                'scavenger': EcosystemRole.SCAVENGER,
                'symbiont': EcosystemRole.SYMBIONT
            }
            agent_role = role_map.get(agent_role_str, EcosystemRole.OMNIVORE)
            
            # Find nearest food source that this agent can actually eat
            for food_source in self.ecosystem_dynamics.food_sources:
                if food_source.amount <= 0.1:  # Skip depleted food sources (matches consumption threshold)
                    continue
                
                # Check if agent can eat this food type
                consumption_efficiency = self.ecosystem_dynamics._get_consumption_efficiency(agent_role, food_source.food_type)
                if consumption_efficiency <= 0.0:
                    continue  # Skip food types this agent cannot eat
                    
                food_x, food_y = food_source.position
                distance = math.sqrt((agent_x - food_x)**2 + (agent_y - food_y)**2)
                
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_food = food_source
            
            if nearest_food is None:
                return {
                    'nearest_food_direction': 0.0,
                    'nearest_food_distance': 10.0,
                    'food_type': 'plants',
                    'food_abundance': 0.0
                }
            
            # Calculate direction to food
            food_x, food_y = nearest_food.position
            direction = math.atan2(food_y - agent_y, food_x - agent_x)
            if direction < 0:
                direction += 2 * math.pi  # Normalize to 0-2π
            
            # Food abundance (0-1)
            abundance = nearest_food.amount / nearest_food.max_capacity
            
            return {
                'nearest_food_direction': direction,
                'nearest_food_distance': nearest_distance,
                'food_type': nearest_food.food_type,
                'food_abundance': abundance
            }
            
        except Exception as e:
            print(f"⚠️ Error getting food info: {e}")
            return {
                'nearest_food_direction': 0.0,
                'nearest_food_distance': 10.0,
                'food_type': 'plants',
                'food_abundance': 0.0
            }
    
    def _get_social_context(self, agent_id: str, agent_position: Tuple[float, float]) -> Dict[str, Any]:
        """Get social context information for the agent."""
        try:
            agent_x, agent_y = agent_position
            nearby_agents = 0
            competition_pressure = 0.0
            
            # Count nearby agents and assess competition
            for other_agent in self.env.agents:
                if getattr(other_agent, '_destroyed', False) or not other_agent.body:
                    continue
                    
                if other_agent.id == agent_id:
                    continue  # Skip self
                
                other_x = other_agent.body.position.x
                other_y = other_agent.body.position.y
                distance = math.sqrt((agent_x - other_x)**2 + (agent_y - other_y)**2)
                
                if distance < 15.0:  # Within social range
                    nearby_agents += 1
                    
                    # Assess competition pressure based on role and energy
                    other_agent_id = other_agent.id
                    other_status = self.agent_statuses.get(other_agent_id, {})
                    other_role = other_status.get('role', 'omnivore')
                    
                    agent_status = self.agent_statuses.get(agent_id, {})
                    agent_role = agent_status.get('role', 'omnivore')
                    
                    # Competition is higher between same roles
                    if other_role == agent_role:
                        competition_pressure += max(0, 1.0 - (distance / 15.0))  # Closer = more competition
                    
                    # Predator-prey relationships
                    if other_role == 'carnivore' and agent_role == 'herbivore':
                        competition_pressure += 2.0 * max(0, 1.0 - (distance / 10.0))  # Threat!
            
            return {
                'nearby_agents': min(nearby_agents, 10),  # Cap at 10 for discretization
                'competition_pressure': min(competition_pressure, 3.0)  # Cap for stability
            }
            
        except Exception as e:
            print(f"⚠️ Error getting social context: {e}")
            return {
                'nearby_agents': 0,
                'competition_pressure': 0.0
            }
    
    def _get_threat_assessment(self, agent_position: Tuple[float, float]) -> Dict[str, Any]:
        """Assess environmental threats for the agent."""
        try:
            # For now, minimal threat assessment
            # Could be expanded to include environmental hazards, predators, etc.
            
            return {
                'threat_level': 0.0,  # 0-1 scale
                'threat_direction': 0.0,  # Direction of nearest threat
                'safe_zones_nearby': 1  # Number of safe zones within range
            }
            
        except Exception as e:
            return {
                'threat_level': 0.0,
                'threat_direction': 0.0,
                'safe_zones_nearby': 1
            }
    
    def get_food_consumption_feedback(self, agent_id: str, agent_position: Tuple[float, float]) -> Dict[str, Any]:
        """
        Get feedback about food consumption attempts.
        Used for reward calculation.
        """
        try:
            # Check if agent recently consumed food
            energy_level = self.agent_energy_levels.get(agent_id, 1.0)
            previous_energy = getattr(self, f'_prev_energy_{agent_id}', energy_level)
            
            energy_change = energy_level - previous_energy
            
            # Store current energy for next comparison
            setattr(self, f'_prev_energy_{agent_id}', energy_level)
            
            consumption_info = {
                'energy_change': energy_change,
                'consumed_food': energy_change > 0.01,  # Threshold for food consumption
                'food_type_consumed': 'unknown'  # Could track specific food types
            }
            
            # If food was consumed, try to identify what type
            if consumption_info['consumed_food']:
                food_info = self._get_nearest_food_info(agent_id, agent_position)
                if food_info['nearest_food_distance'] < 2.0:  # Close enough to have consumed it
                    consumption_info['food_type_consumed'] = food_info['food_type']
            
            return consumption_info
            
        except Exception as e:
            print(f"⚠️ Error getting consumption feedback: {e}")
            return {
                'energy_change': 0.0,
                'consumed_food': False,
                'food_type_consumed': 'unknown'
            }
    
    def update_agent_motivation(self, agent_id: str, motivation_data: Dict[str, Any]):
        """
        Update agent's motivation state based on learning progress.
        Can influence ecosystem behavior.
        """
        try:
            # Update agent status based on learning stage
            if agent_id in self.agent_statuses:
                learning_stage = motivation_data.get('learning_stage', 'basic_movement')
                
                # Map learning stages to status behaviors
                if learning_stage == 'basic_movement':
                    # Focus on movement, less food seeking
                    pass  # Keep current status
                elif learning_stage == 'food_seeking':
                    # More active food seeking
                    if self.agent_statuses[agent_id]['status'] == 'idle':
                        self.agent_statuses[agent_id]['status'] = 'moving'
                elif learning_stage == 'survival_mastery':
                    # Advanced behaviors
                    energy_level = self.agent_energy_levels.get(agent_id, 1.0)
                    if energy_level > 0.8:
                        self.agent_statuses[agent_id]['status'] = 'active'
            
        except Exception as e:
            print(f"⚠️ Error updating agent motivation: {e}")
    
    def get_learning_opportunities(self, agent_position: Tuple[float, float]) -> List[Dict[str, Any]]:
        """
        Identify learning opportunities in the environment.
        Used for curriculum learning.
        """
        try:
            opportunities = []
            
            # Food learning opportunities - use temp agent_id for now
            food_info = self._get_nearest_food_info('temp', agent_position)
            if food_info['nearest_food_distance'] < 5.0:
                opportunities.append({
                    'type': 'food_interaction',
                    'difficulty': 'easy' if food_info['nearest_food_distance'] < 2.0 else 'medium',
                    'reward_potential': food_info['food_abundance'] * 10.0
                })
            
            # Social learning opportunities
            social_info = self._get_social_context('temp', agent_position)
            if social_info['nearby_agents'] > 0:
                opportunities.append({
                    'type': 'social_interaction',
                    'difficulty': 'medium',
                    'reward_potential': min(social_info['nearby_agents'] * 2.0, 8.0)
                })
            
            return opportunities
            
        except Exception as e:
            print(f"⚠️ Error getting learning opportunities: {e}")
            return []
    
    def _get_nearby_agents_data(self, agent_id: str, agent_position: Tuple[float, float]) -> List[Dict[str, Any]]:
        """Get data about nearby agents for predator state calculation."""
        try:
            nearby_agents = []
            agent_x, agent_y = agent_position
            
            for other_agent in self.env.agents:
                if getattr(other_agent, '_destroyed', False) or not other_agent.body:
                    continue
                    
                if other_agent.id == agent_id:
                    continue  # Skip self
                
                other_x = other_agent.body.position.x
                other_y = other_agent.body.position.y
                distance = math.sqrt((agent_x - other_x)**2 + (agent_y - other_y)**2)
                
                # Only include agents within hunting/perception range
                if distance < 20.0:
                    nearby_agents.append({
                        'id': other_agent.id,
                        'position': [other_x, other_y],
                        'distance': distance
                    })
            
            return nearby_agents
            
        except Exception as e:
            print(f"⚠️ Error getting nearby agents data: {e}")
            return []
    
    def _get_food_sources_data(self) -> List[Dict[str, Any]]:
        """Get food sources data for state calculation."""
        try:
            food_data = []
            
            for food_source in self.ecosystem_dynamics.food_sources:
                if food_source.amount > 0:  # Only include non-depleted food
                    food_data.append({
                        'position': list(food_source.position),
                        'type': food_source.food_type,
                        'amount': food_source.amount,
                        'max_capacity': food_source.max_capacity
                    })
            
            return food_data
            
        except Exception as e:
            print(f"⚠️ Error getting food sources data: {e}")
            return [] 