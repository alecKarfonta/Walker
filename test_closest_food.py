#!/usr/bin/env python3
"""
Unit tests for the closest food calculation logic.
Tests various robot types, distances, and food source combinations.
"""

import unittest
import math

def calculate_closest_food_for_agent(agent_pos, agent_role, agent_id, food_sources, other_agents, agent_energy_levels, agent_health, agent_statuses):
    """
    Simplified version of the closest food calculation logic for testing.
    This extracts the core logic without dependencies on the full TrainingEnvironment.
    """
    potential_food_sources = []
    
    # Add environmental food sources for non-carnivore and non-scavenger agents
    if agent_role not in ['carnivore', 'scavenger']:
        for food in food_sources:
            if food['amount'] > 0.1:
                food_pos = food['position']
                distance = ((agent_pos[0] - food_pos[0])**2 + (agent_pos[1] - food_pos[1])**2)**0.5
                potential_food_sources.append({
                    'position': food['position'],
                    'type': food['food_type'],
                    'source': 'environment',
                    'amount': food['amount'],
                    'distance': distance
                })
    
    # For carnivores and scavengers, add other agents as potential prey
    if agent_role in ['carnivore', 'scavenger']:
        for other_agent in other_agents:
            if other_agent['id'] == agent_id:
                continue
            
            other_pos = other_agent['position']
            other_energy = agent_energy_levels.get(other_agent['id'], 1.0)
            other_health = agent_health.get(other_agent['id'], {'health': 1.0})['health']
            
            # SCAVENGER RESTRICTION: Only target robots with energy < 0.3
            if agent_role == 'scavenger' and other_energy >= 0.3:
                continue
            
            # CARNIVORE RESTRICTIONS: Can ONLY hunt herbivore, scavenger, and omnivore robots
            other_role = agent_statuses.get(other_agent['id'], {}).get('role', 'omnivore')
            if agent_role == 'carnivore':
                valid_prey_roles = ['herbivore', 'scavenger', 'omnivore']
                if other_role not in valid_prey_roles:
                    continue
                if other_health <= 0.1:
                    continue  # Don't target dying robots
            
            # Calculate distance for all potential prey
            distance = ((agent_pos[0] - other_pos[0])**2 + (agent_pos[1] - other_pos[1])**2)**0.5
            
            potential_food_sources.append({
                'position': other_pos,
                'type': 'robot',
                'source': 'prey',
                'prey_id': other_agent['id'],
                'prey_energy': other_energy,
                'prey_health': other_health,
                'distance': distance
            })
    
    if not potential_food_sources:
        # ROLE-SPECIFIC MESSAGES
        if agent_role == 'carnivore':
            return {'distance': 999999, 'food_type': 'no valid prey (herbivore/scavenger/omnivore) found', 'source_type': 'prey', 'food_position': None, 'signed_x_distance': 999999}
        elif agent_role == 'scavenger':
            return {'distance': 999999, 'food_type': 'no weakened robots (energy < 30%) found', 'source_type': 'prey', 'food_position': None, 'signed_x_distance': 999999}
        else:
            return {'distance': 999999, 'food_type': 'no environmental food sources found', 'source_type': 'environment', 'food_position': None, 'signed_x_distance': 999999}
    
    # Find the nearest food source
    best_target = None
    best_distance = 999999
    
    for target in potential_food_sources:
        distance = target['distance']
        
        if distance < best_distance:
            best_distance = distance
            best_target = target
    
    if best_target is None:
        return {'distance': 999999, 'food_type': 'none found', 'source_type': 'none', 'food_position': None, 'signed_x_distance': 999999}
    
    # Calculate signed x-axis distance
    target_pos = best_target['position']
    signed_x_distance = target_pos[0] - agent_pos[0]
    
    # Determine food type description
    if best_target.get('source') == 'prey':
        prey_id = best_target.get('prey_id', 'unknown')
        prey_energy = best_target.get('prey_energy', 0.0)
        food_type_desc = f"robot prey {prey_id} (energy: {prey_energy:.2f})"
    else:
        food_type_desc = best_target.get('type', 'unknown')
    
    return {
        'distance': best_distance,
        'signed_x_distance': signed_x_distance,
        'food_type': food_type_desc,
        'source_type': best_target.get('source', 'environment'),
        'food_position': target_pos
    }


class TestClosestFoodFunction(unittest.TestCase):
    
    def setUp(self):
        """Set up test data structures"""
        self.food_sources = []
        self.other_agents = []
        self.agent_energy_levels = {}
        self.agent_health = {}
        self.agent_statuses = {}
    
    def add_food_source(self, x, y, food_type, amount=10.0):
        """Helper to add environmental food source"""
        food = {
            'position': (x, y),
            'food_type': food_type,
            'amount': amount
        }
        self.food_sources.append(food)
        return food
    
    def add_agent(self, agent_id, x, y, role='omnivore', energy=1.0, health=1.0):
        """Helper to add other agent"""
        agent = {
            'id': agent_id,
            'position': (x, y)
        }
        self.other_agents.append(agent)
        self.agent_statuses[agent_id] = {'role': role}
        self.agent_energy_levels[agent_id] = energy
        self.agent_health[agent_id] = {'health': health}
        return agent
    
    def test_herbivore_finds_closest_environmental_food(self):
        """Test that herbivore correctly finds closest environmental food"""
        # Add food sources at different distances
        self.add_food_source(5, 0, 'plants')    # Distance: 5
        self.add_food_source(10, 0, 'seeds')    # Distance: 10
        self.add_food_source(3, 4, 'insects')   # Distance: 5 (3,4,5 triangle)
        
        result = calculate_closest_food_for_agent(
            agent_pos=(0, 0),
            agent_role='herbivore',
            agent_id='herb1',
            food_sources=self.food_sources,
            other_agents=self.other_agents,
            agent_energy_levels=self.agent_energy_levels,
            agent_health=self.agent_health,
            agent_statuses=self.agent_statuses
        )
        
        self.assertEqual(result['distance'], 5.0)
        self.assertEqual(result['food_type'], 'plants')
        self.assertEqual(result['source_type'], 'environment')
        self.assertEqual(result['signed_x_distance'], 5.0)  # Positive = right
        self.assertEqual(result['food_position'], (5, 0))
    
    def test_carnivore_ignores_environmental_food(self):
        """Test that carnivore ignores environmental food and only sees robot prey"""
        # Add environmental food (should be ignored)
        self.add_food_source(2, 0, 'plants')
        self.add_food_source(3, 0, 'meat')
        
        # Add valid prey robots
        self.add_agent('herb1', 8, 6, 'herbivore', energy=0.8, health=0.9)  # Distance: 10
        self.add_agent('omni1', 4, 3, 'omnivore', energy=0.7, health=0.8)   # Distance: 5
        
        result = calculate_closest_food_for_agent(
            agent_pos=(0, 0),
            agent_role='carnivore',
            agent_id='carn1',
            food_sources=self.food_sources,
            other_agents=self.other_agents,
            agent_energy_levels=self.agent_energy_levels,
            agent_health=self.agent_health,
            agent_statuses=self.agent_statuses
        )
        
        # Should find closest valid prey (omnivore at distance 5)
        self.assertEqual(result['distance'], 5.0)
        self.assertIn('robot prey omni1', result['food_type'])
        self.assertEqual(result['source_type'], 'prey')
        self.assertEqual(result['signed_x_distance'], 4.0)  # x-distance to omnivore
        self.assertEqual(result['food_position'], (4, 3))
    
    def test_carnivore_cannot_hunt_other_carnivores(self):
        """Test that carnivores cannot hunt other carnivores or symbionts"""
        # Add invalid prey (carnivore and symbiont)
        self.add_agent('carn2', 2, 0, 'carnivore', energy=0.9, health=0.9)
        self.add_agent('symb1', 3, 0, 'symbiont', energy=0.8, health=0.8)
        
        # Add valid prey farther away
        self.add_agent('herb1', 10, 0, 'herbivore', energy=0.8, health=0.9)
        
        result = calculate_closest_food_for_agent(
            agent_pos=(0, 0),
            agent_role='carnivore',
            agent_id='carn1',
            food_sources=self.food_sources,
            other_agents=self.other_agents,
            agent_energy_levels=self.agent_energy_levels,
            agent_health=self.agent_health,
            agent_statuses=self.agent_statuses
        )
        
        # Should find herbivore, not closer carnivore/symbiont
        self.assertEqual(result['distance'], 10.0)
        self.assertIn('robot prey herb1', result['food_type'])
    
    def test_scavenger_only_hunts_weak_robots(self):
        """Test that scavengers only target robots with energy < 0.3"""
        # Add robots with different energy levels
        self.add_agent('healthy1', 2, 0, 'herbivore', energy=0.8, health=0.9)  # Too healthy
        self.add_agent('weak1', 6, 0, 'omnivore', energy=0.2, health=0.8)      # Valid target
        self.add_agent('dying1', 4, 0, 'herbivore', energy=0.1, health=0.1)    # Valid target
        
        result = calculate_closest_food_for_agent(
            agent_pos=(0, 0),
            agent_role='scavenger',
            agent_id='scav1',
            food_sources=self.food_sources,
            other_agents=self.other_agents,
            agent_energy_levels=self.agent_energy_levels,
            agent_health=self.agent_health,
            agent_statuses=self.agent_statuses
        )
        
        # Should find dying robot (closest valid target)
        self.assertEqual(result['distance'], 4.0)
        self.assertIn('robot prey dying1', result['food_type'])
    
    def test_omnivore_chooses_closest_food_type(self):
        """Test that omnivore correctly chooses between environmental food and robot prey"""
        # Add environmental food
        self.add_food_source(3, 0, 'plants')
        
        # Add robot prey (omnivores can hunt weakened robots)
        self.add_agent('weak1', 8, 0, 'herbivore', energy=0.4, health=0.6)
        
        result = calculate_closest_food_for_agent(
            agent_pos=(0, 0),
            agent_role='omnivore',
            agent_id='omni1',
            food_sources=self.food_sources,
            other_agents=self.other_agents,
            agent_energy_levels=self.agent_energy_levels,
            agent_health=self.agent_health,
            agent_statuses=self.agent_statuses
        )
        
        # Should choose closer environmental food over distant robot
        self.assertEqual(result['distance'], 3.0)
        self.assertEqual(result['food_type'], 'plants')
        self.assertEqual(result['source_type'], 'environment')
    
    def test_no_valid_food_available(self):
        """Test robot with no valid food sources"""
        # Add only environmental food (invalid for carnivore)
        self.add_food_source(2, 0, 'plants')
        
        # Add only invalid robot prey (other carnivores)
        self.add_agent('carn2', 5, 0, 'carnivore', energy=0.8, health=0.9)
        
        result = calculate_closest_food_for_agent(
            agent_pos=(0, 0),
            agent_role='carnivore',
            agent_id='carn1',
            food_sources=self.food_sources,
            other_agents=self.other_agents,
            agent_energy_levels=self.agent_energy_levels,
            agent_health=self.agent_health,
            agent_statuses=self.agent_statuses
        )
        
        self.assertEqual(result['distance'], 999999)
        self.assertIn('no valid prey', result['food_type'])
        self.assertEqual(result['source_type'], 'prey')
        self.assertIsNone(result['food_position'])
    
    def test_signed_x_distance_calculation(self):
        """Test that signed x-distance is calculated correctly"""
        # Food to the left (negative x-distance)
        self.add_food_source(7, 5, 'plants')
        
        result = calculate_closest_food_for_agent(
            agent_pos=(10, 5),
            agent_role='herbivore',
            agent_id='robot1',
            food_sources=self.food_sources,
            other_agents=self.other_agents,
            agent_energy_levels=self.agent_energy_levels,
            agent_health=self.agent_health,
            agent_statuses=self.agent_statuses
        )
        
        self.assertEqual(result['distance'], 3.0)
        self.assertEqual(result['signed_x_distance'], -3.0)  # Negative = left
        
        # Clear and test food to the right
        self.food_sources.clear()
        self.add_food_source(15, 5, 'seeds')
        
        result = calculate_closest_food_for_agent(
            agent_pos=(10, 5),
            agent_role='herbivore',
            agent_id='robot1',
            food_sources=self.food_sources,
            other_agents=self.other_agents,
            agent_energy_levels=self.agent_energy_levels,
            agent_health=self.agent_health,
            agent_statuses=self.agent_statuses
        )
        
        self.assertEqual(result['distance'], 5.0)
        self.assertEqual(result['signed_x_distance'], 5.0)  # Positive = right
    
    def test_distance_calculation_accuracy(self):
        """Test that distance calculations are mathematically correct"""
        # Test Pythagorean theorem: 3-4-5 triangle
        self.add_food_source(3, 4, 'plants')
        
        result = calculate_closest_food_for_agent(
            agent_pos=(0, 0),
            agent_role='herbivore',
            agent_id='robot1',
            food_sources=self.food_sources,
            other_agents=self.other_agents,
            agent_energy_levels=self.agent_energy_levels,
            agent_health=self.agent_health,
            agent_statuses=self.agent_statuses
        )
        
        expected_distance = math.sqrt(3*3 + 4*4)  # Should be 5.0
        self.assertAlmostEqual(result['distance'], expected_distance, places=6)
    
    def test_carnivore_rejects_dying_robots(self):
        """Test that carnivores don't target robots with health <= 0.1"""
        # Add dying robot (should be ignored)
        self.add_agent('dying1', 2, 0, 'herbivore', energy=0.5, health=0.05)
        
        # Add healthy robot farther away
        self.add_agent('healthy1', 10, 0, 'herbivore', energy=0.8, health=0.9)
        
        result = calculate_closest_food_for_agent(
            agent_pos=(0, 0),
            agent_role='carnivore',
            agent_id='carn1',
            food_sources=self.food_sources,
            other_agents=self.other_agents,
            agent_energy_levels=self.agent_energy_levels,
            agent_health=self.agent_health,
            agent_statuses=self.agent_statuses
        )
        
        # Should target healthy robot, not closer dying one
        self.assertEqual(result['distance'], 10.0)
        self.assertIn('healthy1', result['food_type'])
    
    def test_multiple_robots_same_distance(self):
        """Test behavior when multiple valid targets are at the same distance"""
        # Add two herbivores at the same distance
        self.add_agent('herb1', 3, 4, 'herbivore', energy=0.8, health=0.9)  # Distance: 5
        self.add_agent('herb2', -3, -4, 'herbivore', energy=0.7, health=0.8) # Distance: 5
        
        result = calculate_closest_food_for_agent(
            agent_pos=(0, 0),
            agent_role='carnivore',
            agent_id='carn1',
            food_sources=self.food_sources,
            other_agents=self.other_agents,
            agent_energy_levels=self.agent_energy_levels,
            agent_health=self.agent_health,
            agent_statuses=self.agent_statuses
        )
        
        # Should find one of them (function should be deterministic based on order)
        self.assertEqual(result['distance'], 5.0)
        self.assertIn('robot prey', result['food_type'])
        self.assertTrue('herb1' in result['food_type'] or 'herb2' in result['food_type'])


if __name__ == '__main__':
    # Run the tests
    print("Running closest food calculation tests...")
    unittest.main(verbosity=2) 