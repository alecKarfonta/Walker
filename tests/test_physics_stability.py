import unittest
import sys
import os
import Box2D as b2
import numpy as np

# Adjust the path to import from the src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from agents.crawling_crate import CrawlingCrate

class TestPhysicsStability(unittest.TestCase):

    def test_robot_stability_under_max_force(self):
        """
        Tests if the Box2D-based robot remains stable when maximum force is applied.
        """
        # 1. Setup the physics world
        world = b2.b2World(gravity=(0, -10))
        
        # 2. Create the robot
        robot = CrawlingCrate(world, position=(0, 5))

        # 3. Apply maximum torque for a short duration
        max_torque_action = (2.0, 2.0) # Using the new max torque from the Box2D agent
        for _ in range(60): # Apply force for 1 second (60 steps)
            robot.apply_action(max_torque_action)
            world.Step(1.0/60.0, 8, 3)

        # 4. Check velocities
        body_vel = robot.body.linearVelocity
        upper_arm_vel = robot.upper_arm.angularVelocity
        lower_arm_vel = robot.lower_arm.angularVelocity

        # Assert that velocities are within a "sane" range and not exploding
        # These thresholds are arbitrary but should catch runaway physics.
        self.assertLess(np.linalg.norm(body_vel), 50.0, "Body velocity is too high.")
        self.assertLess(abs(upper_arm_vel), 100.0, "Upper arm angular velocity is too high.")
        self.assertLess(abs(lower_arm_vel), 100.0, "Lower arm angular velocity is too high.")

if __name__ == '__main__':
    unittest.main() 